"""
Orchestration: raw data in, fitted models out.

`build_platform()` is the single entry point used by every Streamlit page. It

    loads -> cleans -> engineers features -> applies the minimum-minutes filter
    -> computes positional percentiles and category scores
    -> fits, per position group: a scaler, a similarity engine, K-Means and PCA

and returns a `ScoutingPlatform` holding all of it. The Streamlit layer only
reads from this object; no modelling happens inside a page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import clustering as cl
from . import feature_engineering as fe
from .config import (
    DEFAULT_MIN_MINUTES,
    LEAGUE_STRENGTH,
    METRIC_LABELS,
    PERCENT_METRICS,
    POSITION_GROUPS,
    categories_for,
)
from .data_processing import CleaningReport, clean_players, load_raw_players
from .similarity import SimilarityEngine

STRENGTH_PERCENTILE = 70
WEAKNESS_PERCENTILE = 30


@dataclass
class PositionModel:
    """Fitted models for one position group."""

    group: str
    features: list[str]
    index: pd.Index
    z: pd.DataFrame
    scaler: StandardScaler
    engine: SimilarityEngine
    clusters: cl.ClusterModel
    coords: pd.DataFrame
    pca: PCA
    loadings: dict[str, list[tuple[str, float]]]

    @property
    def archetypes(self) -> pd.Series:
        return self.clusters.labels.map(self.clusters.names)


@dataclass
class ScoutingPlatform:
    """Everything the app needs, built once and cached."""

    features: pd.DataFrame            # every player-season, all engineered columns
    pool: pd.DataFrame                # the filtered comparison pool
    percentiles: pd.DataFrame         # pct_<metric>, within position group
    categories: pd.DataFrame          # cat_<Category>, mean of the basket
    models: dict[str, PositionModel]
    cleaning: CleaningReport
    min_minutes: int
    seasons: list[str] = field(default_factory=list)

    # -- lookups -------------------------------------------------------
    @property
    def player_labels(self) -> pd.Series:
        return (
            self.pool["player"] + "  -  " + self.pool["team"]
            + " (" + self.pool["position"] + ", " + self.pool["season"] + ")"
        )

    def index_for_label(self, label: str):
        matches = self.player_labels[self.player_labels == label]
        return matches.index[0] if len(matches) else None

    def row(self, index) -> pd.Series:
        return self.pool.loc[index]

    def model_for(self, index) -> PositionModel:
        return self.models[self.pool.loc[index, "position_group"]]

    def archetype(self, index) -> tuple[str, str]:
        model = self.model_for(index)
        cluster = int(model.clusters.labels.loc[index])
        return model.clusters.names[cluster], model.clusters.descriptions[cluster]

    def archetype_series(self) -> pd.Series:
        out = pd.Series(index=self.pool.index, dtype=object)
        for model in self.models.values():
            out.loc[model.index] = model.archetypes
        return out

    # -- percentile views ----------------------------------------------
    def percentile(self, index, metric: str) -> float:
        column = f"pct_{metric}"
        if column not in self.percentiles.columns:
            return float("nan")
        return float(self.percentiles.loc[index, column])

    def percentile_frame(self, index, metrics: list[str]) -> pd.DataFrame:
        rows = []
        for metric in metrics:
            column = f"pct_{metric}"
            if column not in self.percentiles.columns:
                continue
            value = self.pool.loc[index, metric] if metric in self.pool.columns else np.nan
            rows.append(
                {
                    "metric": METRIC_LABELS.get(metric, metric),
                    "key": metric,
                    "value": float(value) if pd.notna(value) else np.nan,
                    "percentile": float(self.percentiles.loc[index, column]),
                }
            )
        return pd.DataFrame(rows)

    def category_scores(self, index) -> dict[str, float]:
        group = self.pool.loc[index, "position_group"]
        out = {}
        for category in categories_for(group):
            column = f"cat_{category}"
            if column in self.categories.columns:
                value = self.categories.loc[index, column]
                if pd.notna(value):
                    out[category] = float(value)
        return out

    def strengths(self, index, n: int = 5) -> pd.DataFrame:
        frame = self._profile_frame(index)
        return frame[frame["percentile"] >= STRENGTH_PERCENTILE].head(n)

    def weaknesses(self, index, n: int = 5) -> pd.DataFrame:
        frame = self._profile_frame(index).sort_values("percentile")
        return frame[frame["percentile"] <= WEAKNESS_PERCENTILE].head(n)

    def _profile_frame(self, index) -> pd.DataFrame:
        model = self.model_for(index)
        frame = self.percentile_frame(index, model.features)
        return frame.sort_values("percentile", ascending=False)

    # -- similarity ----------------------------------------------------
    def similar(
        self,
        index,
        n: int = 10,
        metric: str = "cosine",
        candidate_mask: pd.Series | None = None,
    ) -> pd.DataFrame:
        model = self.model_for(index)
        return model.engine.neighbours(index, n=n, metric=metric, candidate_mask=candidate_mask)

    def explain_similarity(self, index_a, index_b, metric: str = "cosine"):
        return self.model_for(index_a).engine.explain(index_a, index_b, metric=metric)

    def pair_similarity(self, index_a, index_b, metric: str = "cosine") -> float:
        model = self.model_for(index_a)
        if self.pool.loc[index_b, "position_group"] != model.group:
            return float("nan")
        distances = model.engine.distance_to_all(index_a, metric=metric)
        position = model.z.index.get_loc(index_b)
        return float(model.engine.similarity_percent(distances, metric=metric)[position])


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build_features(regenerate: bool = False) -> tuple[pd.DataFrame, CleaningReport]:
    """Load, clean and engineer features for every player-season."""
    raw = load_raw_players(regenerate=regenerate)
    clean, report = clean_players(raw)
    return fe.build_features(clean), report


def build_platform(
    features: pd.DataFrame,
    report: CleaningReport,
    min_minutes: int = DEFAULT_MIN_MINUTES,
    seasons: list[str] | None = None,
    random_state: int = 42,
) -> ScoutingPlatform:
    """Filter to a comparison pool and fit every position-group model."""
    pool = features[features["minutes"] >= min_minutes]
    if seasons:
        pool = pool[pool["season"].isin(seasons)]
    pool = pool.reset_index(drop=True)
    pool["league_strength"] = pool["league"].map(LEAGUE_STRENGTH).fillna(pool.get("league_strength", 0.8))

    available = list(pool.columns)
    metrics = sorted(
        {m for group in POSITION_GROUPS for m in fe.metrics_for_percentiles(group, available)}
    )
    percentiles = fe.compute_percentiles(pool, metrics)
    categories = fe.category_scores(percentiles, pool["position_group"])

    models: dict[str, PositionModel] = {}
    for group in POSITION_GROUPS:
        subset = pool[pool["position_group"] == group]
        if len(subset) < 30:
            continue
        group_features = fe.model_features(group, available)
        z, scaler = fe.scale_features(subset, group_features)
        engine = SimilarityEngine(z, subset, random_state=random_state)
        clusters = cl.fit_clusters(z, group, random_state=random_state)
        coords, pca = cl.pca_projection(z, random_state=random_state)
        models[group] = PositionModel(
            group=group,
            features=group_features,
            index=subset.index,
            z=z,
            scaler=scaler,
            engine=engine,
            clusters=clusters,
            coords=coords,
            pca=pca,
            loadings=cl.component_loadings(pca, group_features),
        )

    platform = ScoutingPlatform(
        features=features,
        pool=pool,
        percentiles=percentiles,
        categories=categories,
        models=models,
        cleaning=report,
        min_minutes=min_minutes,
        seasons=seasons or sorted(pool["season"].unique()),
    )
    platform.pool["archetype"] = platform.archetype_series()
    return platform


def format_metric(metric: str, value) -> str:
    """Consistent display formatting for any metric in the registry."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    value = float(value)
    if metric in PERCENT_METRICS:
        return f"{value:.1f}%"
    if metric in {"minutes", "matches", "starts", "height_cm"}:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"

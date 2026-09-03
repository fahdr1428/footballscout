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
    DATA_SOURCES,
    DEFAULT_MIN_MINUTES,
    DEFAULT_WEIGHTS,
    DEFAULT_SOURCE,
    LEAGUE_STRENGTH,
    LEAGUE_TIER,
    METRIC_LABELS,
    PERCENT_METRICS,
    POSITION_GROUPS,
    categories_for,
)
from .data_processing import CleaningReport, clean_players, load_source
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
    dropped_features: list[str] = field(default_factory=list)

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
    source: str = DEFAULT_SOURCE
    leagues: list[str] = field(default_factory=list)
    peer_columns: list[str] = field(default_factory=lambda: ["position_group"])

    @property
    def peer_group_label(self) -> str:
        """How to describe the peer set percentiles are measured against."""
        if "gender" in self.peer_columns:
            return "players in the same position and the same competition type"
        return "players in the same position group"

    def peers(self, index) -> pd.DataFrame:
        """The rows a player's percentiles are actually measured against."""
        mask = pd.Series(True, index=self.pool.index)
        for column in self.peer_columns:
            mask &= self.pool[column] == self.pool.loc[index, column]
        return self.pool[mask]

    # -- source capabilities -------------------------------------------
    @property
    def spec(self):
        return DATA_SOURCES[self.source]

    @property
    def is_real(self) -> bool:
        return self.spec.kind == "real"

    def has(self, column: str) -> bool:
        """Whether this dataset actually supplies a column (age, height, PSxG)."""
        return column in self.pool.columns and bool(self.pool[column].notna().any())

    @property
    def has_age(self) -> bool:
        return self.has("age")

    def league_table(self) -> pd.DataFrame:
        """Leagues present in the pool, with their level and strength coefficient."""
        frame = (
            self.pool.groupby("league")
            .agg(
                players=("player_id", "nunique"),
                seasons=("season", "nunique"),
                clubs=("team", "nunique"),
                level=("league_tier", "first"),
                strength=("league_strength", "first"),
            )
            .reset_index()
        )
        return frame.sort_values(["strength", "players"], ascending=[False, False])

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
    def _enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add any pool column the similarity engine's own copy predates."""
        if frame.empty:
            return frame
        missing = [c for c in self.pool.columns if c not in frame.columns]
        return frame.join(self.pool[missing]) if missing else frame

    def similar(
        self,
        index,
        n: int = 10,
        metric: str = "cosine",
        candidate_mask: pd.Series | None = None,
    ) -> pd.DataFrame:
        model = self.model_for(index)
        return self._enrich(
            model.engine.neighbours(index, n=n, metric=metric, candidate_mask=candidate_mask)
        )

    def replacements(
        self,
        index,
        n: int = 15,
        similarity_weight: float = 0.5,
        candidate_mask: pd.Series | None = None,
        metric: str = "cosine",
    ) -> pd.DataFrame:
        """Who could take this player's place: like-for-like, but at least as good.

        Ranked on a blend the user controls:

            replacement score = w x similarity% + (1 - w) x role fit percentile

        With w at 1 it is a pure style match; at 0 it is "best player available
        for the role", ignoring whether they play like the incumbent at all.
        """
        from .recruitment import fit_scores

        model = self.model_for(index)
        neighbours = self._enrich(
            model.engine.neighbours(
                index, n=max(n * 6, 60), metric=metric, candidate_mask=candidate_mask
            )
        )
        if neighbours.empty:
            return neighbours

        group = self.pool.loc[index, "position_group"]
        weights = DEFAULT_WEIGHTS.get(group, {})
        fit = fit_scores(self.categories.loc[neighbours.index], weights)["fit_score"]
        incumbent_fit = float(
            fit_scores(self.categories.loc[[index]], weights)["fit_score"].iloc[0]
        )

        result = neighbours.copy()
        result["role_fit"] = fit.round(1)
        result["fit_delta"] = (fit - incumbent_fit).round(1)
        result["replacement_score"] = (
            similarity_weight * result["similarity"] + (1 - similarity_weight) * result["role_fit"]
        ).round(1)
        # One row per player: a player with two seasons in the pool should not
        # occupy two places on a shortlist.
        result = (
            result.sort_values("replacement_score", ascending=False)
            .drop_duplicates(subset=["player_id"], keep="first")
            .head(n)
        )
        result["rank"] = range(1, len(result) + 1)
        return result

    def squad(self, team: str, season: str | None = None) -> pd.DataFrame:
        """Every player-season for one club in the current pool."""
        squad = self.pool[self.pool["team"] == team]
        if season:
            squad = squad[squad["season"] == season]
        return squad.sort_values("minutes", ascending=False)

    def team_category_profile(
        self, team: str, season: str | None = None, position_groups: list[str] | None = None
    ) -> pd.DataFrame:
        """Minutes-weighted mean category percentile for a club, against its league.

        Weighting by minutes stops a fringe player's 40-minute cameo counting
        as much as a 3,000-minute season.
        """
        squad = self.squad(team, season)
        if position_groups:
            squad = squad[squad["position_group"].isin(position_groups)]
        if squad.empty:
            return pd.DataFrame()
        league = self.pool[self.pool["league"].isin(squad["league"].unique())]
        if position_groups:
            league = league[league["position_group"].isin(position_groups)]
        rows = []
        for column in [c for c in self.categories.columns if c.startswith("cat_")]:
            club_values = self.categories.loc[squad.index, column]
            weights = squad["minutes"].to_numpy(dtype=float)
            mask = club_values.notna().to_numpy()
            if not mask.any():
                continue
            rows.append(
                {
                    "category": column.removeprefix("cat_"),
                    "club": round(float(np.average(club_values[mask], weights=weights[mask])), 1),
                    "league_median": round(float(self.categories.loc[league.index, column].median()), 1),
                }
            )
        return pd.DataFrame(rows)

    def trajectory(self, index) -> pd.DataFrame:
        """The same player's category scores across every season in the pool."""
        player_id = self.pool.loc[index, "player_id"]
        seasons = self.pool[self.pool["player_id"] == player_id].sort_values("season")
        if len(seasons) < 2:
            return pd.DataFrame()
        rows = []
        for position, row in seasons.iterrows():
            record = {
                "season": row["season"],
                "team": row["team"],
                "league": row["league"],
                "minutes": row["minutes"],
                "position": row["position"],
                "archetype": row.get("archetype"),
            }
            record.update(self.category_scores(position))
            rows.append(record)
        return pd.DataFrame(rows)

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

def build_features(source: str = DEFAULT_SOURCE) -> tuple[pd.DataFrame, CleaningReport, str]:
    """Load, clean and engineer features for every player-season in a source."""
    raw, used = load_source(source)
    clean, report = clean_players(raw)
    report.source = used
    return fe.build_features(clean), report, used


def build_platform(
    features: pd.DataFrame,
    report: CleaningReport,
    min_minutes: int = DEFAULT_MIN_MINUTES,
    seasons: list[str] | None = None,
    leagues: list[str] | None = None,
    random_state: int = 42,
    source: str = DEFAULT_SOURCE,
    min_feature_coverage: float = 0.6,
) -> ScoutingPlatform:
    """Filter to a comparison pool and fit every position-group model.

    `leagues` matters more than it looks: the pool is the peer set for every
    percentile in the app. A dataset spanning men's and women's competitions
    should usually be scoped to one or the other before a percentile is read as
    a statement about a player's standing.
    """
    pool = features[features["minutes"] >= min_minutes]
    if seasons:
        pool = pool[pool["season"].isin(seasons)]
    if leagues:
        pool = pool[pool["league"].isin(leagues)]
    pool = pool.reset_index(drop=True)
    # A real feed carries its own league metadata; the simulated one is looked
    # up from the config table.
    if "league_strength" not in pool.columns or pool["league_strength"].isna().all():
        pool["league_strength"] = pool["league"].map(LEAGUE_STRENGTH).fillna(0.80)
    if "league_tier" not in pool.columns or pool["league_tier"].isna().all():
        pool["league_tier"] = pool["league"].map(LEAGUE_TIER).fillna(1).astype(int)

    available = list(pool.columns)
    metrics = sorted(
        {m for group in POSITION_GROUPS for m in fe.metrics_for_percentiles(group, available)}
    )
    # Peer group for every percentile. Where a dataset spans men's and women's
    # competitions they are separate peer groups: comparing a Frauen Bundesliga
    # midfielder's output against Premier League men would not mean anything.
    peer_columns = ["position_group"]
    if "gender" in pool.columns and pool["gender"].nunique() > 1:
        peer_columns.append("gender")
    percentiles = fe.compute_percentiles(pool, metrics, group_col=peer_columns)
    categories = fe.category_scores(percentiles, pool["position_group"])

    models: dict[str, PositionModel] = {}
    for group in POSITION_GROUPS:
        subset = pool[pool["position_group"] == group]
        if len(subset) < 30:
            continue
        # Drop features this source cannot populate for this position, rather
        # than feeding a column of imputed medians into the distance metric.
        wanted = fe.model_features(group, available)
        group_features = [
            f for f in wanted if subset[f].notna().mean() >= min_feature_coverage
        ]
        dropped = [f for f in wanted if f not in group_features]
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
            dropped_features=dropped,
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
        source=source,
        leagues=leagues or sorted(pool["league"].unique()),
        peer_columns=peer_columns,
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

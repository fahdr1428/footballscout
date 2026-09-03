"""
Model validation and sensitivity testing.

A similarity model that nobody has stress-tested is a random-number generator
with good manners. This module answers four questions:

1. **Do the clusters mean anything?** Silhouette and inertia per k, plus - only
   because the reference dataset is simulated and its generative role profiles
   are known - the adjusted Rand index between K-Means labels and those true
   profiles. On real data that check is unavailable and the report says so.
2. **Does similarity find players who actually do the same job?** For each
   player, how many of their top-k neighbours share their generative role,
   against the rate expected by chance.
3. **Is the model leaning on one or two statistics?** Mean share of pairwise
   distance carried by each feature.
4. **How stable are the rankings?** Drop a metric, or re-weight a category, and
   measure how much of the top ten survives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .config import METRIC_LABELS, VALIDATION_REPORT, categories_for
from .feature_engineering import correlation_analysis
from .pipeline import ScoutingPlatform
from .similarity import SimilarityEngine, jaccard, rank_correlation

TRUE_ROLE_COLUMN = "true_role_profile"


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------

def clustering_diagnostics(platform: ScoutingPlatform) -> pd.DataFrame:
    """Per-position clustering quality, with the true-role check where possible."""
    rows = []
    for group, model in platform.models.items():
        row = {
            "position_group": group,
            "players": len(model.index),
            "features": len(model.features),
            "k": model.clusters.k,
            "elbow_k": model.clusters.elbow_k,
            "silhouette": round(model.clusters.silhouette, 3),
            "inertia": round(model.clusters.inertia, 1),
            "smallest_cluster": int(model.clusters.labels.value_counts().min()),
        }
        truth = platform.pool.loc[model.index, TRUE_ROLE_COLUMN] if TRUE_ROLE_COLUMN in platform.pool else None
        if truth is not None and truth.notna().all():
            row["adjusted_rand_vs_true_role"] = round(
                float(adjusted_rand_score(truth, model.clusters.labels)), 3
            )
            row["cluster_purity"] = round(_purity(truth, model.clusters.labels), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _purity(truth: pd.Series, labels: pd.Series) -> float:
    """Share of players in the modal true role of their assigned cluster."""
    frame = pd.DataFrame({"truth": truth.to_numpy(), "cluster": labels.to_numpy()})
    correct = frame.groupby("cluster")["truth"].agg(lambda s: s.value_counts().iloc[0]).sum()
    return correct / len(frame)


def pca_variance(platform: ScoutingPlatform) -> pd.DataFrame:
    """How much of each position's variance the 2-D archetype map actually shows."""
    return pd.DataFrame(
        [
            {
                "position_group": group,
                "pc1_variance": round(float(model.pca.explained_variance_ratio_[0]), 3),
                "pc2_variance": round(float(model.pca.explained_variance_ratio_[1]), 3),
                "total_shown": round(float(model.pca.explained_variance_ratio_[:2].sum()), 3),
            }
            for group, model in platform.models.items()
        ]
    )


# --------------------------------------------------------------------------
# Similarity: does it match players who do the same job?
# --------------------------------------------------------------------------

def similarity_role_agreement(
    platform: ScoutingPlatform, k: int = 10, metric: str = "cosine", sample: int = 150, seed: int = 0
) -> pd.DataFrame:
    """Share of each player's top-k neighbours sharing their generative role.

    The baseline is what random selection would give: sum of squared role
    shares inside the position group. A model that is no better than the
    baseline has learned nothing about role.
    """
    if TRUE_ROLE_COLUMN not in platform.pool.columns:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    for group, model in platform.models.items():
        roles = platform.pool.loc[model.index, TRUE_ROLE_COLUMN]
        shares = roles.value_counts(normalize=True)
        baseline = float((shares**2).sum())

        queries = rng.choice(model.index.to_numpy(), size=min(sample, len(model.index)), replace=False)
        hits = []
        for index in queries:
            neighbours = model.engine.neighbours(index, n=k, metric=metric)
            if neighbours.empty:
                continue
            same = (neighbours[TRUE_ROLE_COLUMN] == roles.loc[index]).mean()
            hits.append(float(same))
        rows.append(
            {
                "position_group": group,
                "roles": int(roles.nunique()),
                f"top{k}_same_role": round(float(np.mean(hits)), 3),
                "chance_baseline": round(baseline, 3),
                "lift": round(float(np.mean(hits)) / baseline, 2) if baseline else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Real-data checks (no ground-truth labels required)
# --------------------------------------------------------------------------

def self_season_recall(
    platform: ScoutingPlatform, k: int = 10, metric: str = "cosine", seed: int = 0
) -> pd.DataFrame:
    """Does a player's *own other season* come back as one of his closest matches?

    This is the strongest validation available on real data. A player's profile
    in a neighbouring season is the one case where we know the answer: it should
    look like him. If the engine cannot find a player's own second season, the
    similarity it reports between two different players means very little.

    Chance level is k / (pool size - 1), which is why the lift column matters
    more than the raw hit rate.
    """
    pool = platform.pool
    repeats = pool[pool.duplicated(subset=["player_id"], keep=False)]
    if repeats.empty:
        return pd.DataFrame()

    rows = []
    for group, model in platform.models.items():
        members = repeats[
            (repeats["position_group"] == group) & repeats.index.isin(model.index)
        ]
        pairs = [
            (a, b)
            for _, block in members.groupby("player_id")
            for a in block.index
            for b in block.index
            if a != b
        ]
        if not pairs:
            continue
        ranks, hits = [], []
        candidates = len(model.index) - 1
        for a, b in pairs:
            distances = model.engine.distance_to_all(a, metric=metric)
            order = pd.Series(distances, index=model.z.index).drop(index=a).sort_values()
            if b not in order.index:
                continue
            rank = int(order.index.get_loc(b)) + 1
            ranks.append(rank)
            hits.append(rank <= k)
        if not ranks:
            continue
        chance = k / max(candidates, 1)
        rows.append(
            {
                "position_group": group,
                "player_seasons_tested": len(ranks),
                "candidates": candidates,
                f"own_season_in_top{k}": round(float(np.mean(hits)), 3),
                "chance": round(chance, 3),
                "lift": round(float(np.mean(hits)) / chance, 1) if chance else np.nan,
                "median_rank": int(np.median(ranks)),
            }
        )
    return pd.DataFrame(rows)


def team_mate_bias(
    platform: ScoutingPlatform, k: int = 10, metric: str = "cosine", sample: int = 150, seed: int = 0
) -> pd.DataFrame:
    """How often a player's nearest neighbours are his own team-mates.

    Team style leaks into individual numbers - a defender in a possession side
    passes more because of the side, not the defender. If team-mates are wildly
    over-represented in the top ten, the engine is partly matching on club
    rather than on player.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for group, model in platform.models.items():
        members = platform.pool.loc[model.index]
        queries = rng.choice(model.index.to_numpy(), size=min(sample, len(model.index)), replace=False)
        shares, baselines = [], []
        for index in queries:
            team = platform.pool.loc[index, "team"]
            season = platform.pool.loc[index, "season"]
            neighbours = model.engine.neighbours(index, n=k, metric=metric)
            if neighbours.empty:
                continue
            shares.append(float((neighbours["team"] == team).mean()))
            same_club = ((members["team"] == team) & (members["season"] == season)).sum() - 1
            baselines.append(same_club / max(len(members) - 1, 1))
        if not shares:
            continue
        rows.append(
            {
                "position_group": group,
                f"team_mates_in_top{k}": round(float(np.mean(shares)), 3),
                "chance": round(float(np.mean(baselines)), 3),
                "lift": round(float(np.mean(shares)) / max(float(np.mean(baselines)), 1e-9), 1),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Feature dominance
# --------------------------------------------------------------------------

def feature_dominance(
    platform: ScoutingPlatform, group: str, pairs: int = 4000, seed: int = 0
) -> pd.DataFrame:
    """Mean share of the pairwise distance carried by each feature.

    With equal weights and uncorrelated features every metric would carry
    1/n_features of the distance. Anything far above that line is effectively
    steering the similarity model on its own.
    """
    model = platform.models[group]
    z = model.z.to_numpy()
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(z), pairs)
    b = rng.integers(0, len(z), pairs)
    keep = a != b
    squared = (z[a[keep]] - z[b[keep]]) ** 2
    shares = squared / squared.sum(axis=1, keepdims=True)
    even = 1 / len(model.features)
    return (
        pd.DataFrame(
            {
                "feature": model.features,
                "metric": [METRIC_LABELS.get(f, f) for f in model.features],
                "mean_distance_share": shares.mean(axis=0).round(4),
                "vs_even_share": (shares.mean(axis=0) / even).round(2),
            }
        )
        .sort_values("mean_distance_share", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# Sensitivity testing
# --------------------------------------------------------------------------

def _engine_without(platform: ScoutingPlatform, group: str, drop: list[str]) -> SimilarityEngine:
    model = platform.models[group]
    keep = [f for f in model.features if f not in drop]
    return SimilarityEngine(model.z[keep], platform.pool.loc[model.index])


def _engine_weighted(platform: ScoutingPlatform, group: str, weights: dict[str, float]) -> SimilarityEngine:
    model = platform.models[group]
    return SimilarityEngine(model.z, platform.pool.loc[model.index], weights=weights)


def _compare(
    platform: ScoutingPlatform, group: str, other: SimilarityEngine, k: int, sample: int, seed: int,
    metric: str = "cosine",
) -> dict[str, float]:
    model = platform.models[group]
    rng = np.random.default_rng(seed)
    queries = rng.choice(model.index.to_numpy(), size=min(sample, len(model.index)), replace=False)
    overlaps, correlations = [], []
    for index in queries:
        base = list(model.engine.neighbours(index, n=k, metric=metric).index)
        alt = list(other.neighbours(index, n=k, metric=metric).index)
        overlaps.append(jaccard(base, alt))
        correlations.append(rank_correlation(base, alt))
    return {
        "top_k_overlap": round(float(np.mean(overlaps)), 3),
        "rank_correlation": round(float(np.nanmean(correlations)), 3),
    }


def drop_metric_sensitivity(
    platform: ScoutingPlatform, group: str, metrics: list[str], k: int = 10,
    sample: int = 80, seed: int = 0,
) -> pd.DataFrame:
    """How much of the top-k survives when one metric is removed."""
    rows = []
    for metric in metrics:
        if metric not in platform.models[group].features:
            continue
        result = _compare(platform, group, _engine_without(platform, group, [metric]), k, sample, seed)
        rows.append({"removed": METRIC_LABELS.get(metric, metric), **result})
    return pd.DataFrame(rows).sort_values("top_k_overlap")


def reweight_sensitivity(
    platform: ScoutingPlatform, group: str, factor: float = 3.0, k: int = 10,
    sample: int = 80, seed: int = 0,
) -> pd.DataFrame:
    """How much of the top-k survives when one category is up-weighted."""
    model = platform.models[group]
    rows = []
    for category, metrics in categories_for(group).items():
        present = [m for m in metrics if m in model.features]
        if not present:
            continue
        weights = {f: (factor if f in present else 1.0) for f in model.features}
        result = _compare(platform, group, _engine_weighted(platform, group, weights), k, sample, seed)
        rows.append({"category_weighted_x{:g}".format(factor): category, **result})
    return pd.DataFrame(rows).sort_values("top_k_overlap")


def correlation_summary(platform: ScoutingPlatform, group: str, threshold: float = 0.85):
    model = platform.models[group]
    return correlation_analysis(platform.pool.loc[model.index], model.features, threshold=threshold)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "|".join([" --- "] * len(frame.columns)) + "|"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, divider, *rows])


def build_validation_report(
    platform: ScoutingPlatform, sensitivity_groups: list[str] | None = None
) -> str:
    """Assemble the full markdown validation report."""
    sensitivity_groups = sensitivity_groups or ["W", "CB", "DM"]
    parts = ["# Model validation report", ""]
    parts.append(
        f"Pool: **{len(platform.pool):,} player-seasons**, minimum "
        f"**{platform.min_minutes:,} minutes**, seasons {', '.join(platform.seasons)}."
    )
    parts.append(
        "All models are fitted per position group. Nothing below is tuned to make the numbers "
        "look better; where a result is weak it is reported and interpreted."
    )

    parts += ["", "## 1. Clustering", ""]
    parts.append(_md_table(clustering_diagnostics(platform)))
    parts.append("")
    parts.append(
        "Silhouette scores in the 0.10-0.25 range are typical for football style data and should "
        "be read honestly: playing styles form a **continuum**, not well-separated groups. "
        "K-Means here is a useful summary of that continuum, not evidence that discrete player "
        "types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the "
        "best score, with the inertia elbow reported alongside as a cross-check."
    )
    parts.append("")
    parts.append(
        "`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role "
        "profiles used to *generate* the simulated dataset. They are only computable because the "
        "sample data is simulated - on a real feed there is no ground truth, and these columns "
        "would be absent."
    )

    parts += ["", "### Variance shown by the 2-D archetype map", ""]
    parts.append(_md_table(pca_variance(platform)))
    parts.append("")
    parts.append(
        "The cluster map compresses 12-22 features into two axes, so a large share of the "
        "variance is not on screen. Two players sitting close together on the map are not "
        "necessarily close in the full feature space - the similarity table is the authority."
    )

    agreement = similarity_role_agreement(platform)
    if not agreement.empty:
        parts += ["", "## 2. Does similarity find players who do the same job?", ""]
        parts.append(_md_table(agreement))
        parts.append("")
        parts.append(
            "`top10_same_role` is the share of a player's ten nearest neighbours drawn from the "
            "same generative role; `chance_baseline` is what random picking would produce given "
            "the role mix in that position. A lift above 1 means the model is recovering role, "
            "not noise."
        )

    recall = self_season_recall(platform)
    if not recall.empty:
        parts += ["", "## 2b. Does the engine recognise the same player twice?", ""]
        parts.append(_md_table(recall))
        parts.append("")
        parts.append(
            "For every player with two seasons in the pool, this asks where his *other* season "
            "ranks among his nearest neighbours. It is the only case where the right answer is "
            "known without any labels, which makes it the check that also works on real data. "
            "`chance` is what random ordering would give."
        )

    bias = team_mate_bias(platform)
    if not bias.empty:
        parts += ["", "## 2c. Is the engine matching on club rather than player?", ""]
        parts.append(_md_table(bias))
        parts.append("")
        parts.append(
            "Team style leaks into individual numbers: a defender in a possession side passes "
            "more because of the side. Some over-representation of team-mates is expected and "
            "correct; a large lift would mean the model is partly clustering clubs."
        )

    parts += ["", "## 3. Is the model dominated by a few metrics?", ""]
    for group in sensitivity_groups:
        if group not in platform.models:
            continue
        dominance = feature_dominance(platform, group)
        even = 1 / len(platform.models[group].features)
        parts.append(f"**{group}** - even share would be {even:.3f} per feature.")
        parts.append("")
        parts.append(_md_table(dominance.head(6)[["metric", "mean_distance_share", "vs_even_share"]]))
        parts.append("")

    parts += ["", "## 4. Sensitivity of the similarity rankings", ""]
    for group in sensitivity_groups:
        if group not in platform.models:
            continue
        model = platform.models[group]
        candidates = [f for f in model.features][:8]
        parts.append(f"### {group} - removing one metric")
        parts.append("")
        parts.append(_md_table(drop_metric_sensitivity(platform, group, candidates)))
        parts.append("")
        parts.append(f"### {group} - tripling the weight on one category")
        parts.append("")
        parts.append(_md_table(reweight_sensitivity(platform, group)))
        parts.append("")
    parts.append(
        "`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; "
        "`rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose "
        "removal drops the overlap below ~0.5 is effectively steering that position's model."
    )

    parts += ["", "## 5. Correlated features", ""]
    for group in sensitivity_groups:
        if group not in platform.models:
            continue
        _corr, pairs = correlation_summary(platform, group)
        parts.append(f"**{group}** - pairs with |r| >= 0.85:")
        parts.append("")
        if pairs:
            parts.append(
                _md_table(
                    pd.DataFrame(
                        [
                            {
                                "metric A": METRIC_LABELS.get(a, a),
                                "metric B": METRIC_LABELS.get(b, b),
                                "r": r,
                            }
                            for a, b, r in pairs
                        ]
                    )
                )
            )
        else:
            parts.append("_None._")
        parts.append("")
    parts.append(
        "Highly correlated features double-count one idea inside a Euclidean distance. They are "
        "reported rather than silently dropped, because for a scout 'progressive passes' and "
        "'passes into the final third' are different questions even when they move together."
    )

    parts += ["", "## Known limitations", ""]
    parts += [
        "- The bundled dataset is **simulated**. Absolute values are plausible but they are not "
        "real players, and no conclusion about a real footballer can be drawn from them.",
        "- League strength coefficients are editable assumptions in `src/config.py`, not measured "
        "quantities. Every score that uses them says so.",
        "- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.",
        "- Positions come from the dataset. A player who changed role mid-season is compared "
        "against the peer group of his listed position, which will understate him.",
        "- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.",
    ]
    return "\n".join(parts)


def write_validation_report(platform: ScoutingPlatform, path=None) -> str:
    path = path or VALIDATION_REPORT
    report = build_validation_report(platform)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    return report

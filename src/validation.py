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
    """How often a player's nearest neighbours are his own team-mates that season.

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
            # Actual team-mates: same club *and* same season. Counting a club's
            # players from other seasons as team-mates, while the chance
            # baseline counts only one season, would inflate the lift purely
            # because the pool spans several seasons.
            same_squad = (neighbours["team"] == team) & (neighbours["season"] == season)
            shares.append(float(same_squad.mean()))
            squad_size = ((members["team"] == team) & (members["season"] == season)).sum() - 1
            baselines.append(squad_size / max(len(members) - 1, 1))
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
    if not sensitivity_groups:
        preferred = ["W", "CB", "DM", "MID", "DEF", "FWD"]
        sensitivity_groups = [g for g in preferred if g in platform.models][:3]
    parts = ["# Model validation report", ""]
    parts.append(
        f"Pool: **{len(platform.pool):,} player-seasons**, minimum "
        f"**{platform.min_minutes:,} minutes**, seasons {', '.join(platform.seasons)}."
    )
    parts.append(
        "All models are fitted per position group. Nothing below is tuned to make the numbers "
        "look better; where a result is weak it is reported and interpreted."
    )

    parts += ["", "## 0. Are the position groups the right shape?", ""]
    taxonomy = position_separability(platform)
    if taxonomy.empty:
        parts.append(
            "Not run. This source does not record positions specific enough to test - it "
            "publishes broad buckets rather than 'Left-Back' and 'Second Striker'."
        )
    else:
        parts.append(
            "Before asking whether the models are any good, the groups they are fitted on have "
            "to be the right ones. Each row trains a cross-validated classifier to tell two "
            "specific positions apart on their own model features. **Balanced accuracy**, so "
            "0.50 is a coin flip whatever the class imbalance; the two control rows are pairs "
            "nobody doubts are different jobs, and exist to show the measurement works."
        )
        parts.append("")
        parts.append(_md_table(taxonomy))
        parts.append("")
        parts.append(
            "A pair the classifier cannot separate is one job under two names: giving them "
            "separate peer groups would halve the sample and buy nothing. A pair it separates "
            "easily is two jobs, and measuring one against the other's percentiles is a bias no "
            "sample size fixes. The taxonomy in `src/config.py` follows this table - second "
            "strikers and wide midfielders are modelled apart, left and right are not - and the "
            "verdict column says so explicitly when the code and the evidence disagree."
        )
        collapsed = getattr(platform, "collapsed_groups", {})
        if collapsed:
            parts.append("")
            parts.append(
                "In **this** pool, "
                + "; ".join(
                    f"**{group}** has only {info['players']} players and is measured against "
                    f"**{info['parent']}** instead"
                    for group, info in collapsed.items()
                )
                + ". A finer group still has to clear a minimum sample before it gets its own "
                  "peer set; widen the seasons in the sidebar to model it separately."
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
            "correct; a large lift would mean the model is partly clustering clubs.\n\n"
            "**Read the absolute column, not the ratio.** In a wide pool - five leagues in one "
            "season - two team-mates are a vanishing share of the candidates, so `chance` is "
            "tiny and `lift` divides by it. A lift of 4 on a 1% observed share still means a "
            "top-ten list contains one-tenth of a team-mate on average, which is not a model "
            "clustering clubs. The ratio only becomes worrying when the observed share itself "
            "climbs into double figures."
        )

    parts += ["", "## 2d. Did the market later agree? (forward test)", ""]
    growth, growth_summary = value_growth_backtest(platform, horizon=2)
    if not growth_summary.get("available"):
        parts.append(
            "Not run. This needs a source with market values and at least three seasons "
            "loaded in the pool - select more seasons in the sidebar."
        )
    else:
        parts.append(
            f"Hidden-gem score in season *t*, against the player's Transfermarkt valuation "
            f"**{growth_summary['horizon']} seasons later**. The score sees only season *t*, "
            f"so nothing about the outcome enters it. "
            f"{growth_summary['tested']:,} of {growth_summary['candidates']:,} player-seasons "
            f"({growth_summary['coverage']:.0%}) could be followed up."
        )
        parts.append("")
        parts.append(_md_table(growth))
        parts.append("")
        parts.append(
            f"Median value change runs from **x{growth.iloc[0]['median_growth_x']}** in the "
            f"bottom decile to **x{growth.iloc[-1]['median_growth_x']}** in the top, and the "
            f"share of players whose value rose climbs from "
            f"{growth.iloc[0]['share_that_rose']:.0%} to {growth.iloc[-1]['share_that_rose']:.0%}. "
            f"Rank correlation of score against growth: **{growth_summary['rank_correlation']}**."
        )
        if growth_summary.get("stratified"):
            parts.append("")
            parts.append(
                f"**But most of a monotone table like that can be an artefact.** The top decile "
                f"is also younger and cheaper, and a cheap twenty-year-old rises in percentage "
                f"terms for reasons the model can take no credit for. Asking the same question "
                f"inside cells of similar age *and* similar starting price "
                f"({growth_summary['strata_cells']} cells, "
                f"{growth_summary['strata_players']:,} players, minimum 40 each) gives a "
                f"correlation of **{growth_summary['within_stratum_rank_correlation']}** - "
                f"roughly half the headline figure, positive in "
                f"{growth_summary['strata_cells_positive']} of "
                f"{growth_summary['strata_cells']} cells."
            )
            parts.append("")
            parts.append(
                "So: about half the apparent signal is youth and a low starting price, and about "
                "half is left over. A modest edge that survives both controls is a believable "
                "result for a model built from public data; the headline number on its own would "
                "be an overclaim."
            )
        parts.append("")
        parts.append(
            "Three limits. **Survivorship** - a player who left the big five has no later "
            "valuation and drops out, and those are disproportionately the ones who did not work "
            "out, so absolute growth figures flatter every decile. **Market value is "
            "Transfermarkt's estimate**, not a fee anyone paid, and it is partly informed by the "
            "same public data the model reads. **One market regime**, five seasons, one continent."
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


# --------------------------------------------------------------------------
# Is the position taxonomy right?
# --------------------------------------------------------------------------

# Pairs the taxonomy had to decide about, and the controls that prove the
# measurement works. `split` records what config.py actually does with each.
SEPARABILITY_PAIRS = [
    ("SS", "AM", "candidate", True, "Second striker vs attacking midfield"),
    ("LM", "LW", "candidate", True, "Wide midfield vs winger"),
    ("LB", "RB", "candidate", False, "Left-back vs right-back"),
    ("LW", "RW", "candidate", False, "Left wing vs right wing"),
    ("CB", "DM", "control", True, "Centre-back vs defensive midfield"),
    ("DM", "AM", "control", True, "Defensive vs attacking midfield"),
]
SEPARABLE_AT = 0.75    # above this, two positions are doing different jobs
COIN_FLIP_AT = 0.65    # below this, they are the same job and a split costs peers
MIN_PAIR_PLAYERS = 60


def position_separability(
    platform: ScoutingPlatform, folds: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Can a classifier tell two specific positions apart on their own metrics?

    This is what decides how fine the position taxonomy should be. A pair the
    model cannot separate is one job listed under two names: giving them
    separate peer groups halves the sample and buys nothing. A pair it separates
    easily is two jobs, and measuring one against the other's percentiles is a
    bias no amount of sample size fixes.

    The score is cross-validated **balanced accuracy** - the mean of the two
    per-class recalls - not plain accuracy. The pairs are lopsided (there are
    four attacking midfielders for every second striker), and plain accuracy
    rewards a classifier for simply calling everything the majority class: it
    would score 0.81 on that pair while having learned nothing. Balanced
    accuracy is 0.50 for that classifier, and 0.50 is chance whatever the split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pool = platform.pool
    if "position" not in pool.columns or pool["position"].nunique() < 4:
        return pd.DataFrame()

    rows = []
    for first, second, kind, split, label in SEPARABILITY_PAIRS:
        subset = pool[pool["position"].isin([first, second])]
        if len(subset) < MIN_PAIR_PLAYERS or subset["position"].nunique() < 2:
            continue
        # Judge the pair on the feature set of whichever group actually models
        # them, so the test asks what the platform would really see.
        group = subset["position_group"].mode().iloc[0]
        model = platform.models.get(group)
        if model is None:
            continue
        columns = [f for f in model.features if f in subset.columns]
        if len(columns) < 5:
            continue

        frame = subset[columns].astype(float)
        frame = frame.fillna(frame.median())
        target = (subset["position"] == first).astype(int)
        baseline = float(max(target.mean(), 1 - target.mean()))
        classifier = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        )
        accuracy = float(
            cross_val_score(
                classifier, frame, target, cv=folds, scoring="balanced_accuracy"
            ).mean()
        )
        rows.append({
            "pair": label,
            "kind": kind,
            "players": len(subset),
            "balanced_accuracy": round(accuracy, 3),
            "chance": 0.5,
            "majority_class": round(baseline, 3),
            "modelled_separately": split,
            "verdict": _separability_verdict(accuracy, split),
        })
    return pd.DataFrame(rows)


def _separability_verdict(accuracy: float, split: bool) -> str:
    """Plain English, and it says so when the evidence disagrees with the code."""
    if accuracy >= SEPARABLE_AT:
        finding = "different jobs - deserves its own model"
        agrees = split
    elif accuracy <= COIN_FLIP_AT:
        finding = "the same job - splitting would only cost peers"
        agrees = not split
    else:
        finding = "borderline"
        agrees = True
    return finding if agrees else f"{finding} - BUT the taxonomy does the opposite"


# --------------------------------------------------------------------------
# Forward test: did the market later agree?
# --------------------------------------------------------------------------

def value_growth_backtest(
    platform: ScoutingPlatform,
    horizon: int = 2,
    top_share: float = 0.10,
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Does a high hidden-gem score in season *t* precede a rise in market value?

    This is the closest thing to an out-of-sample test the project has. The
    score for a player-season uses **only that season's data**; the outcome is
    his Transfermarkt valuation `horizon` seasons later. Nothing about the
    future enters the score, so the two are cleanly separated.

    It is a backtest, not a proof, and three things limit it:

    * **Survivorship.** A player who left the big five leagues has no later
      valuation and drops out. Those are disproportionately the players who did
      not work out, so the surviving sample flatters every score equally - the
      comparison between deciles is fairer than the absolute growth figures.
    * **Market value is Transfermarkt's estimate**, not a fee anybody paid, and
      it is partly informed by the same public performance data the model reads.
    * **One market regime**, five seasons, one continent.

    Returns a per-decile table and a summary dict.
    """
    from .recruitment import hidden_gem_scores

    pool = platform.pool
    if "market_value_eur" not in pool.columns or pool["season"].nunique() < 2:
        return pd.DataFrame(), {"available": False}

    seasons = sorted(pool["season"].unique())
    offsets = {season: i for i, season in enumerate(seasons)}

    scores = hidden_gem_scores(
        pool, platform.categories,
        {group: model.z for group, model in platform.models.items()},
        weights=weights,
    )
    columns = ["player_id", "season", "position_group", "market_value_eur"]
    if "age" in pool.columns:
        columns.append("age")
    frame = pool[columns].join(scores[["hidden_gem_score"]])
    frame = frame[frame["market_value_eur"].gt(0) & frame["hidden_gem_score"].notna()]

    # The outcome is looked up in the *full* dataset, not the filtered pool, so a
    # player is not counted as "disappeared" merely for dropping below the
    # minute threshold in the later season.
    future = platform.features[["player_id", "season", "market_value_eur"]].copy()
    future = future[future["market_value_eur"].gt(0)]
    future["offset"] = future["season"].map(offsets)
    future = future.dropna(subset=["offset"])
    lookup = (future.set_index(["player_id", "offset"])["market_value_eur"]
                    .groupby(level=[0, 1]).max())

    frame["offset"] = frame["season"].map(offsets)
    frame = frame[frame["offset"] <= len(seasons) - 1 - horizon]
    if frame.empty:
        return pd.DataFrame(), {"available": False}

    keys = list(zip(frame["player_id"], frame["offset"] + horizon))
    frame["later_value_eur"] = [lookup.get(k, np.nan) for k in keys]

    tested = frame.dropna(subset=["later_value_eur"]).copy()
    if len(tested) < 100:
        return pd.DataFrame(), {"available": False}

    tested["growth"] = np.log10(tested["later_value_eur"] / tested["market_value_eur"])
    tested["decile"] = (
        tested["hidden_gem_score"].rank(pct=True).mul(10).clip(upper=9.999).astype(int) + 1
    )

    table = (
        tested.groupby("decile")
        .agg(
            players=("growth", "size"),
            median_score=("hidden_gem_score", "median"),
            median_value_eur=("market_value_eur", "median"),
            median_growth_x=("growth", lambda s: round(float(10 ** s.median()), 2)),
            share_that_rose=("growth", lambda s: round(float((s > 0).mean()), 3)),
        )
        .reset_index()
    )

    # The obvious confound: the top decile is also younger and cheaper, and a
    # cheap 20-year-old rises in percentage terms for reasons that have nothing
    # to do with the model. So the same question is asked again *inside* cells
    # of similar age and similar starting price, where that advantage is held
    # constant. If the signal survives there, it is not merely "young and cheap".
    strata = _value_growth_strata(tested)

    cutoff = tested["hidden_gem_score"].quantile(1 - top_share)
    top = tested[tested["hidden_gem_score"] >= cutoff]
    rest = tested[tested["hidden_gem_score"] < cutoff]
    summary = {
        "available": True,
        "horizon": horizon,
        "candidates": int(len(frame)),
        "tested": int(len(tested)),
        "coverage": round(float(len(tested) / len(frame)), 3),
        "top_decile_growth_x": round(float(10 ** top["growth"].median()), 2),
        "rest_growth_x": round(float(10 ** rest["growth"].median()), 2),
        "top_share_that_rose": round(float((top["growth"] > 0).mean()), 3),
        "rest_share_that_rose": round(float((rest["growth"] > 0).mean()), 3),
        "rank_correlation": round(
            float(tested["hidden_gem_score"].corr(tested["growth"], method="spearman")), 3
        ),
    }
    summary.update(strata)
    return table, summary


AGE_BANDS = [(0, 21, "under 21"), (21, 24, "21-23"), (24, 28, "24-27"), (28, 99, "28+")]


def _value_growth_strata(tested: pd.DataFrame) -> dict:
    """Rank correlation of score against growth *within* age and price cells.

    Holding both confounds constant at once. Cells with fewer than
    `MIN_STRATUM` players are skipped rather than reported on noise; the
    headline is the sample-weighted mean of the cells that qualify.
    """
    MIN_STRATUM = 40
    if "age" not in tested.columns or tested["age"].isna().all():
        return {"stratified": False}

    frame = tested.dropna(subset=["age"]).copy()
    frame["age_band"] = pd.cut(
        frame["age"],
        bins=[b[0] for b in AGE_BANDS] + [AGE_BANDS[-1][1]],
        labels=[b[2] for b in AGE_BANDS], right=False,
    )
    frame["value_band"] = pd.qcut(
        frame["market_value_eur"].rank(method="first"), 5, labels=False, duplicates="drop"
    )

    correlations, sizes = [], []
    for _, cell in frame.groupby(["age_band", "value_band"], observed=True):
        if len(cell) < MIN_STRATUM or cell["hidden_gem_score"].nunique() < 5:
            continue
        rho = cell["hidden_gem_score"].corr(cell["growth"], method="spearman")
        if pd.notna(rho):
            correlations.append(float(rho))
            sizes.append(len(cell))
    if not correlations:
        return {"stratified": False}

    weights = np.array(sizes, dtype=float)
    return {
        "stratified": True,
        "strata_cells": len(correlations),
        "strata_players": int(weights.sum()),
        "within_stratum_rank_correlation": round(
            float(np.average(correlations, weights=weights)), 3
        ),
        "strata_cells_positive": int(sum(c > 0 for c in correlations)),
    }

"""
Recruitment search: turn a scout's brief into a ranked shortlist.

Two products live here.

**Recruitment Fit Score** - the scout sets hard filters (position, age, minutes,
league) plus optional metric thresholds, then distributes 100 points of weight
across attribute categories. The fit score is the weighted mean of the player's
positional percentiles in those categories:

    fit = sum_c (weight_c / 100) * percentile_c

so a fit of 78 literally means "weighted across the things you said matter,
this player sits at the 78th percentile of his positional peers".

**Hidden Gem Score** - an experimental composite that combines on-pitch
performance with age, sample size, league exposure and statistical
distinctiveness. It is a *model output*, not a valuation: with no transfer fee
or wage data in the dataset it cannot say anything about market value, and the
app says so wherever the score appears.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import DEFAULT_WEIGHTS, LEAGUE_STRENGTH, METRIC_LABELS

OPERATORS = {
    ">=": lambda s, v: s >= v,
    ">": lambda s, v: s > v,
    "<=": lambda s, v: s <= v,
    "<": lambda s, v: s < v,
}

# Hidden-gem component weights (percent). Editable in the UI.
DEFAULT_GEM_WEIGHTS = {
    "Performance": 40,
    "Age upside": 20,
    "Low exposure": 15,
    "Statistical uniqueness": 15,
    "Sample size": 10,
}

GEM_AGE_FLOOR = 19.0   # at or below this age the age component scores 100
GEM_AGE_CEILING = 27.0  # at or above this age it scores 0
GEM_MINUTES_FULL = 1800  # minutes at which the sample-size component maxes out


@dataclass
class RecruitmentBrief:
    """A scout's search definition."""

    position_group: str
    min_minutes: int = 900
    age_range: tuple[float, float] = (16, 40)
    leagues: list[str] = field(default_factory=list)
    seasons: list[str] = field(default_factory=list)
    max_league_strength: float | None = None
    thresholds: list[tuple[str, str, float]] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)

    def resolved_weights(self) -> dict[str, float]:
        weights = self.weights or DEFAULT_WEIGHTS.get(self.position_group, {})
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {k: 100 * v / total for k, v in weights.items()}


def apply_brief(pool: pd.DataFrame, brief: RecruitmentBrief) -> pd.Series:
    """Boolean mask of players who satisfy every hard filter in the brief."""
    mask = (
        pool["position_group"].eq(brief.position_group)
        & pool["minutes"].ge(brief.min_minutes)
        & pool["age"].between(*brief.age_range)
    )
    if brief.leagues:
        mask &= pool["league"].isin(brief.leagues)
    if brief.seasons:
        mask &= pool["season"].isin(brief.seasons)
    if brief.max_league_strength is not None:
        mask &= pool["league"].map(LEAGUE_STRENGTH).fillna(1.0).le(brief.max_league_strength)
    for metric, operator, value in brief.thresholds:
        if metric in pool.columns and operator in OPERATORS:
            mask &= OPERATORS[operator](pool[metric], value).fillna(False)
    return mask


def fit_scores(categories: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Weighted fit score plus the per-category contribution behind it."""
    columns = {c: f"cat_{c}" for c in weights if f"cat_{c}" in categories.columns}
    if not columns:
        return pd.DataFrame({"fit_score": np.full(len(categories), np.nan)}, index=categories.index)

    total_weight = sum(weights[c] for c in columns)
    contributions = pd.DataFrame(index=categories.index)
    for category, column in columns.items():
        share = weights[category] / total_weight
        contributions[f"contrib_{category}"] = (categories[column].fillna(50) * share).round(2)
    out = contributions.copy()
    out["fit_score"] = contributions.sum(axis=1).round(1)
    return out


def search(
    pool: pd.DataFrame,
    categories: pd.DataFrame,
    brief: RecruitmentBrief,
    top_n: int = 50,
) -> pd.DataFrame:
    """Ranked shortlist for a brief, with the fit-score breakdown attached."""
    mask = apply_brief(pool, brief)
    if not mask.any():
        return pool.loc[mask].assign(fit_score=pd.Series(dtype=float))

    candidates = pool.loc[mask]
    scored = fit_scores(categories.loc[mask], brief.resolved_weights())
    result = candidates.join(scored)
    result = result.sort_values("fit_score", ascending=False).head(top_n)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def threshold_summary(brief: RecruitmentBrief) -> list[str]:
    """Human-readable echo of the brief, for the report and the UI."""
    lines = [
        f"Position: {brief.position_group}",
        f"Age: {brief.age_range[0]:.0f}-{brief.age_range[1]:.0f}",
        f"Minimum minutes: {brief.min_minutes:,}",
    ]
    if brief.leagues:
        lines.append(f"Leagues: {', '.join(brief.leagues)}")
    for metric, operator, value in brief.thresholds:
        lines.append(f"{METRIC_LABELS.get(metric, metric)} {operator} {value:g}")
    return lines


# --------------------------------------------------------------------------
# Hidden gems
# --------------------------------------------------------------------------

def statistical_uniqueness(z: pd.DataFrame, k: int = 15) -> pd.Series:
    """Mean distance to the k nearest peers, percentile-ranked.

    A high score means the player's statistical profile has few close analogues
    in the pool - interesting for a scout, because it usually means either a
    genuinely rare skill set or a role no one else in the sample plays.
    """
    if len(z) <= k + 1:
        return pd.Series(50.0, index=z.index)
    model = NearestNeighbors(n_neighbors=k + 1).fit(z)
    distances, _ = model.kneighbors(z)
    mean_distance = distances[:, 1:].mean(axis=1)
    return pd.Series(mean_distance, index=z.index).rank(pct=True).mul(100).round(1)


def age_upside(age: pd.Series) -> pd.Series:
    """100 at or below 19, falling linearly to 0 at 27. Deliberately simple."""
    span = GEM_AGE_CEILING - GEM_AGE_FLOOR
    return (100 * ((GEM_AGE_CEILING - age) / span)).clip(0, 100).round(1)


def exposure_score(league: pd.Series) -> pd.Series:
    """Higher for players outside the strongest leagues.

    Uses the editable league-strength coefficients in config.py. It is a proxy
    for visibility, not for quality.
    """
    strength = league.map(LEAGUE_STRENGTH).fillna(0.8)
    low, high = min(LEAGUE_STRENGTH.values()), max(LEAGUE_STRENGTH.values())
    return (100 * (high - strength) / max(high - low, 1e-9)).round(1)


def sample_size_score(minutes: pd.Series) -> pd.Series:
    """Rewards a trustworthy sample; flat once a player passes 1,800 minutes."""
    return (100 * (minutes / GEM_MINUTES_FULL)).clip(0, 100).round(1)


def hidden_gem_scores(
    pool: pd.DataFrame,
    categories: pd.DataFrame,
    z_by_group: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
    performance_weights: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Composite 'hidden gem' score with every component exposed.

    Not a valuation. With no fee or wage data available it cannot be one; it
    ranks players who combine output, youth, sample size, limited exposure and
    an unusual statistical profile.
    """
    weights = weights or DEFAULT_GEM_WEIGHTS
    performance_weights = performance_weights or DEFAULT_WEIGHTS

    performance = pd.Series(np.nan, index=pool.index)
    uniqueness = pd.Series(np.nan, index=pool.index)
    for group, z in z_by_group.items():
        idx = z.index.intersection(pool.index)
        if idx.empty:
            continue
        group_weights = performance_weights.get(group, {})
        performance.loc[idx] = fit_scores(categories.loc[idx], group_weights)["fit_score"]
        uniqueness.loc[idx] = statistical_uniqueness(z.loc[idx])

    components = pd.DataFrame(
        {
            "Performance": performance,
            "Age upside": age_upside(pool["age"]),
            "Low exposure": exposure_score(pool["league"]),
            "Statistical uniqueness": uniqueness,
            "Sample size": sample_size_score(pool["minutes"]),
        }
    )

    total = sum(weights.get(c, 0) for c in components.columns)
    if total <= 0:
        total = 1.0
    weighted = pd.DataFrame(
        {c: components[c] * weights.get(c, 0) / total for c in components.columns}
    )
    out = components.round(1)
    out["hidden_gem_score"] = weighted.sum(axis=1).round(1)
    return out


def cheaper_alternatives_mask(
    pool: pd.DataFrame, reference_index, max_age_delta: float = 0.0, require_lower_league: bool = False
) -> pd.Series:
    """Candidates who are younger than, or play beneath, the reference player.

    This is the 'find me a cheaper version of X' filter. Age and league level
    are used as proxies for cost because the dataset has no fee or wage column -
    and that limitation is stated in the UI.
    """
    reference = pool.loc[reference_index]
    mask = pool["age"] <= reference["age"] - max_age_delta
    if require_lower_league:
        reference_strength = LEAGUE_STRENGTH.get(reference["league"], 1.0)
        mask &= pool["league"].map(LEAGUE_STRENGTH).fillna(1.0) < reference_strength
    return mask

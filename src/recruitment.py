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
from .data_processing import age_mask

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
    "Underpriced for output": 20,
    "Low exposure": 15,
    "Value for money": 15,
    "Statistical uniqueness": 15,
    "Sample size": 10,
}

GEM_AGE_FLOOR = 19.0   # at or below this age the age component scores 100
GEM_AGE_CEILING = 27.0  # at or above this age it scores 0
GEM_MINUTES_FULL = 1800  # minutes at which the sample-size component maxes out
MIN_VALUE_FIT_PLAYERS = 30  # players a position group needs before a price line is fitted


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
    role: str | None = None                 # which template the weights started from
    max_market_value: float | None = None   # euros, where the source has a valuation
    feet: list[str] = field(default_factory=list)
    # Side of the pitch and inverted-foot are filters rather than models: a
    # left-back and a right-back share a peer set because a classifier cannot
    # tell them apart, but a shortlist for a left-back should hold left-backs.
    positions: list[str] = field(default_factory=list)
    flanks: list[str] = field(default_factory=list)
    footed_sides: list[str] = field(default_factory=list)

    def resolved_weights(self) -> dict[str, float]:
        weights = self.weights or DEFAULT_WEIGHTS.get(self.position_group, {})
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {k: 100 * v / total for k, v in weights.items()}


def _league_strength(pool: pd.DataFrame) -> pd.Series:
    """League coefficients from the data where present, else the config table."""
    if "league_strength" in pool.columns and pool["league_strength"].notna().any():
        return pool["league_strength"].astype(float)
    return pool["league"].map(LEAGUE_STRENGTH).fillna(0.80)


def apply_brief(pool: pd.DataFrame, brief: RecruitmentBrief) -> pd.Series:
    """Boolean mask of players who satisfy every hard filter in the brief."""
    mask = pool["position_group"].eq(brief.position_group) & pool["minutes"].ge(brief.min_minutes)
    if "age" in pool.columns and pool["age"].notna().any():
        mask &= age_mask(pool["age"], brief.age_range)
    if brief.leagues:
        mask &= pool["league"].isin(brief.leagues)
    if brief.seasons:
        mask &= pool["season"].isin(brief.seasons)
    if brief.max_league_strength is not None:
        mask &= _league_strength(pool).le(brief.max_league_strength)
    if brief.max_market_value is not None and "market_value_eur" in pool.columns:
        # A player with no recorded valuation is kept: absence of a price is not
        # evidence of an unaffordable one, and the shortlist flags it as unknown.
        value = pd.to_numeric(pool["market_value_eur"], errors="coerce")
        mask &= value.le(brief.max_market_value) | value.isna()
    if brief.feet and "foot" in pool.columns:
        mask &= pool["foot"].astype(str).str.lower().isin([f.lower() for f in brief.feet])
    for column, wanted in (("position", brief.positions), ("flank", brief.flanks),
                           ("footed_side", brief.footed_sides)):
        if wanted and column in pool.columns:
            mask &= pool[column].isin(wanted)
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
    if brief.role:
        lines.append(f"Role: {brief.role}")
    if brief.leagues:
        lines.append(f"Leagues: {', '.join(brief.leagues)}")
    if brief.max_market_value is not None:
        lines.append(f"Maximum market value: EUR {brief.max_market_value / 1e6:,.1f}m")
    if brief.feet:
        lines.append(f"Preferred foot: {', '.join(brief.feet)}")
    if brief.positions:
        lines.append(f"Specific position: {', '.join(brief.positions)}")
    if brief.flanks:
        lines.append(f"Side: {', '.join(brief.flanks)}")
    if brief.footed_sides:
        lines.append(f"Foot vs side: {', '.join(brief.footed_sides)}")
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


def exposure_score(strength: pd.Series) -> pd.Series:
    """Higher for players outside the strongest leagues in the pool.

    A proxy for visibility, not for quality, and only as good as the
    league-strength coefficients it is scaled against.
    """
    strength = strength.astype(float)
    low, high = float(strength.min()), float(strength.max())
    if high - low < 1e-9:
        return pd.Series(50.0, index=strength.index)
    return (100 * (high - strength) / (high - low)).round(1)


def value_for_money(performance: pd.Series, price: pd.Series) -> pd.Series:
    """Percentile of performance per unit of price.

    The obvious scouting question: how much on-pitch output is this player
    returning for what he costs? What "cost" means depends on the source, and
    the app says which one it used - a real Transfermarkt market value in euros
    on the big-five dataset, or the fantasy game's own valuation on the Premier
    League one, which is a popularity signal rather than a fee or a wage.
    """
    price = pd.to_numeric(price, errors="coerce")
    ratio = performance / price.where(price > 0)
    return ratio.rank(pct=True).mul(100).round(1)


def market_value_residual(
    performance: pd.Series, market_value: pd.Series, groups: pd.Series
) -> pd.DataFrame:
    """What the market pays for this level of performance, and who is off the line.

    A ratio of output to price answers "who is cheap", which mostly finds
    players who are cheap because they are not very good. The recruitment
    question is different: **for a player performing this well, is this price
    normal?**

    So, within each position group, log10(market value) is fitted against the
    performance score by ordinary least squares - one straight line, two
    coefficients - and each player's residual is read off it. A player far
    below the line is cheaper than the market usually charges for his output.

    Returns the fitted expectation and the residual as a percentile, both of
    which the app shows, alongside the line itself, so the arithmetic is
    checkable rather than asserted. It is a description of one season's prices,
    not a valuation model: the market may be right and the player limited in
    ways these metrics do not see.
    """
    value = pd.to_numeric(market_value, errors="coerce")
    score = pd.to_numeric(performance, errors="coerce")
    expected = pd.Series(np.nan, index=value.index, dtype=float)
    residual = pd.Series(np.nan, index=value.index, dtype=float)

    for group in groups.dropna().unique():
        mask = (groups == group) & value.gt(0) & score.notna()
        if mask.sum() < MIN_VALUE_FIT_PLAYERS:
            continue
        x = score[mask].to_numpy(dtype=float)
        y = np.log10(value[mask].to_numpy(dtype=float))
        if np.ptp(x) < 1e-9:
            # Every player scored the same: there is no line to fit, and
            # forcing one through a vertical scatter invents a ranking.
            continue
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        expected.loc[mask] = np.power(10.0, fitted)
        residual.loc[mask] = y - fitted

    out = pd.DataFrame({
        "expected_market_value_eur": expected.round(0),
        "value_residual": residual.round(3),
    })
    # Low residual = cheaper than the market charges for that output, so the
    # percentile is inverted to make 100 mean "most underpriced".
    out["value_residual_pct"] = (
        (1 - residual.rank(pct=True)).mul(100).round(1)
    )
    return out


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

    Not a transfer valuation. It ranks players who combine output, youth,
    sample size, limited exposure and an unusual statistical profile - and,
    where the source carries a real market value, how far below the market's
    own price-for-performance line they sit. Every component is listed in the
    app with its weight, and any the source cannot support is dropped and the
    remaining weights renormalised rather than filled in with a placeholder.
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

    columns = {
        "Performance": performance,
        "Statistical uniqueness": uniqueness,
        "Sample size": sample_size_score(pool["minutes"]),
    }
    # Every other component only exists where the source supports it. When one
    # does not, it is dropped and the remaining weights are renormalised rather
    # than a placeholder being invented.
    if "age" in pool.columns and pool["age"].notna().any():
        columns["Age upside"] = age_upside(pool["age"])

    strength = _league_strength(pool)
    if strength.nunique() > 1:
        # Meaningless in a single-league pool: everyone has the same exposure.
        columns["Low exposure"] = exposure_score(strength)

    # A real market value in euros beats a fantasy price wherever one exists.
    if "market_value_eur" in pool.columns and pool["market_value_eur"].notna().any():
        columns["Value for money"] = value_for_money(performance, pool["market_value_eur"])
        priced = market_value_residual(performance, pool["market_value_eur"],
                                       pool["position_group"])
        if priced["value_residual_pct"].notna().any():
            columns["Underpriced for output"] = priced["value_residual_pct"]
    elif "price_m" in pool.columns and pool["price_m"].notna().any():
        columns["Value for money"] = value_for_money(performance, pool["price_m"])

    components = pd.DataFrame(columns)
    # Renormalise per player, over the components *he* has. A column-level
    # renormalisation is not enough once a component is missing for some
    # players only - an unknown age, a player with no valuation - because
    # summing past the gap and dividing by the full weight total scores that
    # player as though he had earned zero on it, which is a penalty for a
    # hole in the data rather than anything about him.
    w = pd.Series({c: float(weights.get(c, 0)) for c in components.columns})
    present = components.notna()
    weighted_sum = components.fillna(0).mul(w, axis=1).sum(axis=1)
    weight_used = present.mul(w, axis=1).sum(axis=1)
    out = components.round(1)
    out["hidden_gem_score"] = (weighted_sum / weight_used.where(weight_used > 0)).round(1)
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
    mask = pd.Series(True, index=pool.index)
    if "age" in pool.columns and pd.notna(reference.get("age")):
        mask &= pool["age"] <= reference["age"] - max_age_delta
    if require_lower_league:
        strength = _league_strength(pool)
        mask &= strength < float(strength.loc[reference_index])
    return mask

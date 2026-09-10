"""
Feature engineering.

Turns cleaned season totals into the quantities the models actually use:

* per-90 rates for every counting stat (totals are never compared directly);
* success percentages computed with **empirical-Bayes shrinkage** so a player
  who won 3 of 3 tackles does not outrank a player who won 60 of 90;
* possession-adjusted defensive volume, because a defender in a 65%-possession
  side simply gets fewer chances to make a tackle;
* positional percentiles, always computed against positional peers inside the
  filtered pool - never against the whole dataset;
* z-scored feature matrices used by the similarity engine and K-Means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    COUNTING_STATS,
    LOWER_IS_BETTER,
    NO_PER90,
    POSITION_FEATURES,
    RATIO_METRICS,
    categories_for,
    per90,
)

# Defensive volume metrics worth expressing per 50% of opponent possession.
PADJ_METRICS = [
    "tackles_per90",
    "interceptions_per90",
    "blocks_per90",
    "clearances_per90",
    "ball_recoveries_per90",
    "pressures_per90",
    "defensive_actions_per90",
]

DEFENSIVE_ACTION_PARTS = ["tackles", "interceptions", "blocks", "clearances"]
PROGRESSIVE_ACTION_PARTS = ["progressive_passes", "progressive_carries"]

# Shrinkage strength for xG-per-shot style ratios (in "pseudo-attempts").
XG_PER_SHOT_PRIOR_SHOTS = 15


def add_per90(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `<stat>_per90` column for every counting stat."""
    df = df.copy()
    exposure = df["minutes"] / 90.0
    rates = {
        per90(stat): df[stat] / exposure
        for stat in COUNTING_STATS
        if stat not in NO_PER90 and stat in df.columns
    }
    if not rates:
        return df
    # Built as one block: a hundred separate inserts fragments the frame.
    return pd.concat([df, pd.DataFrame(rates, index=df.index)], axis=1)


def _shrunk_rate(
    successes: pd.Series, attempts: pd.Series, groups: pd.Series, prior_weight: float
) -> pd.Series:
    """Empirical-Bayes rate: (successes + k*prior) / (attempts + k).

    `prior` is the positional pooled rate, `k` the prior weight in attempts.
    With many attempts the observed rate dominates; with few, the estimate is
    pulled back to what a typical player in that position does.
    """
    successes = pd.to_numeric(successes, errors="coerce")
    attempts = pd.to_numeric(attempts, errors="coerce")
    if attempts.fillna(0).sum() <= 0:
        # The source does not measure this attempt count at all, so the rate is
        # unmeasurable rather than zero - leave it missing and let the model
        # drop the feature.
        return pd.Series(np.nan, index=successes.index)

    group_attempts = attempts.groupby(groups).transform("sum")
    group_successes = successes.groupby(groups).transform("sum")
    pooled = (group_successes / group_attempts.where(group_attempts > 0)).replace(
        [np.inf, -np.inf], np.nan
    )
    pooled = pooled.fillna(successes.sum() / attempts.sum())
    rate = (successes + prior_weight * pooled) / (attempts + prior_weight)
    return rate.replace([np.inf, -np.inf], np.nan)


def add_ratio_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add every success percentage in the registry, with shrinkage applied."""
    df = df.copy()
    df["_aerials_total"] = df.get("aerials_won", 0) + df.get("aerials_lost", 0)
    if {"gk_saves", "gk_goals_against"}.issubset(df.columns):
        df["_saves_plus_conceded"] = df["gk_saves"].fillna(0) + df["gk_goals_against"].fillna(0)
    groups = df["position_group"]
    for key, (num, den, prior_weight, _label) in RATIO_METRICS.items():
        if num not in df.columns or den not in df.columns:
            continue
        rate = _shrunk_rate(df[num], df[den], groups, prior_weight)
        # A goalkeeping ratio is meaningless for an outfielder.
        if key.startswith("gk_"):
            rate = rate.where(groups.eq("GK"))
        df[key] = (rate * 100).round(2)
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Composite and difference metrics that are not plain rates."""
    df = df.copy()
    exposure = df["minutes"] / 90.0
    groups = df["position_group"]

    df["npxg_per_shot"] = _shrunk_rate(
        df["npxg"], df["shots"], groups, XG_PER_SHOT_PRIOR_SHOTS
    ).round(4)
    # Finishing over/under-performance. Descriptive, and noisy in one season:
    # the app says so wherever it is shown.
    df["np_goals_minus_npxg_per90"] = ((df["np_goals"] - df["npxg"]) / exposure).round(3)
    df["aerials_contested_per90"] = (df["_aerials_total"] / exposure).round(3)
    df["defensive_actions_per90"] = (
        df[DEFENSIVE_ACTION_PARTS].sum(axis=1) / exposure
    ).round(3)
    df["progressive_actions_per90"] = (
        df[PROGRESSIVE_ACTION_PARTS].sum(axis=1) / exposure
    ).round(3)

    # How much of a player's passing happens with an opponent closing him down -
    # role and team context as much as skill, which is why it sits beside the
    # under-pressure completion rate rather than inside it.
    if "passes_under_pressure" in df.columns:
        df["pressured_pass_share"] = (
            100 * df["passes_under_pressure"] / df["passes_attempted"].replace(0, np.nan)
        ).round(2)
    if "npxg_open_play" in df.columns:
        df["open_play_npxg_share"] = (
            100 * df["npxg_open_play"] / df["npxg"].replace(0, np.nan)
        ).round(2)

    # Summary-feed derivations. Each is guarded so a source that lacks the
    # inputs simply does not get the column.
    matches = df.get("matches")
    if matches is not None:
        appearances = pd.to_numeric(matches, errors="coerce").replace(0, np.nan)
        if "clean_sheets" in df.columns:
            df["clean_sheet_rate"] = (100 * df["clean_sheets"] / appearances).round(2)
        if "starts" in df.columns:
            df["starts_share"] = (100 * df["starts"] / appearances).clip(upper=100).round(2)
    if {"gk_saves", "gk_goals_against"}.issubset(df.columns):
        # Shots faced is not published by this feed; saves plus goals conceded
        # is the closest honest denominator, and the metric is named for it.
        df["_saves_plus_conceded"] = df["gk_saves"].fillna(0) + df["gk_goals_against"].fillna(0)
    if {"goals_per90", "assists_per90"}.issubset(df.columns):
        df["goal_involvements_per90"] = (df["goals_per90"] + df["assists_per90"]).round(3)
    if {"xg_per90", "xa_per90"}.issubset(df.columns):
        df["xgi_per90"] = (df["xg_per90"] + df["xa_per90"]).round(3)

    # Positional signature. A player's zone shares say where he plays, which is
    # what separates two centre-backs with identical tackle counts - one winning
    # the ball on the halfway line, one on his own six-yard box. These are
    # shares of the player's own activity, so they do not scale with minutes.
    zones = ["touches_def_third", "touches_mid_third", "touches_att_third"]
    if set(zones).issubset(df.columns):
        total_zone_touches = df[zones].sum(axis=1, min_count=len(zones)).replace(0, np.nan)
        df["att_third_touch_share"] = (100 * df["touches_att_third"] / total_zone_touches).round(2)
        df["def_third_touch_share"] = (100 * df["touches_def_third"] / total_zone_touches).round(2)
    if {"touches_att_pen", "touches"}.issubset(df.columns):
        df["box_touch_share"] = (
            100 * df["touches_att_pen"] / df["touches"].replace(0, np.nan)
        ).round(2)
    thirds = ["tackles_def_third", "tackles_mid_third", "tackles_att_third"]
    if set(thirds).issubset(df.columns):
        total_tackles = df[thirds].sum(axis=1, min_count=len(thirds)).replace(0, np.nan)
        df["att_third_tackle_share"] = (100 * df["tackles_att_third"] / total_tackles).round(2)

    # How far the ball actually travels per action - a long-passing centre-back
    # and a sideways one can attempt the same number of passes.
    if {"progressive_pass_distance", "passes_attempted"}.issubset(df.columns):
        df["pass_progress_per_pass"] = (
            df["progressive_pass_distance"] / df["passes_attempted"].replace(0, np.nan)
        ).round(2)
    if {"progressive_carry_distance", "carries"}.issubset(df.columns):
        df["carry_progress_per_carry"] = (
            df["progressive_carry_distance"] / df["carries"].replace(0, np.nan)
        ).round(2)
    if {"sca_from_set_piece", "sca"}.issubset(df.columns):
        df["set_piece_sca_share"] = (
            100 * df["sca_from_set_piece"] / df["sca"].replace(0, np.nan)
        ).round(2)

    gk = groups.eq("GK")
    df["gk_psxg_minus_ga_per90"] = np.where(
        gk, (df.get("gk_psxg", np.nan) - df.get("gk_goals_against", np.nan)) / exposure, np.nan
    ).round(3)
    return df.drop(columns=[c for c in ["_aerials_total", "_saves_plus_conceded"] if c in df.columns])


def add_possession_adjusted(df: pd.DataFrame) -> pd.DataFrame:
    """Possession-adjusted defensive volume.

        padj_x = x_per90 * 50 / (100 - team_possession)

    i.e. the rate the player would post if their side spent half the game out
    of possession. This is the standard pAdj formulation and is only applied to
    defensive *volume* metrics, never to success percentages.
    """
    df = df.copy()
    if "team_possession" not in df.columns:
        return df
    opponent_possession = (100 - df["team_possession"]).clip(lower=20)
    factor = 50.0 / opponent_possession
    df["possession_adjustment_factor"] = factor.round(3)
    for metric in PADJ_METRICS:
        if metric in df.columns:
            df[f"padj_{metric}"] = (df[metric] * factor).round(3)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline: per-90 -> ratios -> derived -> possession-adjusted."""
    out = add_per90(df)
    out = add_ratio_metrics(out)
    out = add_derived_metrics(out)
    out = add_possession_adjusted(out)
    return out


# --------------------------------------------------------------------------
# Percentiles and category scores
# --------------------------------------------------------------------------

def model_features(position_group: str, available: list[str] | None = None) -> list[str]:
    """Feature list for a position group, deduplicated and restricted to columns that exist."""
    features = list(dict.fromkeys(POSITION_FEATURES[position_group]))
    if available is None:
        return features
    present = set(available)
    return [f for f in features if f in present]


def metrics_for_percentiles(position_group: str, available: list[str]) -> list[str]:
    """Every metric we want a positional percentile for."""
    wanted = set(model_features(position_group, available))
    for metrics in categories_for(position_group).values():
        wanted.update(metrics)
    extra = [
        "np_goals_per90", "goals_per90", "assists_per90", "xa_per90", "npxg_per90",
        "shots_per90", "key_passes_per90", "sca_per90", "progressive_passes_per90",
        "progressive_carries_per90", "progressive_actions_per90", "defensive_actions_per90",
        "pass_pct", "aerial_win_pct", "dribbles_completed_per90", "touches_att_pen_per90",
        "ball_recoveries_per90", "interceptions_per90", "tackles_per90",
    ]
    wanted.update(extra)
    return sorted(w for w in wanted if w in set(available))


def compute_percentiles(
    pool: pd.DataFrame, metrics: list[str], group_col: str | list[str] = "position_group"
) -> pd.DataFrame:
    """Percentile rank (0-100) of each metric **within each peer group**.

    The peer group is the position group, and any further column passed in -
    on a dataset spanning men's and women's competitions the peer group is
    position *and* competition, because a percentile is a claim about where a
    player stands among players they could actually be compared with.

    Ranks are used rather than a normal-distribution assumption because most
    football rates are right-skewed. Metrics in `LOWER_IS_BETTER` are inverted
    so that 99 always means "one of the best in this position".
    """
    metrics = [m for m in metrics if m in pool.columns]
    out = pd.DataFrame(index=pool.index)
    grouped = pool.groupby(group_col, observed=True)
    for metric in metrics:
        ranks = grouped[metric].rank(pct=True, na_option="keep") * 100
        if metric in LOWER_IS_BETTER:
            ranks = 100 - ranks
        out[f"pct_{metric}"] = ranks.round(1)
    return out


def category_scores(
    percentiles: pd.DataFrame, position_groups: pd.Series
) -> pd.DataFrame:
    """Attribute-category scores: the mean positional percentile of a basket.

    Every radar axis and every recruitment weight in the app is one of these,
    so a "Dribbling 82" always means "the mean of this player's percentiles for
    the dribbling metrics is 82".
    """
    out = pd.DataFrame(index=percentiles.index)
    for group in position_groups.unique():
        mask = position_groups.eq(group)
        for category, metrics in categories_for(group).items():
            columns = [f"pct_{m}" for m in metrics if f"pct_{m}" in percentiles.columns]
            if not columns:
                continue
            column = f"cat_{category}"
            if column not in out.columns:
                out[column] = np.nan
            out.loc[mask, column] = percentiles.loc[mask, columns].mean(axis=1).round(1)
    return out


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------

def scale_features(
    pool: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, StandardScaler]:
    """Z-score the feature matrix within the pool (median-filled, then scaled)."""
    matrix = pool[features].astype(float).replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median())
    matrix = matrix.fillna(0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return pd.DataFrame(scaled, index=pool.index, columns=features), scaler


def correlation_analysis(
    pool: pd.DataFrame, features: list[str], threshold: float = 0.85
) -> tuple[pd.DataFrame, list[tuple[str, str, float]]]:
    """Correlation matrix plus the pairs above `threshold`.

    Highly correlated features effectively double-weight one idea in a distance
    calculation, so the app surfaces them instead of hiding them.
    """
    matrix = pool[features].astype(float)
    corr = matrix.corr()
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((a, b, round(float(value), 3)))
    pairs.sort(key=lambda item: -abs(item[2]))
    return corr, pairs

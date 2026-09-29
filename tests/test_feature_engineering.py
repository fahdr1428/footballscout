"""Per-90s, shrinkage, possession adjustment and percentiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import LOWER_IS_BETTER
from src.feature_engineering import (
    add_per90, category_scores, compute_percentiles, metrics_for_percentiles, model_features,
    scale_features,
)


def test_per90_is_total_over_minutes(cleaned):
    clean, _ = cleaned
    out = add_per90(clean)
    expected = clean["shots"] / clean["minutes"] * 90
    assert np.allclose(out["shots_per90"], expected)


def test_shrinkage_pulls_small_samples_towards_the_positional_rate(features):
    """A player with very few attempts must not post an extreme success rate."""
    low = features[features["dribbles_attempted"] <= 3]
    high = features[features["dribbles_attempted"] >= 60]
    assert len(low) > 0 and len(high) > 0
    assert low["dribble_success_pct"].std() < high["dribble_success_pct"].std()
    # and nothing reaches an unshrunk 0% or 100%
    assert features["dribble_success_pct"].between(1, 99).all()


def test_ratios_stay_in_range(features):
    for column in ["pass_pct", "aerial_win_pct", "tackle_win_pct", "shot_accuracy_pct"]:
        values = features[column].dropna()
        assert values.between(0, 100).all()


def test_possession_adjustment_direction(features):
    """A side that has more of the ball gives its defenders fewer chances to act."""
    high_possession = features[features["team_possession"] > 58]
    low_possession = features[features["team_possession"] < 44]
    assert (
        (high_possession["padj_tackles_per90"] / high_possession["tackles_per90"]).mean()
        > (low_possession["padj_tackles_per90"] / low_possession["tackles_per90"]).mean()
    )


def test_percentiles_are_computed_within_position_group(features):
    pool = features[features["minutes"] >= 900].reset_index(drop=True)
    percentiles = compute_percentiles(pool, ["progressive_passes_per90"])
    for group, subset in pool.groupby("position_group"):
        values = percentiles.loc[subset.index, "pct_progressive_passes_per90"].dropna()
        if len(values) > 20:
            assert values.min() < 10 and values.max() > 90  # a full 0-100 range per group


def test_lower_is_better_metrics_are_inverted(features):
    """Within a position group, more miscontrols must mean a lower percentile."""
    pool = features[features["minutes"] >= 900].reset_index(drop=True)
    metric = "miscontrols_per90"
    assert metric in LOWER_IS_BETTER
    percentiles = compute_percentiles(pool, [metric])
    joined = pool[[metric, "position_group"]].join(percentiles)
    for _group, subset in joined.groupby("position_group"):
        if len(subset) > 30:
            # Rank-based percentiles invert monotonically, so Spearman is exactly -1.
            assert subset[metric].corr(subset[f"pct_{metric}"], method="spearman") < -0.999


def test_category_scores_are_the_mean_of_their_metric_percentiles(features):
    pool = features[features["minutes"] >= 900].reset_index(drop=True)
    metrics = sorted({m for g in pool["position_group"].unique()
                      for m in metrics_for_percentiles(g, list(pool.columns))})
    percentiles = compute_percentiles(pool, metrics)
    categories = category_scores(percentiles, pool["position_group"])
    wingers = pool.index[pool["position_group"] == "W"]
    index = wingers[0]
    from src.config import OUTFIELD_CATEGORIES
    columns = [f"pct_{m}" for m in OUTFIELD_CATEGORIES["Dribbling"] if f"pct_{m}" in percentiles]
    expected = percentiles.loc[index, columns].mean()
    assert abs(categories.loc[index, "cat_Dribbling"] - expected) < 0.1


def test_scaling_produces_zero_mean_unit_variance(features):
    pool = features[(features["minutes"] >= 900) & (features["position_group"] == "CB")]
    z, _scaler = scale_features(pool, model_features("CB", list(pool.columns)))
    assert np.allclose(z.mean(), 0, atol=1e-8)
    assert np.allclose(z.std(ddof=0), 1, atol=1e-8)


def test_pressure_and_set_piece_shares_are_percentages_of_their_base(features):
    pressured = features[features["passes_attempted"] > 0]
    expected = 100 * pressured["passes_under_pressure"] / pressured["passes_attempted"]
    assert np.allclose(pressured["pressured_pass_share"], expected.round(2), atol=0.01)
    assert pressured["pressured_pass_share"].between(0, 100).all()

    shooting = features[features["npxg"] > 0]
    assert shooting["open_play_npxg_share"].between(0, 100).all()


def test_pass_completion_falls_under_pressure(features):
    """A sanity check on the metric's direction, not on any one player."""
    pool = features[features["passes_under_pressure"] >= 100]
    assert len(pool) > 50
    assert pool["pass_pct_under_pressure"].mean() < pool["pass_pct"].mean()


def test_a_composite_with_an_unmeasured_part_is_missing_not_zero(cleaned):
    """A row sum skips missing parts, so an unmeasured season read "0.00 defensive
    actions" at the 50th percentile. A composite needs every one of its parts."""
    from src.feature_engineering import build_features

    clean, _ = cleaned
    frame = clean.copy()
    blank_all = frame.index[:10]          # nothing counted
    blank_one = frame.index[10:20]        # carries not counted, passes are
    frame.loc[blank_all, ["tackles", "interceptions", "blocks", "clearances"]] = np.nan
    frame.loc[blank_one, "progressive_carries"] = np.nan
    out = build_features(frame)
    assert out.loc[blank_all, "defensive_actions_per90"].isna().all()
    assert out.loc[blank_one, "progressive_actions_per90"].isna().all()
    measured = out.index[20:]
    assert out.loc[measured, "defensive_actions_per90"].notna().all()


def test_a_metric_that_repeats_earns_weight_and_a_noisy_one_does_not():
    """Repeatability is what separates describing a player from describing a season."""
    from src.feature_engineering import feature_reliability, reliability_weights

    rng = np.random.default_rng(1)
    n = 200
    talent = rng.normal(size=n)
    rows = []
    for season in ["2021-22", "2022-23"]:
        rows.append(pd.DataFrame({
            "player_id": [f"p{i}" for i in range(n)], "season": season,
            "position_group": "CB", "minutes": 2000,
            "stable": talent + rng.normal(scale=0.2, size=n),   # the player
            "noise": rng.normal(size=n),                        # the season
        }))
    frame = pd.concat(rows, ignore_index=True)
    rel = feature_reliability(frame, "CB", ["stable", "noise"])
    assert rel["stable"] > 0.9
    assert abs(rel["noise"]) < 0.2
    w = reliability_weights(rel, ["stable", "noise", "unmeasured"])
    assert w["stable"] > 20 * w["noise"]
    assert w["unmeasured"] == pytest.approx(np.median([w["stable"], w["noise"]]))
    assert reliability_weights({}, ["a", "b"]) == {"a": 1.0, "b": 1.0}


def test_a_club_metric_is_not_credited_with_the_clubs_stability():
    """Goals conceded repeats because the keeper stays at the club; it takes the
    median weight of the player metrics however repeatable it looks."""
    from src.feature_engineering import reliability_weights

    rel = {"save_rate": 0.3, "gk_psxg_minus_ga_per90": 0.5, "gk_goals_against_per90": 0.9,
           "clean_sheet_rate": 0.8}
    w = reliability_weights(rel, list(rel))
    player = [0.3 ** 2, 0.5 ** 2]
    assert w["gk_goals_against_per90"] == pytest.approx(np.median(player))
    assert w["clean_sheet_rate"] == pytest.approx(np.median(player))
    assert w["gk_psxg_minus_ga_per90"] == pytest.approx(0.25)


def test_a_shift_across_the_whole_league_is_not_read_as_players_changing():
    """Standardised within season: a provider switch that halves every value
    must not make a perfectly stable metric look unrepeatable."""
    from src.feature_engineering import feature_reliability

    rng = np.random.default_rng(2)
    base = rng.normal(loc=5, size=120)
    frame = pd.concat([
        pd.DataFrame({"player_id": range(120), "season": "2021-22", "position_group": "DM",
                      "minutes": 1500, "m": base}),
        pd.DataFrame({"player_id": range(120), "season": "2022-23", "position_group": "DM",
                      "minutes": 1500, "m": base * 0.5 + 1}),
    ])
    assert feature_reliability(frame, "DM", ["m"])["m"] > 0.99


def test_the_similarity_engine_is_weighted_by_repeatability(platform):
    model = platform.models["CB"]
    assert model.reliability, "a two-season fixture has players to measure repeatability on"
    weights = model.engine.weights
    assert weights.max() > weights.min()        # not the old equal weighting
    assert weights.sum() == pytest.approx(1.0)

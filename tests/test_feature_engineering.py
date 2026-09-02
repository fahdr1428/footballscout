"""Per-90s, shrinkage, possession adjustment and percentiles."""

from __future__ import annotations

import numpy as np
import pandas as pd

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

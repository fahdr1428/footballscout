"""Similarity engine: metric properties, explanations and stability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.similarity import SimilarityEngine, jaccard, rank_correlation


def test_a_player_is_perfectly_similar_to_himself(platform):
    model = platform.models["W"]
    index = model.index[0]
    distances = model.engine.distance_to_all(index)
    similarity = model.engine.similarity_percent(distances)
    position = model.z.index.get_loc(index)
    assert similarity[position] == pytest.approx(100.0)


def test_distance_is_symmetric(platform):
    model = platform.models["CB"]
    a, b = model.index[0], model.index[7]
    ab = model.engine.distance_to_all(a)[model.z.index.get_loc(b)]
    ba = model.engine.distance_to_all(b)[model.z.index.get_loc(a)]
    assert abs(ab - ba) < 1e-9


def test_similarity_is_bounded_and_ordered(platform):
    model = platform.models["FW"]
    results = model.engine.neighbours(model.index[3], n=10)
    assert results["similarity"].between(0, 100).all()
    assert results["similarity"].is_monotonic_decreasing
    assert list(results["rank"]) == list(range(1, len(results) + 1))


def test_neighbours_never_include_the_query_player(platform):
    model = platform.models["CM"]
    index = model.index[5]
    results = model.engine.neighbours(index, n=15)
    assert index not in results.index
    assert platform.pool.loc[index, "player_id"] not in set(results["player_id"])


def test_candidate_filters_do_not_change_the_scores(platform):
    """Filtering candidates must never re-scale anyone's similarity."""
    model = platform.models["W"]
    index = model.index[2]
    everyone = model.engine.neighbours(index, n=200)
    mask = platform.pool["age"] <= 24
    filtered = model.engine.neighbours(index, n=200, candidate_mask=mask)
    shared = everyone.index.intersection(filtered.index)
    assert len(shared) > 5
    assert np.allclose(
        everyone.loc[shared, "similarity"], filtered.loc[shared, "similarity"]
    )


def test_explanation_covers_every_feature_and_sums_to_one(platform):
    model = platform.models["AM"]
    a, b = model.index[0], model.index[4]
    result = platform.explain_similarity(a, b)
    assert set(result.contributions["feature"]) == set(model.features)
    assert abs(result.contributions["distance_share"].sum() - 1.0) < 1e-9
    for row in result.differences:
        assert row["abs_z_gap"] >= 0.6
    for row in result.matches:
        assert row["abs_z_gap"] < 0.6


def test_weighting_a_feature_changes_the_ranking(platform):
    """Sensitivity testing relies on weights actually doing something."""
    model = platform.models["W"]
    index = model.index[1]
    base = list(model.engine.neighbours(index, n=10).index)
    weights = {f: (10.0 if f == "np_goals_per90" else 1.0) for f in model.features}
    weighted = SimilarityEngine(model.z, platform.pool.loc[model.index], weights=weights)
    altered = list(weighted.neighbours(index, n=10).index)
    assert jaccard(base, altered) < 1.0


def test_rank_correlation_helpers():
    assert rank_correlation(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert rank_correlation(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3

"""Clustering, recruitment scores and the report generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clustering import choose_k, elbow_k
from src.recruitment import (
    DEFAULT_GEM_WEIGHTS, RecruitmentBrief, age_upside, apply_brief, exposure_score, fit_scores,
    hidden_gem_scores, sample_size_score, search,
)
from src.reporting import generate_report, ordinal


def test_every_position_gets_its_own_model(platform):
    assert len(platform.models) >= 7
    for group, model in platform.models.items():
        assert len(model.features) >= 10
        assert set(platform.pool.loc[model.index, "position_group"]) == {group}


def test_cluster_labels_are_unique_and_cover_every_player(platform):
    for model in platform.models.values():
        names = list(model.clusters.names.values())
        assert len(names) == len(set(names)) == model.clusters.k
        assert model.clusters.labels.notna().all()
        assert set(model.clusters.labels.unique()) == set(range(model.clusters.k))


def test_choose_k_prefers_granularity_within_tolerance():
    evaluation = pd.DataFrame(
        {"k": [3, 4, 5, 6], "inertia": [100, 80, 70, 65], "silhouette": [0.20, 0.19, 0.14, 0.10]}
    )
    assert choose_k(evaluation) == 4          # 0.19 is within 10% of 0.20; 0.14 is not
    assert elbow_k(evaluation) in {4, 5}


def test_archetype_names_avoid_repeated_adjectives(platform):
    for model in platform.models.values():
        for name in model.clusters.names.values():
            words = name.lower().replace("-", " ").split()
            assert len(words) == len(set(words)), name


def test_fit_score_is_the_weighted_mean_of_category_percentiles(platform):
    weights = {"Finishing": 60, "Dribbling": 40}
    wingers = platform.pool.index[platform.pool["position_group"] == "W"][:20]
    scores = fit_scores(platform.categories.loc[wingers], weights)
    expected = (
        0.6 * platform.categories.loc[wingers, "cat_Finishing"]
        + 0.4 * platform.categories.loc[wingers, "cat_Dribbling"]
    )
    assert np.allclose(scores["fit_score"], expected.round(1), atol=0.15)
    assert scores["fit_score"].between(0, 100).all()


def test_brief_filters_are_all_applied(platform):
    brief = RecruitmentBrief(
        position_group="W", min_minutes=1200, age_range=(18, 23),
        thresholds=[("np_goals_per90", ">=", 0.2)],
    )
    mask = apply_brief(platform.pool, brief)
    selected = platform.pool[mask]
    assert (selected["position_group"] == "W").all()
    assert (selected["minutes"] >= 1200).all()
    assert selected["age"].between(18, 23).all()
    assert (selected["np_goals_per90"] >= 0.2).all()


def test_search_returns_a_ranked_shortlist(platform):
    brief = RecruitmentBrief(position_group="CB", age_range=(18, 30), min_minutes=900)
    results = search(platform.pool, platform.categories, brief, top_n=25)
    assert len(results) <= 25
    assert results["fit_score"].is_monotonic_decreasing
    assert list(results["rank"]) == list(range(1, len(results) + 1))


def test_hidden_gem_components_are_bounded_and_directional(platform):
    scores = hidden_gem_scores(
        platform.pool, platform.categories, {g: m.z for g, m in platform.models.items()}
    )
    for component in DEFAULT_GEM_WEIGHTS:
        assert scores[component].dropna().between(0, 100).all()
    assert scores["hidden_gem_score"].dropna().between(0, 100).all()
    # The age component is strictly decreasing in age until it floors at 27.
    joined = platform.pool[["age"]].join(scores["Age upside"])
    inside_range = joined[joined["age"].between(19, 27, inclusive="neither")]
    assert inside_range["age"].corr(inside_range["Age upside"], method="spearman") < -0.999
    assert joined.loc[joined["age"] < 23, "Age upside"].mean() > (
        joined.loc[joined["age"] >= 27, "Age upside"].mean()
    )


def test_age_and_sample_components_hit_their_documented_bounds():
    assert age_upside(pd.Series([18.0])).iloc[0] == 100
    assert age_upside(pd.Series([27.0])).iloc[0] == 0
    assert sample_size_score(pd.Series([1800])).iloc[0] == 100
    assert sample_size_score(pd.Series([900])).iloc[0] == 50
    scores = exposure_score(pd.Series(["Premier League", "Ekstraklasa"]))
    assert scores.iloc[0] == 0 and scores.iloc[1] == 100


def test_ordinal_suffixes():
    assert [ordinal(n) for n in (1, 2, 3, 11, 12, 13, 21, 92)] == [
        "1st", "2nd", "3rd", "11th", "12th", "13th", "21st", "92nd"
    ]


def test_report_is_built_from_the_data(platform):
    index = platform.pool.index[platform.pool["position_group"] == "FW"][0]
    row = platform.row(index)
    report = generate_report(platform, index)
    for heading in ["## Player", "## Statistical profile", "## Strengths", "## Weaknesses",
                    "## Archetype", "## Similar players", "## Key takeaways"]:
        assert heading in report
    assert row["player"] in report
    assert row["team"] in report
    assert platform.archetype(index)[0] in report
    assert f"{row['minutes']:,.0f}" in report

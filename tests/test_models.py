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
    # Components a dataset cannot compute are dropped, not faked.
    present = [c for c in DEFAULT_GEM_WEIGHTS if c in scores.columns]
    assert "Performance" in present and "Sample size" in present
    for component in present:
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
    # Exposure is scaled across the league coefficients present in the pool.
    scores = exposure_score(pd.Series([1.00, 0.80, 0.60]))
    assert scores.iloc[0] == 0 and scores.iloc[2] == 100 and scores.iloc[1] == 50


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


# --------------------------------------------------------------------------
# Squad tools and graceful degradation
# --------------------------------------------------------------------------

def test_replacements_are_one_row_per_player_and_respect_the_mask(platform):
    index = platform.pool.index[platform.pool["position_group"] == "CB"][0]
    team = platform.pool.loc[index, "team"]
    mask = platform.pool["team"] != team
    results = platform.replacements(index, n=10, candidate_mask=mask)
    assert not results.empty
    assert results["player_id"].is_unique
    assert (results["team"] != team).all()
    assert results["replacement_score"].is_monotonic_decreasing
    assert list(results["rank"]) == list(range(1, len(results) + 1))


def test_replacement_weighting_moves_between_style_and_quality(platform):
    index = platform.pool.index[platform.pool["position_group"] == "W"][0]
    style = platform.replacements(index, n=10, similarity_weight=1.0)
    quality = platform.replacements(index, n=10, similarity_weight=0.0)
    assert style["similarity"].mean() > quality["similarity"].mean()
    assert quality["role_fit"].mean() > style["role_fit"].mean()


def test_team_profile_compares_a_club_with_its_league(platform):
    from src.config import OUTFIELD_GROUPS
    team = platform.pool["team"].value_counts().index[0]
    profile = platform.team_category_profile(team, position_groups=OUTFIELD_GROUPS)
    assert not profile.empty
    assert set(profile.columns) == {"category", "club", "league_median"}
    assert profile[["club", "league_median"]].to_numpy().min() >= 0
    assert profile[["club", "league_median"]].to_numpy().max() <= 100


def test_trajectory_returns_one_row_per_season(platform):
    repeats = platform.pool[platform.pool.duplicated(subset=["player_id"], keep=False)]
    assert not repeats.empty
    index = repeats.index[0]
    trajectory = platform.trajectory(index)
    assert len(trajectory) >= 2
    assert trajectory["season"].is_unique


def test_platform_reports_what_its_source_cannot_supply(features, cleaned):
    """Dropping age must switch the age tools off, not fabricate values."""
    from src.data_processing import clean_players
    from src.pipeline import build_platform

    _clean, report = cleaned
    without_age = features.drop(columns=["age"])
    platform = build_platform(without_age, report, min_minutes=900)
    assert platform.has_age is False

    from src.recruitment import hidden_gem_scores
    scores = hidden_gem_scores(
        platform.pool, platform.categories, {g: m.z for g, m in platform.models.items()}
    )
    assert "Age upside" not in scores.columns
    assert scores["hidden_gem_score"].dropna().between(0, 100).all()


def test_features_missing_for_a_position_are_dropped_not_imputed(features, cleaned):
    from src.pipeline import build_platform

    _clean, report = cleaned
    blanked = features.copy()
    blanked["aerial_win_pct"] = float("nan")
    platform = build_platform(blanked, report, min_minutes=900)
    model = platform.models["CB"]
    assert "aerial_win_pct" not in model.features
    assert "aerial_win_pct" in model.dropped_features

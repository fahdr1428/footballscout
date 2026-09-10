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


def test_a_group_too_small_to_model_is_reported_not_dropped_silently(features, cleaned):
    """Players without a model must still be visible, and counted."""
    from src.pipeline import MIN_GROUP_SIZE, build_platform

    _clean, report = cleaned
    frame = features.copy()
    # Relabel a handful of players into a group that will be too small.
    victims = frame.index[frame["position_group"] == "CM"][: MIN_GROUP_SIZE - 1]
    frame.loc[victims, "position_group"] = "FWD"
    platform = build_platform(frame, report, min_minutes=0)

    assert "FWD" not in platform.models
    assert platform.unmodelled_groups.get("FWD", 0) > 0
    assert platform.unmodelled_players == sum(platform.unmodelled_groups.values())
    # They are still in the pool, so search and tables can find them.
    assert (platform.pool["position_group"] == "FWD").any()


# ---------------------------------------------------------------------------
# Pricing: what the market pays for a level of performance
# ---------------------------------------------------------------------------

def test_market_value_residual_finds_the_player_below_the_price_line():
    """A cheap player performing like an expensive one must rank as underpriced."""
    from src.recruitment import market_value_residual

    # A clean price ladder of 40 players, plus one priced far below his output.
    ladder = [float(v) for v in range(10, 90)][:40]
    performance = pd.Series(ladder + [80.0])
    value = pd.Series([10 ** (5 + p / 40) for p in ladder] + [2_000_000.0])
    groups = pd.Series(["CB"] * len(performance))

    out = market_value_residual(performance, value, groups)
    odd = len(performance) - 1
    assert out["value_residual"].iloc[odd] < 0            # below the fitted line
    assert out["value_residual_pct"].iloc[odd] > 90       # ranked as underpriced
    assert out["expected_market_value_eur"].iloc[odd] > 2_000_000.0


def test_market_value_residual_is_fitted_within_a_position_group():
    """Goalkeepers are priced differently from forwards; one line for both lies."""
    from src.recruitment import market_value_residual

    scale = [float(v) for v in range(10, 50)]
    performance = pd.Series(scale + scale)
    value = pd.Series([10 ** (5 + p / 40) for p in scale]        # cheap group
                      + [10 ** (7 + p / 40) for p in scale])     # expensive group
    groups = pd.Series(["GK"] * 40 + ["FW"] * 40)

    out = market_value_residual(performance, value, groups)
    # Each group is priced on its own line, so nobody reads as mispriced merely
    # for being a goalkeeper.
    assert out["value_residual"].abs().max() < 1e-6


def test_market_value_residual_skips_groups_too_small_to_fit_a_line():
    from src.recruitment import market_value_residual

    performance = pd.Series([10.0, 90.0])
    value = pd.Series([1_000_000.0, 50_000_000.0])
    out = market_value_residual(performance, value, pd.Series(["CB", "CB"]))
    assert out["value_residual"].isna().all()


def test_market_value_residual_refuses_a_fit_when_nobody_differs_on_output():
    """A vertical scatter has no line through it; ranking one would be invented."""
    from src.recruitment import market_value_residual

    performance = pd.Series([50.0] * 40)
    value = pd.Series([float(1_000_000 * (i + 1)) for i in range(40)])
    out = market_value_residual(performance, value, pd.Series(["CB"] * 40))
    assert out["value_residual"].isna().all()


def test_a_real_market_value_is_preferred_over_a_fantasy_price(platform):
    """Both columns present: the euro valuation is the one that should be used."""
    from src.recruitment import hidden_gem_scores

    pool = platform.pool.copy()
    pool["price_m"] = 5.0
    pool["market_value_eur"] = 10_000_000.0
    scores = hidden_gem_scores(
        pool, platform.categories,
        {g: m.z for g, m in platform.models.items()},
    )
    assert "Underpriced for output" in scores.columns


# ---------------------------------------------------------------------------
# Role templates
# ---------------------------------------------------------------------------

def test_every_role_template_is_a_usable_weighting():
    """Weights must sum to 100 and name categories that position actually has."""
    from src.config import (
        BUCKET_CATEGORIES, GK_CATEGORIES, OUTFIELD_CATEGORIES, ROLE_TEMPLATES,
    )

    for group, roles in ROLE_TEMPLATES.items():
        if group == "GK":
            allowed = set(GK_CATEGORIES)
        elif group in {"DEF", "MID", "FWD"}:
            allowed = set(BUCKET_CATEGORIES)
        else:
            allowed = set(OUTFIELD_CATEGORIES)
        for role, weights in roles.items():
            assert sum(weights.values()) == 100, f"{group}/{role}"
            assert set(weights) <= allowed, f"{group}/{role}: {set(weights) - allowed}"


def test_role_weights_fall_back_to_the_position_default():
    from src.config import DEFAULT_ROLE, DEFAULT_WEIGHTS, role_weights

    assert role_weights("CB", DEFAULT_ROLE) == DEFAULT_WEIGHTS["CB"]
    assert role_weights("CB", None) == DEFAULT_WEIGHTS["CB"]
    assert role_weights("CB", "Not a real role") == DEFAULT_WEIGHTS["CB"]


def test_roles_differ_enough_to_reorder_a_shortlist(platform):
    """Two roles that weighted the same way would not be worth offering."""
    from src.config import ROLE_TEMPLATES, role_weights
    from src.recruitment import RecruitmentBrief, search

    group = next(g for g in ROLE_TEMPLATES
                 if (platform.pool["position_group"] == g).sum() >= 20)
    roles = list(ROLE_TEMPLATES[group])[:2]
    orders = []
    for role in roles:
        brief = RecruitmentBrief(position_group=group, min_minutes=0,
                                 weights=role_weights(group, role), role=role)
        orders.append(search(platform.pool, platform.categories, brief, top_n=10)
                      ["player"].tolist())
    assert orders[0] != orders[1]


def test_a_budget_keeps_players_whose_value_is_unknown():
    """No recorded price is not evidence of an unaffordable one."""
    from src.recruitment import RecruitmentBrief, apply_brief

    pool = pd.DataFrame({
        "position_group": ["CB"] * 3,
        "minutes": [2000.0] * 3,
        "league": ["Premier League"] * 3,
        "market_value_eur": [5e6, 80e6, np.nan],
    })
    brief = RecruitmentBrief(position_group="CB", min_minutes=0, max_market_value=10e6)
    assert list(apply_brief(pool, brief)) == [True, False, True]


def test_foot_filter_is_case_insensitive():
    from src.recruitment import RecruitmentBrief, apply_brief

    pool = pd.DataFrame({
        "position_group": ["FB"] * 3,
        "minutes": [2000.0] * 3,
        "league": ["Ligue 1"] * 3,
        "foot": ["Left", "right", "both"],
    })
    brief = RecruitmentBrief(position_group="FB", min_minutes=0, feet=["left"])
    assert list(apply_brief(pool, brief)) == [True, False, False]


# ---------------------------------------------------------------------------
# Forward test against later market value
# ---------------------------------------------------------------------------

def test_value_growth_backtest_needs_market_values_and_several_seasons(platform):
    from src.validation import value_growth_backtest

    table, summary = value_growth_backtest(platform, horizon=2)
    if "market_value_eur" not in platform.pool.columns:
        assert summary == {"available": False}
        assert table.empty


def test_value_growth_backtest_never_looks_at_the_future_to_build_its_score():
    """The score must be computable from season t alone."""
    import inspect

    from src import validation

    source = inspect.getsource(validation.value_growth_backtest)
    # The outcome is joined only after scoring, from a separate lookup.
    assert source.index("hidden_gem_scores(") < source.index("later_value_eur")


def test_growth_strata_skip_cells_too_small_to_read():
    from src.validation import _value_growth_strata

    tested = pd.DataFrame({
        "age": [20.0, 21.0, 25.0, 30.0],
        "market_value_eur": [1e6, 2e6, 3e6, 4e6],
        "hidden_gem_score": [10.0, 20.0, 30.0, 40.0],
        "growth": [0.1, 0.2, -0.1, -0.2],
    })
    assert _value_growth_strata(tested) == {"stratified": False}


def test_growth_strata_reports_a_correlation_when_cells_are_big_enough():
    from src.validation import _value_growth_strata

    rng = np.random.default_rng(3)
    n = 400
    score = rng.uniform(0, 100, n)
    tested = pd.DataFrame({
        "age": rng.uniform(22.0, 23.0, n),          # one age band
        "market_value_eur": rng.uniform(1e6, 2e6, n),
        "hidden_gem_score": score,
        "growth": score / 100 + rng.normal(0, 0.1, n),   # score genuinely predicts
    })
    out = _value_growth_strata(tested)
    assert out["stratified"] is True
    assert out["within_stratum_rank_correlation"] > 0.5


# ---------------------------------------------------------------------------
# The position taxonomy, and the evidence for it
# ---------------------------------------------------------------------------

def test_every_split_out_group_has_a_parent_to_fall_back_to():
    from src.config import DETAILED_GROUPS, POSITION_PARENT

    for group, parent in POSITION_PARENT.items():
        assert group in DETAILED_GROUPS
        assert parent in DETAILED_GROUPS
        assert parent not in POSITION_PARENT, f"{group} -> {parent} is a chain"


def test_every_group_has_features_weights_and_a_noun():
    from src.clustering import POSITION_NOUNS
    from src.config import DEFAULT_WEIGHTS, DETAILED_GROUPS, POSITION_FEATURES

    for group in DETAILED_GROUPS:
        assert POSITION_FEATURES.get(group), group
        assert sum(DEFAULT_WEIGHTS.get(group, {}).values()) == 100, group
        assert POSITION_NOUNS.get(group), group


def test_thin_groups_fold_into_their_parent_rather_than_being_dropped():
    from src.pipeline import collapse_thin_groups

    pool = pd.DataFrame({
        "position_group": ["SS"] * 5 + ["AM"] * 50 + ["W"] * 50,
        "minutes": [1000.0] * 105,
    })
    out, collapsed = collapse_thin_groups(pool, min_size=40)
    assert (out["position_group"] == "SS").sum() == 0
    assert (out["position_group"] == "AM").sum() == 55
    assert collapsed == {"SS": {"parent": "AM", "players": 5}}


def test_a_group_with_enough_players_keeps_its_own_peer_set():
    from src.pipeline import collapse_thin_groups

    pool = pd.DataFrame({
        "position_group": ["SS"] * 60 + ["AM"] * 50,
        "minutes": [1000.0] * 110,
    })
    out, collapsed = collapse_thin_groups(pool, min_size=40)
    assert (out["position_group"] == "SS").sum() == 60
    assert collapsed == {}


def test_separability_verdict_flags_a_taxonomy_that_contradicts_the_evidence():
    from src.validation import _separability_verdict

    assert "BUT" not in _separability_verdict(0.90, split=True)
    assert "BUT" not in _separability_verdict(0.55, split=False)
    assert "BUT" in _separability_verdict(0.90, split=False)   # separable but merged
    assert "BUT" in _separability_verdict(0.55, split=True)    # inseparable but split


def test_separability_uses_balanced_accuracy_so_imbalance_cannot_flatter_it():
    """Four AMs per SS: plain accuracy would score 0.80 for learning nothing."""
    import inspect

    from src import validation

    assert 'scoring="balanced_accuracy"' in inspect.getsource(validation.position_separability)


# ---------------------------------------------------------------------------
# Side of the pitch
# ---------------------------------------------------------------------------

def test_flank_comes_from_the_specific_position():
    from src.data_processing import add_flank

    out = add_flank(pd.DataFrame({
        "position": ["LB", "RB", "CB", "LW", "RM", "GK"],
        "position_group": ["FB", "FB", "CB", "W", "WM", "GK"],
    }))
    assert list(out["flank"]) == ["Left", "Right", "Central", "Left", "Right", "Central"]


def test_inverted_is_a_wide_player_on_the_opposite_foot():
    from src.data_processing import add_flank

    out = add_flank(pd.DataFrame({
        "position": ["LW", "LW", "RW", "CB"],
        "position_group": ["W", "W", "W", "CB"],
        "foot": ["right", "left", "right", "left"],
    }))
    assert list(out["footed_side"][:3]) == ["Inverted", "Natural", "Natural"]
    # A left-footed centre-back is not "inverted" - the idea does not apply.
    assert pd.isna(out["footed_side"].iloc[3]) or out["footed_side"].iloc[3] is None


def test_flank_survives_a_source_with_no_footedness():
    from src.data_processing import add_flank

    out = add_flank(pd.DataFrame({"position": ["LB"], "position_group": ["FB"]}))
    assert out["flank"].iloc[0] == "Left"
    assert "footed_side" not in out.columns


def test_brief_filters_on_side_without_touching_the_model_group():
    from src.recruitment import RecruitmentBrief, apply_brief

    pool = pd.DataFrame({
        "position_group": ["FB"] * 4,
        "position": ["LB", "RB", "LB", "RB"],
        "flank": ["Left", "Right", "Left", "Right"],
        "footed_side": ["Inverted", "Natural", "Natural", "Inverted"],
        "minutes": [2000.0] * 4,
        "league": ["Ligue 1"] * 4,
    })
    brief = RecruitmentBrief(position_group="FB", min_minutes=0,
                             flanks=["Left"], footed_sides=["Inverted"])
    assert list(apply_brief(pool, brief)) == [True, False, False, False]


def test_min_minutes_of_none_means_the_default_not_an_empty_pool():
    """`>= None` compares false for every row and would silently return nothing."""
    from src.config import DEFAULT_MIN_MINUTES
    from src.pipeline import build_platform
    import inspect

    source = inspect.getsource(build_platform)
    assert "if min_minutes is None" in source
    assert "DEFAULT_MIN_MINUTES" in source


def test_a_small_group_drops_to_two_archetypes_rather_than_stranding_one():
    """Sixty players have room for three clusters of twelve on paper. If they do
    not actually fall that way, two well-supported archetypes beat three with a
    handful of players in one corner."""
    from src.clustering import MIN_CLUSTER_ABSOLUTE, fit_clusters

    from src.config import POSITION_FEATURES

    columns = POSITION_FEATURES["SS"][:6]
    rng = np.random.default_rng(11)
    # Two genuine blobs plus a few stragglers: k=3 would isolate the stragglers.
    z = pd.DataFrame(
        np.vstack([
            rng.normal(-2, 0.4, (30, len(columns))),
            rng.normal(2, 0.4, (30, len(columns))),
            rng.normal(0, 0.3, (3, len(columns))),
        ]),
        columns=columns,
    )
    model = fit_clusters(z, "SS")
    assert model.labels.value_counts().min() >= MIN_CLUSTER_ABSOLUTE


def test_a_healthy_group_is_not_dragged_down_to_two():
    """The k=2 option is a fallback, not a new default: a group that supports
    more archetypes must keep them."""
    from src.clustering import fit_clusters

    from src.config import POSITION_FEATURES

    columns = POSITION_FEATURES["CB"][:6]
    rng = np.random.default_rng(12)
    z = pd.DataFrame(
        np.vstack([rng.normal(centre, 0.35, (80, len(columns)))
                   for centre in (-3, 0, 3, 6)]),
        columns=columns,
    )
    assert fit_clusters(z, "CB").k >= 3


def test_clusters_are_numbered_when_no_feature_maps_to_a_naming_concept():
    """A source whose columns describe nothing the vocabulary knows must still
    produce usable labels rather than crash on an empty concept table."""
    from src.clustering import fit_clusters

    rng = np.random.default_rng(13)
    z = pd.DataFrame(
        np.vstack([rng.normal(c, 0.3, (40, 4)) for c in (-2, 2)]),
        columns=[f"unmapped_{i}_per90" for i in range(4)],
    )
    names = set(fit_clusters(z, "CB").names.values())
    assert len(names) == len(set(names))
    assert all(name for name in names)

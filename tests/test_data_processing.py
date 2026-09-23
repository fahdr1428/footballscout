"""Cleaning rules must be strict about anything that corrupts a per-90 rate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_processing import MAX_MINUTES, clean_players, filter_pool


def test_report_accounts_for_every_dropped_row(cleaned, raw):
    clean, report = cleaned
    assert report.rows_in == len(raw)
    assert report.rows_out == len(clean)
    assert report.rows_out < report.rows_in  # the generator injects defects on purpose


def test_impossible_minutes_are_dropped_not_clipped(cleaned):
    clean, report = cleaned
    assert clean["minutes"].between(1, MAX_MINUTES).all()
    assert report.invalid_minutes > 0


def test_no_negative_counting_stats_survive(cleaned):
    clean, _ = cleaned
    numeric = clean.select_dtypes("number").drop(
        columns=[c for c in ["np_goals_minus_npxg_per90"] if c in clean.columns]
    )
    assert (numeric.fillna(0) >= 0).all().all()


def test_exact_duplicates_removed(cleaned):
    clean, report = cleaned
    assert report.exact_duplicates > 0
    assert not clean.duplicated(subset=["player_id", "season"]).any()


def test_goalkeeping_columns_stay_missing_for_outfielders(cleaned):
    clean, _ = cleaned
    outfield = clean[clean["position_group"] != "GK"]
    assert outfield["gk_saves"].isna().all()
    keepers = clean[clean["position_group"] == "GK"]
    assert keepers["gk_saves"].notna().all()


def test_successes_never_exceed_attempts(cleaned):
    clean, _ = cleaned
    assert (clean["passes_completed"] <= clean["passes_attempted"]).all()
    assert (clean["tackles_won"] <= clean["tackles"]).all()


def test_filter_pool_applies_every_filter(features):
    pool = filter_pool(
        features, min_minutes=1200, position_groups=["CB"], age_range=(20, 25)
    )
    assert (pool["minutes"] >= 1200).all()
    assert (pool["position_group"] == "CB").all()
    assert pool["age"].between(20, 25).all()


def test_cleaning_is_idempotent(cleaned):
    clean, _ = cleaned
    again, report = clean_players(clean)
    assert len(again) == len(clean)
    assert report.exact_duplicates == 0
    assert report.invalid_minutes == 0


def test_a_stat_missing_in_one_season_is_partial_not_unavailable(cleaned):
    """Zero-filling an unmeasured metric would claim the player never did it.

    Blanking exactly one season out of several is the case a real source hits
    often - a metric it starts (or stops) counting partway through its run.
    That is "not reliable in 2016-17", not "this source does not have
    tackles": the other seasons still do, so it belongs in `partial`, not the
    flat `unavailable` list that means every season lacks it.
    """
    clean, report = cleaned
    frame = clean.copy()
    season = sorted(frame["season"].unique())[0]
    frame.loc[frame["season"] == season, "tackles"] = np.nan
    cleaned_again, again = clean_players(frame)
    blanked = cleaned_again[cleaned_again["season"] == season]
    kept = cleaned_again[cleaned_again["season"] != season]
    assert blanked["tackles"].isna().all()          # left missing in that season
    assert kept["tackles"].notna().all()            # still measured in the others
    assert "tackles" not in again.unavailable_columns
    assert again.partial_columns.get("tackles") == [season]


def test_a_stat_missing_in_every_season_is_unavailable(cleaned):
    """The flat `unavailable` list is for a column no season measures at all."""
    clean, report = cleaned
    frame = clean.copy()
    frame["tackles"] = np.nan
    _, again = clean_players(frame)
    assert "tackles" in again.unavailable_columns
    assert "tackles" not in again.partial_columns


def test_an_optional_metric_missing_a_whole_season_is_partial_not_imputed(cleaned):
    """A provider switch must not get papered over by a cross-era median.

    `pressures` is imputed from a positional per-90 median when a player is
    missing it - real behaviour for scattered gaps. But FBref's Opta cutover
    blanked it for entire seasons at once; imputing those from the surviving
    seasons' median would hand every player that season a fabricated
    pressures count instead of reporting the loss.
    """
    clean, report = cleaned
    frame = clean.copy()
    season = sorted(frame["season"].unique())[0]
    frame.loc[frame["season"] == season, "pressures"] = np.nan
    cleaned_again, again = clean_players(frame)
    blanked = cleaned_again[cleaned_again["season"] == season]
    kept = cleaned_again[cleaned_again["season"] != season]
    assert blanked["pressures"].isna().all()         # left missing, not imputed
    assert kept["pressures"].notna().all()            # still measured in the others
    assert "pressures" not in again.unavailable_columns
    assert again.partial_columns.get("pressures") == [season]


def test_a_stat_below_the_coverage_floor_is_partial_not_zero_filled(cleaned):
    """~75% coverage in a season is a scrape or join gap, not real zeros.

    The old rule only caught a column once it was *entirely* null for a
    season; this is the case that motivated raising the bar - most players
    measured, some genuinely not, and the gap must not read as a fabricated
    zero for the ones who are missing.
    """
    clean, report = cleaned
    frame = clean.copy()
    season = sorted(frame["season"].unique())[0]
    rows = frame.index[frame["season"] == season]
    # Blank three-quarters of that season's rows for one metric - well under
    # the 90% floor, well above zero.
    gap = rows[: int(len(rows) * 0.75)]
    frame.loc[gap, "tackles"] = np.nan
    _, again = clean_players(frame)
    assert again.partial_columns.get("tackles") == [season]
    kept = frame.loc[rows.difference(gap)]
    # The players who WERE measured keep their real values - nothing about
    # the season being flagged partial should touch rows that have data.
    assert kept["tackles"].notna().all()


def test_age_and_height_are_never_invented(cleaned):
    """A positional median is not anybody's age.

    Age and height are facts about a person, not measurements of a season, and
    no model reads them - they feed filters and the hidden-gem age term, which
    is exactly where a made-up 26.3 does damage. Unknown must stay unknown.
    """
    clean, _ = cleaned
    frame = clean.copy()
    blank = frame.index[:25]
    frame.loc[blank, ["age", "height_cm"]] = np.nan
    again, report = clean_players(frame)
    keys = frame.loc[blank, ["player_id", "season"]]
    hit = again.merge(keys, on=["player_id", "season"])
    assert len(hit) == len(keys)
    assert hit["age"].isna().all() and hit["height_cm"].isna().all()
    assert "age" not in report.imputed and "height_cm" not in report.imputed


def test_an_unknown_age_passes_an_untouched_range_but_not_a_narrowed_one():
    """Nobody should vanish from a list because a source omitted a birth date,
    and nobody should be shortlisted as under-23 without one."""
    from src.data_processing import AgeRange, age_mask

    ages = pd.Series([19.0, 30.0, np.nan])
    untouched = AgeRange(15.0, 40.0, active=False)
    narrowed = AgeRange(15.0, 23.0, active=True)
    assert age_mask(ages, untouched).tolist() == [True, True, True]
    assert age_mask(ages, narrowed).tolist() == [True, False, False]
    # A plain tuple is an explicit request, so it counts as narrowed.
    assert age_mask(ages, (15.0, 40.0)).tolist() == [True, True, False]

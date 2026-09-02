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

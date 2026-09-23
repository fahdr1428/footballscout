"""The Premier League summary-feed ETL: name joining, positions, club of record."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.premier_league import (
    ELEMENT_TYPE_TO_GROUP, UNDERSTAT_COVERAGE_FLOOR, UNDERSTAT_TO_DETAIL, normalise_name,
    season_start_year, understat_coverage,
)


def test_name_normalisation_survives_the_accents_that_break_a_join():
    assert normalise_name("Martin Ødegaard") == "martin odegaard"
    assert normalise_name("Gabriel Magalhães") == "gabriel magalhaes"
    assert normalise_name("Niclas Füllkrug") == "niclas fullkrug"
    assert normalise_name("Mateo Kovačić") == "mateo kovacic"
    assert normalise_name("Dominik Szoboszlai") == "dominik szoboszlai"
    # punctuation and spacing must not create a mismatch
    assert normalise_name("N'Golo Kanté") == normalise_name("N Golo Kante")
    assert normalise_name("  Bruno   Fernandes ") == "bruno fernandes"


def test_season_year_parsing():
    assert season_start_year("2025-26") == 2025
    assert season_start_year("2016-17") == 2016


def test_every_fpl_bucket_maps_to_a_position_group():
    from src.config import BUCKET_GROUPS
    assert set(ELEMENT_TYPE_TO_GROUP.values()) == set(BUCKET_GROUPS)


def test_understat_lineup_codes_map_to_known_positions():
    from src.config import POSITION_TO_GROUP
    for code, detail in UNDERSTAT_TO_DETAIL.items():
        assert detail in POSITION_TO_GROUP, f"{code} -> {detail} is not a known position"


def test_the_club_of_record_is_the_one_a_player_played_for(tmp_path):
    """A season snapshot names a player's *current* club, not that season's."""
    from src.premier_league import appearances_from_gameweeks

    season = tmp_path / "data" / "2025-26" / "gws"
    season.mkdir(parents=True)
    pd.DataFrame(
        {
            "element": [1, 1, 1, 2, 2],
            "team": ["Bournemouth", "Bournemouth", "Bournemouth", "Wolves", "Crystal Palace"],
            "minutes": [90, 90, 45, 90, 90],
        }
    ).to_csv(season / "merged_gw.csv", index=False)

    out = appearances_from_gameweeks(tmp_path, "2025-26").set_index("fpl_element")
    assert out.loc[1, "matches"] == 3
    assert out.loc[1, "gw_team"] == "Bournemouth"
    assert out.loc[1, "clubs_in_season"] == 1
    # A mid-season move is counted, and the club of record is where he played most.
    assert out.loc[2, "clubs_in_season"] == 2
    assert out.loc[2, "gw_team"] in {"Wolves", "Crystal Palace"}


def test_minutes_with_no_appearances_are_not_invented(tmp_path):
    from src.premier_league import appearances_from_gameweeks

    season = tmp_path / "data" / "2025-26" / "gws"
    season.mkdir(parents=True)
    pd.DataFrame({"element": [1], "team": ["Arsenal"], "minutes": [0]}).to_csv(
        season / "merged_gw.csv", index=False
    )
    assert appearances_from_gameweeks(tmp_path, "2025-26").empty


def test_a_season_understat_stopped_early_is_detected():
    """The 2024/25 Understat logs end on 6 April 2025 while FPL minutes run to
    May: shots over a partial season divided by a full season's minutes would
    understate every per-90 by the missing share."""
    data = pd.DataFrame({
        "season_year": [2023, 2023, 2024, 2024],
        "minutes": [3000.0, 1500.0, 3371.0, 1800.0],
        "us_minutes": [3000.0, 1490.0, 2766.0, 1480.0],
    })
    coverage = understat_coverage(data)
    assert coverage[2023] >= UNDERSTAT_COVERAGE_FLOOR
    assert coverage[2024] < UNDERSTAT_COVERAGE_FLOOR

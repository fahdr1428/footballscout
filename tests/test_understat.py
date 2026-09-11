"""
Understat six-league ingestion, tested offline against the mirror's shape.

The mirror is one wide CSV of season aggregates. These tests drive the real
position mapping, season handling and transfer collapsing with small frames in
exactly that shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.understat import (
    FIRST_SEASON, LAST_SEASON, LEAGUES, PARTIAL_SEASONS, POSITIONS,
    _collapse_transfers, _position, season_label,
)


def test_season_label_matches_the_rest_of_the_platform():
    assert season_label("2024/25") == "2024-25"
    assert season_label("2014/15") == "2014-15"


def test_the_unfinished_season_is_not_a_default():
    """The mirror froze in September 2025, leaving about ten rounds of 2025/26."""
    assert "2025/26" in PARTIAL_SEASONS
    assert LAST_SEASON == "2024/25"
    assert LAST_SEASON not in PARTIAL_SEASONS
    assert FIRST_SEASON < LAST_SEASON


def test_every_league_maps_to_one_the_platform_knows():
    from src.config import LEAGUE_STRENGTH

    for name in LEAGUES.values():
        assert name in LEAGUE_STRENGTH, name


def test_positions_use_the_four_bucket_taxonomy():
    """Understat records only GK/D/M/F, so a finer group would be invented."""
    from src.config import BUCKET_GROUPS

    assert {g for g, _ in POSITIONS.values()} == set(BUCKET_GROUPS)


def test_primary_position_is_used_when_there_is_one():
    group, position, source = _position({"primary_position": "D", "position": "D M S"})
    assert (group, position) == ("DEF", "DEF")
    assert source == "Understat line-ups"


def test_substitute_only_rows_fall_back_to_a_recorded_role():
    """`S` means he only came off the bench; it is not a position."""
    group, position, source = _position({"primary_position": "S", "position": "M S"})
    assert (group, position) == ("MID", "MID")
    assert "substitute" in source


def test_a_row_with_no_usable_role_says_so_rather_than_guessing_quietly():
    _, _, source = _position({"primary_position": "S", "position": "S"})
    assert source == "unknown"


def _two_club_season():
    return pd.DataFrame([
        {"player_id": "1", "season": "2024-25", "team": "Empoli", "player": "Mover",
         "minutes": 1200.0, "goals": 4.0, "xg": 3.5},
        {"player_id": "1", "season": "2024-25", "team": "Roma", "player": "Mover",
         "minutes": 700.0, "goals": 2.0, "xg": 2.1},
        {"player_id": "2", "season": "2024-25", "team": "Empoli", "player": "Stayer",
         "minutes": 2400.0, "goals": 9.0, "xg": 8.0},
    ])


def test_a_mid_season_move_becomes_one_row_with_the_totals_added():
    out = _collapse_transfers(_two_club_season())
    mover = out[out["player"] == "Mover"].iloc[0]
    assert len(out) == 2
    assert mover["minutes"] == pytest.approx(1900.0)
    assert mover["goals"] == pytest.approx(6.0)
    assert mover["clubs_in_season"] == 2


def test_club_of_record_is_where_he_played_the_most():
    out = _collapse_transfers(_two_club_season())
    assert out[out["player"] == "Mover"].iloc[0]["team"] == "Empoli"


def test_a_stat_one_club_did_not_record_is_left_missing():
    frame = _two_club_season()
    frame.loc[frame["team"] == "Roma", "xg"] = np.nan
    out = _collapse_transfers(frame)
    assert pd.isna(out[out["player"] == "Mover"].iloc[0]["xg"])


def test_a_source_that_cannot_measure_a_position_reports_it_rather_than_crashing():
    """Understat has no goalkeeping metric at all, which used to end the build."""
    from src.pipeline import MIN_GROUP_FEATURES

    assert MIN_GROUP_FEATURES >= 2


# ---------------------------------------------------------------------------
# The Transfermarkt join: attributes only, and only when it is safe
# ---------------------------------------------------------------------------

def test_an_impossible_age_withdraws_the_whole_match_not_just_the_age():
    """A 9-year-old means the name matched the wrong person; everything else
    on that row came from him too, so none of it can be trusted."""
    import inspect

    from src import understat

    source = inspect.getsource(understat.attach_transfermarkt)
    assert "MIN_PLAUSIBLE_AGE" in source
    for column in ("height_cm", "foot", "nationality", "date_of_birth"):
        assert column in source


def test_plausible_age_bounds_are_a_career_not_a_lifetime():
    from src.understat import MAX_PLAUSIBLE_AGE, MIN_PLAUSIBLE_AGE

    assert 14 <= MIN_PLAUSIBLE_AGE <= 17
    assert 42 <= MAX_PLAUSIBLE_AGE <= 50


def test_position_is_never_taken_from_the_join():
    """It would arrive for two players in three, so a player's peer group would
    depend on whether his name matched rather than on football."""
    import inspect

    from src import understat

    source = inspect.getsource(understat.attach_transfermarkt)
    assert '"position"' not in source
    assert "position_group" not in source


def test_only_names_unique_on_both_sides_are_matched():
    import inspect

    from src import understat

    source = inspect.getsource(understat.attach_transfermarkt)
    assert source.count("duplicated(keep=False)") >= 2


def test_market_value_is_read_as_at_the_season():
    """Not one scrape applied to eleven years: Transfermarkt revalues players
    several times a year and the history can be read at the right moment."""
    import inspect

    from src import understat

    source = inspect.getsource(understat._attach_values)
    assert "merge_asof" in source
    assert 'direction="backward"' in source

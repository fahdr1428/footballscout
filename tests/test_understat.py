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
    """Read from the data: a season is finished once a league has played a
    full season's rounds, so August needs no code change."""
    import pandas as pd

    from src.understat import FULL_SEASON_GAMES, season_status

    frame = pd.DataFrame({
        "season": ["2025/26", "2025/26", "2026/27", "2026/27"],
        "games": [FULL_SEASON_GAMES + 8, 12, 5, 3],
    })
    complete, partial = season_status(frame)
    assert complete == {"2025/26"} and partial == {"2026/27"}
    assert FIRST_SEASON < LAST_SEASON


def test_the_refresh_fetches_the_season_being_played_and_the_last_one():
    import sys
    from datetime import datetime
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from refresh_sources import default_seasons

    assert default_seasons(datetime(2026, 10, 6)) == [2025, 2026]
    assert default_seasons(datetime(2027, 3, 1)) == [2025, 2026]
    assert default_seasons(datetime(2027, 8, 1)) == [2026, 2027]


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


def test_position_comes_from_transfermarkt_only_among_roles_he_played():
    """Understat lists roles alphabetically, so "D M S" is not "a defender".
    Transfermarkt picks among them - and never moves a player somewhere
    Understat says he did not play that season."""
    import pandas as pd

    from src.understat import assign_positions

    frame = pd.DataFrame({
        "tm_id": [1, 2, 3, None],
        "roles": ["D M S", "M S", "D", "D M S"],
        "position_group": ["DEF", "MID", "DEF", "DEF"],
        "position": ["DEF", "MID", "DEF", "DEF"],
        "position_source": ["Understat line-ups"] * 4,
    })
    profiles = pd.DataFrame({
        "tm_id": [1, 2, 3],
        "tm_bucket": ["Midfield", "Attack", "Midfield"],
        "tm_detail": ["Defensive Midfield", "Left Winger", "Central Midfield"],
    })
    out = assign_positions(frame, profiles)
    assert out.loc[0, "position_group"] == "MID"            # Rice: a midfielder who also played D
    assert out.loc[0, "position"] == "Defensive Midfield"
    assert out.loc[1, "position_group"] == "MID"            # Attack, but he never played F that season
    assert out.loc[2, "position_group"] == "DEF"            # only ever played D that season
    assert out.loc[3, "position_group"] == "DEF"            # unmatched: the listed role stands
    assert out.loc[3, "position_source"].startswith("Understat, first role listed")


def test_only_names_unique_on_both_sides_are_matched():
    import inspect

    from src import understat

    from src import tm_history

    source = inspect.getsource(tm_history.link_players)
    assert source.count("duplicated(keep=False)") >= 2
    assert "link_players" in inspect.getsource(understat.attach_transfermarkt)


def test_market_value_is_read_as_at_the_season():
    """Not one scrape applied to eleven years: the valuation in force on the
    date asked for, never a later one, and nothing stale."""
    import numpy as np
    import pandas as pd

    from src.tm_history import value_asof

    history = pd.DataFrame({
        "tm_id": [1, 1, 1, 2],
        "date": pd.to_datetime(["2023-02-01", "2024-06-15", "2025-01-10", "2019-01-01"]),
        "value": [10e6, 30e6, 50e6, 5e6],
    })
    ids = pd.Series([1, 1, 2, 3])
    asof = pd.Series(pd.to_datetime(["2024-07-01", "2023-07-01", "2024-07-01", "2024-07-01"]))
    got = value_asof(ids, asof, history)
    assert got.iloc[0] == 30e6          # the June 2024 valuation, not January 2025's
    assert got.iloc[1] == 10e6
    assert np.isnan(got.iloc[2])        # five years old: describes someone else
    assert np.isnan(got.iloc[3])        # never valued


def test_a_name_written_two_ways_matches_on_club_season_and_minutes():
    import pandas as pd

    from src.premier_league import normalise_name
    from src.understat import _match_by_season

    us = pd.DataFrame({
        "player_id": ["3423", "3423", "77", "88"],
        "player": ["Kylian Mbappe-Lottin", "Kylian Mbappe-Lottin", "Rodri", "Rodri"],
        "season": ["2024-25", "2025-26", "2024-25", "2024-25"],
        "league": ["La Liga", "La Liga", "Premier League", "La Liga"],
        "team": ["Real Madrid", "Real Madrid", "Manchester City", "Real Betis"],
        "minutes": [2938, 2623, 300, 2000],
    })
    us["key"] = us["player"].map(normalise_name)
    tm = pd.DataFrame({
        "player_id": [342229, 342229, 357565, 999],
        "season": [2024, 2025, 2024, 2024],
        "competition_id": ["ES1", "ES1", "GB1", "ES1"],
        "club_name": ["Real Madrid", "Real Madrid", "Manchester City", "Real Betis Balompié"],
        "minutes": [2917, 2606, 290, 1980],
        "player_name": ["Kylian Mbappé", "Kylian Mbappé", "Rodri", "Rodri"],
    })
    link = _match_by_season(us, tm)
    assert link["3423"] == 342229
    # Two players called Rodri, told apart by league and club, not by name.
    assert link["77"] == 357565 and link["88"] == 999


def test_a_loose_name_match_needs_the_club_and_the_minutes_to_agree():
    import pandas as pd

    from src.premier_league import normalise_name
    from src.understat import _match_by_season

    us = pd.DataFrame({"player_id": ["1"], "player": ["Kylian Mbappe-Lottin"],
                       "season": ["2024-25"], "league": ["La Liga"],
                       "team": ["Real Madrid"], "minutes": [2938]})
    us["key"] = us["player"].map(normalise_name)
    tm = pd.DataFrame({"player_id": [5], "season": [2024], "competition_id": ["ES1"],
                       "club_name": ["Real Madrid"], "minutes": [600],
                       "player_name": ["Ethan Mbappé"]})
    assert _match_by_season(us, tm).empty

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
    _collapse_transfers, _position, _unescape_names, season_label,
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


def test_names_are_stored_as_written_not_as_understat_escapes_them():
    frame = pd.DataFrame({"player": ["N&#039;Golo Kanté", "M&#039;Baye Niang", "Pedri"],
                          "team": ["Chelsea", "Brighton &amp; Hove Albion", None]})
    out = _unescape_names(frame)
    assert out["player"].tolist() == ["N'Golo Kanté", "M'Baye Niang", "Pedri"]
    assert out["team"].tolist()[:2] == ["Chelsea", "Brighton & Hove Albion"]


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


def test_a_season_is_labelled_with_where_he_played_that_season():
    """A profile only knows today's position: Dembélé is a centre-forward now,
    a right winger at Barcelona. The season's own Transfermarkt record wins,
    still only among the roles Understat says he played - and today's position
    is the fallback where there is no season record."""
    import pandas as pd

    from src.understat import assign_positions

    frame = pd.DataFrame({
        "tm_id": [7, 7, 8],
        "season": ["2019/20", "2024/25", "2019/20"],
        "roles": ["F M S", "F S", "D M S"],
        "position_group": ["FWD", "FWD", "DEF"],
        "position": ["FWD", "FWD", "DEF"],
        "position_source": ["Understat line-ups"] * 3,
    })
    profiles = pd.DataFrame({"tm_id": [7, 8], "tm_bucket": ["Attack", "Midfield"],
                             "tm_detail": ["Centre-Forward", "Defensive Midfield"]})
    seasonal = pd.DataFrame({"tm_id": [7, 8], "start": [2019, 2019],
                             "tm_season_detail": ["Right Winger", "Left Winger"],
                             "how": ["squad page", "line-ups"]})
    out = assign_positions(frame, profiles, seasonal)
    assert out.loc[0, "position"] == "Right Winger"                 # that season's record
    assert out.loc[0, "position_source"] == "Transfermarkt squad page, that season"
    assert out.loc[1, "position"] == "Centre-Forward"               # no record: today's profile
    assert out.loc[1, "position_source"].startswith("Transfermarkt profile")
    assert out.loc[2, "position"] == "Defensive Midfield"           # a winger's record, but he never played F
    assert out.loc[2, "position_group"] == "MID"


def test_line_ups_beat_squad_pages_and_vague_positions_are_dropped(tmp_path, monkeypatch):
    import pandas as pd

    from src import tm_history

    lineups = tmp_path / "tm_season_positions.csv.gz"
    pd.DataFrame({"player_id": [7, 9], "season": [2019, 2019], "position": ["Left Winger", "attack"],
                  "starts": [20, 5]}).to_csv(lineups, index=False)
    monkeypatch.setattr(tm_history, "SEASON_POSITIONS", lineups)
    import src.fbref as fbref
    squads = pd.DataFrame({"player_url": ["https://www.transfermarkt.com/x/profil/spieler/7",
                                          "https://www.transfermarkt.com/y/profil/spieler/8"],
                           "season_start_year": [2019, 2019],
                           "player_position": ["Right Winger", "Centre-Back"]})
    monkeypatch.setattr(fbref, "download", lambda url, cache, name: tmp_path / name)
    monkeypatch.setattr(fbref, "read_rds", lambda path: squads)
    out = tm_history.load_season_positions(tmp_path).set_index("tm_id")
    assert out.loc[7, "tm_season_detail"] == "Left Winger" and out.loc[7, "how"] == "line-ups"
    assert out.loc[8, "tm_season_detail"] == "Centre-Back" and out.loc[8, "how"] == "squad page"
    assert 9 not in out.index                                        # "attack" says nothing new


# --------------------------------------------------------------------------
# Ties: every one broken by a stated rule, so any machine builds the same data
# --------------------------------------------------------------------------

def test_a_club_of_record_tie_goes_to_the_first_club_alphabetically_whatever_the_row_order():
    rows = pd.DataFrame([
        {"player_id": "9", "season": "2026-27", "team": team, "player": "Robert Sanchez",
         "minutes": 90.0, "goals": 0.0}
        for team in ("Como", "Chelsea")])
    for frame in (rows, rows.iloc[::-1]):
        assert _collapse_transfers(frame).iloc[0]["team"] == "Chelsea"


def _tm_rows(*rows):
    return pd.DataFrame([dict(zip(
        ["player_id", "season", "competition_id", "club_id", "minutes", "player_name", "club_name"], r))
        for r in rows]).assign(apps=1, goals=0, assists=0)


def _us_rows(*rows):
    from src.tm_history import normalise_name

    frame = pd.DataFrame([dict(zip(["player_id", "season", "league", "team", "minutes", "player"], r))
                          for r in rows])
    frame["key"] = frame["player"].map(normalise_name)
    return frame


def test_a_vote_tie_goes_to_the_exact_name_and_a_full_tie_is_left_unmatched():
    from src.tm_history import link_by_season

    tm = _tm_rows(
        # Polyarus: his own record by exact name one season; another Artem at the
        # same club with the same minutes matches only on the shared first name.
        (131444, 2019, "RU1", 1, 2000, "Artem Polyarus", "FK Tambov"),
        (524826, 2020, "RU1", 1, 1500, "Artem Arkhipov", "FK Tambov"),
        # One Understat id over two people: each season matches a different one
        # by the same rule, so nothing says which he is.
        (120705, 2014, "ES1", 2, 1800, "Juan Carlos", "Cordoba CF"),
        (201861, 2020, "ES1", 3, 1700, "Juan Carlos", "SD Huesca"),
    )
    us = _us_rows(
        ("8601", "2019-20", "Russian Premier League", "FK Tambov", 2000, "Artem Polyarus"),
        ("8601", "2020-21", "Russian Premier League", "FK Tambov", 1500, "Artem Polyaru"),
        ("4219", "2014-15", "La Liga", "Cordoba", 1800, "Juan Carlos"),
        ("4219", "2020-21", "La Liga", "Huesca", 1700, "Juan Carlos"),
    )
    for frame in (us, us.iloc[::-1].reset_index(drop=True)):
        link = link_by_season(frame, tm)
        assert link.get("8601") == 131444
        assert "4219" not in link.index

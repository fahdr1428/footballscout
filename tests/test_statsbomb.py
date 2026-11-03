"""The event-level ETL: geometry, minutes and the derived metrics.

These run on hand-built fixtures - no network access - so they pin the
definitions rather than the feed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.statsbomb import (
    POSITION_GROUP_BY_SB, _minutes_from_lineups, in_box, in_final_third, is_progressive,
    parse_match, possession_time,
)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def test_progressive_thresholds_follow_the_stated_rule():
    # Own half to own half needs 30m (about 33 yards) of gain.
    assert not is_progressive([20, 40], [45, 40])          # ~27m gain
    assert is_progressive([10, 40], [50, 40])              # ~37m gain
    # Crossing halfway needs 15m.
    assert is_progressive([50, 40], [70, 40])              # ~18m gain
    assert not is_progressive([55, 40], [64, 40])          # ~8m gain
    # Inside the opposition half needs 10m.
    assert is_progressive([70, 40], [85, 40])
    assert not is_progressive([70, 40], [77, 40])
    # Backwards is never progressive.
    assert not is_progressive([90, 40], [60, 40])


def test_zone_helpers():
    assert in_box([110, 40]) and not in_box([110, 10])
    assert in_final_third([85, 20]) and not in_final_third([70, 20])


def test_every_statsbomb_position_maps_to_a_group():
    from src.config import POSITION_GROUPS
    for group, detailed in POSITION_GROUP_BY_SB.values():
        assert group in POSITION_GROUPS
        assert isinstance(detailed, str) and detailed


# --------------------------------------------------------------------------
# Minutes
# --------------------------------------------------------------------------

def _lineup(positions, player_id=1, name="Test Player"):
    return [{
        "team_id": 1, "team_name": "Test FC",
        "lineup": [{
            "player_id": player_id, "player_name": name, "player_nickname": None,
            "country": {"name": "England"}, "positions": positions,
        }],
    }]


def test_full_match_is_scaled_to_ninety_minutes():
    """StatsBomb's clock runs past 90 with stoppage; per-90 rates must not."""
    players = _minutes_from_lineups(
        _lineup([{"position": "Center Back", "from": "00:00", "to": None,
                  "start_reason": "Starting XI", "end_reason": "Final Whistle"}]),
        match_end=95.5,
    )
    assert players[1].minutes == pytest.approx(90.0)
    assert players[1].started is True


def test_substitute_and_starter_minutes_sum_to_one_slot():
    on = _minutes_from_lineups(
        _lineup([{"position": "Center Forward", "from": "60:00", "to": None,
                  "start_reason": "Substitution - On (Tactical)", "end_reason": "Final Whistle"}]),
        match_end=90.0,
    )
    off = _minutes_from_lineups(
        _lineup([{"position": "Center Forward", "from": "00:00", "to": "60:00",
                  "start_reason": "Starting XI", "end_reason": "Substitution - Off (Tactical)"}]),
        match_end=90.0,
    )
    assert on[1].minutes + off[1].minutes == pytest.approx(90.0)
    assert on[1].started is False


def test_a_spell_after_being_substituted_off_is_ignored():
    """A real defect in the feed: a player re-listed after coming off."""
    players = _minutes_from_lineups(
        _lineup([
            {"position": "Center Midfield", "from": "00:00", "to": "45:00",
             "start_reason": "Starting XI", "end_reason": "Substitution - Off (Tactical)"},
            {"position": "Center Midfield", "from": "47:40", "to": None,
             "start_reason": "Tactical Shift", "end_reason": "Final Whistle"},
        ]),
        match_end=90.0,
    )
    assert players[1].minutes == pytest.approx(45.0)


def test_temporary_exit_for_treatment_still_counts_the_return():
    players = _minutes_from_lineups(
        _lineup([
            {"position": "Left Back", "from": "00:00", "to": "20:00",
             "start_reason": "Starting XI", "end_reason": "Player Off"},
            {"position": "Left Back", "from": "22:00", "to": None,
             "start_reason": "Player On", "end_reason": "Final Whistle"},
        ]),
        match_end=90.0,
    )
    assert players[1].minutes == pytest.approx(88.0)


def test_overlapping_spells_are_never_double_counted():
    players = _minutes_from_lineups(
        _lineup([
            {"position": "Right Wing", "from": "00:00", "to": "60:00",
             "start_reason": "Starting XI", "end_reason": "Tactical Shift"},
            {"position": "Center Forward", "from": "50:00", "to": None,
             "start_reason": "Tactical Shift", "end_reason": "Final Whistle"},
        ]),
        match_end=90.0,
    )
    assert players[1].minutes == pytest.approx(90.0)


# --------------------------------------------------------------------------
# Possession and event parsing
# --------------------------------------------------------------------------

def _event(index, kind, minute, second, team, player=None, **extra):
    event = {
        "id": f"e{index}", "index": index, "period": 1, "minute": minute, "second": second,
        "type": {"name": kind}, "team": {"name": team},
        "possession": extra.pop("possession", 1),
        "possession_team": {"name": extra.pop("possession_team", team)},
    }
    if player:
        event["player"] = {"id": player[0], "name": player[1]}
    event.update(extra)
    return event


def test_possession_is_a_time_share_and_ignores_stoppages():
    events = [
        _event(1, "Pass", 0, 0, "A"),
        _event(2, "Pass", 0, 10, "A"),          # 10s to A
        _event(3, "Pass", 0, 20, "B", possession_team="B"),  # 10s to A (owner at start)
        _event(4, "Pass", 0, 30, "B", possession_team="B"),  # 10s to B
        _event(5, "Pass", 5, 0, "B", possession_team="B"),   # >60s gap: dropped
    ]
    seconds = possession_time(events)
    assert seconds["A"] == pytest.approx(20.0)
    assert seconds["B"] == pytest.approx(10.0)


def test_parse_match_derives_xa_from_the_shot_it_created():
    lineups = [
        {"team_id": 1, "team_name": "A", "lineup": [
            {"player_id": 1, "player_name": "Passer", "player_nickname": None,
             "country": {"name": "England"},
             "positions": [{"position": "Center Attacking Midfield", "from": "00:00", "to": None,
                            "start_reason": "Starting XI", "end_reason": "Final Whistle"}]},
            {"player_id": 2, "player_name": "Shooter", "player_nickname": None,
             "country": {"name": "England"},
             "positions": [{"position": "Striker", "from": "00:00", "to": None,
                            "start_reason": "Starting XI", "end_reason": "Final Whistle"}]},
        ]},
        {"team_id": 2, "team_name": "B", "lineup": [
            {"player_id": 3, "player_name": "Keeper", "player_nickname": None,
             "country": {"name": "England"},
             "positions": [{"position": "Goalkeeper", "from": "00:00", "to": None,
                            "start_reason": "Starting XI", "end_reason": "Final Whistle"}]},
        ]},
    ]
    events = [
        _event(1, "Pass", 10, 0, "A", (1, "Passer"), location=[95, 40],
               **{"pass": {"end_location": [110, 40], "length": 15, "shot_assist": True,
                           "recipient": {"id": 2, "name": "Shooter"}}}),
        _event(2, "Shot", 10, 2, "A", (2, "Shooter"), location=[110, 40],
               shot={"statsbomb_xg": 0.42, "key_pass_id": "e1", "type": {"name": "Open Play"},
                     "outcome": {"name": "Goal"}, "end_location": [120, 40]}),
        _event(3, "Half End", 47, 0, "A"),
    ]
    players, _seconds = parse_match(events, lineups)

    passer, shooter, keeper = players[1].counters, players[2].counters, players[3].counters
    assert passer["key_passes"] == 1
    assert passer["xa"] == pytest.approx(0.42)      # xA is the created shot's xG
    assert passer["sca"] == 1                        # and it counts as a shot-creating action
    assert shooter["goals"] == 1 and shooter["np_goals"] == 1
    assert shooter["npxg"] == pytest.approx(0.42)
    assert shooter["shots_on_target"] == 1
    # The keeper's numbers come from the opponent's shots, not his own events.
    assert keeper["gk_goals_against"] == 1
    assert keeper["gk_shots_on_target_against"] == 1


def test_penalties_are_kept_out_of_non_penalty_totals():
    lineups = [{"team_id": 1, "team_name": "A", "lineup": [
        {"player_id": 1, "player_name": "Taker", "player_nickname": None,
         "country": {"name": "England"},
         "positions": [{"position": "Striker", "from": "00:00", "to": None,
                        "start_reason": "Starting XI", "end_reason": "Final Whistle"}]}]}]
    events = [
        _event(1, "Shot", 10, 0, "A", (1, "Taker"), location=[108, 40],
               shot={"statsbomb_xg": 0.78, "type": {"name": "Penalty"},
                     "outcome": {"name": "Goal"}, "end_location": [120, 40]}),
        _event(2, "Half End", 47, 0, "A"),
    ]
    players, _ = parse_match(events, lineups)
    counters = players[1].counters
    assert counters["goals"] == 1 and counters["pens_scored"] == 1 and counters["pens_taken"] == 1
    assert counters["np_goals"] == 0 and counters["shots"] == 0 and counters["npxg"] == 0
    assert counters["xg"] == pytest.approx(0.78)


def test_under_pressure_passes_are_counted_separately():
    """Pressure is a flag only event data carries; it must not leak into totals."""
    lineups = [{"team_id": 1, "team_name": "A", "lineup": [
        {"player_id": 1, "player_name": "Passer", "player_nickname": None,
         "country": {"name": "England"},
         "positions": [{"position": "Center Midfield", "from": "00:00", "to": None,
                        "start_reason": "Starting XI", "end_reason": "Final Whistle"}]}]}]
    events = [
        _event(1, "Pass", 1, 0, "A", (1, "Passer"), location=[60, 40],
               **{"pass": {"end_location": [70, 40], "length": 10}}),
        _event(2, "Pass", 2, 0, "A", (1, "Passer"), location=[60, 40], under_pressure=True,
               **{"pass": {"end_location": [70, 40], "length": 10}}),
        _event(3, "Pass", 3, 0, "A", (1, "Passer"), location=[60, 40], under_pressure=True,
               **{"pass": {"end_location": [70, 40], "length": 10,
                           "outcome": {"name": "Incomplete"}}}),
        _event(4, "Half End", 47, 0, "A"),
    ]
    players, _ = parse_match(events, lineups)
    counters = players[1].counters
    assert counters["passes_attempted"] == 3
    assert counters["passes_completed"] == 2
    assert counters["passes_under_pressure"] == 2
    assert counters["passes_completed_under_pressure"] == 1


def test_set_piece_chances_are_split_from_open_play():
    lineups = [{"team_id": 1, "team_name": "A", "lineup": [
        {"player_id": 1, "player_name": "Striker", "player_nickname": None,
         "country": {"name": "England"},
         "positions": [{"position": "Striker", "from": "00:00", "to": None,
                        "start_reason": "Starting XI", "end_reason": "Final Whistle"}]}]}]
    events = [
        _event(1, "Shot", 10, 0, "A", (1, "Striker"), location=[110, 40],
               play_pattern={"name": "From Corner"},
               shot={"statsbomb_xg": 0.20, "type": {"name": "Open Play"},
                     "outcome": {"name": "Saved"}, "end_location": [120, 40]}),
        _event(2, "Shot", 20, 0, "A", (1, "Striker"), location=[108, 40],
               play_pattern={"name": "Regular Play"},
               shot={"statsbomb_xg": 0.30, "type": {"name": "Open Play"},
                     "outcome": {"name": "Goal"}, "end_location": [120, 40]}),
        _event(3, "Half End", 47, 0, "A"),
    ]
    players, _ = parse_match(events, lineups)
    counters = players[1].counters
    assert counters["npxg"] == pytest.approx(0.50)
    assert counters["npxg_set_piece"] == pytest.approx(0.20)
    assert counters["npxg_open_play"] == pytest.approx(0.30)


# --------------------------------------------------------------------------
# The men's 2015/16 season and its Transfermarkt join
# --------------------------------------------------------------------------

def _season_rows():
    return pd.DataFrame({
        "player_id": ["SB1", "SB2", "SB3", "SB4", "SB5"],
        "player": ["Lionel Messi", "Sam Kerr", "Jamie Vardy", "Sunil Chhetri", "Old Timer"],
        "gender": ["male", "female", "male", "male", "male"],
        "season": ["2015-16", "2019-20", "2015-16", "2021-22", "2015-16"],
        "league": ["La Liga", "FA WSL", "Premier League", "Indian Super League", "Serie A"],
        "team": ["Barcelona", "Chelsea", "Leicester City", "Bengaluru", "Juventus"],
        "minutes": [2729, 1500, 3143, 1600, 1000],
        "age": [None] * 5, "height_cm": [None] * 5,
    })


def test_the_2015_16_season_is_the_four_mens_leagues_only():
    from src.statsbomb import season_2015_16

    kept = season_2015_16(_season_rows())
    assert kept["player"].tolist() == ["Lionel Messi", "Jamie Vardy", "Old Timer"]


def test_transfermarkt_fields_come_from_the_matched_player_and_bad_ages_are_withdrawn(monkeypatch, tmp_path):
    from src import statsbomb, tm_history

    rows = statsbomb.season_2015_16(_season_rows())
    profiles = pd.DataFrame({
        "tm_id": [28003, 1, 2], "tm_name": ["Lionel Messi", "Jamie Vardy", "Old Timer"],
        "date_of_birth": pd.to_datetime(["1987-06-24", "1987-01-11", "1950-01-01"]),
        "height_cm": [170.0, 179.0, 180.0], "foot": ["left", "right", "right"],
        "nationality": ["Argentina", "England", "Italy"], "tm_bucket": None, "tm_detail": None,
    })
    history = pd.DataFrame({"tm_id": [28003, 28003, 1], "date": pd.to_datetime(
        ["2016-01-01", "2026-05-01", "2016-02-01"]), "value": [120e6, 15e6, 18e6]})
    monkeypatch.setattr(tm_history, "load_profiles", lambda cache, progress=print: profiles)
    monkeypatch.setattr(tm_history, "load_valuations", lambda cache, progress=print: history)
    monkeypatch.setattr(tm_history, "link_players", lambda frame, prof: pd.DataFrame({
        "player_id": ["SB1", "SB3", "SB5"], "tm_id": [28003, 1, 2], "how": "season"}))

    out, coverage = statsbomb.attach_transfermarkt(rows, tmp_path, progress=lambda *_: None)
    messi = out.set_index("player").loc["Lionel Messi"]
    assert messi["age"] == 28.5 and messi["foot"] == "left" and messi["height_cm"] == 170
    assert messi["market_value_eur"] == 120e6          # in force on 1 July 2016
    assert messi["latest_value_eur"] == 15e6
    old = out.set_index("player").loc["Old Timer"]     # 66 years old: the wrong man
    assert pd.isna(old["age"]) and pd.isna(old["height_cm"]) and pd.isna(old["market_value_eur"])
    assert coverage["rejected_implausible_age"] == 1


def test_the_built_2015_16_file_is_matched_for_nearly_every_minute():
    from src.config import STATSBOMB_2015_16_CSV

    if not STATSBOMB_2015_16_CSV.exists():
        pytest.skip("run scripts/build_statsbomb_2015_16.py first")
    frame = pd.read_csv(STATSBOMB_2015_16_CSV)
    assert set(frame["league"]) == {"Premier League", "La Liga", "Serie A", "Ligue 1"}
    assert set(frame["season"]) == {"2015-16"} and set(frame["gender"]) == {"male"}
    assert frame.loc[frame["age"].notna(), "minutes"].sum() / frame["minutes"].sum() > 0.97
    assert frame["age"].dropna().between(15, 45).all()

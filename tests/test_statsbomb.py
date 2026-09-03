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

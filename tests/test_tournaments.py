"""Tournament event data: the defensive counts read off the events, and the link to Transfermarkt."""

from __future__ import annotations

import pandas as pd

from src.statsbomb import TOURNAMENTS, link_internationals, parse_match


def _event(index, kind, minute, team, player=None, **extra):
    event = {"id": f"e{index}", "index": index, "period": 1, "minute": minute, "second": 0,
             "type": {"name": kind}, "team": {"name": team}, "possession": 1,
             "possession_team": {"name": extra.pop("possession_team", team)}}
    if player:
        event["player"] = {"id": player[0], "name": player[1]}
    event.update(extra)
    return event


def _starter(pid, name, position, country="Spain", nickname=None):
    return {"player_id": pid, "player_name": name, "player_nickname": nickname, "country": {"name": country},
            "positions": [{"position": position, "from": "00:00", "to": None,
                           "start_reason": "Starting XI", "end_reason": "Final Whistle"}]}


LINEUPS = [
    {"team_id": 1, "team_name": "Spain", "lineup": [_starter(1, "Rodrigo Hernández Cascante", "Center Defensive Midfield", nickname="Rodri")]},
    {"team_id": 2, "team_name": "France", "lineup": [_starter(2, "Attacker", "Center Forward", country="France")]},
]


def test_defending_is_counted_where_it_happens_and_against_whom():
    rodri, attacker = (1, "Rodri"), (2, "Attacker")
    events = [
        # A failed dribble, stopped by his tackle in his own third.
        _event(1, "Dribble", 5, "France", attacker, location=[90, 40], dribble={"outcome": {"name": "Incomplete"}},
               related_events=["e2"]),
        _event(2, "Duel", 5, "Spain", rodri, location=[30, 40], related_events=["e1"],
               duel={"type": {"name": "Tackle"}, "outcome": {"name": "Won"}}),
        # Beaten once.
        _event(3, "Dribbled Past", 10, "Spain", rodri, location=[50, 40]),
        # A tackle high up the pitch, not on a dribbler.
        _event(4, "Duel", 12, "Spain", rodri, location=[95, 40], duel={"type": {"name": "Tackle"}, "outcome": {"name": "Lost In Play"}}),
        # A shot blocked and a pass blocked.
        _event(5, "Shot", 20, "France", attacker, location=[100, 40], related_events=["e6"],
               shot={"statsbomb_xg": 0.1, "type": {"name": "Open Play"}, "outcome": {"name": "Blocked"}}),
        _event(6, "Block", 20, "Spain", rodri, location=[20, 40], related_events=["e5"]),
        _event(7, "Block", 25, "Spain", rodri, location=[45, 40]),
        # Pressing: once high, once straight after losing the ball.
        _event(8, "Pressure", 30, "Spain", rodri, location=[85, 40], possession_team="France"),
        _event(9, "Pressure", 31, "Spain", rodri, location=[55, 40], counterpress=True, possession_team="France"),
        # Ball won back in the opponent's half, and in his own.
        _event(10, "Ball Recovery", 40, "Spain", rodri, location=[70, 40]),
        _event(11, "Ball Recovery", 41, "Spain", rodri, location=[35, 40]),
        _event(12, "Foul Won", 42, "Spain", rodri, location=[60, 40]),
        _event(13, "Half End", 47, "Spain"),
    ]
    players, _ = parse_match(events, LINEUPS)
    c = players[1].counters
    assert (c["tackles"], c["tackles_won"]) == (2, 1)
    assert (c["tackles_def_third"], c["tackles_mid_third"], c["tackles_att_third"]) == (1, 0, 1)
    assert (c["dribblers_tackled"], c["dribblers_challenged"]) == (1, 2)     # one stopped, one got past
    assert (c["blocks"], c["shots_blocked"]) == (2, 1)
    assert (c["pressures"], c["pressures_att_third"], c["counterpressures"]) == (2, 1, 1)
    assert (c["ball_recoveries"], c["recoveries_opp_half"]) == (2, 1)
    assert c["fouls_won"] == 1
    # Touches by third, from the same possession gains as touches.
    assert c["touches_def_third"] + c["touches_mid_third"] + c["touches_att_third"] == c["touches"]
    assert players[1].player == "Rodri" and players[1].full_name == "Rodrigo Hernández Cascante"


def _profiles(rows):
    return pd.DataFrame(rows, columns=["tm_id", "tm_name", "date_of_birth", "nationality"])


def test_a_tournament_player_is_linked_only_when_one_person_fits():
    frame = pd.DataFrame([
        {"player_id": "SB1", "player": "Rodri", "full_name": "Rodrigo Hernández Cascante", "season": "2023-24", "nationality": "Spain"},
        {"player_id": "SB2", "player": "Dayotchanculle Upamecano", "full_name": "Dayotchanculle Upamecano", "season": "2023-24", "nationality": "France"},
        {"player_id": "SB3", "player": "João Silva", "full_name": "João Silva", "season": "2023-24", "nationality": "Portugal"},
        {"player_id": "SB4", "player": "Evan N'Dicka", "full_name": "Obite Evan Ndicka", "season": "2023-24", "nationality": "Côte d'Ivoire"},
    ])
    profiles = _profiles([
        (357565, "Rodri", "1996-06-22", "Spain"),
        (999, "Rodri", "1960-01-01", "Spain"),             # same name, an impossible age: not him
        (344695, "Dayot Upamecano", "1998-10-27", "France"),
        (11, "João Silva", "1999-01-01", "Portugal"),      # two João Silvas of an age: neither is taken
        (12, "João Silva", "2001-01-01", "Portugal"),
        (588, "Evan Ndicka", "1999-08-20", "Cote d'Ivoire"),
    ])
    link = link_internationals(frame, profiles).set_index("player_id")["tm_id"].to_dict()
    assert link == {"SB1": 357565, "SB2": 344695, "SB4": 588}


def test_tournaments_are_filed_under_the_club_season_they_close():
    seasons = {t.league: t.season for t in TOURNAMENTS}
    assert seasons["World Cup 2022"] == "2022-23"
    assert seasons["Euro 2024"] == seasons["Copa America 2024"] == seasons["Africa Cup of Nations 2023"] == "2023-24"

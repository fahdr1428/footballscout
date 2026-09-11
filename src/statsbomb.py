"""
Real player data from the StatsBomb Open Data event feed.

    Data provided by StatsBomb - https://github.com/statsbomb/open-data
    Free for public use under the StatsBomb Open Data user agreement.

This module turns raw event data into the same player-season schema the rest of
the platform uses. Nothing is downloaded from a paid feed and nothing is
scraped: the open-data repository publishes complete, licensed event files.

Every derived metric is defined here in code rather than taken on trust from a
provider's summary table, so each one can be argued with:

* **Minutes** come from the lineup position spells (absolute match clock),
  rescaled so a full match is 90 minutes - StatsBomb's clock runs to ~95 with
  stoppage, and per-90 rates would otherwise be ~5% low.
* **Progressive passes and carries** use the Wyscout thresholds: the ball must
  end at least 30m closer to goal when the action starts and ends in the
  player's own half, 15m when it crosses halfway, and 10m inside the
  opposition half.
* **xA** is expected assists in its literal sense: the StatsBomb xG of the shot
  each key pass created (`shot.key_pass_id` links the two).
* **Shot-creating actions** are the last two attacking actions (completed pass,
  completed take-on, foul won) by the shooting team inside the same possession.
* **Successful pressures** are pressures after which the pressing team holds
  possession within five seconds.
* **Errors** are a miscontrol, dispossession or failed pass followed within
  five seconds by a shot for the opposition - i.e. an error leading to a shot.
* **Team possession** is the team's share of on-ball involvements across the
  season, used only by the possession-adjusted defensive metrics.

Post-shot xG is a paid StatsBomb feature and is absent here, so goalkeeper
`gk_psxg` stays missing rather than being invented; the platform drops features
that are missing for a whole position group.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

OPEN_DATA = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
ATTRIBUTION = (
    "Data provided by StatsBomb Open Data "
    "(https://github.com/statsbomb/open-data), free for public use."
)

# --------------------------------------------------------------------------
# Which competitions to ingest
# --------------------------------------------------------------------------
# Only competition-seasons where the open data covers the *whole* league are
# included. Seasons where the feed carries a single club (Barcelona's La Liga
# years, Leverkusen 2023/24) would give every other team a phantom squad, so
# they are deliberately left out.

@dataclass(frozen=True)
class Competition:
    competition_id: int
    season_id: int
    league: str
    country: str
    gender: str
    season: str
    tier: int
    strength: float


COMPETITIONS: list[Competition] = [
    Competition(2, 27, "Premier League", "England", "male", "2015-16", 1, 1.00),
    Competition(11, 27, "La Liga", "Spain", "male", "2015-16", 1, 0.98),
    Competition(12, 27, "Serie A", "Italy", "male", "2015-16", 1, 0.95),
    Competition(7, 27, "Ligue 1", "France", "male", "2015-16", 1, 0.90),
    Competition(1238, 108, "Indian Super League", "India", "male", "2021-22", 3, 0.55),
    Competition(37, 4, "FA WSL", "England", "female", "2018-19", 1, 0.90),
    Competition(37, 42, "FA WSL", "England", "female", "2019-20", 1, 0.92),
    Competition(37, 90, "FA WSL", "England", "female", "2020-21", 1, 0.94),
    Competition(37, 281, "FA WSL", "England", "female", "2023-24", 1, 1.00),
    Competition(49, 3, "NWSL", "United States", "female", "2018", 1, 0.90),
    Competition(49, 107, "NWSL", "United States", "female", "2023", 1, 0.98),
    Competition(182, 281, "Liga F", "Spain", "female", "2023-24", 1, 0.94),
    Competition(131, 281, "Serie A Women", "Italy", "female", "2023-24", 2, 0.82),
    Competition(135, 281, "Frauen Bundesliga", "Germany", "female", "2023-24", 1, 0.96),
]

# --------------------------------------------------------------------------
# Pitch geometry (StatsBomb coordinates: 120 x 80 yards, attacking left to right)
# --------------------------------------------------------------------------

PITCH_X, PITCH_Y = 120.0, 80.0
GOAL = np.array([120.0, 40.0])
YARD_M = 0.9144
FINAL_THIRD_X = 80.0
BOX_X, BOX_Y_LOW, BOX_Y_HIGH = 102.0, 18.0, 62.0
HALFWAY_X = 60.0
LONG_PASS_YARDS = 30.0          # FBref's long-pass threshold
GK_LAUNCH_YARDS = 40.0
PRESSURE_WINDOW_SECONDS = 5.0
ERROR_WINDOW_SECONDS = 5.0

# Wyscout progressive thresholds, in metres of gain towards the goal.
PROGRESSIVE_OWN_HALF_M = 30.0
PROGRESSIVE_CROSSING_M = 15.0
PROGRESSIVE_FINAL_M = 10.0

POSITION_GROUP_BY_SB = {
    "Goalkeeper": ("GK", "GK"),
    "Right Back": ("FB", "RB"),
    "Left Back": ("FB", "LB"),
    "Right Wing Back": ("FB", "RWB"),
    "Left Wing Back": ("FB", "LWB"),
    "Center Back": ("CB", "CB"),
    "Right Center Back": ("CB", "RCB"),
    "Left Center Back": ("CB", "LCB"),
    "Right Defensive Midfield": ("DM", "DM"),
    "Center Defensive Midfield": ("DM", "DM"),
    "Left Defensive Midfield": ("DM", "DM"),
    "Right Center Midfield": ("CM", "CM"),
    "Center Midfield": ("CM", "CM"),
    "Left Center Midfield": ("CM", "CM"),
    "Right Midfield": ("WM", "RM"),
    "Left Midfield": ("WM", "LM"),
    "Right Wing": ("W", "RW"),
    "Left Wing": ("W", "LW"),
    "Right Attacking Midfield": ("AM", "AM"),
    "Center Attacking Midfield": ("AM", "AM"),
    "Left Attacking Midfield": ("AM", "AM"),
    "Secondary Striker": ("SS", "SS"),
    "Right Center Forward": ("FW", "CF"),
    "Center Forward": ("FW", "CF"),
    "Left Center Forward": ("FW", "CF"),
    "Striker": ("FW", "ST"),
}

# Events that count as the player getting the ball (our "touches" definition).
POSSESSION_GAIN_TYPES = {"Ball Receipt*", "Ball Recovery", "Interception", "Clearance", "Block"}

SHOT_ON_TARGET = {"Goal", "Saved", "Saved to Post", "Saved Off Target"}
# Play patterns that mean the chance came from a dead ball. Throw-ins and goal
# kicks are deliberately excluded: a shot several passes after a throw-in is
# open play by any normal reading, and counting it as a set piece would put the
# open-play share far below the ~75-80% the game actually produces.
SET_PIECE_PATTERNS = {"From Corner", "From Free Kick"}
GK_SAVE_TYPES = {"Shot Saved", "Save", "Penalty Saved", "Shot Saved to Post", "Shot Saved Off Target"}
GK_CLAIM_TYPES = {"Collected", "Punch", "Claim", "Smother", "Keeper Sweeper"}
DUEL_WON = {"Won", "Success", "Success In Play", "Success Out"}


# --------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------

def fetch_json(url: str, retries: int = 5, backoff: float = 2.0):
    """GET a JSON document, retrying transient network failures."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "scouting-platform/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
            last = error
            if isinstance(error, urllib.error.HTTPError) and error.code == 404:
                raise
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def list_matches(competition: Competition) -> list[dict]:
    return fetch_json(f"{OPEN_DATA}/matches/{competition.competition_id}/{competition.season_id}.json")


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _distance_to_goal(point) -> float:
    return float(np.linalg.norm(np.asarray(point[:2], dtype=float) - GOAL)) * YARD_M


def is_progressive(start, end) -> bool:
    """Wyscout progressive-action definition, applied to passes and carries."""
    if start is None or end is None:
        return False
    gain = _distance_to_goal(start) - _distance_to_goal(end)
    start_own_half = start[0] < HALFWAY_X
    end_own_half = end[0] < HALFWAY_X
    if start_own_half and end_own_half:
        return gain >= PROGRESSIVE_OWN_HALF_M
    if start_own_half:
        return gain >= PROGRESSIVE_CROSSING_M
    return gain >= PROGRESSIVE_FINAL_M


def in_box(point) -> bool:
    return point is not None and point[0] >= BOX_X and BOX_Y_LOW <= point[1] <= BOX_Y_HIGH


def in_final_third(point) -> bool:
    return point is not None and point[0] >= FINAL_THIRD_X


def _seconds(event: dict) -> float:
    """Absolute match clock in seconds (periods are stacked, as StatsBomb does)."""
    return float(event.get("minute", 0)) * 60 + float(event.get("second", 0))


def _clock_minutes(stamp: str | None) -> float | None:
    """Lineup timestamps are the absolute match clock, e.g. '74:32'."""
    if not stamp:
        return None
    parts = stamp.split(":")
    try:
        return int(parts[0]) + int(parts[1]) / 60
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------
# Per-match parsing
# --------------------------------------------------------------------------

COUNTER_KEYS = [
    "goals", "np_goals", "pens_scored", "pens_taken", "xg", "npxg", "assists", "xa",
    "shots", "shots_on_target", "sca", "gca", "key_passes", "touches_att_pen",
    "progressive_carries", "carries_into_final_third", "carries_into_pen_area",
    "dribbles_completed", "dribbles_attempted", "passes_attempted", "passes_completed",
    "progressive_passes", "passes_into_final_third", "passes_into_pen_area", "through_balls",
    "crosses", "switches", "long_passes_attempted", "long_passes_completed",
    "progressive_receptions", "touches", "miscontrols", "dispossessed", "tackles",
    "passes_under_pressure", "passes_completed_under_pressure", "npxg_open_play", "npxg_set_piece",
    "tackles_won", "interceptions", "blocks", "clearances", "ball_recoveries", "pressures",
    "pressures_successful", "aerials_won", "aerials_lost", "fouls_committed", "errors",
    "gk_shots_on_target_against", "gk_saves", "gk_goals_against", "gk_crosses_faced",
    "gk_crosses_stopped", "gk_def_actions_outside_box", "gk_launches_attempted",
    "gk_launches_completed",
]


@dataclass
class PlayerMatch:
    """One player's involvement in one match."""

    player_id: int
    player: str
    team: str
    nationality: str = ""
    minutes: float = 0.0
    started: bool = False
    position_minutes: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counters: dict[str, float] = field(default_factory=lambda: dict.fromkeys(COUNTER_KEYS, 0.0))


# Once one of these ends a spell the player has left the pitch for good.
# "Player Off" is a temporary exit for treatment and is paired with "Player On",
# so it is deliberately not in this set.
TERMINAL_END_REASONS = ("Substitution - Off", "Sent Off", "Red Card", "Second Yellow")


def _minutes_from_lineups(lineups: list[dict], match_end: float) -> dict[int, PlayerMatch]:
    """Minutes and positions per player, rescaled so a full match is 90 minutes.

    The open-data lineup feed occasionally lists a spell that starts *after* the
    player was substituted off - a recording artefact rather than a re-entry,
    which the laws of the game do not allow. Spells are therefore taken in
    order, clamped so they cannot overlap, and dropped once a spell has ended
    with the player leaving the pitch.
    """
    scale = (90.0 / match_end) if match_end > 0 else 1.0
    players: dict[int, PlayerMatch] = {}
    for team in lineups:
        for entry in team["lineup"]:
            spells = entry.get("positions") or []
            if not spells:
                continue
            record = PlayerMatch(
                player_id=entry["player_id"],
                player=entry.get("player_nickname") or entry["player_name"],
                team=team["team_name"],
                nationality=(entry.get("country") or {}).get("name", ""),
            )
            ordered_spells = sorted(spells, key=lambda s: _clock_minutes(s.get("from")) or 0.0)
            cursor, finished = 0.0, False
            for spell in ordered_spells:
                if finished:
                    break
                start = max(_clock_minutes(spell.get("from")) or 0.0, cursor)
                end = _clock_minutes(spell.get("to"))
                end = match_end if end is None else min(end, match_end)
                played = max(0.0, end - start) * scale
                record.minutes += played
                record.position_minutes[spell.get("position") or "Unknown"] += played
                cursor = max(cursor, end)
                if spell.get("start_reason") == "Starting XI":
                    record.started = True
                reason = spell.get("end_reason") or ""
                if any(token in reason for token in TERMINAL_END_REASONS):
                    finished = True
            record.minutes = min(record.minutes, 90.0)
            if record.minutes > 0:
                players[record.player_id] = record
    return players


def _match_end_minute(events: list[dict]) -> float:
    ends = [e for e in events if e["type"]["name"] == "Half End"]
    if ends:
        return max(float(e.get("minute", 0)) + float(e.get("second", 0)) / 60 for e in ends)
    return max((float(e.get("minute", 0)) for e in events), default=90.0)


MAX_GAP_SECONDS = 60.0  # a longer gap between events is a stoppage, not possession


def possession_time(ordered: list[dict]) -> dict[str, float]:
    """Seconds of possession per team, from the gaps between consecutive events.

    Each interval between two events in the same period is credited to whichever
    team was in possession at its start. Gaps longer than a minute are dropped
    as stoppages, and period changes end an interval, so half-time is never
    counted for anyone. This is a time share - the way possession is normally
    quoted - rather than a share of touches, which exaggerates the spread.
    """
    seconds: dict[str, float] = defaultdict(float)
    for current, following in zip(ordered, ordered[1:]):
        if current.get("period") != following.get("period"):
            continue
        team = (current.get("possession_team") or {}).get("name")
        if not team:
            continue
        gap = _seconds(following) - _seconds(current)
        if 0 < gap <= MAX_GAP_SECONDS:
            seconds[team] += gap
    return dict(seconds)


def parse_match(events: list[dict], lineups: list[dict]) -> tuple[dict[int, PlayerMatch], dict[str, float]]:
    """Aggregate one match's events into per-player counters and possession time."""
    match_end = _match_end_minute(events)
    players = _minutes_from_lineups(lineups, match_end)
    team_touches: dict[str, float] = defaultdict(float)

    def bump(player, key, value=1.0):
        if player and player["id"] in players:
            players[player["id"]].counters[key] += value

    pass_owner: dict[str, dict] = {}
    for event in events:
        if event["type"]["name"] == "Pass" and event.get("player"):
            pass_owner[event["id"]] = event

    # Index events by team-second for the pressure and error look-aheads.
    ordered = sorted(events, key=lambda e: e.get("index", 0))
    times = np.array([_seconds(e) for e in ordered])

    for position, event in enumerate(ordered):
        kind = event["type"]["name"]
        player = event.get("player")
        team = event.get("team", {}).get("name")
        location = event.get("location")

        if kind in POSSESSION_GAIN_TYPES and team:
            incomplete = kind == "Ball Receipt*" and event.get("ball_receipt", {}).get("outcome")
            if not incomplete:
                team_touches[team] += 1
                bump(player, "touches")
                if kind == "Ball Receipt*" and in_box(location):
                    bump(player, "touches_att_pen")

        if kind == "Pass":
            detail = event.get("pass", {})
            end = detail.get("end_location")
            complete = "outcome" not in detail
            length = float(detail.get("length") or 0.0)
            bump(player, "passes_attempted")
            # StatsBomb flags an event as under_pressure when an opponent is
            # actively closing the player down - a distinction only event data
            # can make, and one of the more useful things it buys a scout.
            if event.get("under_pressure"):
                bump(player, "passes_under_pressure")
                if complete:
                    bump(player, "passes_completed_under_pressure")
            if complete:
                bump(player, "passes_completed")
                if is_progressive(location, end):
                    bump(player, "progressive_passes")
                    recipient = detail.get("recipient")
                    if recipient and recipient.get("id") in players:
                        players[recipient["id"]].counters["progressive_receptions"] += 1
                if in_final_third(end) and not in_final_third(location):
                    bump(player, "passes_into_final_third")
                if in_box(end) and not in_box(location):
                    bump(player, "passes_into_pen_area")
            if detail.get("cross"):
                bump(player, "crosses")
            if detail.get("switch"):
                bump(player, "switches")
            if detail.get("through_ball") or (detail.get("technique", {}) or {}).get("name") == "Through Ball":
                bump(player, "through_balls")
            if length >= LONG_PASS_YARDS:
                bump(player, "long_passes_attempted")
                if complete:
                    bump(player, "long_passes_completed")
            if detail.get("shot_assist") or detail.get("goal_assist"):
                bump(player, "key_passes")
            if detail.get("goal_assist"):
                bump(player, "assists")
            if detail.get("aerial_won"):
                bump(player, "aerials_won")

        elif kind == "Carry":
            end = event.get("carry", {}).get("end_location")
            if is_progressive(location, end):
                bump(player, "progressive_carries")
                if in_final_third(end) and not in_final_third(location):
                    bump(player, "carries_into_final_third")
            if in_box(end) and not in_box(location):
                bump(player, "carries_into_pen_area")
                bump(player, "touches_att_pen")

        elif kind == "Shot":
            detail = event.get("shot", {})
            outcome = (detail.get("outcome") or {}).get("name")
            shot_type = (detail.get("type") or {}).get("name")
            xg = float(detail.get("statsbomb_xg") or 0.0)
            penalty = shot_type == "Penalty"
            bump(player, "xg", xg)
            if penalty:
                bump(player, "pens_taken")
                if outcome == "Goal":
                    bump(player, "pens_scored")
                    bump(player, "goals")
            else:
                bump(player, "shots")
                bump(player, "npxg", xg)
                # Chances built in open play are a different skill from chances
                # that arrive from a corner, so the two are kept apart.
                pattern = (event.get("play_pattern") or {}).get("name", "")
                set_piece = pattern in SET_PIECE_PATTERNS or shot_type in {"Free Kick", "Corner"}
                bump(player, "npxg_set_piece" if set_piece else "npxg_open_play", xg)
                if outcome in SHOT_ON_TARGET:
                    bump(player, "shots_on_target")
                if outcome == "Goal":
                    bump(player, "goals")
                    bump(player, "np_goals")
            if in_box(location):
                bump(player, "touches_att_pen")
            if detail.get("aerial_won"):
                bump(player, "aerials_won")

            # xA: the creating pass inherits this shot's xG.
            key_pass = pass_owner.get(detail.get("key_pass_id"))
            if key_pass is not None:
                bump(key_pass.get("player"), "xa", xg)

            # Shot-creating actions: the last two attacking actions in this possession.
            credited: set[int] = set()
            for back in range(position - 1, max(-1, position - 40), -1):
                previous = ordered[back]
                if previous.get("possession") != event.get("possession"):
                    break
                if previous.get("team", {}).get("name") != team or not previous.get("player"):
                    continue
                previous_kind = previous["type"]["name"]
                creative = (
                    (previous_kind == "Pass" and "outcome" not in previous.get("pass", {}))
                    or (previous_kind == "Dribble"
                        and (previous.get("dribble", {}).get("outcome") or {}).get("name") == "Complete")
                    or previous_kind == "Foul Won"
                )
                if not creative or previous["player"]["id"] in credited:
                    continue
                credited.add(previous["player"]["id"])
                bump(previous["player"], "sca")
                if outcome == "Goal":
                    bump(previous["player"], "gca")
                if len(credited) >= 2:
                    break

        elif kind == "Dribble":
            bump(player, "dribbles_attempted")
            if (event.get("dribble", {}).get("outcome") or {}).get("name") == "Complete":
                bump(player, "dribbles_completed")

        elif kind == "Duel":
            detail = event.get("duel", {})
            duel_type = (detail.get("type") or {}).get("name")
            outcome = (detail.get("outcome") or {}).get("name")
            if duel_type == "Tackle":
                bump(player, "tackles")
                if outcome in DUEL_WON:
                    bump(player, "tackles_won")
            elif duel_type == "Aerial Lost":
                bump(player, "aerials_lost")

        elif kind == "Interception":
            outcome = (event.get("interception", {}).get("outcome") or {}).get("name")
            if outcome not in {"Lost", "Lost Out"}:
                bump(player, "interceptions")

        elif kind == "Block":
            bump(player, "blocks")
        elif kind == "Clearance":
            bump(player, "clearances")
            if event.get("clearance", {}).get("aerial_won"):
                bump(player, "aerials_won")
        elif kind == "Ball Recovery":
            if not event.get("ball_recovery", {}).get("recovery_failure"):
                bump(player, "ball_recoveries")
        elif kind == "Miscontrol":
            bump(player, "miscontrols")
            if event.get("miscontrol", {}).get("aerial_won"):
                bump(player, "aerials_won")
        elif kind == "Dispossessed":
            bump(player, "dispossessed")
        elif kind == "Foul Committed":
            bump(player, "fouls_committed")

        elif kind == "Pressure":
            bump(player, "pressures")
            # Successful when the pressing team holds the ball within five seconds.
            window_end = times[position] + PRESSURE_WINDOW_SECONDS
            for ahead in range(position + 1, len(ordered)):
                if times[ahead] > window_end:
                    break
                later = ordered[ahead]
                if later.get("possession_team", {}).get("name") == team and later.get("player"):
                    if later["type"]["name"] in {"Pass", "Carry", "Ball Receipt*", "Shot", "Dribble"}:
                        bump(player, "pressures_successful")
                        break

        elif kind == "Goal Keeper":
            detail = event.get("goalkeeper", {})
            gk_type = (detail.get("type") or {}).get("name")
            if gk_type in GK_SAVE_TYPES:
                bump(player, "gk_saves")
            if gk_type in GK_CLAIM_TYPES:
                bump(player, "gk_crosses_stopped")
            if gk_type == "Keeper Sweeper" or (location and location[0] > 18.0 and gk_type not in GK_SAVE_TYPES):
                if location and location[0] > 18.0:
                    bump(player, "gk_def_actions_outside_box")

    _apply_errors(ordered, times, players)
    _apply_goalkeeper_context(ordered, lineups, players)
    return players, possession_time(ordered)


def _apply_errors(ordered, times, players) -> None:
    """An error is a loss of the ball followed by an opposition shot within 5s."""
    loss_types = {"Miscontrol", "Dispossessed"}
    for position, event in enumerate(ordered):
        player = event.get("player")
        team = event.get("team", {}).get("name")
        kind = event["type"]["name"]
        failed_pass = kind == "Pass" and "outcome" in event.get("pass", {})
        if not (kind in loss_types or failed_pass) or not player:
            continue
        window_end = times[position] + ERROR_WINDOW_SECONDS
        for ahead in range(position + 1, len(ordered)):
            if times[ahead] > window_end:
                break
            later = ordered[ahead]
            if later["type"]["name"] == "Shot" and later.get("team", {}).get("name") != team:
                if player["id"] in players:
                    players[player["id"]].counters["errors"] += 1
                break


def _apply_goalkeeper_context(ordered, lineups, players) -> None:
    """Shots and crosses faced are read from the *opponent's* events."""
    keepers: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for team in lineups:
        for entry in team["lineup"]:
            for spell in entry.get("positions") or []:
                if spell.get("position") == "Goalkeeper":
                    start = _clock_minutes(spell.get("from")) or 0.0
                    end = _clock_minutes(spell.get("to"))
                    keepers[team["team_name"]].append(
                        (start, end if end is not None else 1e9, entry["player_id"])
                    )

    def keeper_on(team: str, minute: float) -> int | None:
        for start, end, player_id in keepers.get(team, []):
            if start <= minute < end:
                return player_id
        return None

    # The opponent is taken from the match's teams, not only from the teams a
    # goalkeeper was found for, so one missing keeper never blanks the whole match.
    teams = [t["team_name"] for t in lineups]
    for event in ordered:
        team = event.get("team", {}).get("name")
        if team is None or len(teams) < 2:
            continue
        defending = next((t for t in teams if t != team), None)
        if defending is None:
            continue
        minute = float(event.get("minute", 0)) + float(event.get("second", 0)) / 60
        keeper_id = keeper_on(defending, minute)
        if keeper_id is None or keeper_id not in players:
            continue
        kind = event["type"]["name"]
        if kind == "Shot":
            detail = event.get("shot", {})
            outcome = (detail.get("outcome") or {}).get("name")
            if outcome in SHOT_ON_TARGET:
                players[keeper_id].counters["gk_shots_on_target_against"] += 1
            if outcome == "Goal":
                players[keeper_id].counters["gk_goals_against"] += 1
        elif kind == "Pass" and event.get("pass", {}).get("cross"):
            end = event.get("pass", {}).get("end_location")
            if in_box(end):
                players[keeper_id].counters["gk_crosses_faced"] += 1


def _apply_gk_distribution(players: dict[int, PlayerMatch], events: list[dict]) -> None:
    """Goal-kicks and long keeper passes, for the distribution metrics."""
    for event in events:
        if event["type"]["name"] != "Pass":
            continue
        player = event.get("player")
        if not player or player["id"] not in players:
            continue
        record = players[player["id"]]
        if not any(pos == "Goalkeeper" for pos in record.position_minutes):
            continue
        detail = event.get("pass", {})
        if float(detail.get("length") or 0) >= GK_LAUNCH_YARDS:
            record.counters["gk_launches_attempted"] += 1
            if "outcome" not in detail:
                record.counters["gk_launches_completed"] += 1


# --------------------------------------------------------------------------
# Season aggregation
# --------------------------------------------------------------------------

def collect_match(match_id: int) -> tuple[list[dict], dict[str, float]]:
    """Download one match and return per-player rows plus team touch counts."""
    events = fetch_json(f"{OPEN_DATA}/events/{match_id}.json")
    lineups = fetch_json(f"{OPEN_DATA}/lineups/{match_id}.json")
    players, team_touches = parse_match(events, lineups)
    _apply_gk_distribution(players, events)

    # Possession is measured match by match, as a share of playing time, so an
    # unbalanced fixture sample can never distort a team's season figure.
    total_touches = sum(team_touches.values()) or 1.0

    rows = []
    for record in players.values():
        row = {
            "match_id": match_id,
            "player_id": record.player_id,
            "player": record.player,
            "nationality": record.nationality,
            "team": record.team,
            "minutes": round(record.minutes, 2),
            "started": int(record.started),
            "positions": dict(record.position_minutes),
            "match_possession": round(100 * team_touches.get(record.team, 0.0) / total_touches, 2),
        }
        row.update({k: v for k, v in record.counters.items()})
        rows.append(row)
    return rows, team_touches


def _cache_path(competition: Competition, cache_dir: Path) -> Path:
    return cache_dir / f"{competition.competition_id}_{competition.season_id}.jsonl.gz"


def build_competition(
    competition: Competition,
    cache_dir: Path,
    workers: int = 8,
    limit: int | None = None,
    progress=print,
) -> pd.DataFrame:
    """Ingest one competition-season, resuming from the on-disk cache."""
    import concurrent.futures

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(competition, cache_dir)

    done: set[int] = set()
    rows: list[dict] = []
    team_touches: dict[str, float] = defaultdict(float)
    if path.exists():
        with gzip.open(path, "rt") as handle:
            for line in handle:
                payload = json.loads(line)
                done.add(payload["match_id"])
                rows.extend(payload["rows"])
                for team, touches in payload["team_touches"].items():
                    team_touches[team] += touches

    matches = list_matches(competition)
    if limit:
        matches = matches[:limit]
    pending = [m["match_id"] for m in matches if m["match_id"] not in done]
    progress(
        f"{competition.league} {competition.season}: {len(matches)} matches "
        f"({len(done)} cached, {len(pending)} to fetch)"
    )

    if pending:
        with gzip.open(path, "at") as handle:
            with concurrent.futures.ThreadPoolExecutor(workers) as pool:
                futures = {pool.submit(collect_match, mid): mid for mid in pending}
                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    match_id = futures[future]
                    try:
                        match_rows, touches = future.result()
                    except Exception as error:  # a single bad match must not kill the run
                        progress(f"  ! match {match_id} failed: {error}")
                        continue
                    handle.write(
                        json.dumps({"match_id": match_id, "rows": match_rows, "team_touches": touches})
                        + "\n"
                    )
                    rows.extend(match_rows)
                    for team, value in touches.items():
                        team_touches[team] += value
                    if i % 50 == 0:
                        progress(f"  {competition.league} {competition.season}: {i}/{len(pending)}")

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["league"] = competition.league
        frame["season"] = competition.season
        frame["country"] = competition.country
        frame["gender"] = competition.gender
        frame["league_tier"] = competition.tier
        frame["league_strength"] = competition.strength
    return frame


def _dominant_position(position_minutes: dict[str, float]) -> tuple[str, str]:
    """Position group and detailed position, weighted by minutes played there."""
    totals: dict[str, float] = defaultdict(float)
    for name, minutes in position_minutes.items():
        totals[name] += minutes
    if not totals:
        return "CM", "CM"
    best = max(totals, key=totals.get)
    return POSITION_GROUP_BY_SB.get(best, ("CM", "CM"))


def aggregate_player_seasons(match_rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse player-match rows into the platform's player-season schema."""
    if match_rows.empty:
        return match_rows

    counter_columns = [c for c in COUNTER_KEYS if c in match_rows.columns]
    grouped = match_rows.groupby(["player_id", "league", "season"], sort=False)

    records = []
    for (player_id, league, season), block in grouped:
        positions: dict[str, float] = defaultdict(float)
        for mapping in block["positions"]:
            for name, minutes in mapping.items():
                positions[name] += minutes
        group, detailed = _dominant_position(positions)
        club_minutes = block.groupby("team")["minutes"].sum()
        team = club_minutes.idxmax()

        weights = block["minutes"].to_numpy()
        possession = (
            float(np.average(block["match_possession"], weights=weights)) if weights.sum() else 50.0
        )
        record = {
            "player_id": f"SB{int(player_id)}",
            "player": block["player"].iloc[-1],
            "nationality": block["nationality"].mode().iloc[0] if block["nationality"].notna().any() else "",
            "position": detailed,
            "position_group": group,
            "team": team,
            "league": league,
            "season": season,
            "country": block["country"].iloc[0],
            "gender": block["gender"].iloc[0],
            "league_tier": int(block["league_tier"].iloc[0]),
            "league_strength": float(block["league_strength"].iloc[0]),
            "minutes": float(block["minutes"].sum()),
            "matches": int(len(block)),
            "starts": int(block["started"].sum()),
            "clubs_in_season": int(club_minutes.size),
            "team_possession": round(possession, 2),
            "age": np.nan,       # not published in the open-data feed
            "height_cm": np.nan,
        }
        for column in counter_columns:
            record[column] = float(block[column].sum())
        records.append(record)

    frame = pd.DataFrame(records)
    frame["gk_psxg"] = np.nan   # post-shot xG is not in the open-data feed
    frame["minutes"] = frame["minutes"].round(0).astype(int)
    return frame.sort_values(["league", "season", "minutes"], ascending=[True, True, False])


def build_dataset(
    competitions: list[Competition] | None = None,
    cache_dir: Path | None = None,
    workers: int = 8,
    limit: int | None = None,
    progress=print,
) -> pd.DataFrame:
    """Ingest every configured competition and return the player-season table."""
    from .config import RAW_DIR

    competitions = competitions or COMPETITIONS
    cache_dir = cache_dir or (RAW_DIR / "statsbomb_cache")

    frames = []
    for competition in competitions:
        frame = build_competition(
            competition, cache_dir=cache_dir, workers=workers, limit=limit, progress=progress
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return aggregate_player_seasons(pd.concat(frames, ignore_index=True))

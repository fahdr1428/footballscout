"""
Transfermarkt ingestion, tested against the API's documented response shapes.

The service scrapes Transfermarkt, so these tests drive the real code with a
fake client returning payloads in the exact schema `transfermarkt-api` declares
(camelCase aliases included). That pins the parsing without hitting the network.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.transfermarkt import (
    POSITION_MAP, TOP_COMPETITIONS, TransfermarktClient, attach_stats, build_dataset,
    build_squads, enrich, map_position, season_label,
)


class FakeClient:
    """Stands in for a transfermarkt-api instance."""

    def __init__(self, clubs=None, players=None, stats=None):
        self.clubs = clubs or {}
        self.players = players or {}
        self.stats = stats or {}
        self.calls: list[tuple[str, str]] = []

    def competition_clubs(self, competition_id, season_id=None):
        self.calls.append(("clubs", competition_id))
        return self.clubs.get(competition_id, [])

    def club_players(self, club_id, season_id=None):
        self.calls.append(("players", club_id))
        return self.players.get(club_id, [])

    def player_stats(self, player_id):
        self.calls.append(("stats", player_id))
        return self.stats.get(player_id, [])


def _client() -> FakeClient:
    return FakeClient(
        clubs={"GB1": [{"id": "11", "name": "Arsenal"}, {"id": "281", "name": "Manchester City"}]},
        players={
            "11": [
                {
                    "id": "316264", "name": "William Saliba", "position": "Centre-Back",
                    "dateOfBirth": "2001-03-24", "age": 24, "nationality": ["France"],
                    "height": 1920, "foot": "right", "marketValue": 80000000,
                    "contract": "2027-06-30",
                },
                {
                    "id": "433177", "name": "Bukayo Saka", "position": "Right Winger",
                    "dateOfBirth": "2001-09-05", "age": 24, "nationality": ["England"],
                    "height": 178, "foot": "left", "marketValue": 130000000,
                },
            ],
            "281": [
                {
                    "id": "418560", "name": "Erling Haaland", "position": "Centre-Forward",
                    "dateOfBirth": "2000-07-21", "age": 25, "nationality": ["Norway"],
                    "height": 195, "foot": "left", "marketValue": 180000000,
                }
            ],
        },
        stats={
            "316264": [
                {"competitionId": "GB1", "competitionName": "Premier League", "seasonId": "2025",
                 "clubId": "11", "appearances": 36, "goals": 3, "assists": 1,
                 "yellowCards": 4, "redCards": 0, "minutesPlayed": 3180},
                # A different competition in the same season must not be counted.
                {"competitionId": "CL", "competitionName": "Champions League", "seasonId": "2025",
                 "clubId": "11", "appearances": 8, "goals": 1, "assists": 0,
                 "yellowCards": 1, "redCards": 0, "minutesPlayed": 720},
                # ...nor the same competition in a different season.
                {"competitionId": "GB1", "competitionName": "Premier League", "seasonId": "2024",
                 "clubId": "11", "appearances": 30, "goals": 2, "assists": 0,
                 "yellowCards": 3, "redCards": 0, "minutesPlayed": 2700},
            ],
            "433177": [
                {"competitionId": "GB1", "seasonId": "2025", "clubId": "11", "appearances": 33,
                 "goals": 12, "assists": 14, "yellowCards": 2, "redCards": 0, "minutesPlayed": 2790}
            ],
            "418560": [
                {"competitionId": "GB1", "seasonId": "2025", "clubId": "281", "appearances": 34,
                 "goals": 27, "assists": 8, "yellowCards": 3, "redCards": 0, "minutesPlayed": 2953}
            ],
        },
    )


# --------------------------------------------------------------------------
# Positions and seasons
# --------------------------------------------------------------------------

def test_every_transfermarkt_position_maps_to_a_real_group():
    from src.config import DETAILED_GROUPS, POSITION_TO_GROUP
    for position, (group, detailed) in POSITION_MAP.items():
        assert group in DETAILED_GROUPS, position
        assert POSITION_TO_GROUP.get(detailed) == group, position


def test_position_mapping_covers_the_defence_split_a_fantasy_feed_cannot():
    assert map_position("Centre-Back") == ("CB", "CB")
    assert map_position("Left-Back") == ("FB", "LB")
    assert map_position("Right-Back") == ("FB", "RB")
    assert map_position("Defensive Midfield")[0] == "DM"
    assert map_position("Left Winger")[0] == "W"
    assert map_position("Second Striker")[0] == "AM"
    # An unknown string falls back rather than raising.
    assert map_position("Sweeper")[0] == "CM"
    assert map_position(None)[0] == "CM"


def test_season_labels_follow_the_starting_year():
    assert season_label(2025) == "2025-26"
    assert season_label("2016") == "2016-17"
    assert season_label(1999) == "1999-00"


# --------------------------------------------------------------------------
# Squads
# --------------------------------------------------------------------------

def test_squads_carry_the_attributes_no_other_source_here_has():
    squads = build_squads(_client(), TOP_COMPETITIONS[:1], 2025, workers=1, progress=lambda *a: None)
    assert len(squads) == 3
    saliba = squads[squads["player"] == "William Saliba"].iloc[0]
    assert saliba["position_group"] == "CB" and saliba["position"] == "CB"
    assert saliba["team"] == "Arsenal" and saliba["league"] == "Premier League"
    assert saliba["market_value_eur"] == 80_000_000
    assert saliba["market_value_m"] == 80.0
    assert saliba["age"] == 24
    assert saliba["nationality"] == "France"
    assert saliba["foot"] == "right"
    assert saliba["season"] == "2025-26"


def test_heights_given_in_millimetres_are_converted():
    """Transfermarkt reports height in mm on some pages and cm on others."""
    squads = build_squads(_client(), TOP_COMPETITIONS[:1], 2025, workers=1, progress=lambda *a: None)
    heights = dict(zip(squads["player"], squads["height_cm"]))
    assert heights["William Saliba"] == 192.0      # arrived as 1920
    assert heights["Bukayo Saka"] == 178.0         # arrived as 178


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------

def test_stats_count_only_the_requested_competition_and_season():
    client = _client()
    squads = build_squads(client, TOP_COMPETITIONS[:1], 2025, workers=1, progress=lambda *a: None)
    with_stats = attach_stats(client, squads, 2025, workers=1, progress=lambda *a: None)
    saliba = with_stats[with_stats["player"] == "William Saliba"].iloc[0]
    # League only: not the Champions League rows, not last season's.
    assert saliba["appearances"] == 36
    assert saliba["goals"] == 3
    assert saliba["minutes"] == 3180
    assert saliba["yellow_cards"] == 4


def test_build_dataset_drops_players_who_did_not_play():
    client = _client()
    client.stats["433177"] = [
        {"competitionId": "GB1", "seasonId": "2025", "clubId": "11", "appearances": 0,
         "goals": 0, "assists": 0, "yellowCards": 0, "redCards": 0, "minutesPlayed": 0}
    ]
    frame = build_dataset(client, TOP_COMPETITIONS[:1], 2025, workers=1, progress=lambda *a: None)
    assert "Bukayo Saka" not in set(frame["player"])
    assert set(frame["player"]) == {"William Saliba", "Erling Haaland"}
    assert (frame["matches"] == frame["appearances"]).all()


def test_dataset_can_be_built_without_the_slow_per_player_stats_calls():
    client = _client()
    build_squads(client, TOP_COMPETITIONS[:1], 2025, workers=1, progress=lambda *a: None)
    assert not any(kind == "stats" for kind, _ in client.calls)


# --------------------------------------------------------------------------
# Enrichment - the highest-value use
# --------------------------------------------------------------------------

def test_enrichment_adds_value_and_a_true_position_to_a_bucketed_feed():
    market = build_squads(_client(), TOP_COMPETITIONS[:1], 2025, workers=1, progress=lambda *a: None)
    performance = pd.DataFrame(
        {
            "player": ["William Saliba", "Bukayo Saka", "Some Trialist"],
            "season": ["2025-26"] * 3,
            "position_group": ["DEF", "MID", "MID"],     # a fantasy feed's buckets
            "position": ["DEF", "MID", "MID"],
            "minutes": [3180, 2790, 900],
            "position_source": ["FPL bucket only"] * 3,
        }
    )
    enriched, report = enrich(performance, market)

    assert report == {"matched": 2, "of": 3}
    rows = enriched.set_index("player")
    # A four-bucket DEF becomes a centre-back; a MID becomes a winger.
    assert rows.loc["William Saliba", "position_group"] == "CB"
    assert rows.loc["Bukayo Saka", "position_group"] == "W"
    assert rows.loc["Bukayo Saka", "position"] == "RW"
    assert rows.loc["William Saliba", "market_value_m"] == 80.0
    assert rows.loc["William Saliba", "position_source"] == "Transfermarkt position"
    # An unmatched player is left exactly as he was, never guessed at.
    assert rows.loc["Some Trialist", "position_group"] == "MID"
    assert pd.isna(rows.loc["Some Trialist", "market_value_m"])
    assert rows.loc["Some Trialist", "position_source"] == "FPL bucket only"


def test_enrichment_can_take_the_value_without_the_position():
    market = build_squads(_client(), TOP_COMPETITIONS[:1], 2025, workers=1, progress=lambda *a: None)
    performance = pd.DataFrame(
        {"player": ["Erling Haaland"], "season": ["2025-26"],
         "position_group": ["FWD"], "position": ["FWD"], "minutes": [2953]}
    )
    enriched, _ = enrich(performance, market, use_positions=False)
    assert enriched.loc[0, "position_group"] == "FWD"          # untouched
    assert enriched.loc[0, "market_value_m"] == 180.0          # still enriched


def test_enrichment_of_an_empty_frame_is_a_no_op():
    empty = pd.DataFrame(columns=["player", "season"])
    out, report = enrich(empty, pd.DataFrame())
    assert out.empty and report["matched"] == 0


# --------------------------------------------------------------------------
# Client behaviour
# --------------------------------------------------------------------------

def test_the_client_rate_limits_itself():
    """The hosted instance allows about two requests every three seconds."""
    client = TransfermarktClient(rate=20.0)
    started = time.monotonic()
    for _ in range(4):
        client._wait_turn()
    assert time.monotonic() - started >= 0.10      # 3 gaps of 1/20s


def test_a_missing_resource_is_empty_rather_than_an_error(monkeypatch):
    import urllib.error

    client = TransfermarktClient(rate=0)

    def raise_404(*args, **kwargs):
        raise urllib.error.HTTPError("url", 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", raise_404)
    assert client.get("/clubs/999999/players") == {}
    assert client.club_players("999999") == []

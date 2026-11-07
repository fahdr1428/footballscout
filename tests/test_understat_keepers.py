"""Understat keepers: full matches in goal, joined to the league's match list by id."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refresh_sources as rs  # noqa: E402
from src.feature_engineering import add_keeper_full_match_rates  # noqa: E402
from src.understat import KEEPER_COLUMNS, attach_keepers  # noqa: E402

# The league's match list, as getLeagueData sends it ("dates").
DATES = [
    {"id": "1", "isResult": True, "h": {"title": "Arsenal"}, "a": {"title": "Chelsea"},
     "goals": {"h": "2", "a": "1"}, "xG": {"h": "1.9", "a": "0.8"}},
    {"id": "2", "isResult": True, "h": {"title": "Fulham"}, "a": {"title": "Arsenal"},
     "goals": {"h": "0", "a": "0"}, "xG": {"h": "1.4", "a": "0.6"}},
    {"id": "3", "isResult": True, "h": {"title": "Arsenal"}, "a": {"title": "Everton"},
     "goals": {"h": "3", "a": "2"}, "xG": {"h": "2.5", "a": "1.1"}},
    {"id": "4", "isResult": False, "h": {"title": "Arsenal"}, "a": {"title": "Spurs"},
     "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None}},
]
# A keeper's own match list, as getPlayerData sends it ("matches").
MATCHES = [
    {"id": "1", "season": "2025", "position": "GK", "time": "90"},   # home, let in 1 from 0.8 xG
    {"id": "2", "season": "2025", "position": "GK", "time": "90"},   # away, clean sheet from 1.4 xG
    {"id": "3", "season": "2025", "position": "GK", "time": "46"},   # came off at half-time: left out
    {"id": "4", "season": "2025", "position": "GK", "time": "90"},   # not played yet: left out
    {"id": "9", "season": "2024", "position": "GK", "time": "90"},   # a season not asked for
]


def test_full_matches_are_summed_exactly_and_partial_ones_left_out():
    table = rs.match_table(DATES)
    assert set(table) == {"1", "2", "3"}
    rows = rs.keeper_rows("77", MATCHES, {"2025-26": {"Arsenal"}}, table)
    assert rows == [{"player_id": "77", "season": "2025-26", "gk_full_matches": 2, "gk_full_xga": 2.2,
                     "gk_full_goals_against": 1, "gk_full_clean_sheets": 1}]


def test_a_match_whose_side_cannot_be_told_is_left_out():
    table = rs.match_table(DATES)
    # Listed for both clubs (a mid-season move between them): which side was his is unknown.
    rows = rs.keeper_rows("77", MATCHES[:1], {"2025-26": {"Arsenal", "Chelsea"}}, table)
    assert rows[0]["gk_full_matches"] == 0


def test_keepers_reach_the_dataset_and_become_rates_per_full_match(tmp_path):
    path = tmp_path / "keepers.csv"
    pd.DataFrame([{"player_id": "77", "season": "2025-26", "gk_full_matches": 2, "gk_full_xga": 2.2,
                   "gk_full_goals_against": 1, "gk_full_clean_sheets": 1}]).to_csv(path, index=False)
    frame = pd.DataFrame({"player_id": ["77", "78"], "season": ["2025-26", "2025-26"],
                          "position_group": ["GK", "GK"], "minutes": [200, 900]})
    out = attach_keepers(frame, path, progress=lambda *_: None)
    assert out.loc[0, "gk_full_matches"] == 2 and np.isnan(out.loc[1, "gk_full_matches"])
    assert set(KEEPER_COLUMNS) <= set(out.columns)
    derived = add_keeper_full_match_rates(out)
    row = derived.iloc[0]
    assert row["xgc_per90"] == 1.1 and row["gk_goals_against_per90"] == 0.5
    assert row["gk_xgc_minus_ga_per90"] == 0.6 and row["clean_sheet_rate"] == 50.0
    # A keeper with no full matches fetched has no reading at all, not a zero.
    assert np.isnan(derived.iloc[1]["gk_xgc_minus_ga_per90"])


def test_an_answer_that_is_not_json_falls_back_to_the_page(monkeypatch):
    """A changed endpoint is not a host that is down: the page is read instead, and the log says what came back."""
    calls, logged = [], []
    def fake_fetch(url, **_):
        calls.append(url)
        if "getLeagueData" in url:
            return b"<!doctype html><title>Just a moment</title>"
        return (b"<html><title>EPL</title><script src='/js/app.js'></script>"
                b"<script>var playersData = JSON.parse('[{\\x22id\\x22:\\x221\\x22}]');</script></html>")
    monkeypatch.setattr(rs, "fetch", fake_fetch)
    monkeypatch.setattr(rs, "log", logged.append)
    monkeypatch.setattr(rs.time, "sleep", lambda *_: None)
    players, route = rs.understat_players("EPL", 2026)
    assert route == "embedded JSON" and players == [{"id": "1"}]
    assert len(calls) == 2 and any("not JSON" in line and "Just a moment" in line for line in logged)
    assert not rs.unreachable(RuntimeError("not JSON")) and rs.unreachable(TimeoutError())

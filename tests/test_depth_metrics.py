"""Depth metrics: sums, differences and ratios of a source's own counts, and team shares that know their limits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import CONVERSION_PRIOR_SHOTS, add_depth_metrics


def squad(team: str, shots: list[float], minutes: float = 900.0, league: str = "Premier League") -> pd.DataFrame:
    """Eleven players of one club, every one on the pitch for the same minutes."""
    return pd.DataFrame({
        "player_id": [f"{team}-{i}" for i in range(len(shots))], "team": team, "league": league,
        "season": "2025-26", "position_group": "MID", "minutes": minutes, "shots": shots,
    })


def test_a_team_share_is_his_rate_over_the_teams_rate_per_player():
    # Ten players with 10 shots and one with 20: 120 shots over 9,900 player-minutes.
    frame = squad("Arsenal", [20.0] + [10.0] * 10)
    out = add_depth_metrics(frame)
    # The team's rate per player-minute is 120 / (9900 / 11) = 120 / 900.
    assert out.loc[0, "team_shot_share"] == pytest.approx(100 * (20 / 900) / (120 / 900), abs=0.01)
    assert out.loc[1, "team_shot_share"] == pytest.approx(100 * (10 / 900) / (120 / 900), abs=0.01)
    # Shares of players on the pitch all season average one eleventh of the team each.
    assert out["team_shot_share"].sum() == pytest.approx(100.0, abs=0.1)


def test_a_player_with_no_single_team_gets_no_share():
    frame = squad("Roma", [10.0] * 11)
    frame.loc[0, "team"] = "Cagliari,Roma"          # how Understat lists a mid-season move
    frame["clubs_in_season"] = 1
    frame.loc[1, "clubs_in_season"] = 2
    out = add_depth_metrics(frame)
    assert np.isnan(out.loc[0, "team_shot_share"]) and np.isnan(out.loc[1, "team_shot_share"])
    assert out.loc[2:, "team_shot_share"].notna().all()


def test_a_team_only_partly_in_the_data_gets_no_share():
    # Shots recorded for two of eleven players: the club's shots are not known.
    partial = squad("Everton", [12.0, 8.0] + [np.nan] * 9)
    assert add_depth_metrics(partial)["team_shot_share"].isna().all()
    # Under five matches of minutes between its players: not a team's season.
    brief = squad("Luton", [2.0] * 11, minutes=300.0)
    assert add_depth_metrics(brief)["team_shot_share"].isna().all()


def test_conversion_takes_penalties_out_where_they_can_be_and_is_pulled_to_the_position():
    frame = pd.DataFrame({
        "player_id": ["a", "b"], "position_group": ["FWD", "FWD"], "minutes": [1800.0, 1800.0],
        "shots": [100.0, 10.0], "pens_taken": [5.0, 0.0], "goals": [25.0, 0.0], "np_goals": [20.0, 0.0],
    })
    out = add_depth_metrics(frame, shots_include_penalties=True)
    taken = np.array([95.0, 10.0]); scored = np.array([20.0, 0.0])
    pooled = scored.sum() / taken.sum()
    expected = 100 * (scored + CONVERSION_PRIOR_SHOTS * pooled) / (taken + CONVERSION_PRIOR_SHOTS)
    assert out["shot_conversion_pct"].tolist() == pytest.approx(expected.tolist(), abs=0.01)
    # Ten shots without a goal is not a 0% finisher: the estimate sits between nothing and the position's rate.
    assert 0 < out.loc[1, "shot_conversion_pct"] < 100 * pooled


def test_composites_are_exact_sums_and_differences_per_90():
    frame = pd.DataFrame({
        "player_id": ["a"], "position_group": ["DM"], "minutes": [1800.0], "tackles": [40.0], "interceptions": [30.0],
        "dribblers_challenged": [50.0], "dribblers_tackled": [28.0], "blocks": [24.0], "shots_blocked": [6.0],
        "npxg": [2.0], "xa": [4.0], "assists": [6.0], "touches": [1500.0], "miscontrols": [20.0], "dispossessed": [10.0],
        "passes_completed": [1000.0], "progressive_passes": [130.0],
    })
    row = add_depth_metrics(frame).iloc[0]
    assert row["tackles_interceptions_per90"] == 3.5          # (40 + 30) / 20 matches
    assert row["dribbled_past_per90"] == 1.1                  # (50 - 28) / 20
    assert row["passes_blocked_per90"] == 0.9                 # (24 - 6) / 20
    assert row["npxg_xa_per90"] == 0.3 and row["assists_minus_xa_per90"] == 0.1
    assert row["turnover_rate"] == 2.0                        # 30 lost in 1,500 touches
    assert row["progressive_pass_rate"] == 13.0               # 130 of 1,000 completed


def test_a_source_without_the_counts_does_not_get_the_column():
    frame = pd.DataFrame({"player_id": ["a"], "position_group": ["MID"], "minutes": [900.0], "xa": [1.0], "assists": [2.0]})
    out = add_depth_metrics(frame)
    assert "assists_minus_xa_per90" in out.columns
    for absent in ("tackles_interceptions_per90", "dribbled_past_per90", "shot_conversion_pct", "team_shot_share", "turnover_rate"):
        assert absent not in out.columns

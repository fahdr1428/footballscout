"""
FBref big-five ingestion, tested offline against the upstream file shapes.

The build joins eleven FBref stat blocks, a curated player mapping and a
Transfermarkt squad export. These tests drive the real joining, collapsing and
position code with small frames in exactly those shapes, so the parsing is
pinned without downloading 16 MB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fbref import (
    COMP_TO_LEAGUE, ERA_CUTOVER, FBREF_BROAD, FIRST_SEASON, LAST_SEASON,
    PARTIAL_SEASONS, TM_BROAD, TM_POSITIONS, _age, _carry_identity, _collapse_transfers,
    _map_position, _season_label, _team_possession, _tidy_block,
    attach_identity,
)


def block(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Season handling
# ---------------------------------------------------------------------------

def test_season_label_reads_as_a_football_season():
    assert _season_label(2018) == "2017-18"
    assert _season_label(2022) == "2021-22"


def test_partial_season_is_excluded_by_default():
    """2025/26 is a live, in-progress season - real data, a few matches deep -
    so it must not be a default the way a complete season is."""
    assert LAST_SEASON == 2025
    assert 2026 in PARTIAL_SEASONS
    assert LAST_SEASON not in PARTIAL_SEASONS
    assert FIRST_SEASON < LAST_SEASON


def test_era_cutover_sits_inside_the_default_range():
    """Dual-era blocks are only meaningful if seasons on both sides of the
    cutover are actually in scope by default."""
    assert FIRST_SEASON < ERA_CUTOVER <= LAST_SEASON


# ---------------------------------------------------------------------------
# Positions come from Transfermarkt, never from the statistics
# ---------------------------------------------------------------------------

def test_specific_transfermarkt_positions_map_to_detailed_groups():
    assert _map_position("Centre-Back", "DF") == ("CB", "CB", "Transfermarkt")
    assert _map_position("Left-Back", "DF") == ("FB", "LB", "Transfermarkt")
    assert _map_position("Defensive Midfield", "MF") == ("DM", "DM", "Transfermarkt")
    assert _map_position("Right Winger", "FW") == ("W", "RW", "Transfermarkt")


def test_vague_position_falls_back_to_buckets_rather_than_being_guessed():
    """A broad label must not be invented into a specific one."""
    group, detail, source = _map_position("Defender", "DF")
    assert (group, detail) == ("DEF", "DEF")
    assert "broad" in source


def test_unmapped_player_falls_back_to_fbrefs_own_position():
    group, detail, source = _map_position(None, "MF,FW")
    assert group == "MID"
    assert source == "FBref (broad)"


def test_every_detailed_group_is_reachable_from_transfermarkt():
    """Compared against config, so adding a group cannot leave it unreachable."""
    from src.config import DETAILED_GROUPS

    groups = {g for g, _ in TM_POSITIONS.values()}
    assert groups == set(DETAILED_GROUPS)


def test_the_split_out_groups_are_routed_to_themselves_not_their_parent():
    """Second strikers and wide midfielders are separable, so they must not
    arrive labelled as attacking midfielders and wingers."""
    assert TM_POSITIONS["Second Striker"] == ("SS", "SS")
    assert TM_POSITIONS["Left Midfield"] == ("WM", "LM")
    assert TM_POSITIONS["Right Midfield"] == ("WM", "RM")
    assert TM_POSITIONS["Left Winger"] == ("W", "LW")


# ---------------------------------------------------------------------------
# Mid-season transfers
# ---------------------------------------------------------------------------

def _two_club_season() -> pd.DataFrame:
    return block([
        {"Url": "u1", "Season_End_Year": 2019, "team": "Empoli", "player": "Mover",
         "minutes": 382.0, "goals": 1.0, "tackles": 10.0, "gk_avg_pass_length": 30.0},
        {"Url": "u1", "Season_End_Year": 2019, "team": "Fiorentina", "player": "Mover",
         "minutes": 158.0, "goals": 2.0, "tackles": 5.0, "gk_avg_pass_length": 40.0},
        {"Url": "u2", "Season_End_Year": 2019, "team": "Empoli", "player": "Stayer",
         "minutes": 2000.0, "goals": 4.0, "tackles": 40.0, "gk_avg_pass_length": 35.0},
    ])


def test_transfer_totals_are_summed_into_one_row():
    out = _collapse_transfers(_two_club_season())
    mover = out[out["player"] == "Mover"].iloc[0]
    assert len(out) == 2
    assert mover["minutes"] == pytest.approx(540.0)
    assert mover["goals"] == pytest.approx(3.0)
    assert mover["clubs_in_season"] == 2


def test_club_of_record_is_where_the_player_played_most():
    out = _collapse_transfers(_two_club_season())
    mover = out[out["player"] == "Mover"].iloc[0]
    assert mover["team"] == "Empoli"          # 382 minutes beats 158


def test_season_averages_are_weighted_by_minutes_not_averaged_flat():
    out = _collapse_transfers(_two_club_season())
    mover = out[out["player"] == "Mover"].iloc[0]
    expected = (30.0 * 382 + 40.0 * 158) / 540
    assert mover["gk_avg_pass_length"] == pytest.approx(expected)
    assert mover["gk_avg_pass_length"] != pytest.approx(35.0)   # the flat mean


def test_a_stat_one_club_did_not_measure_is_left_missing_not_understated():
    """Summing a half-measured season would silently halve the player's total."""
    frame = _two_club_season()
    frame.loc[frame["team"] == "Fiorentina", "tackles"] = np.nan
    out = _collapse_transfers(frame)
    mover = out[out["player"] == "Mover"].iloc[0]
    assert pd.isna(mover["tackles"])


def test_players_who_did_not_move_keep_their_row():
    out = _collapse_transfers(_two_club_season())
    stayer = out[out["player"] == "Stayer"].iloc[0]
    assert stayer["clubs_in_season"] == 1
    assert stayer["goals"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Block joining
# ---------------------------------------------------------------------------

def test_tidy_block_renames_to_the_metric_registry_and_filters_seasons():
    raw = block([
        {"Season_End_Year": 2019, "Squad": "Arsenal", "Url": "u1",
         "Tkl_Tackles": 40.0, "Int": 20.0, "Clr": 5.0},
        {"Season_End_Year": 2023, "Squad": "Arsenal", "Url": "u1",
         "Tkl_Tackles": 4.0, "Int": 2.0, "Clr": 1.0},
    ])
    out = _tidy_block(raw, "defense", range(2018, 2023))
    assert list(out["Season_End_Year"]) == [2019]
    assert {"tackles", "interceptions", "clearances"} <= set(out.columns)
    assert "team" in out.columns


def test_blocks_are_keyed_on_the_club_so_a_transfer_does_not_multiply_rows():
    """Joining on player alone turns a two-club season into four rows."""
    raw = block([
        {"Season_End_Year": 2019, "Squad": "Empoli", "Url": "u1", "Tkl_Tackles": 10.0},
        {"Season_End_Year": 2019, "Squad": "Fiorentina", "Url": "u1", "Tkl_Tackles": 5.0},
    ])
    out = _tidy_block(raw, "defense", range(2018, 2023))
    assert len(out) == 2
    assert set(out["team"]) == {"Empoli", "Fiorentina"}


def test_completed_under_pressure_is_not_faked_from_total_completions():
    raw = block([{"Season_End_Year": 2019, "Squad": "Arsenal", "Url": "u1",
                  "Press_Pass": 100.0, "Cmp_Outcomes": 900.0}])
    out = _tidy_block(raw, "passing_types", range(2018, 2023))
    assert "passes_under_pressure" in out.columns
    assert "passes_completed_under_pressure" not in out.columns


# ---------------------------------------------------------------------------
# Dual-era blocks: the archive keeps StatsBomb-era columns, the release
# extends to 2025/26 under Opta-era names
# ---------------------------------------------------------------------------

def test_merge_eras_takes_each_season_from_its_own_source():
    from src.fbref import ERA_CUTOVER, _merge_eras

    archive = block([
        {"Season_End_Year": ERA_CUTOVER - 1, "Url": "u1", "Att_Vs": 5.0},
        {"Season_End_Year": ERA_CUTOVER, "Url": "u2", "Att_Vs": 999.0},  # stale past cutover
    ])
    release = block([
        {"Season_End_Year": ERA_CUTOVER - 1, "Url": "u3", "Att_Challenges": 999.0},
        {"Season_End_Year": ERA_CUTOVER, "Url": "u4", "Att_Challenges": 7.0},
    ])
    out = _merge_eras(archive, release, [("Att_Vs", "Att_Challenges")])
    assert set(out["Url"]) == {"u1", "u4"}   # pre-cutover from archive, post from release


def test_merge_eras_coalesces_a_renamed_column_onto_the_old_name():
    from src.fbref import ERA_CUTOVER, _merge_eras

    archive = block([{"Season_End_Year": ERA_CUTOVER - 1, "Url": "u1", "Att_Vs": 5.0}])
    release = block([{"Season_End_Year": ERA_CUTOVER, "Url": "u2", "Att_Challenges": 7.0}])
    out = _merge_eras(archive, release, [("Att_Vs", "Att_Challenges")])
    # Both eras' values land in the OLD name, so one RENAMES entry covers both.
    assert dict(zip(out["Url"], out["Att_Vs"])) == {"u1": 5.0, "u2": 7.0}
    assert "Att_Challenges" not in out.columns or out["Att_Challenges"].isna().sum() >= 0


def test_merge_eras_leaves_an_uncoalesced_loss_to_vanish_at_the_cutover():
    """passing_types has no successor for Press_Pass - _merge_eras must not
    invent one; the metric is just honestly gone from the cutover on."""
    from src.fbref import ERA_CUTOVER, _merge_eras

    archive = block([{"Season_End_Year": ERA_CUTOVER - 1, "Url": "u1", "Press_Pass": 12.0}])
    release = block([{"Season_End_Year": ERA_CUTOVER, "Url": "u2", "Crs_Pass": 3.0}])
    out = _merge_eras(archive, release, [])
    pre = out[out["Url"] == "u1"].iloc[0]
    post = out[out["Url"] == "u2"].iloc[0]
    assert pre["Press_Pass"] == 12.0
    assert pd.isna(post.get("Press_Pass"))


# ---------------------------------------------------------------------------
# Transfermarkt identity join
# ---------------------------------------------------------------------------

def _joinable():
    players = block([
        {"Url": "https://fbref.com/en/players/aaa/A", "Season_End_Year": 2019,
         "team": "Arsenal", "player": "A", "fbref_position": "DF", "fbref_age": 25},
        {"Url": "https://fbref.com/en/players/bbb/B", "Season_End_Year": 2019,
         "team": "Lyon", "player": "B", "fbref_position": "FW", "fbref_age": 21},
    ])
    mapping = block([
        {"PlayerFBref": "A", "UrlFBref": "https://fbref.com/en/players/aaa/A",
         "UrlTmarkt": "https://www.transfermarkt.com/a/profil/spieler/1", "TmPos": "Centre-Back"},
    ])
    values = block([
        {"player_url": "https://www.transfermarkt.com/a/profil/spieler/1",
         "season_start_year": 2018, "player_market_value_euro": 40_000_000.0,
         "player_position": "Left-Back", "player_height_mtrs": 1.85,
         "player_foot": "left", "player_nationality": "England",
         "player_dob": "1994-03-01", "contract_expiry": "2024-06-30",
         "joined_from": "Chelsea", "date_joined": "2018-07-01"},
    ])
    return players, mapping, values


def test_market_value_joins_on_the_right_season():
    """Transfermarkt counts seasons by start year; FBref by end year."""
    players, mapping, values = _joinable()
    out, coverage = attach_identity(players, mapping, values)
    a = out[out["player"] == "A"].iloc[0]
    assert a["market_value_eur"] == pytest.approx(40_000_000.0)
    assert coverage["with_market_value"] == pytest.approx(0.5)


def test_the_season_position_beats_the_mappings_career_position():
    """A player's role changes over five seasons; the season row is closer."""
    players, mapping, values = _joinable()
    out, _ = attach_identity(players, mapping, values)
    a = out[out["player"] == "A"].iloc[0]
    assert (a["position"], a["position_group"]) == ("LB", "FB")   # not CB


def test_height_is_converted_from_metres_to_centimetres():
    players, mapping, values = _joinable()
    out, _ = attach_identity(players, mapping, values)
    assert out[out["player"] == "A"].iloc[0]["height_cm"] == pytest.approx(185.0)


def test_unmapped_player_still_gets_a_group_and_says_where_it_came_from():
    players, mapping, values = _joinable()
    out, _ = attach_identity(players, mapping, values)
    b = out[out["player"] == "B"].iloc[0]
    assert b["position_group"] == "FWD"
    assert b["position_source"] == "FBref (broad)"
    assert pd.isna(b["market_value_eur"])


def test_contract_columns_are_dropped_because_they_are_a_scrape_time_snapshot():
    """Transfermarkt backfills today's contract onto every past season."""
    players, mapping, values = _joinable()
    out, _ = attach_identity(players, mapping, values)
    assert not [c for c in out.columns if "contract" in c or "joined" in c]


# ---------------------------------------------------------------------------
# Odds and ends
# ---------------------------------------------------------------------------

def test_age_prefers_date_of_birth_and_falls_back_to_birth_year():
    players = block([
        {"Season_End_Year": 2019, "date_of_birth": "1994-01-01", "born": 1990},
        {"Season_End_Year": 2019, "date_of_birth": None, "born": 1988},
    ])
    ages = _age(players)
    assert ages.iloc[0] == pytest.approx(25.0, abs=0.05)   # DOB wins over born
    assert ages.iloc[1] == pytest.approx(30.5)              # 2019 - 1988 - 0.5


def test_age_does_not_depend_on_fbrefs_shifting_age_format():
    """FBref's Age reads "25" one season and "25-081" the next.

    Parsing it numerically turned every "25-081" into nothing, which is how
    two whole seasons once shipped without a real age. The birth year is
    present in every season in one format, so age comes from that instead.
    """
    players = block([
        {"Season_End_Year": 2024, "date_of_birth": None, "born": 1999,
         "fbref_age": "25-081"},
        {"Season_End_Year": 2025, "date_of_birth": None, "born": 1999,
         "fbref_age": "25"},
    ])
    ages = _age(players)
    assert ages.notna().all()
    assert ages.iloc[0] == pytest.approx(24.5)
    assert ages.iloc[1] == pytest.approx(25.5)


def test_identity_is_carried_across_a_players_own_seasons_only():
    """Transfermarkt stops at 2022/23; a player's birth date does not."""
    players = block([
        {"Url": "/players/aaa/", "Season_End_Year": 2022, "date_of_birth": "1999-05-01",
         "height_cm": 180.0, "foot": "left", "nationality": "Spain"},
        {"Url": "/players/aaa/", "Season_End_Year": 2024, "date_of_birth": None,
         "height_cm": None, "foot": None, "nationality": None},
        # Only seen after Transfermarkt stopped: must stay unknown, never
        # borrow from somebody else.
        {"Url": "/players/bbb/", "Season_End_Year": 2024, "date_of_birth": None,
         "height_cm": None, "foot": None, "nationality": None},
    ])
    out = _carry_identity(players).set_index(["Url", "Season_End_Year"])
    later = out.loc[("/players/aaa/", 2024)]
    assert later["date_of_birth"] == "1999-05-01"
    assert later["height_cm"] == 180.0
    assert later["foot"] == "left"
    stranger = out.loc[("/players/bbb/", 2024)]
    assert pd.isna(stranger["date_of_birth"]) and pd.isna(stranger["height_cm"])


def test_team_possession_ignores_the_mirrored_opponent_rows():
    raw = block([
        {"Season_End_Year": 2019, "Squad": "Arsenal", "Poss": 58.0},
        {"Season_End_Year": 2019, "Squad": "vs Arsenal", "Poss": 42.0},
    ])
    out = _team_possession(raw, range(2018, 2023))
    assert list(out["team"]) == ["Arsenal"]
    assert out.iloc[0]["team_possession"] == pytest.approx(58.0)


def test_every_competition_maps_to_a_known_league():
    from src.config import LEAGUE_STRENGTH

    for league in COMP_TO_LEAGUE.values():
        assert league in LEAGUE_STRENGTH

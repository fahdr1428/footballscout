"""
The big five European leagues, from FBref match data joined to Transfermarkt.

    Premier League - La Liga - Serie A - Bundesliga - Ligue 1
    2017/18 -> 2022/23, ~16,000 player-seasons, ~5,700 players

WHERE IT COMES FROM
-------------------
Three public files, all mirrored in `JaseZiv/worldfootballR_data` (the data
repository behind the `worldfootballR` R package):

* **FBref season stats**, eleven blocks per player - standard, shooting,
  passing, pass types, goal- and shot-creating actions, defence, possession,
  playing time, miscellaneous, and two goalkeeping blocks. This is Opta/
  StatsBomb match data aggregated to a season, and it is the deepest openly
  available statistical picture of these leagues.
* **A curated FBref -> Transfermarkt player mapping**, 15,440 rows of
  hand-checked URL pairs.
* **Transfermarkt season squads**, which carry what FBref does not: a real
  market value in euros for that season, a *specific* position, height,
  preferred foot and nationality.

WHY THE JOIN MATTERS
--------------------
FBref knows only DF / MF / FW / GK. Transfermarkt knows "Centre-Back",
"Left-Back", "Defensive Midfield", "Right Winger". Taking the position from
Transfermarkt means the eight position groups are set by an **independent
source**, not inferred from the same statistics the models then read - which
would be circular. 99.9% of player-seasons here carry a Transfermarkt position
and 98.1% a market value for that exact season.

That combination is what makes this the closest thing here to a working
recruitment database: real players, deep performance metrics, and a real
valuation that moves season by season - Messi's runs 180 -> 150 -> 112 -> 80 ->
50 million euro across the five seasons, which is what actually happened.

WHAT IT DOES NOT HAVE
---------------------
It stops at 2022/23. FBref changed data provider from StatsBomb to Opta in
October 2022, and the upstream repository was archived in September 2025.
Two consequences, both handled rather than hidden:

* **Pressures disappear in 2022/23** - Opta does not count them. So do
  progressive carries and miscontrols under their old definitions.
* **xA becomes xAG** in 2022/23.
* **No contract data.** See `attach_identity` - Transfermarkt's contract dates
  are a scrape-time snapshot and cannot be trusted per season, so they are
  dropped rather than shown.

Columns a season cannot supply are reported unavailable for that season and
never imputed, so a model fitted on 2022/23 simply has fewer features than one
fitted on 2021/22. For the current season, use the Premier League source.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "https://github.com/JaseZiv/worldfootballR_data"
RAW_BASE = "https://raw.githubusercontent.com/JaseZiv/worldfootballR_data/master"
ATTRIBUTION = (
    "FBref season statistics and Transfermarkt squad records, mirrored by the "
    f"open-source worldfootballR_data repository ({REPO})."
)

# Season_End_Year: 2018 is the 2017/18 season. The advanced blocks start there.
#
# 2022/23 is deliberately excluded. The upstream snapshot of it stops after
# about 13 rounds (median 498 minutes against ~1,250 in a full season), and it
# is also the season FBref switched provider, so it loses pressures and
# progressive carries. Pooling a third of a season with five whole ones would
# put every 2022/23 player at the bottom of every volume metric for a reason
# that has nothing to do with the player. Set `seasons` explicitly to include it.
FIRST_SEASON, LAST_SEASON = 2018, 2022
PARTIAL_SEASONS = {2023}

STATS_DIR = "data/fb_big5_advanced_season_stats"
PLAYER_BLOCKS = [
    "standard", "shooting", "passing", "passing_types", "gca",
    "defense", "possession", "playing_time", "misc", "keepers", "keepers_adv",
]
TEAM_BLOCKS = ["possession", "standard"]

MAPPING_PATH = "raw-data/fbref-tm-player-mapping/output/fbref_to_tm_mapping.csv"
VALUES_PATH = "data/tm_player_vals/big5_player_vals.rds"

# FBref competition names -> the league names in config.LEAGUES.
COMP_TO_LEAGUE = {
    "Premier League": "Premier League",
    "La Liga": "La Liga",
    "Serie A": "Serie A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue 1",
}

# Transfermarkt's specific positions -> (position group, detailed position).
# Taken from an independent source, so grouping is not circular.
TM_POSITIONS: dict[str, tuple[str, str]] = {
    "Goalkeeper": ("GK", "GK"),
    "Centre-Back": ("CB", "CB"),
    "Left-Back": ("FB", "LB"),
    "Right-Back": ("FB", "RB"),
    "Defensive Midfield": ("DM", "DM"),
    "Central Midfield": ("CM", "CM"),
    "Attacking Midfield": ("AM", "AM"),
    "Second Striker": ("SS", "SS"),
    "Left Winger": ("W", "LW"),
    "Right Winger": ("W", "RW"),
    "Left Midfield": ("WM", "LM"),
    "Right Midfield": ("WM", "RM"),
    "Centre-Forward": ("FW", "CF"),
}

# Transfermarkt sometimes records only a broad position. Those are NOT guessed
# into a detailed group - they fall back to the four-bucket taxonomy, which the
# platform treats as first-class, and `position_source` records which happened.
TM_BROAD = {
    "Defender": ("DEF", "DEF"), "defence": ("DEF", "DEF"),
    "Midfield": ("MID", "MID"), "midfield": ("MID", "MID"),
    "Attack": ("FWD", "FWD"), "attack": ("FWD", "FWD"),
}

# FBref's own position field, the last resort when a player is unmapped.
FBREF_BROAD = {"GK": ("GK", "GK"), "DF": ("DEF", "DEF"),
               "MF": ("MID", "MID"), "FW": ("FWD", "FWD")}

# ---------------------------------------------------------------------------
# Column mapping: FBref name -> the platform's metric registry name.
# Only *counting totals* are carried across. Every percentage and rate the app
# shows is recomputed from these totals by feature_engineering, so a player
# with three attempts can never surface as "100%".
# ---------------------------------------------------------------------------

RENAMES: dict[str, dict[str, str]] = {
    "standard": {
        "Gls": "goals", "Ast": "assists", "G_minus_PK": "np_goals",
        "PK": "pens_scored", "PKatt": "pens_taken",
        "xG_Expected": "xg", "npxG_Expected": "npxg",
        "CrdY": "yellow_cards", "CrdR": "red_cards",
    },
    "shooting": {
        "Sh_Standard": "shots", "SoT_Standard": "shots_on_target",
        "FK_Standard": "free_kick_shots",
    },
    "passing": {
        "Att_Total": "passes_attempted", "Cmp_Total": "passes_completed",
        "Att_Short": "short_passes_attempted", "Cmp_Short": "short_passes_completed",
        "Att_Medium": "medium_passes_attempted", "Cmp_Medium": "medium_passes_completed",
        "Att_Long": "long_passes_attempted", "Cmp_Long": "long_passes_completed",
        "PrgDist_Total": "progressive_pass_distance",
        "KP": "key_passes", "Final_Third": "passes_into_final_third",
        "PPA": "passes_into_pen_area", "CrsPA": "crosses_into_pen_area",
        "Prog": "progressive_passes",
    },
    "passing_types": {
        "TB_Pass": "through_balls", "Sw_Pass": "switches", "Crs_Pass": "crosses",
        "CK_Pass": "corners_taken", "FK_Pass": "free_kick_passes",
        "Press_Pass": "passes_under_pressure",
        "Cmp_Outcomes": "passes_completed_under_pressure",   # replaced below
    },
    "gca": {
        "SCA_SCA": "sca", "GCA_GCA": "gca",
        "PassLive_SCA": "sca_from_open_play_pass", "PassDead_SCA": "sca_from_set_piece",
        "Drib_SCA": "sca_from_dribble", "Def_SCA": "sca_from_defensive_action",
    },
    "defense": {
        "Tkl_Tackles": "tackles", "TklW_Tackles": "tackles_won",
        "Def 3rd_Tackles": "tackles_def_third", "Mid 3rd_Tackles": "tackles_mid_third",
        "Att 3rd_Tackles": "tackles_att_third",
        "Att_Vs": "dribblers_challenged", "Tkl_Vs": "dribblers_tackled",
        "Press_Pressures": "pressures", "Succ_Pressures": "pressures_successful",
        "Att 3rd_Pressures": "pressures_att_third",
        "Blocks_Blocks": "blocks", "Sh_Blocks": "shots_blocked",
        "Int": "interceptions", "Clr": "clearances", "Err": "errors",
    },
    "possession": {
        "Touches_Touches": "touches", "Def Pen_Touches": "touches_def_pen",
        "Def 3rd_Touches": "touches_def_third", "Mid 3rd_Touches": "touches_mid_third",
        "Att 3rd_Touches": "touches_att_third", "Att Pen_Touches": "touches_att_pen",
        "Att_Dribbles": "dribbles_attempted", "Succ_Dribbles": "dribbles_completed",
        "Carries_Carries": "carries", "PrgDist_Carries": "progressive_carry_distance",
        "Prog_Carries": "progressive_carries",
        "Final_Third_Carries": "carries_into_final_third",
        "CPA_Carries": "carries_into_pen_area",
        "Prog_Receiving": "progressive_receptions",
    },
    "misc": {
        "Fls": "fouls_committed", "Fld": "fouls_won", "Off": "offsides",
        "Recov": "ball_recoveries", "Won_Aerial": "aerials_won",
        "Lost_Aerial": "aerials_lost", "PKwon": "pens_won", "PKcon": "pens_conceded",
    },
    "keepers": {
        "GA": "gk_goals_against", "SoTA": "gk_shots_on_target_against",
        "Saves": "gk_saves", "CS": "clean_sheets",
        "PKatt_Penalty": "gk_pens_faced", "PKsv_Penalty": "gk_pens_saved",
    },
    "keepers_adv": {
        "PSxG_Expected": "gk_psxg",
        "Opp_Crosses": "gk_crosses_faced", "Stp_Crosses": "gk_crosses_stopped",
        "#OPA_Sweeper": "gk_def_actions_outside_box",
        "Att_Launched": "gk_launches_attempted", "Cmp_Launched": "gk_launches_completed",
        "AvgLen_Passes": "gk_avg_pass_length", "AvgDist_Sweeper": "gk_avg_sweeper_distance",
    },
    "playing_time": {
        "PPM_Team.Success": "team_points_per_match",
    },
}

# Columns that are season averages rather than totals: weighted by minutes when
# a mid-season transfer means two rows have to become one.
AVERAGED = {"gk_avg_pass_length", "gk_avg_sweeper_distance", "team_points_per_match"}

# FBref publishes one row per club, so a mid-season transfer is two rows. Blocks
# are joined on the club as well as the player, then collapsed to one row each.
KEYS = ["Url", "Season_End_Year", "team"]
PLAYER_KEYS = ["Url", "Season_End_Year"]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def download(path: str, cache: Path, retries: int = 4, backoff: float = 2.0) -> Path:
    """Fetch one file from the mirror into `cache`, skipping if already there."""
    target = cache / Path(path).name
    if target.exists() and target.stat().st_size > 0:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_BASE}/{path}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            target.write_bytes(payload)
            return target
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))
    return target


def fetch_all(cache: Path, progress=print) -> dict[str, Path]:
    """Download every file the build needs. About 16 MB in total."""
    wanted = {f"player_{b}": f"{STATS_DIR}/big5_player_{b}.rds" for b in PLAYER_BLOCKS}
    wanted.update({f"team_{b}": f"{STATS_DIR}/big5_team_{b}.rds" for b in TEAM_BLOCKS})
    wanted["mapping"] = MAPPING_PATH
    wanted["values"] = VALUES_PATH

    paths = {}
    for i, (name, path) in enumerate(wanted.items(), 1):
        progress(f"  [{i:>2}/{len(wanted)}] {Path(path).name}")
        paths[name] = download(path, cache)
    return paths


def read_rds(path: Path) -> pd.DataFrame:
    """Read an R .rds data frame into pandas."""
    import pyreadr

    return list(pyreadr.read_r(str(path)).values())[0]


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def _tidy_block(frame: pd.DataFrame, block: str, seasons: range) -> pd.DataFrame:
    """Restrict one FBref block to the wanted seasons and rename its columns."""
    frame = frame[frame["Season_End_Year"].astype(int).isin(seasons)].copy()
    frame["Season_End_Year"] = frame["Season_End_Year"].astype(int)

    frame = frame.rename(columns={"Squad": "team"})
    mapping = {k: v for k, v in RENAMES.get(block, {}).items() if k in frame.columns}
    keep = KEYS + list(mapping)
    out = frame[keep].rename(columns=mapping).drop_duplicates(KEYS)

    # `Cmp_Outcomes` counts completed passes overall, not completed-under-pressure.
    # Only `Press_Pass` (attempts under pressure) is a real pressure column, so the
    # completed-under-pressure counterpart is left out rather than faked.
    out = out.drop(columns=["passes_completed_under_pressure"], errors="ignore")
    return out


def _identity(standard: pd.DataFrame, seasons: range) -> pd.DataFrame:
    """Per player-season identity: club, league, minutes, appearances, age."""
    df = standard[standard["Season_End_Year"].astype(int).isin(seasons)].copy()
    df["Season_End_Year"] = df["Season_End_Year"].astype(int)
    return df.rename(columns={
        "Player": "player", "Squad": "team", "Comp": "competition",
        "Nation": "nation_code", "Pos": "fbref_position",
        "Min_Playing": "minutes", "MP_Playing": "matches",
        "Starts_Playing": "starts", "Age": "fbref_age", "Born": "born",
    })[KEYS + ["competition", "player", "nation_code", "fbref_position",
               "minutes", "matches", "starts", "fbref_age", "born"]]


def _collapse_transfers(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season.

    FBref lists a mid-season transfer as one row per club. Counting totals are
    summed, season averages are weighted by minutes, and the club and league of
    record are those the player spent the most minutes at - the same rule the
    Premier League source uses. A stat is only summed when *every* club-row
    measured it, so a player who moved between leagues that count different
    things is left missing rather than silently understated.
    """
    frame = frame.sort_values("minutes", ascending=False)
    sizes = frame.groupby(PLAYER_KEYS)["minutes"].transform("size")

    single = frame[sizes == 1].copy()
    single["clubs_in_season"] = 1

    moved = frame[sizes > 1]
    if moved.empty:
        return single.reset_index(drop=True)

    skip = set(PLAYER_KEYS) | AVERAGED | {"fbref_age", "born"}
    counts = [c for c in moved.columns
              if c not in skip and pd.api.types.is_numeric_dtype(moved[c])]

    rows = []
    for _, block in moved.groupby(PLAYER_KEYS, sort=False):
        row = block.iloc[0].copy()          # most minutes: club and league of record
        for column in counts:
            values = block[column]
            row[column] = values.sum() if values.notna().all() else np.nan
        weights = block["minutes"].fillna(0)
        for column in AVERAGED & set(block.columns):
            values = block[column]
            ok = values.notna() & (weights > 0)
            row[column] = (np.average(values[ok], weights=weights[ok])
                           if ok.any() else np.nan)
        row["clubs_in_season"] = len(block)
        rows.append(row)

    combined = pd.DataFrame(rows)
    return pd.concat([single, combined], ignore_index=True)


def _season_label(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[-2:]}"


def _map_position(tm_position, fbref_position) -> tuple[str, str, str]:
    """(group, detailed position, where the position came from)."""
    key = str(tm_position).strip() if pd.notna(tm_position) else ""
    if key in TM_POSITIONS:
        group, detail = TM_POSITIONS[key]
        return group, detail, "Transfermarkt"
    if key in TM_BROAD:
        group, detail = TM_BROAD[key]
        return group, detail, "Transfermarkt (broad)"
    first = str(fbref_position).split(",")[0].strip() if pd.notna(fbref_position) else ""
    group, detail = FBREF_BROAD.get(first, ("MID", "MID"))
    return group, detail, "FBref (broad)"


def attach_identity(players: pd.DataFrame, mapping: pd.DataFrame,
                    values: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join the Transfermarkt position, market value and contract data."""
    link = mapping.dropna(subset=["UrlFBref"]).drop_duplicates("UrlFBref")
    players = players.merge(
        link[["UrlFBref", "UrlTmarkt", "TmPos"]],
        left_on="Url", right_on="UrlFBref", how="left",
    ).drop(columns=["UrlFBref"])

    values = values.copy()
    values["Season_End_Year"] = values["season_start_year"].astype(int) + 1
    columns = {
        "player_market_value_euro": "market_value_eur",
        "player_position": "tm_season_position",
        "player_height_mtrs": "_height_m",
        "player_foot": "foot",
        "player_nationality": "nationality",
        "player_dob": "date_of_birth",
    }
    # Contract expiry, date joined and joined-from are NOT carried across.
    # Transfermarkt records them as at the time the squad page was read, so the
    # 2017/18 rows carry contracts signed years later - Harry Kane's every
    # season reads 2024-06-30. A contract filter is one of the things a real
    # recruitment tool is used for, which is exactly why shipping a wrong one
    # is worse than shipping none.
    slim = (values.dropna(subset=["player_url"])
                  .sort_values("player_market_value_euro", ascending=False)
                  .drop_duplicates(["player_url", "Season_End_Year"])
                  [["player_url", "Season_End_Year"] + list(columns)]
                  .rename(columns=columns))

    players = players.merge(
        slim, left_on=["UrlTmarkt", "Season_End_Year"],
        right_on=["player_url", "Season_End_Year"], how="left",
    ).drop(columns=["player_url"], errors="ignore")

    # A season-specific Transfermarkt position beats the mapping's single
    # career position, because a player's role changes over six seasons.
    source_position = players["tm_season_position"].fillna(players["TmPos"])
    mapped = [_map_position(t, f) for t, f in
              zip(source_position, players["fbref_position"])]
    players["position_group"] = [m[0] for m in mapped]
    players["position"] = [m[1] for m in mapped]
    players["position_source"] = [m[2] for m in mapped]

    players["height_cm"] = pd.to_numeric(players["_height_m"], errors="coerce") * 100
    coverage = {
        "mapped_to_transfermarkt": float(players["UrlTmarkt"].notna().mean()),
        "with_market_value": float(players["market_value_eur"].notna().mean()),
        "detailed_position": float((players["position_source"] == "Transfermarkt").mean()),
    }
    return players.drop(columns=["_height_m", "TmPos", "tm_season_position"]), coverage


def _age(players: pd.DataFrame) -> pd.Series:
    """Age at 1 January of the season's second half, from date of birth."""
    dob = pd.to_datetime(players["date_of_birth"], errors="coerce")
    reference = pd.to_datetime(players["Season_End_Year"].astype(str) + "-01-01")
    age = (reference - dob).dt.days / 365.25
    return age.fillna(pd.to_numeric(players["fbref_age"], errors="coerce"))


def _team_possession(team_possession: pd.DataFrame, seasons: range) -> pd.DataFrame:
    """Season possession share per club, for possession-adjusting defensive volume."""
    df = team_possession.copy()
    df["Season_End_Year"] = df["Season_End_Year"].astype(int)
    df = df[df["Season_End_Year"].isin(seasons)]
    column = "Poss" if "Poss" in df.columns else next(
        (c for c in df.columns if c.startswith("Poss")), None)
    if column is None:
        return pd.DataFrame(columns=["team", "Season_End_Year", "team_possession"])
    # FBref publishes a row per squad and a mirrored "vs " opponent row; keep the squad.
    df = df[~df["Squad"].astype(str).str.startswith("vs ")]
    out = df[["Squad", "Season_End_Year", column]].rename(
        columns={"Squad": "team", column: "team_possession"})
    return out.drop_duplicates(["team", "Season_End_Year"])


def build_dataset(cache: Path, seasons: range | None = None,
                  progress=print) -> tuple[pd.DataFrame, dict]:
    """Build the big-five dataset. Returns the frame and a coverage summary."""
    seasons = seasons or range(FIRST_SEASON, LAST_SEASON + 1)
    progress("fetching source files")
    paths = fetch_all(cache, progress=progress)

    progress("reading FBref blocks")
    blocks = {name: read_rds(paths[f"player_{name}"]) for name in PLAYER_BLOCKS}

    players = _identity(blocks["standard"], seasons)
    for block in PLAYER_BLOCKS:
        tidy = _tidy_block(blocks[block], block, seasons)
        new = [c for c in tidy.columns if c not in players.columns or c in KEYS]
        players = players.merge(tidy[new], on=KEYS, how="left")

    # xA (StatsBomb era) and xAG (Opta era) are the same idea from two providers.
    passing = _raw_block(blocks["passing"], seasons, ["xA", "xAG"])
    players = players.merge(passing, on=KEYS, how="left")
    players["xa"] = players["xA"].combine_first(players["xAG"])
    players["xa_definition"] = np.where(
        players["xA"].notna(), "xA (StatsBomb)",
        np.where(players["xAG"].notna(), "xAG (Opta)", None))
    players = players.drop(columns=["xA", "xAG"])

    # The Opta era renamed two possession columns; take whichever season has one.
    possession = _raw_block(blocks["possession"], seasons,
                            ["Mis_Carries", "Mis_Dribbles", "Dis_Carries", "Dis_Dribbles"])
    players = players.merge(possession, on=KEYS, how="left")
    players["miscontrols"] = players["Mis_Carries"].combine_first(players["Mis_Dribbles"])
    players["dispossessed"] = players["Dis_Carries"].combine_first(players["Dis_Dribbles"])
    players = players.drop(columns=["Mis_Carries", "Mis_Dribbles",
                                    "Dis_Carries", "Dis_Dribbles"])

    progress(f"collapsing {len(players):,} club-rows into player-seasons")
    players = _collapse_transfers(players)

    progress("joining Transfermarkt positions, values and contracts")
    mapping = pd.read_csv(paths["mapping"], encoding="latin-1")
    values = read_rds(paths["values"])
    players, coverage = attach_identity(players, mapping, values)

    players["age"] = _age(players)
    players["season"] = players["Season_End_Year"].map(_season_label)
    players["league"] = players["competition"].map(COMP_TO_LEAGUE)
    players["player_id"] = players["Url"].str.extract(r"/players/([0-9a-f]+)/")[0]
    players["gender"] = "male"

    team_possession = _team_possession(read_rds(paths["team_possession"]), seasons)
    players = players.merge(team_possession, on=["team", "Season_End_Year"], how="left")

    players = players.drop(columns=["Url", "UrlTmarkt", "Season_End_Year",
                                    "competition", "fbref_age", "fbref_position",
                                    "born", "nation_code"], errors="ignore")
    players = players[players["league"].notna() & (players["minutes"] > 0)]

    ordered = ["player", "player_id", "team", "league", "season", "position",
               "position_group", "position_source", "age", "minutes", "matches",
               "starts", "clubs_in_season"]
    rest = [c for c in players.columns if c not in ordered]
    players = players[ordered + rest].sort_values(
        ["season", "league", "team", "player"]).reset_index(drop=True)

    coverage["player_seasons"] = len(players)
    coverage["players"] = int(players["player_id"].nunique())
    return players, coverage


def _raw_block(frame: pd.DataFrame, seasons: range, columns: list[str]) -> pd.DataFrame:
    """Pull named columns straight out of a block, without renaming."""
    df = frame.copy()
    df["Season_End_Year"] = df["Season_End_Year"].astype(int)
    df = df[df["Season_End_Year"].isin(seasons)]
    df = df.rename(columns={"Squad": "team"})
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df[KEYS + columns].drop_duplicates(KEYS)

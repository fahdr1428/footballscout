"""
The big five European leagues, from FBref match data joined to Transfermarkt.

    Premier League - La Liga - Serie A - Bundesliga - Ligue 1
    2017/18 -> 2024/25, ~21,400 player-seasons, plus 2025/26's opening weeks

WHERE IT COMES FROM
-------------------
Four public files, all published by `JaseZiv/worldfootballR_data` (the data
repository behind the `worldfootballR` R package, whose code was archived in
September 2025):

* **FBref season stats**, up to eleven blocks per player - standard,
  shooting, passing, pass types, goal- and shot-creating actions, defence,
  possession, playing time, miscellaneous, and two goalkeeping blocks. Most
  of these are published as a **GitHub Release asset**, which reaches
  further than the repository's committed snapshot - but it is frozen too,
  and not on one date. Standard, shooting, passing, pass types and playing
  time were last written on 18 September 2025 (all of 2024/25, about five
  rounds of 2025/26); defence, possession, misc, both goalkeeping blocks
  and the team blocks on 17 October 2024 (all of 2023/24, eight rounds of
  2024/25). `_drop_stale_seasons` detects that from the data and drops any
  season a block does not cover in full. The archived repository's
  committed snapshot is used only for the two things the release never
  carried: goal/shot-creating actions, and the older StatsBomb-era column
  names for a handful of defensive and possession metrics (see
  ERA_CUTOVER below).
* **A curated FBref -> Transfermarkt player mapping**, 15,440 rows of
  hand-checked URL pairs. Not part of the release, so it is a snapshot: it
  still matches roughly 88% of players in 2025/26's opening weeks by identity
  (players do not change FBref URLs), just not new debutants since it was
  last refreshed.
* **Transfermarkt season squads**, which carry a real market value in euros,
  a *specific* position, height, preferred foot and nationality - current
  through 2022/23, after which no further seasons were ever published.

WHY THE JOIN MATTERS
--------------------
FBref knows only DF / MF / FW / GK. Transfermarkt knows "Centre-Back",
"Left-Back", "Defensive Midfield", "Right Winger". Taking the position from
Transfermarkt means the eight position groups are set by an **independent
source**, not inferred from the same statistics the models then read - which
would be circular.

That combination is what makes this the closest thing here to a working
recruitment database: real players, deep performance metrics, and - for the
seasons Transfermarkt covers - a real valuation that moves season by season,
Messi's running 180 -> 150 -> 112 -> 80 -> 50 million euro, which is what
actually happened.

WHAT IT DOES NOT HAVE
---------------------
FBref changed data provider from StatsBomb to Opta in October 2022, and
several metrics simply stopped being published anywhere on the site - not
just for new seasons, but retroactively, since the release reflects FBref's
current display rather than an archived one. `ERA_CUTOVER` (season-end 2023)
is where this module switches from the archived StatsBomb-era snapshot to the
Opta-era release for the affected blocks, so the deep StatsBomb-only metrics
- pressures, and the old dribble/carry-progression column names - stay
available for 2017/18-2021/22 exactly as before, and are honestly reported
unavailable from 2022/23 on rather than quietly dropped or approximated:

* **Pressures never come back.** Opta does not publish anything equivalent,
  in any season, on FBref's site today.
* **xA becomes xAG** in 2022/23 (the same idea, a different provider) and the
  two are combined into one `xa` column.
* **No contract data.** See `attach_identity` - Transfermarkt's contract dates
  are a scrape-time snapshot and cannot be trusted per season, so they are
  dropped rather than shown.
* **No market value from 2023/24 on** - Transfermarkt values stop where the
  mapping does.
* **2025/26 is a fragment.** The mirror stopped about five rounds in, on 18
  September 2025, and the season has since finished without it. Real data,
  but a few matches per player - pooling it with complete seasons would rank
  everyone in it on a fraction of a season. It stays out of
  `default_seasons` and needs `--seasons` to select.
* **2024/25 is thinner than the seasons before it.** The blocks that froze
  in October 2024 are dropped for it outright, so it has shooting, passing,
  pass types and playing time, but no defending, possession, misc or
  goalkeeping block. The most recent season with every block is 2023/24.

Columns a season cannot supply are reported unavailable for that season and
never imputed, so a model fitted on 2022/23 simply has fewer features than one
fitted on 2021/22. For a complete 2025/26, use the Premier League source.
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
# The release asset reaches further than the archived repository's committed
# snapshot, but it stopped too - see STALE_BLOCK_RATIO for why "the release
# has 2024/25" is not the same as "every block has all of 2024/25".
RELEASE_BASE = (
    "https://github.com/JaseZiv/worldfootballR_data/releases/download/"
    "fb_big5_advanced_season_stats"
)
ATTRIBUTION = (
    "FBref season statistics and Transfermarkt squad records, mirrored by the "
    f"open-source worldfootballR_data repository ({REPO})."
)

# Season_End_Year: 2018 is the 2017/18 season. The advanced blocks start there.
#
# 2025/26 is deliberately excluded from the default range - it is real data,
# but the mirror stopped about five rounds in, and pooling it with complete
# seasons would rank every player in it by a fraction of a season. Set
# `seasons` explicitly to include it.
FIRST_SEASON, LAST_SEASON = 2018, 2025
PARTIAL_SEASONS = {2026}

# FBref switched its advanced-stats provider from StatsBomb to Opta partway
# through the 2022/23 season, and the site's own display changed retroactively
# for every season, not just new ones - the release asset (current-display)
# uses Opta-era column names throughout, while the archived repository's
# committed snapshot (frozen before the switch) uses the old StatsBomb-era
# names. Seasons before this cutover are read from the archived snapshot, so
# they keep metrics Opta never replaced (pressures, in particular); seasons
# from it on are read from the release, which is the only place 2023/24
# onward exists at all.
ERA_CUTOVER = 2023
# Blocks that have to be read from BOTH sources and stitched at the cutover,
# rather than from whichever one covers a season - because the release does
# not just add seasons, it drops or renames columns these blocks already
# shipped from the archive. Two kinds of loss, handled the same way:
#   - a straight rename ("Att_Vs" became "Att_Challenges") is coalesced back
#     onto the old name via _merge_eras, so RENAMES only has to know one of
#     them and the metric survives across the cutover unchanged;
#   - a column with no successor ("Press_Pass", pressure-passing, simply is
#     not in the release) is left to vanish at the cutover on its own -
#     _merge_eras does not invent a replacement for it, so it is available
#     for 2017/18-2021/22 exactly as already shipped, and honestly reported
#     unavailable from ERA_CUTOVER on.
# Every other block either matches exactly or only gained columns, so it is
# read from whichever source actually covers a season and passed straight
# through with no stitching.
DUAL_ERA_BLOCKS = {"defense", "possession", "passing_types"}
# Coalesced column pairs per dual-era block: (old name, new name) -> merged
# onto the old name, old value preferred where both exist (they should not
# overlap - the split is by season - but combine_first is the safe default).
ERA_COALESCE: dict[str, list[tuple[str, str]]] = {
    "defense": [
        ("Att_Vs", "Att_Challenges"), ("Tkl_Vs", "Tkl_Challenges"),
    ],
    "possession": [
        ("Att_Dribbles", "Att_Take"), ("Succ_Dribbles", "Succ_Take"),
        ("Prog_Carries", "PrgC_Carries"), ("Prog_Receiving", "PrgR_Receiving"),
    ],
    "passing_types": [],   # only a straight loss (Press_Pass) - nothing to coalesce
}
# Published only in the archived snapshot; the release never carried it, so it
# is unavailable from ERA_CUTOVER on rather than approximated from something else.
ARCHIVE_ONLY_BLOCKS = {"gca"}

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

def download(url: str, cache: Path, name: str,
            retries: int = 4, backoff: float = 2.0) -> Path:
    """Fetch one file into `cache` under `name`, skipping if already there.

    `name` rather than the URL's own basename, because a release download and
    an archive download for the same block would otherwise collide on the
    same filename (both are called e.g. big5_player_defense.rds at source).
    """
    target = cache / name
    if target.exists() and target.stat().st_size > 0:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            # Release downloads are larger and go through a redirect to a
            # signed storage URL; urllib follows it, just give it more room.
            with urllib.request.urlopen(url, timeout=180) as response:
                payload = response.read()
            target.write_bytes(payload)
            return target
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))
    return target


def fetch_all(cache: Path, progress=print) -> dict[str, Path]:
    """Download every file the build needs. About 25 MB in total.

    Most player blocks come from the release (fresh through 2025/26). The
    dual-era blocks are fetched from *both* the release and the archive, so
    build_dataset can stitch them at ERA_CUTOVER; ARCHIVE_ONLY_BLOCKS and the
    Transfermarkt files are archive-only, since neither exists in the release.
    """
    wanted: dict[str, str] = {}
    for block in PLAYER_BLOCKS:
        if block in ARCHIVE_ONLY_BLOCKS:
            wanted[f"player_{block}"] = f"{RAW_BASE}/{STATS_DIR}/big5_player_{block}.rds"
        elif block in DUAL_ERA_BLOCKS:
            wanted[f"player_{block}_archive"] = f"{RAW_BASE}/{STATS_DIR}/big5_player_{block}.rds"
            wanted[f"player_{block}_release"] = f"{RELEASE_BASE}/big5_player_{block}.rds"
        else:
            wanted[f"player_{block}"] = f"{RELEASE_BASE}/big5_player_{block}.rds"
    for block in TEAM_BLOCKS:
        wanted[f"team_{block}"] = f"{RELEASE_BASE}/big5_team_{block}.rds"
    wanted["mapping"] = f"{RAW_BASE}/{MAPPING_PATH}"
    wanted["values"] = f"{RAW_BASE}/{VALUES_PATH}"

    paths = {}
    for i, (name, url) in enumerate(wanted.items(), 1):
        progress(f"  [{i:>2}/{len(wanted)}] {name}")
        paths[name] = download(url, cache, f"{name}{Path(url).suffix}")
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


# Attributes of a person rather than of a season. Transfermarkt stops at
# 2022/23, but a player's date of birth, preferred foot and nationality do not
# change after it (and an adult's height does not either), so a value recorded
# in any of his seasons is the right value for all of them.
INVARIANT_IDENTITY = ["date_of_birth", "height_cm", "foot", "nationality"]


def _carry_identity(players: pd.DataFrame) -> pd.DataFrame:
    """Fill a player's invariant attributes from his other seasons.

    Nearest season first in each direction - forward from the last season
    that recorded one, then back from the first - so nothing is ever taken
    from a different player, and a player Transfermarkt never saw keeps none.
    """
    players = players.sort_values(["Url", "Season_End_Year"]).copy()
    present = [c for c in INVARIANT_IDENTITY if c in players.columns]
    by_player = players.groupby("Url", sort=False)[present]
    players[present] = by_player.ffill().combine_first(by_player.bfill())
    return players


def _age(players: pd.DataFrame) -> pd.Series:
    """Age at 1 January of the season's second half.

    Exact from a Transfermarkt date of birth where there is one. Otherwise
    from FBref's birth year, which is present for ~100% of players in every
    season: someone born in year B is between SEY-B-1 and SEY-B on 1 January
    of season-end year SEY, so SEY-B-0.5 is right to within six months.

    FBref's own `Age` column is deliberately not used. It reads "25" in some
    seasons and "25-081" (years-days) in others, at a reference date that
    also moves - the second form parses to nothing, which is how 2023/24 and
    2025/26 once shipped with no real ages at all.
    """
    dob = pd.to_datetime(players["date_of_birth"], errors="coerce")
    season_end = players["Season_End_Year"].astype(int)
    reference = pd.to_datetime(season_end.astype(str) + "-01-01")
    exact = (reference - dob).dt.days / 365.25
    born = pd.to_numeric(players["born"], errors="coerce")
    return exact.fillna(season_end - born - 0.5)


def _team_possession(team_possession: pd.DataFrame, seasons: range) -> pd.DataFrame:
    """Season possession share per club, for possession-adjusting defensive volume."""
    df = team_possession.copy()
    df["Season_End_Year"] = df["Season_End_Year"].astype(int)
    df = df[df["Season_End_Year"].isin(seasons)]
    column = "Poss" if "Poss" in df.columns else next(
        (c for c in df.columns if c.startswith("Poss")), None)
    if column is None:
        return pd.DataFrame(columns=["team", "Season_End_Year", "team_possession"])
    # FBref publishes a mirrored "possession by this squad's opponents" row
    # alongside each squad's own. The release marks it with a clean
    # Team_or_Opponent column; the older archived snapshot instead prefixed
    # the mirrored row's own Squad name with "vs " - both are handled so this
    # keeps working whichever source a caller passes in.
    if "Team_or_Opponent" in df.columns:
        df = df[df["Team_or_Opponent"].astype(str).str.lower() == "team"]
    else:
        df = df[~df["Squad"].astype(str).str.startswith("vs ")]
    out = df[["Squad", "Season_End_Year", column]].rename(
        columns={"Squad": "team", column: "team_possession"})
    return out.drop_duplicates(["team", "Season_End_Year"])


# The mirror's blocks did not all stop on the same day. Shooting, passing and
# the standard block (which carries minutes) ran to September 2025; defence,
# possession, misc, both goalkeeping blocks and the team blocks stopped on 17
# October 2024, eight rounds into 2024/25. Joined naively, a 2024/25 row paired
# eight rounds of tackles with a whole season of minutes, so every such per-90
# read five to six times too low - a defender making 1.5 tackles a game showed
# 0.27. The archive's 2022/23 snapshot has the same problem for the goal- and
# shot-creation block (a third of the season).
#
# Rather than hard-coding those dates, each block is checked against the
# standard block using the minutes it carries itself: a block whose playing
# time for a season falls materially short of the standard block's is stale for
# that season and is dropped for it, whole. What remains is a season measured
# in full or not at all - never a fraction dressed as a rate. If the mirror is
# ever refreshed, the check passes and the season comes back on its own.
STALE_BLOCK_RATIO = 0.95
BLOCK_MINUTES = {"standard": "Mins_Per_90_Playing", "playing_time": "Mins_Per_90_Playing.Time"}


def _block_coverage(block: pd.DataFrame, standard: pd.DataFrame, minutes: str) -> pd.Series:
    """Per season: the block's playing time over the standard block's, same players."""
    keys = ["Url", "Squad", "Season_End_Year"]
    ref = standard[keys + ["Mins_Per_90_Playing"]].copy()
    other = block[keys + [minutes]].drop_duplicates(keys).copy()
    for frame in (ref, other):
        frame["Season_End_Year"] = frame["Season_End_Year"].astype(int)
    ref["_ref"] = pd.to_numeric(ref.pop("Mins_Per_90_Playing"), errors="coerce")
    other["_own"] = pd.to_numeric(other.pop(minutes), errors="coerce")
    both = ref.merge(other, on=keys, how="inner")
    sums = both.groupby("Season_End_Year")[["_own", "_ref"]].sum()
    return (sums["_own"] / sums["_ref"].where(sums["_ref"] > 0)).dropna()


def _drop_stale_seasons(block: pd.DataFrame, standard: pd.DataFrame, name: str,
                        stale: dict) -> pd.DataFrame:
    """Remove every season this block does not cover in full; record which."""
    minutes = BLOCK_MINUTES.get(name, "Mins_Per_90")
    if minutes not in block.columns or name == "standard":
        return block
    coverage = _block_coverage(block, standard, minutes)
    bad = coverage[coverage < STALE_BLOCK_RATIO]
    if bad.empty:
        return block
    stale[name] = {_season_label(int(s)): round(float(r), 3) for s, r in bad.items()}
    return block[~block["Season_End_Year"].astype(int).isin(bad.index)]


def _drop_stale_team_seasons(team: pd.DataFrame, standard: pd.DataFrame,
                             stale: dict) -> pd.DataFrame:
    """The same test for a team block: its matches against its players' minutes.

    Eleven players are on the pitch, so a squad's summed player 90s divided by
    eleven is how many matches it has played in the standard block.
    """
    if "Mins_Per_90" not in team.columns:
        return team
    own = team.copy()
    own["Season_End_Year"] = own["Season_End_Year"].astype(int)
    if "Team_or_Opponent" in own.columns:
        own = own[own["Team_or_Opponent"].astype(str).str.lower() == "team"]
    else:
        own = own[~own["Squad"].astype(str).str.startswith("vs ")]
    played = own.groupby("Season_End_Year")["Mins_Per_90"].apply(
        lambda s: pd.to_numeric(s, errors="coerce").sum())
    ref = standard.assign(Season_End_Year=standard["Season_End_Year"].astype(int))
    implied = ref.groupby("Season_End_Year")["Mins_Per_90_Playing"].apply(
        lambda s: pd.to_numeric(s, errors="coerce").sum()) / 11
    ratio = (played / implied.reindex(played.index)).dropna()
    bad = ratio[ratio < STALE_BLOCK_RATIO]
    if bad.empty:
        return team
    stale["team_possession"] = {_season_label(int(s)): round(float(r), 3) for s, r in bad.items()}
    return team[~team["Season_End_Year"].astype(int).isin(bad.index)]


def _merge_eras(
    archive: pd.DataFrame, release: pd.DataFrame, coalesce: list[tuple[str, str]],
) -> pd.DataFrame:
    """Stitch a dual-era block: the archive below ERA_CUTOVER, the release from it.

    Each source is trusted only for the seasons it is authoritative for - the
    archive is a frozen StatsBomb-era snapshot, so used past its cutover it
    would either be missing seasons entirely or (for playing-time-style
    columns that exist in both) silently stale relative to the release. After
    concatenating, `coalesce` pairs of (old name, new name) are merged onto
    the old name, so a single rename at the source (e.g. "Att_Vs" becoming
    "Att_Challenges") does not fragment one metric into two columns that
    RENAMES only half recognises.
    """
    old = archive[archive["Season_End_Year"].astype(int) < ERA_CUTOVER].copy()
    new = release[release["Season_End_Year"].astype(int) >= ERA_CUTOVER].copy()
    merged = pd.concat([old, new], ignore_index=True, sort=False)
    for old_name, new_name in coalesce:
        if old_name in merged.columns and new_name in merged.columns:
            merged[old_name] = merged[old_name].combine_first(merged[new_name])
        elif new_name in merged.columns:
            merged[old_name] = merged[new_name]
    return merged


def build_dataset(cache: Path, seasons: range | None = None,
                  progress=print) -> tuple[pd.DataFrame, dict]:
    """Build the big-five dataset. Returns the frame and a coverage summary."""
    seasons = seasons or range(FIRST_SEASON, LAST_SEASON + 1)
    progress("fetching source files")
    paths = fetch_all(cache, progress=progress)

    progress("reading FBref blocks")
    blocks: dict[str, pd.DataFrame] = {}
    for block in PLAYER_BLOCKS:
        if block in DUAL_ERA_BLOCKS:
            archive = read_rds(paths[f"player_{block}_archive"])
            release = read_rds(paths[f"player_{block}_release"])
            blocks[block] = _merge_eras(archive, release, ERA_COALESCE.get(block, []))
        else:
            blocks[block] = read_rds(paths[f"player_{block}"])

    progress("checking every block covers each season in full")
    stale: dict[str, dict[str, float]] = {}
    for block in PLAYER_BLOCKS:
        blocks[block] = _drop_stale_seasons(blocks[block], blocks["standard"], block, stale)
    for name, seasons_lost in stale.items():
        progress(f"  {name}: dropped " + ", ".join(
            f"{s} ({r:.0%} of the season)" for s, r in seasons_lost.items()))

    players = _identity(blocks["standard"], seasons)
    for block in PLAYER_BLOCKS:
        tidy = _tidy_block(blocks[block], block, seasons)
        new = [c for c in tidy.columns if c not in players.columns or c in KEYS]
        players = players.merge(tidy[new], on=KEYS, how="left")

    # xA (StatsBomb era) and xAG (Opta era) are the same idea from two providers.
    passing = _raw_block(blocks["passing"], seasons, ["xA", "xAG", "PrgP"])
    players = players.merge(passing, on=KEYS, how="left")
    players["xa"] = players["xA"].combine_first(players["xAG"])
    players["xa_definition"] = np.where(
        players["xA"].notna(), "xA (StatsBomb)",
        np.where(players["xAG"].notna(), "xAG (Opta)", None))
    # FBref relabelled progressive passes the same way mid-release: RENAMES
    # already turned the old "Prog" header into `progressive_passes` above,
    # which covers seasons through 2021/22; "PrgP" is the newer header for
    # the same metric, populated from 2022/23, fetched raw here since RENAMES
    # only recognises one name per column and would otherwise fragment one
    # metric into two.
    players["progressive_passes"] = players["progressive_passes"].combine_first(
        players["PrgP"]
    )
    players = players.drop(columns=["xA", "xAG", "PrgP"])

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
    players = _carry_identity(players)

    players["age"] = _age(players)
    players["season"] = players["Season_End_Year"].map(_season_label)
    players["league"] = players["competition"].map(COMP_TO_LEAGUE)
    players["player_id"] = players["Url"].str.extract(r"/players/([0-9a-f]+)/")[0]
    players["gender"] = "male"

    team_raw = _drop_stale_team_seasons(read_rds(paths["team_possession"]),
                                        blocks["standard"], stale)
    team_possession = _team_possession(team_raw, seasons)
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

    coverage["stale_blocks"] = stale
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

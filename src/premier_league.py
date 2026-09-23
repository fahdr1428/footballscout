"""
Premier League, ten seasons to 2025/26, from public season data.

Two public sources, joined:

* **Fantasy Premier League** season and gameweek exports, mirrored season by
  season in `vaastav/Fantasy-Premier-League`. This is the official FPL feed:
  minutes, starts, goals, assists, cards, saves, clean sheets, the Opta-derived
  Influence / Creativity / Threat indices, the bonus-point score, price and
  ownership - and, from 2022/23, Opta expected goals and expected assists, and
  from 2025/26 the tackles, recoveries and clearances-blocks-interceptions
  counts the game added with its Defensive Contribution scoring.
* **Understat** per-player match logs from the same repository, which add
  shots, key passes, non-penalty goals and xG, xGChain, xGBuildup - and, most
  usefully, the **position a player actually lined up in each match**.

WHAT THIS SOURCE IS GOOD AND BAD AT
-----------------------------------
Good: it is current (2025/26 complete, 38 rounds), it covers ten seasons, and
it carries three things the event feed cannot - **age**, **price** and
**ownership**.

Bad: it is a summary feed, not event data. There are no progressive passes, no
pass completion, no dribbles, no aerials. And FPL knows only four positional
buckets, so this source groups players as GK / DEF / MID / FWD. Understat's
line-up position rides along as an attribute you can filter on, but it is not
used to group: deriving a finer position from the same statistics the models
then read would be circular.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import UNDERSTAT_POSITIONS

REPO_URL = "https://github.com/vaastav/Fantasy-Premier-League"
ATTRIBUTION = (
    "Fantasy Premier League and Understat data, mirrored by the open-source "
    f"Fantasy-Premier-League repository ({REPO_URL})."
)

SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]
CURRENT_SEASON = "2025-26"

ELEMENT_TYPE_TO_GROUP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Understat line-up codes -> the readable position carried as an attribute.
UNDERSTAT_TO_DETAIL = {
    "GK": "GK", "DC": "CB", "DR": "RB", "DL": "LB",
    "DMC": "DM", "DMR": "DM", "DML": "DM",
    "MC": "CM", "MR": "RM", "ML": "LM",
    "AMC": "AM", "AMR": "RW", "AML": "LW",
    "FW": "CF", "FWR": "RW", "FWL": "LW",
}

# Columns copied straight across from the FPL season export.
FPL_DIRECT = {
    "minutes": "minutes",
    "starts": "starts",
    "goals_scored": "goals",
    "assists": "assists",
    "clean_sheets": "clean_sheets",
    "goals_conceded": "gk_goals_against",
    "saves": "gk_saves",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "own_goals": "own_goals",
    "penalties_missed": "pens_missed",
    "penalties_saved": "pens_saved",
    "expected_goals": "xg",
    "expected_assists": "xa",
    "expected_goals_conceded": "xgc",
    "tackles": "tackles",
    "recoveries": "ball_recoveries",
    "clearances_blocks_interceptions": "cbi",
    "defensive_contribution": "defensive_contribution",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "bps": "bps",
}

UNDERSTAT_DIRECT = {
    "shots": "shots",
    "key_passes": "key_passes",
    "npg": "np_goals",
    "npxG": "npxg",
    "xGChain": "xg_chain",
    "xGBuildup": "xg_buildup",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def read_csv(path: Path) -> pd.DataFrame:
    """The mirror mixes utf-8 and latin-1 files; try both before giving up."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"could not decode {path}")


_TRANSLIT = str.maketrans({"ø": "o", "Ø": "o", "đ": "d", "Đ": "d", "ł": "l", "Ł": "l",
                           "ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ð": "d"})


def normalise_name(value: str) -> str:
    """Accent- and punctuation-free lower-case name, for joining two feeds."""
    text = str(value).translate(_TRANSLIT)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", text.lower())).strip()


def season_start_year(season: str) -> int:
    return int(season.split("-")[0])


# --------------------------------------------------------------------------
# Fantasy Premier League season exports
# --------------------------------------------------------------------------

def load_team_names(repo: Path) -> dict[tuple[str, int], str]:
    """Season + FPL team id -> club name.

    The repository's master list does not cover every season, so each season's
    own `teams.csv` is read first and the master list fills the gaps.
    """
    names: dict[tuple[str, int], str] = {}
    master = repo / "data" / "master_team_list.csv"
    if master.exists():
        table = read_csv(master)
        names.update({(r.season, int(r.team)): r.team_name for r in table.itertuples()})
    for path in sorted((repo / "data").glob("*/teams.csv")):
        season = path.parent.name
        table = read_csv(path)
        if {"id", "name"}.issubset(table.columns):
            names.update({(season, int(r.id)): r.name for r in table.itertuples()})
    return names


def load_fpl_season(repo: Path, season: str, team_names: dict) -> pd.DataFrame:
    """One season of FPL player totals, in this platform's column names."""
    raw = read_csv(repo / "data" / season / "players_raw.csv")
    frame = pd.DataFrame(index=raw.index)

    frame["player"] = (
        raw["first_name"].astype(str).str.strip() + " " + raw["second_name"].astype(str).str.strip()
    ).str.strip()
    frame["known_as"] = raw["web_name"].astype(str)
    frame["season"] = season
    frame["position_group"] = raw["element_type"].map(ELEMENT_TYPE_TO_GROUP)
    frame["team"] = [team_names.get((season, int(t)), f"Team {t}") for t in raw["team"]]
    frame["league"] = "Premier League"

    for source, target in FPL_DIRECT.items():
        frame[target] = pd.to_numeric(raw[source], errors="coerce") if source in raw.columns else np.nan

    # Price is the fantasy game's own valuation, in tenths of a million.
    frame["price_m"] = pd.to_numeric(raw.get("now_cost"), errors="coerce") / 10
    frame["ownership_pct"] = pd.to_numeric(raw.get("selected_by_percent"), errors="coerce")
    frame["total_points"] = pd.to_numeric(raw.get("total_points"), errors="coerce")
    frame["birth_date"] = raw.get("birth_date")
    for duty, column in [
        ("penalty_duty", "penalties_order"),
        ("corner_duty", "corners_and_indirect_freekicks_order"),
        ("freekick_duty", "direct_freekicks_order"),
    ]:
        frame[duty] = pd.to_numeric(raw.get(column), errors="coerce")
    frame["fpl_element"] = raw["id"]
    return frame


def appearances_from_gameweeks(repo: Path, season: str) -> pd.DataFrame:
    """Appearances, and the club actually played for, from the gameweek export.

    The season snapshot in `players_raw.csv` carries a player's *current* club,
    which for a recently finished season is the club he moved to afterwards -
    in 2025/26 that put Semenyo and Guehi at Manchester City. The gameweek file
    records the club he played each match for, so the club shown here is the
    one he played the most minutes for that season, and mid-season transfers
    are counted rather than hidden.
    """
    path = repo / "data" / season / "gws" / "merged_gw.csv"
    empty = pd.DataFrame(columns=["fpl_element", "matches", "gw_team", "clubs_in_season"])
    if not path.exists():
        return empty
    gameweeks = read_csv(path)
    if "element" not in gameweeks.columns:
        return empty
    gameweeks["minutes"] = pd.to_numeric(gameweeks["minutes"], errors="coerce").fillna(0)
    played = gameweeks[gameweeks["minutes"] > 0]
    if played.empty:
        return empty

    out = played.groupby("element").size().rename("matches").reset_index()
    out = out.rename(columns={"element": "fpl_element"})
    if "team" in played.columns:
        minutes_by_club = played.groupby(["element", "team"])["minutes"].sum()
        modal = minutes_by_club.groupby(level=0).idxmax().map(lambda pair: pair[1]).rename("gw_team")
        clubs = minutes_by_club.groupby(level=0).size().rename("clubs_in_season")
        out = out.merge(
            pd.concat([modal, clubs], axis=1).reset_index().rename(columns={"element": "fpl_element"}),
            on="fpl_element", how="left",
        )
    return out


# --------------------------------------------------------------------------
# Understat match logs
# --------------------------------------------------------------------------

def load_understat(repo: Path) -> pd.DataFrame:
    """Every Understat per-player match log in the mirror, aggregated by season.

    Each file holds a player's whole match history, so the union across season
    folders reaches back further than the folders themselves. A player's
    position for a season is the one he spent the most minutes in.
    """
    files: dict[str, Path] = {}
    for folder in sorted((repo / "data").glob("*/understat")):
        for path in folder.glob("*.csv"):
            if path.name == "understat_player.csv":
                continue
            files[path.name] = path        # later seasons overwrite: keep the newest file

    rows: list[dict] = []
    positions: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for name, path in files.items():
        try:
            log = read_csv(path)
        except ValueError:
            continue
        if log.empty or "season" not in log.columns:
            continue
        player = " ".join(name[:-4].split("_")[:-1])
        key = normalise_name(player)
        log = log.copy()
        log["minutes"] = pd.to_numeric(log["time"], errors="coerce").fillna(0)
        for season_year, block in log.groupby(log["season"].astype(int)):
            record = {"key": key, "understat_player": player, "season_year": int(season_year)}
            for source, target in UNDERSTAT_DIRECT.items():
                record[f"us_{target}"] = float(
                    pd.to_numeric(block.get(source), errors="coerce").fillna(0).sum()
                )
            record["us_xa"] = float(pd.to_numeric(block.get("xA"), errors="coerce").fillna(0).sum())
            record["us_xg"] = float(pd.to_numeric(block.get("xG"), errors="coerce").fillna(0).sum())
            record["us_minutes"] = float(block["minutes"].sum())
            rows.append(record)
            for code, minutes in block.groupby(block["position"].astype(str))["minutes"].sum().items():
                if code != "Sub":
                    positions[(key, int(season_year))][code] += float(minutes)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.groupby(["key", "season_year"], as_index=False).max(numeric_only=False)

    detail, raw_code = [], []
    for row in frame.itertuples():
        played = positions.get((row.key, row.season_year), {})
        best = max(played, key=played.get) if played else None
        raw_code.append(best)
        detail.append(UNDERSTAT_TO_DETAIL.get(best) if best else None)
    frame["detailed_position"] = detail
    frame["lineup_code"] = raw_code
    return frame


UNDERSTAT_COVERAGE_FLOOR = 0.95


def understat_coverage(data: pd.DataFrame) -> dict[int, float]:
    """Per season: Understat's minutes over FPL's, for the players matched in both."""
    matched = data[data["us_minutes"].notna() & (data["minutes"] > 0)]
    sums = matched.groupby("season_year")[["us_minutes", "minutes"]].sum()
    return (sums["us_minutes"] / sums["minutes"]).round(3).to_dict()


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def _age_lookup(seasons: list[pd.DataFrame]) -> dict[str, pd.Timestamp]:
    """Birth dates, taken from whichever seasons publish them.

    A date of birth does not change, so a player whose DOB appears in a recent
    season also gives us his age in every earlier season he played.
    """
    lookup: dict[str, pd.Timestamp] = {}
    for frame in seasons:
        if "birth_date" not in frame.columns:
            continue
        known = frame[frame["birth_date"].notna()]
        for row in known.itertuples():
            born = pd.to_datetime(row.birth_date, errors="coerce")
            if pd.notna(born):
                lookup.setdefault(normalise_name(row.player), born)
    return lookup


def build_dataset(
    repo: Path, seasons: list[str] | None = None, progress=print
) -> pd.DataFrame:
    """Assemble the ten-season Premier League player-season table."""
    repo = Path(repo)
    seasons = seasons or SEASONS
    team_names = load_team_names(repo)

    progress("reading Understat match logs...")
    understat = load_understat(repo)
    progress(f"  {len(understat):,} player-seasons of Understat data")

    frames = []
    for season in seasons:
        if not (repo / "data" / season / "players_raw.csv").exists():
            progress(f"  {season}: not in the mirror, skipped")
            continue
        frame = load_fpl_season(repo, season, team_names)
        appearances = appearances_from_gameweeks(repo, season)
        if not appearances.empty:
            frame = frame.merge(appearances, on="fpl_element", how="left")
            if "gw_team" in frame.columns:
                frame["team"] = frame["gw_team"].fillna(frame["team"])
                frame = frame.drop(columns=["gw_team"])
        else:
            frame["matches"] = np.nan
        frames.append(frame)
        progress(f"  {season}: {len(frame):,} registered players")

    data = pd.concat(frames, ignore_index=True)
    data["key"] = data["player"].map(normalise_name)
    data["season_year"] = data["season"].map(season_start_year)

    # ---- join Understat -------------------------------------------------
    if not understat.empty:
        data = data.merge(
            understat.drop(columns=["understat_player"]), on=["key", "season_year"], how="left"
        )
        played = data["minutes"] >= 900
        matched = data["us_minutes"].notna()
        progress(
            f"  Understat matched on {matched[played].mean():.0%} of player-seasons "
            f"above 900 minutes ({matched.mean():.0%} of all registered players)"
        )
        # The Understat logs were captured part-way through some seasons - the
        # 2024/25 files stop on 6 April 2025, seven gameweeks short - while FPL
        # minutes run to the last day. Dividing a partial season's shots by a
        # whole season's minutes understates every Understat per-90 by the
        # missing share, so a season whose Understat minutes fall short of
        # FPL's is not used for Understat's numbers at all. FPL's own xG and xA,
        # where the season has them, fill in; the rest is honestly missing.
        for season_year, ratio in understat_coverage(data).items():
            if ratio >= UNDERSTAT_COVERAGE_FLOOR:
                continue
            stats = [c for c in data.columns if c.startswith("us_")]
            data.loc[data["season_year"] == season_year, stats] = np.nan
            progress(f"  {season_year}-{str(season_year + 1)[-2:]}: Understat covers only "
                     f"{ratio:.0%} of the season's minutes - its numbers are not used")
        for target in list(UNDERSTAT_DIRECT.values()) + ["xa", "xg"]:
            column = f"us_{target}"
            if column not in data.columns:
                continue
            if target in data.columns:
                # Understat is the richer measurement; FPL fills the gaps.
                data[target] = data[column].where(data[column].notna(), data[target])
            else:
                data[target] = data[column]
        data = data.drop(columns=[c for c in data.columns if c.startswith("us_")])
    else:
        data["detailed_position"] = None
        data["lineup_code"] = None

    # ---- identity and attributes ---------------------------------------
    births = _age_lookup(frames)
    reference = pd.to_datetime(data["season_year"].astype(str) + "-12-31")
    born = data["key"].map(births)
    data["age"] = ((reference - born).dt.days / 365.25).round(1)
    data["height_cm"] = np.nan
    data["player_id"] = "PL" + pd.Series(data["key"].factorize()[0], index=data.index).astype(str).str.zfill(5)
    # A player's line-up position is only published up to the last season the
    # Understat mirror covers. For later seasons his most recent known position
    # is carried forward and labelled as such - positions do change, so the
    # label matters. This is a displayed attribute only: the groups the models
    # use are always the FPL buckets, never this.
    data = data.sort_values(["player_id", "season"]).reset_index(drop=True)
    data["position_source"] = np.where(
        data["detailed_position"].notna(), "Understat line-up data", None
    )
    carried = data.groupby("player_id")["detailed_position"].ffill()
    filled_now = data["detailed_position"].isna() & carried.notna()
    data["detailed_position"] = carried
    data.loc[filled_now, "position_source"] = "Last known Understat line-up position"
    data["position_source"] = data["position_source"].fillna("FPL bucket only")
    data["position"] = data["detailed_position"].fillna(data["position_group"])
    data["lineup_position"] = data["lineup_code"].map(UNDERSTAT_POSITIONS)
    data["league_tier"] = 1
    data["league_strength"] = 1.00
    data["gender"] = "male"
    data["team_possession"] = np.nan

    # ---- housekeeping ---------------------------------------------------
    data["matches"] = data["matches"].fillna(0)
    if "clubs_in_season" in data.columns:
        data["clubs_in_season"] = data["clubs_in_season"].fillna(1).astype(int)
    data["starts"] = data["starts"].fillna(data["matches"])
    data["pens_scored"] = (data["goals"] - data["np_goals"]).clip(lower=0)
    data["pens_taken"] = data["pens_scored"] + data["pens_missed"].fillna(0)
    data["ict"] = pd.to_numeric(data.get("influence"), errors="coerce").fillna(0) + \
        pd.to_numeric(data.get("creativity"), errors="coerce").fillna(0) + \
        pd.to_numeric(data.get("threat"), errors="coerce").fillna(0)

    data = data[data["minutes"] > 0].reset_index(drop=True)
    drop = ["key", "season_year", "fpl_element", "birth_date", "lineup_code", "known_as",
            "pens_missed", "pens_saved", "own_goals"]
    return data.drop(columns=[c for c in drop if c in data.columns])

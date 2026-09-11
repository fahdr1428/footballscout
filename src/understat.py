"""
Six leagues, eleven complete seasons, to 2024/25.

    Premier League - La Liga - Serie A - Bundesliga - Ligue 1 - Russian Premier League
    2014/15 -> 2024/25, ~20,300 player-seasons past 900 minutes

WHERE IT COMES FROM
-------------------
Understat's per-player season aggregates, mirrored as CSV in
`vibedatascience/understat_players_aggregated`. Understat models every shot in
these six leagues and has done since 2014/15, which is why this is the longest
and most recent run of real data in the project.

WHAT IT IS GOOD AT
------------------
**Recency and reach.** It is the only source here carrying 2022/23, 2023/24 and
2024/25, and the only one covering six leagues. Eleven seasons is enough to
follow a career: a player's 2016/17 and his 2024/25 sit in the same table.

**The expected-goals family, done properly.** xG and npxG from a shot model,
xA from the chance created, and - unusually - **xGChain** and **xGBuildup**,
which credit every player in a possession that ended in a shot. xGBuildup
excludes the shot and the assist, so it is the closest thing in open data to
"how much did this player contribute to attacks without finishing them".

WHAT IT CANNOT DO
-----------------
**It does not measure defending. At all.** There are no tackles, interceptions,
clearances, duels, pressures or blocks in this feed - Understat models shots,
not the rest of the game. Just under half the pool are defenders, and they are
ranked here purely on what they contribute going forward. For defending, use
the big-five (FBref) source, which has 44 metrics including all of the above.

**Four positional buckets, not ten.** Understat's season aggregate records only
GK / D / M / F. There is no centre-back versus full-back, no left versus right
wing. Inferring a finer position from the same statistics the models then read
would be circular, so this source uses the four-bucket taxonomy, which the
platform treats as first class.

**2025/26 is a fragment.** The mirror stopped updating in September 2025, so
that season holds about ten rounds. It is excluded by default for the same
reason the FBref source excludes its partial year: pooling a tenth of a season
with whole ones puts those players at the bottom of every volume metric for a
reason that has nothing to do with them.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "https://github.com/vibedatascience/understat_players_aggregated"
RAW = ("https://raw.githubusercontent.com/vibedatascience/understat_players_aggregated/"
       "main/understat_players_aggregated_2014_td.csv")
ATTRIBUTION = (
    "Understat per-player season aggregates, mirrored by the open-source "
    f"understat_players_aggregated repository ({REPO}). Ages, heights, "
    "preferred feet and market values from Transfermarkt profiles, mirrored by "
    "salimt/football-datasets (https://github.com/salimt/football-datasets)."
)

# Transfermarkt biography and market-value history, used to fill in what
# Understat does not record. Both are plain CSV in the mirror (the two files
# that are Git LFS there are not needed).
TM_BASE = ("https://raw.githubusercontent.com/salimt/football-datasets/"
           "main/datalake/transfermarkt")
TM_PROFILES = f"{TM_BASE}/player_profiles/player_profiles.csv"
TM_VALUES = f"{TM_BASE}/player_market_value/player_market_value.csv"

# The mirror froze in September 2025, so its newest season is ~10 rounds deep.
PARTIAL_SEASONS = {"2025/26"}
FIRST_SEASON, LAST_SEASON = "2014/15", "2024/25"

# Understat's league keys -> the names in config.LEAGUES.
LEAGUES = {
    "EPL": "Premier League",
    "La_Liga": "La Liga",
    "Serie_A": "Serie A",
    "Bundesliga": "Bundesliga",
    "Ligue_1": "Ligue 1",
    "RFPL": "Russian Premier League",
}

# Understat's single-letter role -> the platform's four-bucket taxonomy.
# "S" means the player only ever came off the bench with no recorded role; every
# such row is under 400 minutes, so the minimum-minutes filter removes them all.
POSITIONS = {"GK": ("GK", "GK"), "D": ("DEF", "DEF"),
             "M": ("MID", "MID"), "F": ("FWD", "FWD")}

# Outside this range the name matched the wrong player, not an unusual career.
MIN_PLAUSIBLE_AGE, MAX_PLAUSIBLE_AGE = 15.0, 45.0

RENAMES = {
    "player_name": "player", "team_title": "team", "time": "minutes",
    "games": "matches", "npg": "np_goals", "xG": "xg", "npxG": "npxg",
    "xA": "xa", "xGChain": "xg_chain", "xGBuildup": "xg_buildup",
}


def download(url: str, cache: Path, name: str,
             retries: int = 4, backoff: float = 2.0) -> Path:
    """Fetch one CSV into the cache, skipping if it is already there."""
    target = cache / name
    if target.exists() and target.stat().st_size > 0:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                target.write_bytes(response.read())
            return target
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))
    return target


def season_label(season: str) -> str:
    """Understat writes 2024/25; the rest of the platform writes 2024-25."""
    return str(season).replace("/", "-")


def _position(row) -> tuple[str, str, str]:
    """(group, position, where it came from) from Understat's role letters."""
    primary = str(row.get("primary_position") or "").strip()
    if primary in POSITIONS:
        group, position = POSITIONS[primary]
        return group, position, "Understat line-ups"
    # Only ever a substitute: fall back to whatever role the season recorded.
    for letter in str(row.get("position") or "").split():
        if letter in POSITIONS:
            group, position = POSITIONS[letter]
            return group, position, "Understat line-ups (substitute only)"
    return "MID", "MID", "unknown"


def attach_transfermarkt(frame: pd.DataFrame, profiles: Path, values: Path,
                         progress=print) -> tuple[pd.DataFrame, dict]:
    """Fill in what Understat does not record, from Transfermarkt profiles.

    Understat publishes no age, height, foot, nationality or valuation - and
    without an age the whole youth side of scouting is unavailable: no age
    filter, no "younger equivalent", no age component in the hidden-gem score.

    The two feeds share no id, so players are matched **on name, and only when
    the normalised name is unique on both sides**. A name held by two players in
    either dataset is left unmatched rather than guessed at: a wrong age on a
    shortlist is worse than a missing one. That costs coverage - about 73% of
    players match, covering 79% of the minutes played - and the app reports it.

    Position is deliberately NOT taken from here, even though Transfermarkt has
    a specific one. It would arrive for two players in three, so a player's peer
    group would depend on whether his name happened to match rather than on
    football. Grouping stays on Understat's own four buckets.
    """
    from .premier_league import normalise_name

    progress("matching players to Transfermarkt profiles")
    tm = pd.read_csv(profiles, low_memory=False, usecols=[
        "player_id", "player_name", "date_of_birth", "height", "foot", "citizenship"])
    tm = tm.rename(columns={"player_id": "tm_id"})
    tm["clean"] = tm["player_name"].astype(str).str.replace(r"\s*\(\d+\)\s*$", "", regex=True)
    tm["key"] = tm["clean"].map(normalise_name)
    tm = tm[~tm["key"].duplicated(keep=False)]

    frame = frame.copy()
    frame["key"] = frame["player"].map(normalise_name)
    ambiguous = frame["key"].duplicated(keep=False) & frame["player_id"].duplicated(keep=False)
    keys = frame[["player_id", "key"]].drop_duplicates("player_id")
    keys = keys[~keys["key"].duplicated(keep=False)]

    link = keys.merge(tm, on="key", how="inner")[
        ["player_id", "tm_id", "date_of_birth", "height", "foot", "citizenship"]]
    frame = frame.merge(link, on="player_id", how="left")

    frame["age"] = _age(frame)
    frame["height_cm"] = pd.to_numeric(
        frame["height"].astype(str).str.replace(",", ".").str.extract(r"([\d.]+)")[0],
        errors="coerce")
    # Transfermarkt writes heights in metres; anything under 3 is metres.
    frame["height_cm"] = np.where(frame["height_cm"] < 3,
                                  frame["height_cm"] * 100, frame["height_cm"]).round()
    # Transfermarkt lists every citizenship a player holds, run together; the
    # first is the one he represents or was born to, which is what a scout reads.
    frame["nationality"] = (frame["citizenship"].astype(str)
                            .str.split(r"\s{2,}", regex=True).str[0]
                            .str.strip().replace({"nan": np.nan, "": np.nan}))
    frame = frame.drop(columns=["citizenship"])
    frame["foot"] = frame["foot"].astype(str).str.lower().replace("nan", np.nan)

    # An impossible age is not a bad age - it is evidence the name matched the
    # wrong person, and everything else attached to that row came from him too.
    # So the whole enrichment is withdrawn for those rows rather than patched.
    wrong = frame["age"].notna() & ~frame["age"].between(MIN_PLAUSIBLE_AGE, MAX_PLAUSIBLE_AGE)
    rejected = int(wrong.sum())
    frame.loc[wrong, ["age", "height_cm", "foot", "nationality", "tm_id",
                      "date_of_birth"]] = np.nan

    progress("attaching market values as at each season")
    frame = _attach_values(frame, values)

    coverage = {
        "rejected_implausible_age": rejected,
        "matched_players": int(frame.loc[frame["tm_id"].notna(), "player_id"].nunique()),
        "total_players": int(frame["player_id"].nunique()),
        "minutes_covered": float(
            frame.loc[frame["tm_id"].notna(), "minutes"].sum() / frame["minutes"].sum()),
        "with_age": float(frame["age"].notna().mean()),
        "with_market_value": float(frame["market_value_eur"].notna().mean()),
    }
    return frame.drop(columns=["key", "height", "tm_id"], errors="ignore"), coverage


def _age(frame: pd.DataFrame) -> pd.Series:
    """Age at 1 January inside the season, from date of birth."""
    born = pd.to_datetime(frame["date_of_birth"], errors="coerce", format="mixed")
    end_year = frame["season"].str.slice(0, 4).astype(int) + 1
    reference = pd.to_datetime(end_year.astype(str) + "-01-01")
    return ((reference - born).dt.days / 365.25).round(1)


def _attach_values(frame: pd.DataFrame, values: Path) -> pd.DataFrame:
    """Market value as at each season, not a single scrape-time snapshot.

    Transfermarkt revalues players a few times a year, so the history can be
    read at the right moment: the most recent valuation on or before 1 January
    inside that season. That is what the player was considered worth *then*,
    which is the only version of the number worth putting next to that season's
    output.
    """
    history = pd.read_csv(values, low_memory=False)
    history["date"] = pd.to_datetime(history["date_unix"], errors="coerce")
    history = history.dropna(subset=["date", "value"]).sort_values("date")
    history["tm_id"] = pd.to_numeric(history["player_id"], errors="coerce")
    history = history.dropna(subset=["tm_id"])
    history["tm_id"] = history["tm_id"].astype("int64")

    frame = frame.copy()
    frame["_asof"] = pd.to_datetime(
        (frame["season"].str.slice(0, 4).astype(int) + 1).astype(str) + "-01-01")
    left = frame[["tm_id", "_asof"]].copy()
    left["tm_id"] = pd.to_numeric(left["tm_id"], errors="coerce")
    # merge_asof needs the `by` key to be the same dtype on both sides.
    left = left.reset_index().dropna(subset=["tm_id"]).sort_values("_asof")
    left["tm_id"] = left["tm_id"].astype("int64")

    merged = pd.merge_asof(
        left, history[["tm_id", "date", "value"]].rename(columns={"date": "_asof"}),
        on="_asof", by="tm_id", direction="backward",
    )
    frame["market_value_eur"] = np.nan
    frame.loc[merged["index"], "market_value_eur"] = merged["value"].to_numpy()
    return frame.drop(columns=["_asof"])


def build_dataset(cache: Path, seasons: list[str] | None = None,
                  enrich: bool = True, progress=print) -> tuple[pd.DataFrame, dict]:
    """Build the six-league dataset. Returns the frame and a coverage summary."""
    progress("fetching the Understat mirror")
    frame = pd.read_csv(download(RAW, cache, "understat_players_aggregated.csv"))
    raw_rows = len(frame)

    frame = frame[frame["league"].isin(LEAGUES)].copy()
    frame["league"] = frame["league"].map(LEAGUES)

    if seasons is None:
        seasons = [s for s in sorted(frame["season"].unique())
                   if s not in PARTIAL_SEASONS and FIRST_SEASON <= s <= LAST_SEASON]
    frame = frame[frame["season"].isin(seasons)]

    mapped = [_position(row) for _, row in frame.iterrows()]
    frame["position_group"] = [m[0] for m in mapped]
    frame["position"] = [m[1] for m in mapped]
    frame["position_source"] = [m[2] for m in mapped]

    frame = frame.rename(columns=RENAMES)
    frame["season"] = frame["season"].map(season_label)
    frame["gender"] = "male"
    # Understat ids are stable across seasons, which is what the trajectory and
    # self-season-recall checks need to recognise the same player twice.
    frame["player_id"] = frame["id"].astype(str)

    keep = ["player", "player_id", "team", "league", "season", "position",
            "position_group", "position_source", "gender", "minutes", "matches",
            "goals", "np_goals", "assists", "xg", "npxg", "xa", "shots",
            "key_passes", "xg_chain", "xg_buildup", "yellow_cards", "red_cards"]
    frame = frame[[c for c in keep if c in frame.columns]]
    frame = frame[frame["minutes"].gt(0) & frame["league"].notna()]

    # One row per player-season: a mid-season move is two Understat rows, and the
    # counting stats add up while the club of record is where he played most.
    frame = _collapse_transfers(frame)

    if enrich:
        profiles = download(TM_PROFILES, cache, "tm_player_profiles.csv")
        history = download(TM_VALUES, cache, "tm_market_values.csv")
        frame, matched = attach_transfermarkt(frame, profiles, history, progress=progress)
    else:
        matched = {}

    frame = frame.sort_values(["season", "league", "team", "player"]).reset_index(drop=True)
    coverage = {
        "rows_in": raw_rows,
        "player_seasons": len(frame),
        "players": int(frame["player_id"].nunique()),
        "seasons": sorted(frame["season"].unique()),
        "leagues": sorted(frame["league"].unique()),
        **matched,
    }
    return frame, coverage


def _collapse_transfers(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season, totals summed, club of record by minutes."""
    keys = ["player_id", "season"]
    frame = frame.sort_values("minutes", ascending=False)
    sizes = frame.groupby(keys)["minutes"].transform("size")

    single = frame[sizes == 1].copy()
    single["clubs_in_season"] = 1

    moved = frame[sizes > 1]
    if moved.empty:
        return single.reset_index(drop=True)

    counts = [c for c in moved.columns
              if c not in keys and pd.api.types.is_numeric_dtype(moved[c])]
    rows = []
    for _, block in moved.groupby(keys, sort=False):
        row = block.iloc[0].copy()           # most minutes: club and league of record
        for column in counts:
            values = block[column]
            row[column] = values.sum() if values.notna().all() else np.nan
        row["clubs_in_season"] = len(block)
        rows.append(row)
    return pd.concat([single, pd.DataFrame(rows)], ignore_index=True)

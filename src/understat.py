"""
Six leagues, twelve complete seasons, and the one being played.

    Premier League - La Liga - Serie A - Bundesliga - Ligue 1 - Russian Premier League
    2014/15 -> 2025/26 complete, plus the season in progress

WHERE IT COMES FROM
-------------------
Understat's per-player season aggregates: to 2024/25 from the CSV mirror
`vibedatascience/understat_players_aggregated`, and from 2025/26 fetched from
understat.com itself by scripts/refresh_sources.py on GitHub's runners.
Understat models every shot in these six leagues and has done since 2014/15,
which is why this is the longest
and most recent run of real data in the project.

WHAT IT IS GOOD AT
------------------
**Recency and reach.** It is the only source here carrying the season being
played, and the only one covering six leagues. Twelve seasons is enough to
follow a career: a player's 2016/17 and his 2025/26 sit in the same table.

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
the roles a player appeared in - GK / D / M / F - and lists them alphabetically,
not by how often he played each. Transfermarkt's position picks among the roles
he actually played that season (`assign_positions`); inferring anything finer
from the statistics the models then read would be circular, so this source uses
the four-bucket taxonomy, which the platform treats as first class.

**The season being played** is built and flagged, but is not in the default
pool: a few rounds in, nobody has 900 minutes, and pooling it with whole
seasons would put every player at the bottom of every volume metric for a
reason that has nothing to do with him. `season_status` reads which seasons
are finished from the data, so a new season needs no change here.
"""

from __future__ import annotations

import html
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import tm_history
from .config import RAW_DIR

REPO = "https://github.com/vibedatascience/understat_players_aggregated"
RAW = ("https://raw.githubusercontent.com/vibedatascience/understat_players_aggregated/"
       "main/understat_players_aggregated_2014_td.csv")
ATTRIBUTION = (
    "Understat per-player season aggregates (https://understat.com) - to 2024/25 "
    f"from the open-source understat_players_aggregated mirror ({REPO}), from "
    "2025/26 fetched from understat.com directly. Ages, heights, preferred feet "
    "and market values from Transfermarkt, via dcaribou/transfermarkt-datasets "
    "(https://github.com/dcaribou/transfermarkt-datasets, CC0) and "
    "salimt/football-datasets (https://github.com/salimt/football-datasets)."
)

# The mirror froze in September 2025, a few rounds into 2025/26. The seasons
# after that come from understat.com itself, fetched on GitHub's runners by
# scripts/refresh_sources.py, and replace the mirror's rows for any season both
# carry. A season is complete once its deepest league has played
# FULL_SEASON_GAMES rounds; the one in progress is kept, flagged, and left out
# of the default pool.
LIVE_CSV = RAW_DIR / "live" / "understat_live.csv"
# Keepers' full matches in goal, from Understat's match lists: the xG their team
# faced and the goals it let in while they played every minute (written by
# scripts/refresh_sources.py on GitHub's runners).
KEEPERS_CSV = RAW_DIR / "live" / "understat_keepers.csv"
KEEPER_COLUMNS = ["gk_full_matches", "gk_full_xga", "gk_full_goals_against", "gk_full_clean_sheets"]
FULL_SEASON_GAMES = 30
PARTIAL_SEASONS = {"2026/27"}
FIRST_SEASON, LAST_SEASON = "2014/15", "2025/26"

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


def attach_transfermarkt(frame: pd.DataFrame, cache: Path,
                         progress=print) -> tuple[pd.DataFrame, dict]:
    """Fill in what Understat does not record, from Transfermarkt.

    Understat publishes no age, height, foot, nationality or valuation - and
    without an age the whole youth side of scouting is unavailable: no age
    filter, no "younger equivalent", no age component in the hidden-gem score.

    The two feeds share no id, so a player is matched in two passes:

    1. **Name, season, league and club.** Transfermarkt's minutes per club per
       season (`data/raw/tm/tm_season_minutes.csv.gz`) say who played where.
       An Understat row is matched to a Transfermarkt row with the same
       normalised name in the same league-season - or the same surname and
       first initial *and* the same club - and where two candidates remain,
       the one whose minutes are closest wins. A player's Transfermarkt id is
       then the one most of his seasons agree on, which also carries it to
       seasons Transfermarkt has not reached yet.
    2. **Unique name**, for whoever the first pass missed: matched only when
       the normalised name is held by one player on both sides. A name held by
       two players in either dataset is left unmatched rather than guessed at:
       a wrong age on a shortlist is worse than a missing one.

    Position comes from here only as a choice among the roles Understat says
    he played that season (`assign_positions`): Understat lists those roles
    alphabetically, so on its own it cannot say which one he mostly played.
    """
    frame = frame.copy()
    frame["key"] = frame["player"].map(tm_history.normalise_name)
    profiles = tm_history.load_profiles(cache, progress=progress)

    progress("matching players to Transfermarkt by name, season and club")
    link = tm_history.link_players(frame, profiles)

    frame = frame.merge(link[["player_id", "tm_id", "how"]], on="player_id", how="left")
    progress("reading where each player played, season by season")
    seasonal = tm_history.load_season_positions(cache, progress=progress)
    frame = assign_positions(frame, profiles, seasonal)
    frame = attach_positions_played(frame, seasonal)
    frame = frame.merge(
        profiles[["tm_id", "date_of_birth", "height_cm", "foot", "nationality"]
                 + (["contract_expiry"] if "contract_expiry" in profiles.columns else [])],
        on="tm_id", how="left")
    frame["date_of_birth"] = pd.to_datetime(frame["date_of_birth"]).dt.strftime("%Y-%m-%d")
    frame["age"] = _age(frame)

    # An impossible age is not a bad age - it is evidence the name matched the
    # wrong person, and everything else attached to that row came from him too.
    # So the whole enrichment is withdrawn for those rows rather than patched.
    wrong = frame["age"].notna() & ~frame["age"].between(MIN_PLAUSIBLE_AGE, MAX_PLAUSIBLE_AGE)
    rejected = int(wrong.sum())
    frame.loc[wrong, [c for c in ["age", "height_cm", "foot", "nationality", "tm_id",
                                  "date_of_birth", "contract_expiry", *POSITIONS_PLAYED] if c in frame.columns]] = np.nan

    progress("attaching market values")
    frame = _attach_values(frame, cache, progress=progress)

    matched = frame["tm_id"].notna()
    coverage = {
        "rejected_implausible_age": rejected,
        "matched_players": int(frame.loc[matched, "player_id"].nunique()),
        "matched_by_season_and_club": int(
            frame.loc[matched & frame["how"].eq("season"), "player_id"].nunique()),
        "total_players": int(frame["player_id"].nunique()),
        "minutes_covered": float(frame.loc[matched, "minutes"].sum() / frame["minutes"].sum()),
        "with_age": float(frame["age"].notna().mean()),
        "with_market_value": float(frame["market_value_eur"].notna().mean()),
    }
    return frame.drop(columns=["key", "tm_id", "how"], errors="ignore"), coverage


# Transfermarkt's buckets, and the Understat role letter each one corresponds to.
TM_BUCKETS = {"Goalkeeper": "GK", "Defender": "DEF", "Midfield": "MID", "Attack": "FWD"}
ROLE_LETTER = {"GK": "GK", "DEF": "D", "MID": "M", "FWD": "F"}


POSITIONS_PLAYED = ["position_starts", "position_share", "positions_played", "captain_starts"]


def attach_positions_played(frame: pd.DataFrame, seasonal: pd.DataFrame | None) -> pd.DataFrame:
    """Transfermarkt's line-up record for each player-season: how many league
    games he started in his main position, what share of his starts that was,
    every position he started in, and how many he started as captain. Line-ups
    only - a squad page names one position and counts nothing."""
    frame = frame.drop(columns=[c for c in POSITIONS_PLAYED if c in frame.columns])
    if seasonal is None or not len(seasonal) or "tm_starts" not in seasonal.columns:
        return frame
    record = seasonal[seasonal["how"].eq("line-ups")].rename(columns={
        "tm_starts": "position_starts", "tm_share": "position_share",
        "tm_played": "positions_played", "tm_captain": "captain_starts"})
    record = record[["tm_id", "start"] + [c for c in POSITIONS_PLAYED if c in record.columns]]
    key = pd.DataFrame({"tm_id": pd.to_numeric(frame["tm_id"], errors="coerce"),
                        "start": pd.to_numeric(frame["season"].astype(str).str.slice(0, 4), errors="coerce")},
                       index=frame.index)
    hit = key.merge(record, on=["tm_id", "start"], how="left")
    for column in POSITIONS_PLAYED:
        frame[column] = hit[column].to_numpy() if column in hit.columns else np.nan
    return frame


def assign_positions(frame: pd.DataFrame, profiles: pd.DataFrame,
                     seasonal: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each player's bucket: Transfermarkt's, among the roles he played that season.

    Understat records the roles a player appeared in during a season - "D M S"
    - but lists them **alphabetically**, not by how often he played each, so
    taking the first one puts every midfielder who ever dropped into defence
    in the defenders' pool (Declan Rice, "D M S", read as a defender). Which
    role he mostly played is not in the feed at all.

    Transfermarkt's position answers that, and it is a separate source, so the
    grouping is still not inferred from the statistics the models read. It is
    used only when it names a role Understat says he played *that* season, so
    a player whose position changed over his career is never put somewhere
    he did not play. Where it does not - or he is unmatched - the first role
    listed stands, and `position_source` says which rule decided.
    """
    frame = frame.copy()
    found = frame[["tm_id"]].merge(profiles[["tm_id", "tm_bucket", "tm_detail"]],
                                   on="tm_id", how="left")
    bucket = found["tm_bucket"].map(TM_BUCKETS).to_numpy()
    detail = found["tm_detail"].to_numpy()
    roles = frame["roles"].fillna("").astype(str).str.split()

    # Where he played *that* season, when Transfermarkt recorded it: his
    # profile only knows today's position.
    then_detail = np.full(len(frame), None, dtype=object)
    then_how = np.full(len(frame), None, dtype=object)
    if seasonal is not None and len(seasonal):
        start = pd.to_numeric(frame["season"].astype(str).str.slice(0, 4), errors="coerce")
        hit = (pd.DataFrame({"tm_id": pd.to_numeric(frame["tm_id"], errors="coerce"), "start": start})
               .merge(seasonal, on=["tm_id", "start"], how="left"))
        then_detail = hit["tm_season_detail"].to_numpy(dtype=object)
        then_how = hit["how"].to_numpy(dtype=object)
    then_bucket = np.array([TM_BUCKETS.get(tm_history.DETAIL_BUCKET.get(d)) if isinstance(d, str) else None
                            for d in then_detail], dtype=object)

    group, position, source = [], [], []
    for i, r in enumerate(roles):
        if isinstance(then_bucket[i], str) and ROLE_LETTER[then_bucket[i]] in r:
            group.append(then_bucket[i]); position.append(then_detail[i])
            source.append(f"Transfermarkt {then_how[i]}, that season")
        elif isinstance(bucket[i], str) and ROLE_LETTER[bucket[i]] in r:
            group.append(bucket[i]); position.append(detail[i] if isinstance(detail[i], str) else bucket[i])
            source.append("Transfermarkt profile (today's position), among the roles he played that season")
        else:
            group.append(None); position.append(None); source.append(None)
    use = np.array([g is not None for g in group])
    frame.loc[use, "position_group"] = np.array(group, dtype=object)[use]
    frame.loc[use, "position"] = np.array(position, dtype=object)[use]
    frame.loc[use, "position_source"] = np.array(source, dtype=object)[use]
    listed = ~use & frame["position_source"].eq("Understat line-ups")
    frame.loc[listed, "position_source"] = "Understat, first role listed (alphabetical)"
    return frame


# The name-season-club matcher is shared with the Premier League source.
_match_by_season = tm_history.link_by_season


def _age(frame: pd.DataFrame) -> pd.Series:
    """Age at 1 January inside the season, from date of birth."""
    born = pd.to_datetime(frame["date_of_birth"], errors="coerce", format="mixed")
    end_year = frame["season"].str.slice(0, 4).astype(int) + 1
    reference = pd.to_datetime(end_year.astype(str) + "-01-01")
    return ((reference - born).dt.days / 365.25).round(1)


def _attach_values(frame: pd.DataFrame, cache: Path, progress=print) -> pd.DataFrame:
    """Market value as each season closed, plus his latest one.

    Transfermarkt revalues players a few times a year, so the history is read
    at a stated moment rather than one scrape applied to every season: the
    valuation in force on 1 July after the season ends - what he was worth as
    it closed - which is the same rule the big-five source uses, so a price
    means one thing wherever it appears. A season still in progress gets the
    latest valuation there is. `latest_value_eur` and its date are the price
    a scout reads as today's.
    """
    history = tm_history.load_valuations(cache, progress=progress)
    frame = frame.copy()
    asof = pd.to_datetime(
        (frame["season"].str.slice(0, 4).astype(int) + 1).astype(str) + "-07-01")
    frame["market_value_eur"] = tm_history.value_asof(frame["tm_id"], asof, history)
    latest = tm_history.latest_values(history)
    joined = frame[["tm_id"]].merge(latest, on="tm_id", how="left")
    frame["latest_value_eur"] = joined["latest_value_eur"].to_numpy()
    frame["latest_value_date"] = joined["latest_value_date"].to_numpy()
    return frame


def season_status(frame: pd.DataFrame) -> tuple[set[str], set[str]]:
    """(complete, in progress): complete once its deepest league reaches FULL_SEASON_GAMES."""
    depth = pd.to_numeric(frame["games"], errors="coerce").groupby(frame["season"]).max()
    complete = {s for s, games in depth.items() if games >= FULL_SEASON_GAMES}
    return complete, set(depth.index) - complete


def build_dataset(cache: Path, seasons: list[str] | None = None,
                  enrich: bool = True, include_partial: bool = False,
                  progress=print) -> tuple[pd.DataFrame, dict]:
    """Build the six-league dataset. Returns the frame and a coverage summary."""
    progress("fetching the Understat mirror")
    frame = pd.read_csv(download(RAW, cache, "understat_players_aggregated.csv"))
    if LIVE_CSV.exists():
        live = pd.read_csv(LIVE_CSV)
        progress(f"  + {len(live):,} rows fetched from understat.com for "
                 + ", ".join(sorted(live["season"].unique())))
        frame = pd.concat([frame[~frame["season"].isin(set(live["season"]))], live],
                          ignore_index=True)
    raw_rows = len(frame)

    frame = frame[frame["league"].isin(LEAGUES)].copy()
    frame["league"] = frame["league"].map(LEAGUES)

    if seasons is None:
        # Which seasons are finished is read from the data, not a constant, so
        # a new season starting in August needs no edit here. The season in
        # progress is built in the same pass when asked for, so a player's
        # Transfermarkt match - voted on across his seasons - reaches it too;
        # Transfermarkt's minutes for it do not exist yet.
        complete, partial = season_status(frame)
        seasons = [s for s in sorted(frame["season"].unique())
                   if FIRST_SEASON <= s and (s in complete or (include_partial and s in partial))]
    frame = frame[frame["season"].isin(seasons)]

    # Understat's role string, kept so Transfermarkt can choose among it later.
    frame["roles"] = frame["position"].astype(str)
    mapped = [_position(row) for _, row in frame.iterrows()]
    frame["position_group"] = [m[0] for m in mapped]
    frame["position"] = [m[1] for m in mapped]
    frame["position_source"] = [m[2] for m in mapped]

    frame = _unescape_names(frame.rename(columns=RENAMES))
    frame["season"] = frame["season"].map(season_label)
    frame["gender"] = "male"
    # Understat ids are stable across seasons, which is what the trajectory and
    # self-season-recall checks need to recognise the same player twice.
    frame["player_id"] = frame["id"].astype(str)

    keep = ["roles", "player", "player_id", "team", "league", "season", "position",
            "position_group", "position_source", "gender", "minutes", "matches",
            "goals", "np_goals", "assists", "xg", "npxg", "xa", "shots",
            "key_passes", "xg_chain", "xg_buildup", "yellow_cards", "red_cards"]
    frame = frame[[c for c in keep if c in frame.columns]]
    frame = frame[frame["minutes"].gt(0) & frame["league"].notna()]

    # One row per player-season: a mid-season move is two Understat rows, and the
    # counting stats add up while the club of record is where he played most.
    frame = _collapse_transfers(frame)
    frame = attach_keepers(frame, progress=progress)

    if enrich:
        frame, matched = attach_transfermarkt(frame, cache, progress=progress)
    else:
        matched = {}

    frame = frame.drop(columns=["roles"], errors="ignore")
    frame = frame.sort_values(["season", "league", "team", "player", "player_id"],
                              kind="stable").reset_index(drop=True)
    coverage = {
        "rows_in": raw_rows,
        "player_seasons": len(frame),
        "players": int(frame["player_id"].nunique()),
        "seasons": sorted(frame["season"].unique()),
        "leagues": sorted(frame["league"].unique()),
        **matched,
    }
    return frame, coverage


def attach_keepers(frame: pd.DataFrame, path: Path = KEEPERS_CSV, progress=print) -> pd.DataFrame:
    """Each keeper-season's full matches in goal, where they have been fetched."""
    if not path.exists():
        return frame
    keepers = pd.read_csv(path, dtype={"player_id": str})
    keepers = keepers[["player_id", "season"] + KEEPER_COLUMNS].drop_duplicates(["player_id", "season"], keep="last")
    out = frame.drop(columns=[c for c in KEEPER_COLUMNS if c in frame.columns]).merge(
        keepers, on=["player_id", "season"], how="left")
    found = out["gk_full_matches"].notna() & out["position_group"].eq("GK")
    progress(f"  keepers' full matches attached to {int(found.sum()):,} goalkeeper-seasons")
    return out


def _unescape_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Understat serves names HTML-escaped ("N&#039;Golo Kanté"); store them as written."""
    for column in ("player", "team"):
        if column in frame:
            frame[column] = frame[column].map(lambda v: html.unescape(v) if isinstance(v, str) else v)
    return frame


def _collapse_transfers(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season, totals summed, club of record by minutes."""
    keys = ["player_id", "season"]
    # Most minutes first; level on minutes (one game at each club), the club
    # that comes first alphabetically - a stated rule, so every machine agrees.
    frame = frame.sort_values(keys + ["minutes", "team"], ascending=[True, True, False, True],
                              kind="stable")
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

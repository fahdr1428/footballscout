#!/usr/bin/env python3
"""
Fetch the current seasons the mirrors stopped short of, from the primary sources.

    python scripts/refresh_sources.py                 # both, into data/raw
    python scripts/refresh_sources.py --only understat --seasons 2025 2026

The GitHub mirrors this project was built on froze in September 2025: the
Understat aggregate a few rounds into 2025/26, FBref's release asset the same
week. This reaches past them:

* **Understat** - every player's season aggregate for the six leagues, read
  from understat.com itself, written in exactly the mirror's CSV layout so
  `src/understat.py` reads the two as one table. Understat has served this two
  ways over the years - a JSON endpoint behind the league page, and before that
  JSON embedded in the page - so both are tried, and the log says which worked.
* **Transfermarkt** - dcaribou/transfermarkt-datasets, a CC0 public-domain
  build of Transfermarkt published as CSV (valuations to June 2026 at the time
  of writing). Only what this project uses is kept: profiles, valuation
  history, and minutes per club per season in the six leagues, which is what
  lets a player be matched on name *and* club rather than name alone.

It runs on GitHub's runners (`.github/workflows/refresh-data.yml`), which can
reach both hosts; the build machine this project is usually developed on
cannot. Every request is spaced out and identifies itself, and nothing here is
fetched more than once a run.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

UNDERSTAT = "https://understat.com"
UNDERSTAT_LEAGUES = ["EPL", "La_Liga", "Bundesliga", "Serie_A", "Ligue_1", "RFPL"]
# Understat names a season by the year it starts: 2025 is 2025/26. The run
# fetches the season being played and the one before it (which may have
# finished since the last run), so the schedule needs no editing each summer.
def default_seasons(today: datetime | None = None) -> list[int]:
    today = today or datetime.now(timezone.utc)
    current = today.year if today.month >= 7 else today.year - 1
    return [current - 1, current]
MIRROR_COLUMNS = [
    "assists", "games", "goals", "id", "key_passes", "npg", "npxG", "player_name",
    "position", "red_cards", "shots", "team_title", "time", "xA", "xG", "xGBuildup",
    "xGChain", "yellow_cards", "league", "year", "season", "primary_position",
    "scrape_timestamp",
]

TM_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
# Transfermarkt's ids for the six leagues Understat covers.
TM_COMPETITIONS = {"GB1": "Premier League", "ES1": "La Liga", "IT1": "Serie A",
                   "L1": "Bundesliga", "FR1": "Ligue 1", "RU1": "Russian Premier League"}
TM_FIRST_SEASON = 2014
TM_PLAYER_COLUMNS = [
    "player_id", "name", "first_name", "last_name", "date_of_birth", "height_in_cm",
    "foot", "position", "sub_position", "country_of_citizenship", "current_club_id",
    "current_club_name", "market_value_in_eur", "highest_market_value_in_eur",
    "contract_expiration_date", "last_season", "url",
]

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36 footballscout-refresh "
              "(+https://github.com/fahdr1428/footballscout)")
PAUSE_SECONDS = 2.5
# A fixed mtime keeps the gzip bytes identical when the table is, so a run
# that finds nothing new commits nothing.
GZIP = {"method": "gzip", "mtime": 0}


def log(message: str) -> None:
    print(message, flush=True)


def fetch(url: str, headers: dict | None = None, retries: int = 4, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept-Encoding": "gzip", **(headers or {})})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
            # Understat compresses its JSON whether or not it was asked to.
            if body[:2] == b"\x1f\x8b" and not url.endswith(".gz"):
                body = gzip.decompress(body)
            return body
        except (urllib.error.URLError, TimeoutError) as error:
            status = getattr(error, "code", None)
            log(f"  attempt {attempt + 1} failed for {url}: {status or error}")
            if status in (403, 404) or attempt == retries - 1:
                raise
            time.sleep(PAUSE_SECONDS * (2 ** attempt))
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------
# Understat
# --------------------------------------------------------------------------

def _decode_embedded(page: str, variable: str) -> list | None:
    """The older layout: `var playersData = JSON.parse('\\x5B...')` in the page."""
    match = re.search(variable + r"\s*=\s*JSON\.parse\('(.*?)'\)", page, re.S)
    if not match:
        return None
    text = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), match.group(1))
    text = text.replace("\\'", "'").replace('\\\\', '\\')
    return json.loads(text)


def understat_players(league: str, season: int) -> tuple[list[dict], str]:
    """One league-season of player aggregates, and which route returned it."""
    page_url = f"{UNDERSTAT}/league/{league}/{season}"
    try:
        raw = fetch(f"{UNDERSTAT}/getLeagueData/{league}/{season}", headers={
            "X-Requested-With": "XMLHttpRequest", "Referer": page_url,
            "Accept": "application/json, text/javascript, */*; q=0.01"})
        data = json.loads(raw.decode("utf-8"))
        players = data.get("players") if isinstance(data, dict) else None
        if players:
            return players, "getLeagueData"
        log(f"  getLeagueData returned no players (keys: {list(data)[:6] if isinstance(data, dict) else type(data)})")
    except Exception as error:  # noqa: BLE001 - fall through to the page
        log(f"  getLeagueData unavailable: {error}")
    time.sleep(PAUSE_SECONDS)
    page = fetch(page_url).decode("utf-8", "replace")
    players = _decode_embedded(page, "playersData")
    if players:
        return players, "embedded JSON"
    log(f"  page fetched ({len(page):,} chars) but no player data found in it")
    return [], "none"


def refresh_understat(seasons: list[int], out: Path) -> dict:
    # The day, not the second: a re-run the same day writes identical rows.
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    frames, summary = [], {}
    for season in seasons:
        for league in UNDERSTAT_LEAGUES:
            log(f"understat {league} {season}")
            players, route = understat_players(league, season)
            time.sleep(PAUSE_SECONDS)
            if not players:
                summary[f"{league} {season}"] = {"players": 0, "route": route}
                continue
            frame = pd.DataFrame(players)
            frame["league"] = league
            frame["year"] = season
            frame["season"] = f"{season}/{str(season + 1)[-2:]}"
            # The mirror's primary position is the first letter of the role
            # string Understat records (it lists them sorted), kept identical
            # here so old and new seasons are grouped by the same rule.
            frame["primary_position"] = frame["position"].astype(str).str.split().str[0]
            frame["scrape_timestamp"] = stamp
            frames.append(frame)
            summary[f"{league} {season}"] = {
                "players": len(frame), "route": route,
                "max_games": int(pd.to_numeric(frame["games"], errors="coerce").max()),
                "max_minutes": int(pd.to_numeric(frame["time"], errors="coerce").max()),
            }
            log(f"  {len(frame)} players via {route}; most games "
                f"{summary[f'{league} {season}']['max_games']}")
    if not frames:
        raise SystemExit("Understat returned nothing for any league-season")
    table = pd.concat(frames, ignore_index=True)
    missing = [c for c in MIRROR_COLUMNS if c not in table.columns]
    if missing:
        log(f"  columns Understat did not send: {missing}")
    table = table[[c for c in MIRROR_COLUMNS if c in table.columns]]
    target = out / "live" / "understat_live.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    log(f"wrote {target.relative_to(ROOT)}: {len(table):,} rows")
    return {"fetched_at": stamp, "rows": len(table), "by_league_season": summary}


# --------------------------------------------------------------------------
# Transfermarkt (dcaribou/transfermarkt-datasets, CC0)
# --------------------------------------------------------------------------

def tm_table(name: str, usecols=None) -> pd.DataFrame:
    log(f"transfermarkt {name}")
    raw = fetch(f"{TM_BASE}/{name}.csv.gz", timeout=600)
    frame = pd.read_csv(io.BytesIO(gzip.decompress(raw)), low_memory=False)
    log(f"  {len(frame):,} rows; columns: {list(frame.columns)}")
    if usecols:
        frame = frame[[c for c in usecols if c in frame.columns]]
    return frame


def refresh_transfermarkt(out: Path) -> dict:
    target = out / "tm"
    target.mkdir(parents=True, exist_ok=True)

    games = tm_table("games")
    games = games[games["competition_id"].isin(TM_COMPETITIONS)]
    games = games[["game_id", "season"]]

    appearances = tm_table("appearances")
    appearances = appearances[appearances["competition_id"].isin(TM_COMPETITIONS)]
    appearances = appearances.merge(games, on="game_id", how="inner")
    appearances = appearances[appearances["season"] >= TM_FIRST_SEASON]
    clubs = tm_table("clubs")[["club_id", "name"]].rename(columns={"name": "club_name"})
    minutes = (appearances
               .groupby(["player_id", "season", "competition_id", "player_club_id"], as_index=False)
               .agg(apps=("game_id", "size"), minutes=("minutes_played", "sum"),
                    goals=("goals", "sum"), assists=("assists", "sum"),
                    player_name=("player_name", "last"))
               .rename(columns={"player_club_id": "club_id"})
               .merge(clubs, on="club_id", how="left"))
    minutes.to_csv(target / "tm_season_minutes.csv.gz", index=False, compression=GZIP)
    wanted = set(minutes["player_id"])
    log(f"  {len(minutes):,} player-club-seasons for {len(wanted):,} players")

    players = tm_table("players", TM_PLAYER_COLUMNS)
    players = players[players["player_id"].isin(wanted)]
    players.to_csv(target / "tm_players.csv.gz", index=False, compression=GZIP)

    values = tm_table("player_valuations")
    values = values[values["player_id"].isin(wanted)]
    values = values[["player_id", "date", "market_value_in_eur"]]
    values.to_csv(target / "tm_valuations.csv.gz", index=False, compression=GZIP)

    meta = {
        "source": "dcaribou/transfermarkt-datasets (CC0 1.0)",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "players": len(players),
        "valuations": len(values),
        "latest_valuation": str(pd.to_datetime(values["date"]).max().date()),
        "latest_appearance": str(pd.to_datetime(appearances["date"]).max().date()),
        "seasons": sorted(int(s) for s in minutes["season"].unique()),
    }
    log(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--only", choices=["understat", "transfermarkt"], default=None)
    parser.add_argument("--seasons", type=int, nargs="*", default=None,
                        help="Understat season start years, e.g. 2025 for 2025/26")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.seasons = args.seasons or default_seasons()

    meta_path = args.out / "live" / "refresh_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    failures = []
    if args.only in (None, "understat"):
        try:
            meta["understat"] = refresh_understat(args.seasons, args.out)
        except BaseException as error:  # noqa: BLE001 - report, carry on to the other source
            failures.append(f"understat: {error}")
    if args.only in (None, "transfermarkt"):
        try:
            meta["transfermarkt"] = refresh_transfermarkt(args.out)
        except BaseException as error:  # noqa: BLE001
            failures.append(f"transfermarkt: {error}")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    for failure in failures:
        log(f"FAILED {failure}")
    return 1 if len(failures) == (2 if args.only is None else 1) else 0


if __name__ == "__main__":
    sys.exit(main())

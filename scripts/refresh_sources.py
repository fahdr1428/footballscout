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


def unreachable(error: BaseException) -> bool:
    """No answer at all - a failed DNS lookup, a refused or hung connection - as
    opposed to an HTTP status, which is the host answering. An answer that could
    not be read (not JSON, say) is an answer too."""
    return isinstance(error, (urllib.error.URLError, TimeoutError, OSError)) and getattr(error, "code", None) is None


def fetch(url: str, headers: dict | None = None, retries: int = 6, timeout: int = 60,
          deadline: float | None = None) -> bytes:
    """GET with backoff for the blips a runner meets - a failed DNS lookup, a reset
    connection - over about two minutes; a 403 or 404 is an answer, not a blip.

    `timeout` bounds each wait on the socket, not the download, so a large file
    on a slow link still arrives. `deadline` (a time.monotonic() value) caps the
    whole thing: a host that hangs every connection would otherwise cost
    retries x timeout, and the job has an hour for everything.
    """
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept-Encoding": "gzip", **(headers or {})})
    for attempt in range(retries):
        wait = timeout if deadline is None else max(1.0, min(timeout, deadline - time.monotonic()))
        try:
            with urllib.request.urlopen(request, timeout=wait) as response:
                body = response.read()
            # Understat compresses its JSON whether or not it was asked to.
            if body[:2] == b"\x1f\x8b" and not url.endswith(".gz"):
                body = gzip.decompress(body)
            return body
        except (urllib.error.URLError, TimeoutError) as error:
            status = getattr(error, "code", None)
            log(f"  attempt {attempt + 1} failed for {url}: {status or error}")
            pause = min(60, PAUSE_SECONDS * (2 ** attempt))
            if status in (403, 404) or attempt == retries - 1 or (
                    deadline is not None and time.monotonic() + pause >= deadline):
                raise
            time.sleep(pause)
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


# understat.com's pages are small: if it has not answered in this long across
# every league-season, it is down, and the last good fetch stands.
UNDERSTAT_BUDGET_SECONDS = 8 * 60
UNDERSTAT_TIMEOUT = 30


# Every league-season fetched this run - its players and its matches - so the
# keeper step reads the matches without asking understat.com twice.
LEAGUE_DATA: dict[tuple[str, int], dict] = {}


def _glimpse(body: bytes, n: int = 400) -> str:
    """What an answer that was not what was asked for says, on one line, for the log:
    for a page, its visible text (an error page's message), not its markup."""
    text = body.decode("utf-8", "replace")
    if "<html" in text[:2000].lower():
        text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:n] or "(empty)"


def _page_hints(page: str) -> str:
    """What a page loads its data from, when it no longer embeds it: for the log."""
    endpoints = sorted(set(re.findall(r"(get[A-Z]\w*Data[\w/{}$.+'\"-]*)", page)))[:10]
    scripts = sorted(set(re.findall(r"<script[^>]+src=[\"']([^\"']+)", page)))[:10]
    title = re.search(r"<title>(.*?)</title>", page, re.S)
    return (f"title {title.group(1).strip()[:80]!r}; " if title else "") + f"endpoints {endpoints}; scripts {scripts}"


def understat_players(league: str, season: int,
                      deadline: float | None = None) -> tuple[list[dict], str]:
    """One league-season of player aggregates, and which route returned it."""
    page_url = f"{UNDERSTAT}/league/{league}/{season}"
    try:
        raw = fetch(f"{UNDERSTAT}/getLeagueData/{league}/{season}", headers={
            "X-Requested-With": "XMLHttpRequest", "Referer": page_url,
            "Accept": "application/json, text/javascript, */*; q=0.01"},
            retries=4, timeout=UNDERSTAT_TIMEOUT, deadline=deadline)
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            # An answer, but not JSON - a changed endpoint or a challenge page. Say
            # what it was and read the page instead, rather than treating the host as down.
            raise RuntimeError(f"getLeagueData answered {len(raw):,} bytes that are not JSON: {_glimpse(raw)}")
        players = data.get("players") if isinstance(data, dict) else None
        if players:
            LEAGUE_DATA[(league, season)] = {"players": players, "dates": data.get("dates") or []}
            return players, "getLeagueData"
        log(f"  getLeagueData returned no players (keys: {list(data)[:6] if isinstance(data, dict) else type(data)})")
    except Exception as error:  # noqa: BLE001 - fall through to the page
        if unreachable(error):
            # The page is on the same host: asking it a second way only doubles the wait.
            raise
        log(f"  getLeagueData unavailable: {error}")
    time.sleep(PAUSE_SECONDS)
    page = fetch(page_url, retries=4, timeout=UNDERSTAT_TIMEOUT,
                 deadline=deadline).decode("utf-8", "replace")
    players = _decode_embedded(page, "playersData")
    if players:
        LEAGUE_DATA[(league, season)] = {"players": players, "dates": _decode_embedded(page, "datesData") or []}
        return players, "embedded JSON"
    log(f"  page fetched ({len(page):,} chars) but no player data found in it: {_page_hints(page)}; "
        f"it says: {_glimpse(page.encode())}")
    return [], "none"


def refresh_understat(seasons: list[int], out: Path) -> dict:
    # The day, not the second: a re-run the same day writes identical rows.
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    deadline = time.monotonic() + UNDERSTAT_BUDGET_SECONDS
    frames, summary = [], {}
    for season in seasons:
        for league in UNDERSTAT_LEAGUES:
            log(f"understat {league} {season}")
            players, route = understat_players(league, season, deadline)
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
# Understat goalkeepers
# --------------------------------------------------------------------------
# Understat's season aggregate records nothing a keeper does. Its match list
# does: every league match with both sides' xG and goals. A keeper's own match
# list says which matches he played and for how long, so for every match he
# played in full, the xG of the shots his team faced and the goals it let in
# are his - exactly, with nothing apportioned. A match he came on in or went
# off in is left out rather than split by a guess, and the file says how many
# full matches each figure rests on.
KEEPERS_CSV = "understat_keepers.csv"
# Finished seasons whose keepers have all been fetched: not asked for again.
KEEPERS_STATE = "understat_keepers_meta.json"
KEEPER_FIRST_SEASON = 2014
KEEPER_COLUMNS = ["player_id", "season", "gk_full_matches", "gk_full_xga", "gk_full_goals_against",
                  "gk_full_clean_sheets"]
KEEPER_BUDGET_MINUTES = 15


def _season_label(start: int) -> str:
    return f"{start}-{str(start + 1)[-2:]}"


def match_table(dates: list[dict]) -> dict[str, dict]:
    """Finished league matches by id: each side's club, xG and goals."""
    out = {}
    for m in dates:
        if not m.get("isResult") or "h" not in m or "a" not in m:
            continue
        try:
            out[str(m["id"])] = {
                "h": m["h"]["title"], "a": m["a"]["title"],
                "xg": {k: float(m["xG"][k]) for k in ("h", "a")},
                "goals": {k: int(m["goals"][k]) for k in ("h", "a")},
            }
        except (KeyError, TypeError, ValueError):
            continue
    return out


def keeper_rows(player_id: str, matches: list[dict], clubs: dict[str, set], table: dict[str, dict]) -> list[dict]:
    """One keeper's full matches in goal, summed season by season.

    `clubs` maps each season label to the clubs he played for that season (from
    the league's player list), which says which side of each match was his.
    """
    # A row for every season he is listed in, so one with no full match reads 0, not "not fetched".
    seasons: dict[str, dict] = {season: {"player_id": str(player_id), "season": season, "gk_full_matches": 0,
                                         "gk_full_xga": 0.0, "gk_full_goals_against": 0, "gk_full_clean_sheets": 0}
                                for season in sorted(clubs)}
    for m in matches:
        try:
            season = _season_label(int(m["season"]))
        except (KeyError, TypeError, ValueError):
            continue
        if season not in clubs:
            continue
        row = seasons[season]
        game = table.get(str(m.get("id")))
        if game is None or str(m.get("position", "")).strip() != "GK":
            continue
        try:
            if int(float(m.get("time", 0))) < 90:
                continue
        except (TypeError, ValueError):
            continue
        mine = {side for side in ("h", "a") if game[side] in clubs[season]}
        if len(mine) != 1:
            continue   # his club cannot be told apart from the opponent's name: left out
        other = "a" if mine == {"h"} else "h"
        row["gk_full_matches"] += 1
        row["gk_full_xga"] += game["xg"][other]
        row["gk_full_goals_against"] += game["goals"][other]
        row["gk_full_clean_sheets"] += int(game["goals"][other] == 0)
    for row in seasons.values():
        row["gk_full_xga"] = round(row["gk_full_xga"], 3)
    return list(seasons.values())


def understat_player_matches(player_id: str, deadline: float | None = None) -> list[dict]:
    """A player's league matches, from the JSON endpoint or, failing that, his page."""
    page_url = f"{UNDERSTAT}/player/{player_id}"
    try:
        raw = fetch(f"{UNDERSTAT}/getPlayerData/{player_id}", headers={
            "X-Requested-With": "XMLHttpRequest", "Referer": page_url,
            "Accept": "application/json, text/javascript, */*; q=0.01"},
            retries=3, timeout=UNDERSTAT_TIMEOUT, deadline=deadline)
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise RuntimeError(f"getPlayerData answered {len(raw):,} bytes that are not JSON: {_glimpse(raw)}")
        if isinstance(data, dict) and isinstance(data.get("matches"), list):
            return data["matches"]
    except Exception as error:  # noqa: BLE001 - fall through to the page
        if unreachable(error):
            raise
        log(f"  getPlayerData unavailable for {player_id}: {error}")
    time.sleep(PAUSE_SECONDS)
    page = fetch(page_url, retries=3, timeout=UNDERSTAT_TIMEOUT, deadline=deadline).decode("utf-8", "replace")
    return _decode_embedded(page, "matchesData") or []


def refresh_understat_keepers(seasons: list[int], out: Path, current: int,
                              budget_minutes: float = KEEPER_BUDGET_MINUTES) -> dict:
    """Every keeper's full matches for these seasons, added to a running file.

    Finished seasons are fetched once and kept; the season being played is
    fetched again every run. A run stops at its time budget and the next one
    carries on, so a backfill of past seasons can span several runs.
    """
    target = out / "live" / KEEPERS_CSV
    state_path = out / "live" / KEEPERS_STATE
    have = pd.read_csv(target, dtype={"player_id": str}) if target.exists() else pd.DataFrame(columns=KEEPER_COLUMNS)
    done = set(zip(have["player_id"].astype(str), have["season"]))
    complete = set(json.loads(state_path.read_text()).get("complete", [])) if state_path.exists() else set()
    now = _season_label(current)
    # A finished season whose keepers are all in already is not fetched again.
    seasons = [s for s in seasons if _season_label(s) == now or _season_label(s) not in complete]
    deadline = time.monotonic() + budget_minutes * 60
    table: dict[str, dict] = {}
    clubs: dict[str, dict[str, set]] = {}
    for season in seasons:
        for league in UNDERSTAT_LEAGUES:
            if (league, season) not in LEAGUE_DATA:
                log(f"understat {league} {season} (for its keepers)")
                try:
                    understat_players(league, season, deadline)
                except Exception as error:  # noqa: BLE001
                    log(f"  skipped: {error}")
                time.sleep(PAUSE_SECONDS)
            data = LEAGUE_DATA.get((league, season))
            if not data:
                continue
            table.update(match_table(data["dates"]))
            for p in data["players"]:
                if "GK" in str(p.get("position", "")).split():
                    titles = {t.strip() for t in str(p.get("team_title", "")).split(",") if t.strip()}
                    clubs.setdefault(str(p["id"]), {}).setdefault(_season_label(season), set()).update(titles)
    # The season being played first, then the most recent finished ones.
    todo = sorted((pid for pid, by in clubs.items() if any(s == now or (pid, s) not in done for s in by)),
                  key=lambda pid: (now not in clubs[pid], tuple(-int(s[:4]) for s in sorted(clubs[pid], reverse=True)), pid))
    log(f"keepers: {len(clubs)} in these seasons, {len(todo)} to fetch")
    rows, fetched = [], 0
    for pid in todo:
        if time.monotonic() >= deadline:
            log(f"  time budget spent after {fetched} keepers; the next run carries on")
            break
        try:
            matches = understat_player_matches(pid, deadline)
        except Exception as error:  # noqa: BLE001
            log(f"  keeper {pid}: {error}")
            continue
        rows += keeper_rows(pid, matches, clubs[pid], table)
        fetched += 1
        time.sleep(PAUSE_SECONDS)
    new = pd.DataFrame(rows, columns=KEEPER_COLUMNS)
    if len(new):
        key = set(zip(new["player_id"], new["season"]))
        have = have[[k not in key for k in zip(have["player_id"].astype(str), have["season"])]]
    table_out = (pd.concat([have, new], ignore_index=True)
                 .sort_values(["season", "player_id"], kind="stable").reset_index(drop=True))
    target.parent.mkdir(parents=True, exist_ok=True)
    table_out.to_csv(target, index=False)
    # A finished season is complete once every keeper it lists has a row.
    have_now = set(zip(table_out["player_id"].astype(str), table_out["season"]))
    for label in {s for by in clubs.values() for s in by} - {now}:
        if all((pid, label) in have_now for pid, by in clubs.items() if label in by):
            complete.add(label)
    state_path.write_text(json.dumps({"complete": sorted(complete)}, indent=1) + "\n")
    log(f"wrote {target.relative_to(ROOT)}: {len(table_out):,} keeper-seasons ({len(new):,} from this run)")
    return {"keepers_fetched": fetched, "keeper_seasons": len(table_out), "to_fetch": len(todo)}


# --------------------------------------------------------------------------
# Transfermarkt (dcaribou/transfermarkt-datasets, CC0)
# --------------------------------------------------------------------------

def tm_table(name: str, usecols=None) -> pd.DataFrame:
    log(f"transfermarkt {name}")
    # 120 seconds of silence mid-download is a stalled connection, not a slow one.
    raw = fetch(f"{TM_BASE}/{name}.csv.gz", timeout=120)
    frame = pd.read_csv(io.BytesIO(gzip.decompress(raw)), low_memory=False)
    log(f"  {len(frame):,} rows; columns: {list(frame.columns)}")
    if usecols:
        frame = frame[[c for c in usecols if c in frame.columns]]
    return frame


def season_positions(games: pd.DataFrame, wanted: set, target: Path) -> int:
    """Where each player started most league games, season by season.

    A Transfermarkt profile only carries today's position; the match line-ups
    record the one he started in every game. The position he started most often
    in a season is what that season is labelled with - Ousmane Dembélé a right
    winger at Barcelona, not today's centre-forward. Written to
    tm_season_positions.csv.gz; if the line-ups cannot be read the refresh
    carries on and the build falls back to squad pages and profiles.
    """
    try:
        lineups = tm_table("game_lineups")
        needed = {"game_id", "player_id", "position", "type"}
        if not needed <= set(lineups.columns):
            log(f"  game_lineups lacks {sorted(needed - set(lineups.columns))}; skipped")
            return 0
        lineups = lineups.merge(games, on="game_id", how="inner")
        lineups = lineups[(lineups["season"] >= TM_FIRST_SEASON) & lineups["player_id"].isin(wanted)
                          & lineups["type"].astype(str).str.contains("start", case=False)]
        counts = (lineups.groupby(["player_id", "season", "position"]).size()
                         .rename("starts").reset_index())
        total = counts.groupby(["player_id", "season"])["starts"].transform("sum")
        counts["share"] = (counts["starts"] / total).round(3)
        best = (counts.sort_values(["player_id", "season", "starts", "position"],
                                   ascending=[True, True, False, True])
                      .drop_duplicates(["player_id", "season"]))
        best.to_csv(target / "tm_season_positions.csv.gz", index=False, compression=GZIP)
        log(f"  {len(best):,} player-seasons with a most-started position")
        return len(best)
    except Exception as error:  # noqa: BLE001 - one optional table must not stop the refresh
        log(f"  game_lineups unavailable ({error}); positions fall back to squad pages and profiles")
        return 0


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

    positions = season_positions(games, wanted, target)

    meta = {
        "source": "dcaribou/transfermarkt-datasets (CC0 1.0)",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "players": len(players),
        "valuations": len(values),
        "season_positions": positions,
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
    parser.add_argument("--keeper-seasons", type=int, nargs="*", default=None,
                        help="season start years to fetch keepers' matches for (default: --seasons); "
                             "past seasons are fetched once and kept")
    parser.add_argument("--keeper-budget", type=float, default=KEEPER_BUDGET_MINUTES,
                        help="minutes this run may spend on keepers before leaving the rest to the next")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.seasons = args.seasons or default_seasons()

    meta_path = args.out / "live" / "refresh_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    failures = []
    understat_failed = None
    if args.only in (None, "understat"):
        try:
            meta["understat"] = refresh_understat(args.seasons, args.out)
        except BaseException as error:  # noqa: BLE001 - report, carry on to the other source
            understat_failed = error
    if args.only in (None, "transfermarkt"):
        try:
            meta["transfermarkt"] = refresh_transfermarkt(args.out)
        except BaseException as error:  # noqa: BLE001
            failures.append(f"transfermarkt: {error}")
    if understat_failed is not None:
        # Once more after the Transfermarkt download: a runner's network blip has
        # usually cleared in the minutes that takes.
        log(f"understat failed ({understat_failed}); trying once more")
        try:
            meta["understat"] = refresh_understat(args.seasons, args.out)
            understat_failed = None
        except BaseException as error:  # noqa: BLE001
            failures.append(f"understat: {error}")
    if args.only in (None, "understat") and understat_failed is None:
        # Keepers are extra: a failure here is logged and the rest of the refresh stands.
        try:
            # Every season since 2014/15 by default: finished ones are fetched once,
            # a little each run within the budget, then left alone.
            wanted = args.keeper_seasons or range(KEEPER_FIRST_SEASON, max(args.seasons) + 1)
            meta["understat_keepers"] = refresh_understat_keepers(
                sorted(set(wanted) | set(args.seasons)), args.out, max(args.seasons), args.keeper_budget)
        except BaseException as error:  # noqa: BLE001
            log(f"keepers not refreshed: {error}")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    for failure in failures:
        log(f"FAILED {failure}")
    return 1 if len(failures) == (2 if args.only is None else 1) else 0


if __name__ == "__main__":
    sys.exit(main())

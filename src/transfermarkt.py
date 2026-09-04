"""
Transfermarkt ingestion, through the open-source `transfermarkt-api` service.

    https://github.com/felipeall/transfermarkt-api

That project wraps Transfermarkt in a small FastAPI service. This module is the
client for it: it walks competitions -> clubs -> players, and optionally each
player's season stats, and emits rows in this platform's schema.

WHAT IT ADDS THAT NOTHING ELSE HERE HAS
---------------------------------------
* **The top leagues in one pool.** Six competitions by default (the big five
  plus Liga Portugal), any season, so players can be compared across leagues.
* **Real market values**, in euros - the first genuine valuation in this
  project. Everything else is a proxy.
* **True positions.** Transfermarkt records "Centre-Back", "Left Winger",
  "Defensive Midfield" and so on, which map straight onto this platform's
  detailed position groups - unlike a fantasy feed's four buckets.
* **Age, height, preferred foot, nationality, contract expiry.**

WHAT IT DOES NOT HAVE
---------------------
Transfermarkt is a market and biographical database, not a performance one. Its
per-season stats are appearances, goals, assists, cards and minutes. There is no
xG, no passing, no defending. So this source is strong for market and squad
questions and thin for style: used alone, its similarity model has few features
to work with, and the app says so.

Its best use is the second one below - **enriching** a performance dataset with
market value and a real position.

NETWORK
-------
Transfermarkt is not reachable from every environment. Run the service
yourself, which is a single command:

    docker run -d -p 8000:8000 --name transfermarkt-api \\
        $(docker build -q https://github.com/felipeall/transfermarkt-api.git)

then point this module at `http://localhost:8000`. The maintainer's hosted
instance at https://transfermarkt-api.fly.dev works too, but it is rate limited
to roughly two requests every three seconds, so pass `--rate 0.6`.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

PROJECT_URL = "https://github.com/felipeall/transfermarkt-api"
ATTRIBUTION = (
    "Squad, market value and biographical data from Transfermarkt, read through the "
    f"open-source transfermarkt-api service ({PROJECT_URL})."
)
DEFAULT_BASE_URL = "http://localhost:8000"


@dataclass(frozen=True)
class TransfermarktCompetition:
    competition_id: str
    league: str
    country: str
    tier: int
    strength: float


# Transfermarkt competition codes. The default six are the big five plus Liga
# Portugal; the extras are there so `--competitions` can widen the pool.
TOP_COMPETITIONS: list[TransfermarktCompetition] = [
    TransfermarktCompetition("GB1", "Premier League", "England", 1, 1.00),
    TransfermarktCompetition("ES1", "LaLiga", "Spain", 1, 0.97),
    TransfermarktCompetition("IT1", "Serie A", "Italy", 1, 0.95),
    TransfermarktCompetition("L1", "Bundesliga", "Germany", 1, 0.95),
    TransfermarktCompetition("FR1", "Ligue 1", "France", 1, 0.90),
    TransfermarktCompetition("PO1", "Liga Portugal", "Portugal", 2, 0.80),
]

EXTRA_COMPETITIONS: list[TransfermarktCompetition] = [
    TransfermarktCompetition("NL1", "Eredivisie", "Netherlands", 2, 0.80),
    TransfermarktCompetition("BE1", "Jupiler Pro League", "Belgium", 2, 0.76),
    TransfermarktCompetition("TR1", "Super Lig", "Turkey", 2, 0.74),
    TransfermarktCompetition("GB2", "Championship", "England", 2, 0.74),
    TransfermarktCompetition("SC1", "Scottish Premiership", "Scotland", 3, 0.66),
    TransfermarktCompetition("A1", "Austrian Bundesliga", "Austria", 3, 0.66),
    TransfermarktCompetition("C1", "Swiss Super League", "Switzerland", 3, 0.64),
    TransfermarktCompetition("DK1", "Danish Superliga", "Denmark", 3, 0.65),
]

COMPETITIONS_BY_ID = {c.competition_id: c for c in TOP_COMPETITIONS + EXTRA_COMPETITIONS}

# Transfermarkt's own position vocabulary -> (position group, detailed position).
POSITION_MAP: dict[str, tuple[str, str]] = {
    "Goalkeeper": ("GK", "GK"),
    "Centre-Back": ("CB", "CB"),
    "Left-Back": ("FB", "LB"),
    "Right-Back": ("FB", "RB"),
    "Defensive Midfield": ("DM", "DM"),
    "Central Midfield": ("CM", "CM"),
    "Left Midfield": ("W", "LM"),
    "Right Midfield": ("W", "RM"),
    "Attacking Midfield": ("AM", "AM"),
    "Left Winger": ("W", "LW"),
    "Right Winger": ("W", "RW"),
    "Second Striker": ("AM", "SS"),
    "Centre-Forward": ("FW", "CF"),
}


def map_position(position: str | None) -> tuple[str, str]:
    """Position group and detailed position for a Transfermarkt position string."""
    if not position:
        return "CM", "CM"
    return POSITION_MAP.get(position.strip(), ("CM", "CM"))


def season_label(season_id: str | int) -> str:
    """Transfermarkt season ids are the starting year: 2025 -> '2025-26'."""
    year = int(season_id)
    return f"{year}-{str(year + 1)[-2:]}"


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

class TransfermarktClient:
    """Minimal client for a transfermarkt-api instance.

    Deliberately dependency-free (urllib), with retries and a rate limiter -
    the maintainer's hosted instance allows about two requests every three
    seconds, and a scraper behind an API still deserves to be treated gently.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        rate: float = 4.0,
        timeout: int = 30,
        retries: int = 3,
        backoff: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval = 1.0 / rate if rate > 0 else 0.0
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def _wait_turn(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.min_interval
        if wait:
            time.sleep(wait)

    def get(self, path: str, **params) -> dict:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        last: Exception | None = None
        for attempt in range(self.retries):
            self._wait_turn()
            try:
                request = urllib.request.Request(
                    url, headers={"Accept": "application/json", "User-Agent": "scouting-platform/1.0"}
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return {}
                last = error
                time.sleep(self.backoff * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
                last = error
                time.sleep(self.backoff * (attempt + 1))
        raise RuntimeError(f"transfermarkt-api request failed: {url}: {last}")

    # -- endpoints -----------------------------------------------------
    def competition_clubs(self, competition_id: str, season_id: str | int | None = None) -> list[dict]:
        payload = self.get(f"/competitions/{competition_id}/clubs", season_id=season_id)
        return payload.get("clubs", [])

    def club_players(self, club_id: str, season_id: str | int | None = None) -> list[dict]:
        payload = self.get(f"/clubs/{club_id}/players", season_id=season_id)
        return payload.get("players", [])

    def player_stats(self, player_id: str) -> list[dict]:
        payload = self.get(f"/players/{player_id}/stats")
        return payload.get("stats", [])


# --------------------------------------------------------------------------
# Squad spine: who plays where, what they are worth
# --------------------------------------------------------------------------

def _to_number(value) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def build_squads(
    client: TransfermarktClient,
    competitions: list[TransfermarktCompetition],
    season_id: str | int,
    workers: int = 4,
    progress=print,
) -> pd.DataFrame:
    """Every player in every club of every competition, with their attributes."""
    import concurrent.futures

    season = season_label(season_id)
    rows: list[dict] = []

    for competition in competitions:
        clubs = client.competition_clubs(competition.competition_id, season_id)
        progress(f"{competition.league} {season}: {len(clubs)} clubs")
        if not clubs:
            continue

        def squad(club: dict) -> list[dict]:
            return [(club, player) for player in client.club_players(club["id"], season_id)]

        with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
            for squad_rows in pool.map(squad, clubs):
                for club, player in squad_rows:
                    group, detailed = map_position(player.get("position"))
                    nationality = player.get("nationality") or []
                    rows.append(
                        {
                            "player_id": f"TM{player['id']}",
                            "transfermarkt_id": player["id"],
                            "player": player.get("name"),
                            "position_group": group,
                            "position": detailed,
                            "transfermarkt_position": player.get("position"),
                            "team": club.get("name"),
                            "club_id": club.get("id"),
                            "league": competition.league,
                            "country": competition.country,
                            "league_tier": competition.tier,
                            "league_strength": competition.strength,
                            "competition_id": competition.competition_id,
                            "season": season,
                            "age": _to_number(player.get("age")),
                            "date_of_birth": player.get("dateOfBirth") or player.get("date_of_birth"),
                            "height_cm": _to_number(player.get("height")),
                            "foot": player.get("foot"),
                            "nationality": nationality[0] if nationality else None,
                            "market_value_eur": _to_number(player.get("marketValue") or player.get("market_value")),
                            "contract_expires": player.get("contract"),
                            "joined_on": player.get("joinedOn") or player.get("joined_on"),
                            "signed_from": player.get("signedFrom") or player.get("signed_from"),
                        }
                    )
        progress(f"  {len(rows):,} players so far")

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Height arrives in millimetres on some Transfermarkt pages.
    tall = frame["height_cm"] > 250
    frame.loc[tall, "height_cm"] = frame.loc[tall, "height_cm"] / 10
    frame["market_value_m"] = (frame["market_value_eur"] / 1_000_000).round(2)
    frame["gender"] = "male"
    return frame


def attach_stats(
    client: TransfermarktClient,
    squads: pd.DataFrame,
    season_id: str | int,
    workers: int = 4,
    progress=print,
) -> pd.DataFrame:
    """Add the season's appearances, goals, assists, cards and minutes.

    One request per player, so this is the slow half. `build_dataset` can skip
    it when only the squad spine is wanted.
    """
    import concurrent.futures

    if squads.empty:
        return squads
    season = str(season_id)
    wanted = dict(zip(squads["transfermarkt_id"], squads["competition_id"]))

    def fetch(player_id: str) -> tuple[str, dict]:
        totals = {"appearances": 0, "goals": 0, "assists": 0,
                  "yellow_cards": 0, "red_cards": 0, "minutes": 0}
        for stat in client.player_stats(player_id):
            if str(stat.get("seasonId") or stat.get("season_id")) != season:
                continue
            if str(stat.get("competitionId") or stat.get("competition_id")) != wanted[player_id]:
                continue
            totals["appearances"] += int(stat.get("appearances") or 0)
            totals["goals"] += int(stat.get("goals") or 0)
            totals["assists"] += int(stat.get("assists") or 0)
            totals["yellow_cards"] += int(stat.get("yellowCards") or stat.get("yellow_cards") or 0)
            totals["red_cards"] += int(stat.get("redCards") or stat.get("red_cards") or 0)
            totals["minutes"] += int(stat.get("minutesPlayed") or stat.get("minutes_played") or 0)
        return player_id, totals

    collected: dict[str, dict] = {}
    ids = list(squads["transfermarkt_id"])
    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        for i, (player_id, totals) in enumerate(pool.map(fetch, ids), 1):
            collected[player_id] = totals
            if i % 200 == 0:
                progress(f"  stats {i:,}/{len(ids):,}")

    stats = pd.DataFrame.from_dict(collected, orient="index")
    stats.index.name = "transfermarkt_id"
    return squads.merge(stats.reset_index(), on="transfermarkt_id", how="left")


def build_dataset(
    client: TransfermarktClient,
    competitions: list[TransfermarktCompetition] | None = None,
    season_id: str | int = 2025,
    with_stats: bool = True,
    workers: int = 4,
    progress=print,
) -> pd.DataFrame:
    """Squads, attributes and (optionally) season stats, in the platform schema."""
    competitions = competitions or TOP_COMPETITIONS
    squads = build_squads(client, competitions, season_id, workers=workers, progress=progress)
    if squads.empty:
        return squads
    if with_stats:
        squads = attach_stats(client, squads, season_id, workers=workers, progress=progress)

    frame = squads.copy()
    for column in ["appearances", "goals", "assists", "yellow_cards", "red_cards", "minutes"]:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["matches"] = frame["appearances"]
    frame["starts"] = np.nan          # Transfermarkt does not separate starts here
    frame["team_possession"] = np.nan
    frame = frame.rename(columns={"minutes": "minutes"})
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce")
    return frame[frame["minutes"].fillna(0) > 0].reset_index(drop=True)


# --------------------------------------------------------------------------
# Enrichment: the highest-value use
# --------------------------------------------------------------------------

def enrich(
    features: pd.DataFrame,
    market: pd.DataFrame,
    use_positions: bool = True,
    normaliser=None,
) -> tuple[pd.DataFrame, dict]:
    """Attach market value, height, foot, nationality and a true position.

    A performance feed with four positional buckets and no valuation becomes a
    great deal more useful with Transfermarkt's position and price joined onto
    it. The join is on normalised name plus season; anything unmatched is left
    alone rather than guessed at, and the returned report says how much matched.
    """
    from .premier_league import normalise_name

    normaliser = normaliser or normalise_name
    if features.empty or market.empty:
        return features, {"matched": 0, "of": len(features)}

    left = features.copy()
    left["_key"] = left["player"].map(normaliser)
    right = market.copy()
    right["_key"] = right["player"].map(normaliser)

    columns = ["market_value_eur", "market_value_m", "height_cm", "foot", "nationality",
               "contract_expires", "transfermarkt_position", "transfermarkt_id"]
    if use_positions:
        columns += ["position_group", "position"]
    columns = [c for c in columns if c in right.columns]

    keys = ["_key", "season"] if "season" in right.columns and "season" in left.columns else ["_key"]
    right = right.drop_duplicates(subset=keys)[keys + columns]
    merged = left.merge(right, on=keys, how="left", suffixes=("", "_tm"))

    matched = merged["transfermarkt_id"].notna()
    report = {"matched": int(matched.sum()), "of": len(merged)}

    for column in columns:
        target = f"{column}_tm" if f"{column}_tm" in merged.columns else column
        if target != column and column in merged.columns:
            # Prefer the Transfermarkt value where we have one, keep ours otherwise.
            merged[column] = merged[target].where(merged[target].notna(), merged[column])
            merged = merged.drop(columns=[target])

    if use_positions and "position_group" in columns:
        merged["position_source"] = np.where(
            matched, "Transfermarkt position", merged.get("position_source", "source default")
        )
    return merged.drop(columns=["_key"]), report

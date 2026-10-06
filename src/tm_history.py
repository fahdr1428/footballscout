"""
Transfermarkt valuation history and profiles, from the open builds on hand.

Two public builds of Transfermarkt are read and merged:

* **dcaribou/transfermarkt-datasets** (CC0 1.0) - valuations to June 2026 at
  the time of writing. Its host is not reachable from every machine, so
  `scripts/refresh_sources.py` fetches a compact extract on GitHub's runners
  and commits it under `data/raw/tm/`; it is used when that extract is there.
* **salimt/football-datasets** - valuations to September 2025, plain CSV on
  GitHub, downloaded on first use and cached.

Where both carry a valuation for the same player on the same day, dcaribou's
is kept. Every source reads values the same way - the most recent valuation on
or before a stated date, never one from after it - and a valuation more than
eighteen months older than that date is not used at all: by then the player
has usually left the leagues Transfermarkt revalues, and the number describes
someone who no longer exists.
"""

from __future__ import annotations

import http.client
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR

DCARIBOU_DIR = RAW_DIR / "tm"
DCARIBOU_ATTRIBUTION = (
    "Transfermarkt market values and profiles from dcaribou/transfermarkt-datasets "
    "(https://github.com/dcaribou/transfermarkt-datasets, CC0 1.0)"
)
SALIMT_BASE = ("https://raw.githubusercontent.com/salimt/football-datasets/"
               "main/datalake/transfermarkt")
SALIMT_VALUES = f"{SALIMT_BASE}/player_market_value/player_market_value.csv"
SALIMT_PROFILES = f"{SALIMT_BASE}/player_profiles/player_profiles.csv"

MAX_VALUE_AGE = pd.Timedelta(days=548)


_NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError,
                   http.client.IncompleteRead)


def _download(url: str, cache: Path, name: str, retries: int = 4) -> Path:
    """Fetch once into the cache; a cut-off transfer is retried, never kept."""
    target = cache / name
    if target.exists() and target.stat().st_size > 0:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                partial.write_bytes(response.read())
            partial.replace(target)
            return target
        except _NETWORK_ERRORS:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (2 ** attempt))
    return target


def tm_id_from_url(urls: pd.Series) -> pd.Series:
    """Transfermarkt's player id from a profile URL (".../spieler/418560")."""
    return pd.to_numeric(urls.astype(str).str.extract(r"/spieler/(\d+)")[0], errors="coerce")


def load_valuations(cache: Path, progress=print) -> pd.DataFrame:
    """Every valuation on hand: tm_id, date, value, source."""
    frames = []
    extract = DCARIBOU_DIR / "tm_valuations.csv.gz"
    if extract.exists():
        d = pd.read_csv(extract)
        frames.append(pd.DataFrame({
            "tm_id": pd.to_numeric(d["player_id"], errors="coerce"),
            "date": pd.to_datetime(d["date"], errors="coerce"),
            "value": pd.to_numeric(d["market_value_in_eur"], errors="coerce"),
            "source": "dcaribou",
        }))
    try:
        s = pd.read_csv(_download(SALIMT_VALUES, cache, "tm_market_values.csv"))
        frames.append(pd.DataFrame({
            "tm_id": pd.to_numeric(s["player_id"], errors="coerce"),
            "date": pd.to_datetime(s["date_unix"], errors="coerce"),
            "value": pd.to_numeric(s["value"], errors="coerce"),
            "source": "salimt",
        }))
    except _NETWORK_ERRORS as error:
        progress(f"  salimt valuations unavailable: {error}")
    if not frames:
        return pd.DataFrame(columns=["tm_id", "date", "value", "source"])
    history = pd.concat(frames, ignore_index=True).dropna(subset=["tm_id", "date", "value"])
    history = history[history["value"] > 0]
    history["tm_id"] = history["tm_id"].astype("int64")
    # concat put dcaribou first, so keep="first" prefers it on a shared day.
    history = history.drop_duplicates(["tm_id", "date"], keep="first")
    progress(f"  {len(history):,} valuations for {history['tm_id'].nunique():,} players, "
             f"latest {history['date'].max().date()}")
    return history.sort_values("date").reset_index(drop=True)


def value_asof(tm_ids: pd.Series, asof: pd.Series, history: pd.DataFrame) -> pd.Series:
    """The valuation in force on each date, or NaN if none recent enough."""
    left = pd.DataFrame({"tm_id": pd.to_numeric(tm_ids, errors="coerce"),
                         "asof": pd.to_datetime(asof)}, index=tm_ids.index)
    left = left.dropna().reset_index().sort_values("asof")
    if left.empty or history.empty:
        return pd.Series(np.nan, index=tm_ids.index)
    left["tm_id"] = left["tm_id"].astype("int64")
    right = history[["tm_id", "date", "value"]].rename(columns={"date": "asof"})
    right = right.astype({"tm_id": "int64"}).sort_values("asof")
    merged = pd.merge_asof(
        left, right,
        on="asof", by="tm_id", direction="backward", tolerance=MAX_VALUE_AGE,
    )
    out = pd.Series(np.nan, index=tm_ids.index)
    out.loc[merged["index"]] = merged["value"].to_numpy()
    return out


def latest_values(history: pd.DataFrame) -> pd.DataFrame:
    """Each player's most recent valuation and its date."""
    last = history.sort_values("date").drop_duplicates("tm_id", keep="last")
    return pd.DataFrame({"tm_id": last["tm_id"].to_numpy(),
                         "latest_value_eur": last["value"].to_numpy(),
                         "latest_value_date": last["date"].dt.strftime("%Y-%m-%d").to_numpy()})


def load_profiles(cache: Path, progress=print) -> pd.DataFrame:
    """tm_id, name, date_of_birth, height_cm, foot, nationality - one row a player."""
    frames = []
    extract = DCARIBOU_DIR / "tm_players.csv.gz"
    if extract.exists():
        d = pd.read_csv(extract, low_memory=False)
        frames.append(pd.DataFrame({
            "tm_id": pd.to_numeric(d["player_id"], errors="coerce"),
            "tm_name": d.get("name"),
            "date_of_birth": pd.to_datetime(d.get("date_of_birth"), errors="coerce"),
            "height_cm": pd.to_numeric(d.get("height_in_cm"), errors="coerce"),
            "foot": d.get("foot"),
            "nationality": d.get("country_of_citizenship"),
        }))
    try:
        s = pd.read_csv(_download(SALIMT_PROFILES, cache, "tm_player_profiles.csv"),
                        low_memory=False, usecols=["player_id", "player_name", "date_of_birth",
                                                   "height", "foot", "citizenship"])
        height = pd.to_numeric(s["height"].astype(str).str.replace(",", ".")
                               .str.extract(r"([\d.]+)")[0], errors="coerce")
        frames.append(pd.DataFrame({
            "tm_id": pd.to_numeric(s["player_id"], errors="coerce"),
            "tm_name": s["player_name"].astype(str).str.replace(r"\s*\(\d+\)\s*$", "", regex=True),
            "date_of_birth": pd.to_datetime(s["date_of_birth"], errors="coerce", format="mixed"),
            "height_cm": np.where(height < 3, height * 100, height).round(),
            "foot": s["foot"],
            "nationality": s["citizenship"].astype(str).str.split(r"\s{2,}", regex=True).str[0],
        }))
    except _NETWORK_ERRORS as error:
        progress(f"  salimt profiles unavailable: {error}")
    if not frames:
        return pd.DataFrame(columns=["tm_id", "tm_name", "date_of_birth", "height_cm",
                                     "foot", "nationality"])
    profiles = pd.concat(frames, ignore_index=True).dropna(subset=["tm_id"])
    profiles["tm_id"] = profiles["tm_id"].astype("int64")
    profiles["foot"] = profiles["foot"].astype(str).str.lower().replace(
        {"nan": np.nan, "none": np.nan, "": np.nan})
    profiles["nationality"] = profiles["nationality"].replace({"nan": np.nan, "": np.nan})
    # A field one build lacks is taken from the other, never invented.
    profiles = profiles.groupby("tm_id", sort=False).first().reset_index()
    return profiles


# --------------------------------------------------------------------------
# Matching a source's players to Transfermarkt
# --------------------------------------------------------------------------

def normalise_name(value) -> str:
    """The platform's one name normaliser (premier_league.normalise_name)."""
    from .premier_league import normalise_name as canonical

    return canonical(value)


# Transfermarkt's ids for the six leagues, as in scripts/refresh_sources.py.
TM_COMPETITIONS = {"GB1": "Premier League", "ES1": "La Liga", "IT1": "Serie A",
                   "L1": "Bundesliga", "FR1": "Ligue 1", "RU1": "Russian Premier League"}
_CLUB_STOPWORDS = {"fc", "cf", "ac", "as", "sc", "club", "de", "afc", "ssc", "us", "sv", "vfl",
                   "vfb", "rc", "ud", "cd", "rcd", "ogc", "fk", "tsg", "and", "calcio"}


def _club_tokens(name) -> set[str]:
    return {t for t in normalise_name(name).split() if t not in _CLUB_STOPWORDS and len(t) > 1}


def _initial_and_surname(key: str) -> str:
    parts = key.split()
    return f"{parts[0][0]} {parts[-1]}" if len(parts) > 1 else key


def link_by_season(frame: pd.DataFrame, tm: pd.DataFrame | None = None) -> pd.Series:
    """Understat player_id -> Transfermarkt id, from same league-season rows."""
    if tm is None:
        path = DCARIBOU_DIR / "tm_season_minutes.csv.gz"
        if not path.exists():
            return pd.Series(dtype="int64", name="tm_id", index=pd.Index([], name="player_id"))
        tm = pd.read_csv(path)
    tm = tm.copy()
    tm["league"] = tm["competition_id"].map(TM_COMPETITIONS)
    tm = tm.dropna(subset=["league", "player_name"])
    tm["key"] = tm["player_name"].map(normalise_name)
    tm["start"] = tm["season"].astype(int)
    tm["short"] = tm["key"].map(_initial_and_surname)
    tm = tm.rename(columns={"player_id": "tm_player", "minutes": "tm_minutes"})[
        ["tm_player", "start", "league", "key", "short", "club_name", "tm_minutes"]]

    us = frame[["player_id", "season", "league", "team", "minutes", "key"]].copy()
    us["start"] = us["season"].str.slice(0, 4).astype(int)
    us["short"] = us["key"].map(_initial_and_surname)
    candidates = pd.concat([
        us.merge(tm.drop(columns="short"), on=["start", "league", "key"]).assign(exact=True),
        us.merge(tm.drop(columns="key"), on=["start", "league", "short"]).assign(exact=False),
    ], ignore_index=True)
    if candidates.empty:
        return pd.Series(dtype="int64", name="tm_id", index=pd.Index([], name="player_id"))
    candidates["same_club"] = [bool(_club_tokens(a) & _club_tokens(b))
                               for a, b in zip(candidates["team"], candidates["club_name"])]
    # A surname-and-initial match must also agree on the club.
    candidates = candidates[candidates["exact"] | candidates["same_club"]].copy()
    candidates["gap"] = (candidates["minutes"] - candidates["tm_minutes"]).abs()
    candidates = candidates.sort_values(["exact", "same_club", "gap"],
                                        ascending=[False, False, True])
    best = candidates.drop_duplicates(["player_id", "season"])

    # Names written differently on each side ("Kylian Mbappe-Lottin" against
    # "Kylian Mbappé"): same league-season, same club, a shared name token of
    # four letters or more, and minutes within 15% of each other.
    rest = us[~us.set_index(["player_id", "season"]).index.isin(
        best.set_index(["player_id", "season"]).index)]
    loose = rest.merge(tm.drop(columns=["key", "short"]).merge(
        tm[["tm_player", "start", "key"]].rename(columns={"key": "tm_key"}),
        on=["tm_player", "start"]).drop_duplicates(["tm_player", "start", "league", "club_name"]),
        on=["start", "league"])
    if not loose.empty:
        shared = [bool({t for t in a.split() if len(t) >= 4} & set(b.split()))
                  for a, b in zip(loose["key"], loose["tm_key"])]
        loose = loose[shared].copy()
        loose["same_club"] = [bool(_club_tokens(a) & _club_tokens(b))
                              for a, b in zip(loose["team"], loose["club_name"])]
        loose["gap"] = (loose["minutes"] - loose["tm_minutes"]).abs()
        loose = loose[loose["same_club"] & (loose["gap"] <= 0.15 * loose["minutes"].clip(lower=600))]
        # Only when exactly one Transfermarkt player fits.
        loose = loose[~loose.duplicated(["player_id", "season"], keep=False)]
        best = pd.concat([best, loose.assign(exact=False)], ignore_index=True)
    # One Transfermarkt id per Understat player: the one most seasons agree on...
    votes = best.groupby(["player_id", "tm_player"]).size().rename("n").reset_index()
    votes = votes.sort_values("n", ascending=False).drop_duplicates("player_id")
    # ...and never the same Transfermarkt player for two Understat players.
    votes = votes[~votes["tm_player"].duplicated(keep=False)]
    return votes.set_index("player_id")["tm_player"].astype("int64")


def link_players(frame: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """player_id -> tm_id, by name-season-club first and a unique name after.

    `frame` needs player_id, season ("2024-25"), league, team, minutes and a
    normalised-name `key`. The second pass matches only a name held by one
    player on both sides: a wrong match is worse than none.
    """
    link = link_by_season(frame).rename("tm_id").reset_index()
    names = profiles[["tm_id", "tm_name"]].dropna().copy()
    names["key"] = names["tm_name"].map(normalise_name)
    names = names[~names["key"].duplicated(keep=False)]
    keys = frame[["player_id", "key"]].drop_duplicates("player_id")
    keys = keys[~keys["key"].duplicated(keep=False) & ~keys["player_id"].isin(link["player_id"])]
    unique = keys.merge(names, on="key", how="inner")[["player_id", "tm_id"]]
    return pd.concat([link.assign(how="season"), unique.assign(how="name")], ignore_index=True)

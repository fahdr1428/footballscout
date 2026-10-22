"""
Check the committed datasets against each other.

Understat, FBref, the Fantasy Premier League feed and Transfermarkt count the
same matches independently, so the same player-season should carry the same
goals and minutes in each. This joins every pair on the player's name, season
and league - only names that appear once on both sides, so a join never has to
guess between two people - and reports how often they agree. Nothing here feeds
the models; it is a check on the inputs, and the figures are the ones quoted in
the README and on the site's About page.

    python scripts/cross_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.premier_league import normalise_name  # noqa: E402

RAW = ROOT / "data" / "raw"
MIN_MINUTES = 900
TM_LEAGUES = {"GB1": "Premier League", "ES1": "La Liga", "IT1": "Serie A",
              "L1": "Bundesliga", "FR1": "Ligue 1"}


def _load(name: str, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(RAW / f"{name}.csv.gz", usecols=lambda c: c in columns)
    frame["key"] = frame["player"].map(normalise_name)
    return frame


def understat() -> pd.DataFrame:
    frame = _load("understat_big6", ["player", "season", "league", "minutes", "goals", "assists", "xg"])
    return frame[frame["league"].isin(TM_LEAGUES.values())]


def fbref() -> pd.DataFrame:
    return _load("fbref_big5", ["player", "season", "league", "minutes", "goals", "assists", "xg"])


def fpl() -> pd.DataFrame:
    # Goals, assists and minutes are the fantasy game's own. Its xG is not used:
    # the Premier League build takes xG from Understat where it can, so comparing
    # it with Understat's would be comparing a feed with itself.
    return _load("premier_league", ["player", "season", "league", "minutes", "goals", "assists"])


def transfermarkt() -> pd.DataFrame:
    """Transfermarkt's minutes and goals per player, season and league, clubs added up."""
    frame = pd.read_csv(RAW / "tm" / "tm_season_minutes.csv.gz")
    frame = frame[frame["competition_id"].isin(TM_LEAGUES)]
    frame = (frame.groupby(["player_id", "season", "competition_id"], as_index=False)
             .agg(player=("player_name", "last"), minutes=("minutes", "sum"),
                  goals=("goals", "sum"), assists=("assists", "sum")))
    frame["league"] = frame["competition_id"].map(TM_LEAGUES)
    frame["season"] = frame["season"].map(lambda y: f"{y}-{str(y + 1)[2:]}")
    frame["key"] = frame["player"].map(normalise_name)
    return frame


def agreement(left: pd.DataFrame, right: pd.DataFrame, keys: list[str],
              min_minutes: int = MIN_MINUTES) -> dict:
    """How often two feeds agree on the same player-season, past `min_minutes` in both."""
    left = left[~left.duplicated(keys, keep=False)]
    right = right[~right.duplicated(keys, keep=False)]
    both = left.merge(right, on=keys, suffixes=("_a", "_b"))
    both = both[both["minutes_a"].ge(min_minutes) & both["minutes_b"].ge(min_minutes)]
    out = {"pairs": len(both), "seasons": f"{both['season'].min()} to {both['season'].max()}"}
    out["minutes_within_5pct"] = float(
        ((both["minutes_a"] - both["minutes_b"]).abs()
         <= 0.05 * both[["minutes_a", "minutes_b"]].max(axis=1)).mean())
    for column in ("goals", "assists"):
        if f"{column}_a" in both and f"{column}_b" in both:
            gap = (both[f"{column}_a"] - both[f"{column}_b"]).abs().dropna()
            out[f"{column}_same"] = float(gap.eq(0).mean())
            out[f"{column}_within_one"] = float(gap.le(1).mean())
    if "xg_a" in both and "xg_b" in both:
        pair = both[["xg_a", "xg_b"]].dropna()
        out["xg_pairs"] = len(pair)
        out["xg_correlation"] = float(np.corrcoef(pair["xg_a"], pair["xg_b"])[0, 1])
    return out


def checks() -> dict[str, dict]:
    us, fb, pl, tm = understat(), fbref(), fpl(), transfermarkt()
    league = ["key", "season", "league"]
    return {
        "Understat vs FBref": agreement(us, fb, league),
        "Understat vs Transfermarkt": agreement(us, tm, league),
        "FBref vs Transfermarkt": agreement(fb, tm, league),
        "Understat vs FPL": agreement(us[us["league"] == "Premier League"], pl, ["key", "season"]),
        "FBref vs FPL": agreement(fb[fb["league"] == "Premier League"], pl, ["key", "season"]),
    }


def _pct(value) -> str:
    return "-" if value is None or pd.isna(value) else f"{100 * value:.1f}%"


def main() -> int:
    print(f"Same player-season, both feeds past {MIN_MINUTES} minutes, "
          "names unique on both sides.\n")
    print("| Feeds | Player-seasons | Seasons | Goals identical | Goals within one "
          "| Minutes within 5% | Assists identical | xG correlation |")
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for name, row in checks().items():
        xg = f"{row['xg_correlation']:.3f}" if "xg_correlation" in row else "-"
        print(f"| {name} | {row['pairs']:,} | {row['seasons']} | {_pct(row.get('goals_same'))} "
              f"| {_pct(row.get('goals_within_one'))} | {_pct(row['minutes_within_5pct'])} "
              f"| {_pct(row.get('assists_same'))} | {xg} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

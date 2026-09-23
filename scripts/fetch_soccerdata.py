#!/usr/bin/env python3
"""
Pull current FBref seasons straight from the source, via `soccerdata`.

    pip install soccerdata
    python scripts/fetch_soccerdata.py --seasons 2024-25 2025-26
    python scripts/fetch_soccerdata.py --leagues "ENG-Premier League" --seasons 2025-26
    python scripts/fetch_soccerdata.py --dry-run      # show the column mapping only

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
Every other fetch script in this project reads a **mirror** - a repository that
already holds the data - so it runs anywhere. This one goes to fbref.com at
request time, which means it only runs somewhere fbref.com is reachable. It is
not reachable from a sandboxed CI container or from Claude Code's cloud
environment, where the egress proxy answers 403 to the CONNECT; run it on your
own machine.

It is the only path to the **current** season with full FBref depth. The
bundled big-five dataset reads a mirror that froze in stages - defending,
possession and goalkeeping in October 2024, shooting and passing in September
2025 - so its latest season with every block is 2023/24. Pressures and the
shot/goal-creating-action type breakdown are gone from FBref's own display for
every season from 2022/23 on, so this script cannot bring those back either.

WHAT IT WRITES
--------------
* A parquet cache per league-season-table under `--cache`, so a re-run costs no
  requests. `soccerdata` also caches, and rate-limits itself; leave that alone.
  FBref's terms ask for modest automated use, which is what this does - it is a
  personal research tool, not a service.
* `data/raw/fbref_live.csv.gz` in this platform's schema, which registers as a
  data source and flows through exactly the same cleaning, features and models
  as everything else.

MAPPING
-------
`soccerdata` returns a MultiIndex of (group, statistic) that FBref changes from
time to time. Rather than assume, the script flattens those columns, matches
them against the aliases below, and **prints what it found and what it did not**
before writing anything. Check that report on the first run: a metric that did
not map is left missing rather than filled with something else.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_DIR  # noqa: E402

OUT = RAW_DIR / "fbref_live.csv.gz"
DEFAULT_CACHE = Path("~/.cache/footballscout-soccerdata").expanduser()

LEAGUES = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A",
           "GER-Bundesliga", "FRA-Ligue 1"]
STAT_TYPES = ["standard", "shooting", "passing", "passing_types",
              "goal_shot_creation", "defense", "possession", "misc", "keeper", "keeper_adv"]

# Normalised fragments of a flattened "group_statistic" column -> our metric.
# Matching is on the normalised string, so capitalisation and punctuation drift
# in FBref's headers does not silently break the mapping.
ALIASES: dict[str, str] = {
    "performance_gls": "goals", "performance_ast": "assists",
    "performance_g-pk": "np_goals", "performance_pk": "pens_scored",
    "performance_pkatt": "pens_taken", "performance_crdy": "yellow_cards",
    "performance_crdr": "red_cards", "expected_xg": "xg", "expected_npxg": "npxg",
    "expected_xag": "xa",
    "playingtime_min": "minutes", "playingtime_mp": "matches",
    "playingtime_starts": "starts",
    "standard_sh": "shots", "standard_sot": "shots_on_target",
    "standard_fk": "free_kick_shots",
    "total_att": "passes_attempted", "total_cmp": "passes_completed",
    "total_prgdist": "progressive_pass_distance",
    "short_att": "short_passes_attempted", "short_cmp": "short_passes_completed",
    "medium_att": "medium_passes_attempted", "medium_cmp": "medium_passes_completed",
    "long_att": "long_passes_attempted", "long_cmp": "long_passes_completed",
    "_kp": "key_passes", "_1/3": "passes_into_final_third", "_ppa": "passes_into_pen_area",
    "_crspa": "crosses_into_pen_area", "_prgp": "progressive_passes",
    "pass_types_tb": "through_balls", "pass_types_sw": "switches",
    "pass_types_crs": "crosses", "pass_types_ck": "corners_taken",
    "sca_sca": "sca", "gca_gca": "gca",
    "sca_types_passlive": "sca_from_open_play_pass", "sca_types_passdead": "sca_from_set_piece",
    "sca_types_to": "sca_from_dribble", "sca_types_def": "sca_from_defensive_action",
    "tackles_tkl": "tackles", "tackles_tklw": "tackles_won",
    "tackles_def3rd": "tackles_def_third", "tackles_mid3rd": "tackles_mid_third",
    "tackles_att3rd": "tackles_att_third",
    "challenges_att": "dribblers_challenged", "challenges_tkl": "dribblers_tackled",
    "blocks_blocks": "blocks", "blocks_sh": "shots_blocked",
    "_int": "interceptions", "_clr": "clearances", "_err": "errors",
    "touches_touches": "touches", "touches_defpen": "touches_def_pen",
    "touches_def3rd": "touches_def_third", "touches_mid3rd": "touches_mid_third",
    "touches_att3rd": "touches_att_third", "touches_attpen": "touches_att_pen",
    "takeons_att": "dribbles_attempted", "takeons_succ": "dribbles_completed",
    "carries_carries": "carries", "carries_prgdist": "progressive_carry_distance",
    "carries_prgc": "progressive_carries", "carries_1/3": "carries_into_final_third",
    "carries_cpa": "carries_into_pen_area", "carries_mis": "miscontrols",
    "carries_dis": "dispossessed", "receiving_prgr": "progressive_receptions",
    "performance_fls": "fouls_committed", "performance_fld": "fouls_won",
    "performance_off": "offsides", "performance_recov": "ball_recoveries",
    "aerialduels_won": "aerials_won", "aerialduels_lost": "aerials_lost",
    "performance_ga": "gk_goals_against", "performance_sota": "gk_shots_on_target_against",
    "performance_saves": "gk_saves", "performance_cs": "clean_sheets",
    "expected_psxg": "gk_psxg", "crosses_opp": "gk_crosses_faced",
    "crosses_stp": "gk_crosses_stopped", "sweeper_#opa": "gk_def_actions_outside_box",
    "launched_att": "gk_launches_attempted", "launched_cmp": "gk_launches_completed",
}


def normalise(column) -> str:
    parts = [str(p) for p in (column if isinstance(column, tuple) else (column,))]
    parts = [p for p in parts if p and not p.lower().startswith("unnamed")]
    return re.sub(r"[^a-z0-9/#]+", "", "_".join(parts).lower().replace(" ", ""))


def map_columns(frame: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """Which of our metrics this table supplies, and which aliases went unused."""
    found, seen = {}, set()
    for column in frame.columns:
        key = normalise(column)
        for alias, target in ALIASES.items():
            if target in seen:
                continue
            if key == alias.strip("_") or key.endswith(alias):
                found[column] = target
                seen.add(target)
                break
    return found, sorted(set(ALIASES.values()) - seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leagues", nargs="*", default=LEAGUES)
    parser.add_argument("--seasons", nargs="*", default=["2024-25", "2025-26"])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the column mapping without writing the dataset")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    try:
        import soccerdata as sd
    except ImportError:
        print("this script needs soccerdata:\n    pip install soccerdata")
        return 1

    args.cache.mkdir(parents=True, exist_ok=True)
    fbref = sd.FBref(leagues=args.leagues, seasons=args.seasons,
                     data_dir=args.cache / "soccerdata")

    tables, missing_report = [], {}
    for stat_type in STAT_TYPES:
        cached = args.cache / f"{stat_type}.parquet"
        try:
            if cached.exists():
                table = pd.read_parquet(cached)
            else:
                print(f"  pulling {stat_type} …")
                table = fbref.read_player_season_stats(stat_type=stat_type)
                table.to_parquet(cached)
        except Exception as error:            # noqa: BLE001 - FBref drops tables
            print(f"  {stat_type}: unavailable ({type(error).__name__}: {error})")
            continue

        table = table.reset_index()
        mapping, absent = map_columns(table)
        missing_report[stat_type] = absent
        print(f"  {stat_type:20} {len(table):6,} rows, mapped {len(mapping):2} metrics")
        keep = {c: n for c, n in mapping.items()}
        slim = table[list(keep)].rename(columns=keep)
        for key in ("league", "season", "team", "player", "pos", "age"):
            if key in table.columns:
                slim[key] = table[key]
        tables.append(slim)

    if not tables:
        print("\nNothing was returned. fbref.com has to be reachable from this machine:\n"
              "    curl -sI https://fbref.com/ | head -1\n"
              "A 403 on the CONNECT means an egress policy is blocking it, not FBref.")
        return 1

    if args.dry_run:
        print("\nunmapped metrics per table (left missing, never substituted):")
        for stat_type, absent in missing_report.items():
            if absent:
                print(f"  {stat_type}: {', '.join(absent[:12])}"
                      + (" …" if len(absent) > 12 else ""))
        return 0

    frame = tables[0]
    for extra in tables[1:]:
        shared = [c for c in ("league", "season", "team", "player") if c in extra.columns]
        new = [c for c in extra.columns if c not in frame.columns or c in shared]
        frame = frame.merge(extra[new], on=shared, how="outer")

    frame["position_group"] = frame.get("pos", pd.Series(index=frame.index, dtype=object))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"\n{len(frame):,} player-seasons x {frame.shape[1]} columns -> {args.out}")
    print("Positions arrive as FBref's DF/MF/FW/GK. To model the ten detailed groups,\n"
          "join Transfermarkt positions the way src/fbref.py does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Build the real player-season dataset from StatsBomb Open Data.

    python scripts/fetch_statsbomb.py                  # every configured competition
    python scripts/fetch_statsbomb.py --limit 20       # 20 matches each, for a smoke test
    python scripts/fetch_statsbomb.py --leagues "FA WSL" "NWSL"

Match-level results are cached under data/raw/statsbomb_cache/, so an
interrupted run resumes instead of re-downloading.

    Data provided by StatsBomb - https://github.com/statsbomb/open-data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_DIR, STATSBOMB_PLAYERS_CSV  # noqa: E402
from src.statsbomb import ATTRIBUTION, COMPETITIONS, build_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="matches per competition")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--leagues", nargs="*", default=None, help="filter by league name")
    parser.add_argument("--out", type=Path, default=STATSBOMB_PLAYERS_CSV)
    args = parser.parse_args()

    competitions = COMPETITIONS
    if args.leagues:
        wanted = {name.lower() for name in args.leagues}
        competitions = [c for c in competitions if c.league.lower() in wanted]

    print(ATTRIBUTION)
    print(f"{len(competitions)} competition-seasons\n")
    started = time.time()

    frame = build_dataset(
        competitions,
        cache_dir=RAW_DIR / "statsbomb_cache",
        workers=args.workers,
        limit=args.limit,
    )
    if frame.empty:
        print("no data ingested")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(f"\nelapsed {time.time() - started:.0f}s")
    print(f"{len(frame):,} player-seasons x {frame.shape[1]} columns -> {args.out}")
    print(
        frame.groupby(["league", "season"])
        .agg(players=("player_id", "size"), minutes=("minutes", "sum"))
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

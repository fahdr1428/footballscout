#!/usr/bin/env python3
"""
Men's international tournaments from StatsBomb Open Data: the 2018 and 2022
World Cups, Euro 2020 and Euro 2024, the 2024 Copa America and the 2023 Africa
Cup of Nations, every metric derived here from the raw events, with
Transfermarkt's ages, heights, feet and prices joined on.

    python scripts/build_tournaments.py   # -> data/raw/statsbomb_tournaments.csv.gz

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

from src.config import RAW_DIR, STATSBOMB_TOURNAMENTS_CSV  # noqa: E402
from src.statsbomb import ATTRIBUTION, TOURNAMENTS, build_dataset, tournaments_table  # noqa: E402

DEFAULT_CACHE = Path("/tmp/understat-big6")   # Transfermarkt downloads, shared with fetch_understat.py


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="matches per tournament, for a smoke test")
    parser.add_argument("--out", type=Path, default=STATSBOMB_TOURNAMENTS_CSV)
    args = parser.parse_args()

    print(ATTRIBUTION)
    started = time.time()
    frame = build_dataset(TOURNAMENTS, cache_dir=RAW_DIR / "statsbomb_cache", workers=args.workers, limit=args.limit)
    print(f"{len(frame):,} player-tournaments in {time.time() - started:.0f}s")
    frame, coverage = tournaments_table(frame, args.cache)
    for key, value in coverage.items():
        print(f"  {key}: {value:.1%}" if isinstance(value, float) else f"  {key}: {value:,}")
    frame["tm_id"] = frame["tm_id"].astype("Int64")
    # A fixed gzip timestamp: an unchanged table is an unchanged file.
    frame.to_csv(args.out, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

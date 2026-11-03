#!/usr/bin/env python3
"""
The men's 2015/16 season from StatsBomb Open Data, with Transfermarkt's ages
and prices joined on.

    python scripts/fetch_statsbomb.py           # the event-derived player table, once
    python scripts/build_statsbomb_2015_16.py   # -> data/raw/statsbomb_2015_16.csv.gz

    Data provided by StatsBomb - https://github.com/statsbomb/open-data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import STATSBOMB_2015_16_CSV, STATSBOMB_PLAYERS_CSV  # noqa: E402
from src.statsbomb import attach_transfermarkt, season_2015_16  # noqa: E402

DEFAULT_CACHE = Path("/tmp/understat-big6")   # shared with fetch_understat.py


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=STATSBOMB_2015_16_CSV)
    args = parser.parse_args()

    frame = season_2015_16(pd.read_csv(STATSBOMB_PLAYERS_CSV))
    print(f"{len(frame):,} player-seasons in {sorted(frame['league'].unique())}")
    frame, coverage = attach_transfermarkt(frame, args.cache)
    for key, value in coverage.items():
        print(f"  {key}: {value:.1%}" if isinstance(value, float) else f"  {key}: {value:,}")
    # A fixed gzip timestamp: an unchanged table is an unchanged file, so a
    # refresh that found nothing new has nothing to commit.
    frame.to_csv(args.out, index=False, compression=(
        {"method": "gzip", "mtime": 0} if str(args.out).endswith(".gz") else "infer"))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

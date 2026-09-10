#!/usr/bin/env python3
"""
Build the big-five-leagues dataset (2017/18 - 2021/22) from public season data.

    python scripts/fetch_fbref.py                     # download + build
    python scripts/fetch_fbref.py --cache /tmp/fbref  # reuse a download
    python scripts/fetch_fbref.py --seasons 2020 2021 2022

Sources, all mirrored by https://github.com/JaseZiv/worldfootballR_data:
FBref season statistics (eleven blocks per player), a curated FBref ->
Transfermarkt player mapping, and Transfermarkt season squad records for the
market value, position, height, foot and nationality.

About 16 MB is downloaded once and cached, so re-runs are fast.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FBREF_BIG5_CSV  # noqa: E402
from src.fbref import ATTRIBUTION, FIRST_SEASON, LAST_SEASON, build_dataset  # noqa: E402

DEFAULT_CACHE = Path("/tmp/fbref-big5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="where the downloaded source files are kept")
    parser.add_argument("--seasons", nargs="*", type=int, default=None,
                        metavar="END_YEAR",
                        help=f"season end years (default {FIRST_SEASON}-{LAST_SEASON}); "
                             "2023 exists upstream but is a partial season")
    parser.add_argument("--out", type=Path, default=FBREF_BIG5_CSV)
    args = parser.parse_args()

    print(ATTRIBUTION)
    try:
        import pyreadr  # noqa: F401
    except ImportError:
        print("\nthis build needs pyreadr to read R data files:\n    pip install pyreadr")
        return 1

    seasons = sorted(args.seasons) if args.seasons else None
    if seasons:
        seasons = range(seasons[0], seasons[-1] + 1)

    started = time.time()
    frame, coverage = build_dataset(args.cache, seasons=seasons)
    if frame.empty:
        print("no data built")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(f"\nelapsed {time.time() - started:.0f}s")
    print(f"{len(frame):,} player-seasons x {frame.shape[1]} columns -> {args.out}")
    print("\njoin coverage")
    for key in ("mapped_to_transfermarkt", "detailed_position", "with_market_value"):
        if key in coverage:
            print(f"  {key.replace('_', ' '):26} {coverage[key]:.1%}")

    summary = (
        frame.groupby(["season", "league"])
        .agg(players=("player", "size"), median_minutes=("minutes", "median"))
        .reset_index()
        .pivot(index="season", columns="league", values="players")
    )
    print("\nplayers per league and season")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

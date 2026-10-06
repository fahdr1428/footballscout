#!/usr/bin/env python3
"""
Build the six-league Understat dataset (2014/15 - 2024/25).

    python scripts/fetch_understat.py                    # download + build
    python scripts/fetch_understat.py --seasons 2023/24 2024/25
    python scripts/fetch_understat.py --include-partial  # add the ~10-round 2025/26

Source: Understat per-player season aggregates, mirrored as CSV by
https://github.com/vibedatascience/understat_players_aggregated. About 7 MB is
downloaded once and cached, so re-runs are instant.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import UNDERSTAT_CSV  # noqa: E402
from src.understat import ATTRIBUTION, PARTIAL_SEASONS, build_dataset  # noqa: E402

DEFAULT_CACHE = Path("/tmp/understat-big6")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="Understat form, e.g. 2024/25")
    parser.add_argument("--include-partial", action="store_true",
                        help=f"also include {', '.join(sorted(PARTIAL_SEASONS))}, "
                             "the season in progress")
    parser.add_argument("--no-enrich", action="store_true",
                        help="skip the Transfermarkt join (ages, heights, feet, values)")
    parser.add_argument("--out", type=Path, default=UNDERSTAT_CSV)
    args = parser.parse_args()

    print(ATTRIBUTION)
    started = time.time()

    frame, coverage = build_dataset(args.cache, seasons=args.seasons,
                                    enrich=not args.no_enrich,
                                    include_partial=args.include_partial)

    if frame.empty:
        print("no data built")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(f"\nelapsed {time.time() - started:.0f}s")
    print(f"{len(frame):,} player-seasons x {frame.shape[1]} columns -> {args.out}")
    print(f"{coverage['players']:,} unique players, "
          f"{len(coverage['seasons'])} seasons, {len(coverage['leagues'])} leagues")
    if "matched_players" in coverage:
        print(f"\nTransfermarkt join (name + season + club, then unique name)")
        print(f"  players matched      {coverage['matched_players']:,} of "
              f"{coverage['total_players']:,}")
        print(f"  minutes covered      {coverage['minutes_covered']:.1%}")
        print(f"  rows with an age     {coverage['with_age']:.1%}")
        print(f"  rows with a value    {coverage['with_market_value']:.1%}")

    table = (frame[frame["minutes"] >= 900]
             .groupby(["season", "league"]).size().unstack(fill_value=0))
    print("\nplayers past 900 minutes, by season and league")
    print(table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

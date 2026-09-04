#!/usr/bin/env python3
"""
Build the top-leagues dataset from a transfermarkt-api instance.

The service is https://github.com/felipeall/transfermarkt-api. Run it yourself:

    docker run -d -p 8000:8000 --name transfermarkt-api \\
        $(docker build -q https://github.com/felipeall/transfermarkt-api.git)

then:

    # squads, market values and season stats for the top six leagues
    python scripts/fetch_transfermarkt.py --season 2025

    # squads only - fast, and all that the enrichment layer needs
    python scripts/fetch_transfermarkt.py --season 2025 --no-stats --market-only

    # widen the pool
    python scripts/fetch_transfermarkt.py --season 2025 --competitions GB1 ES1 IT1 L1 FR1 PO1 NL1 GB2

    # against the maintainer's hosted instance, which is rate limited
    python scripts/fetch_transfermarkt.py --api https://transfermarkt-api.fly.dev --rate 0.6

Data is Transfermarkt's, read through that service. It is a market and
biographical database: expect market values, ages, heights and true positions,
and expect appearances / goals / assists / cards / minutes rather than xG.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TRANSFERMARKT_CSV, TRANSFERMARKT_MARKET_CSV  # noqa: E402
from src.transfermarkt import (  # noqa: E402
    ATTRIBUTION, COMPETITIONS_BY_ID, DEFAULT_BASE_URL, TOP_COMPETITIONS, TransfermarktClient,
    build_dataset, build_squads,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=DEFAULT_BASE_URL, help="transfermarkt-api base URL")
    parser.add_argument("--season", default="2025", help="season id: the starting year")
    parser.add_argument("--competitions", nargs="*", default=None,
                        help=f"Transfermarkt codes; default {[c.competition_id for c in TOP_COMPETITIONS]}")
    parser.add_argument("--rate", type=float, default=4.0,
                        help="requests per second (use 0.6 against the hosted instance)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-stats", action="store_true",
                        help="skip the per-player season stats (one request per player)")
    parser.add_argument("--market-only", action="store_true",
                        help="write the market spine used to enrich the other datasets")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.competitions:
        unknown = [c for c in args.competitions if c not in COMPETITIONS_BY_ID]
        if unknown:
            print(f"unknown competition codes: {unknown}")
            print(f"known: {sorted(COMPETITIONS_BY_ID)}")
            return 2
        competitions = [COMPETITIONS_BY_ID[c] for c in args.competitions]
    else:
        competitions = TOP_COMPETITIONS

    print(ATTRIBUTION)
    print(f"api: {args.api}  season: {args.season}  "
          f"competitions: {[c.competition_id for c in competitions]}\n")

    client = TransfermarktClient(base_url=args.api, rate=args.rate)
    started = time.time()

    try:
        if args.market_only:
            frame = build_squads(client, competitions, args.season, workers=args.workers)
            destination = args.out or TRANSFERMARKT_MARKET_CSV
        else:
            frame = build_dataset(
                client, competitions, args.season,
                with_stats=not args.no_stats, workers=args.workers,
            )
            destination = args.out or TRANSFERMARKT_CSV
    except RuntimeError as error:
        print(f"\ncould not reach the API: {error}")
        print("Is the service running? See the header of this file for the docker command.")
        return 1

    if frame.empty:
        print("no rows returned - check the season id and competition codes")
        return 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"\nelapsed {time.time() - started:.0f}s")
    print(f"{len(frame):,} players x {frame.shape[1]} columns -> {destination}")

    summary = frame.groupby("league").agg(
        players=("player", "size"),
        clubs=("team", "nunique"),
        median_value_m=("market_value_m", "median"),
        median_age=("age", "median"),
    )
    print()
    print(summary.round(1).to_string())
    print()
    print(frame["position_group"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

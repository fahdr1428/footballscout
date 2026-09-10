#!/usr/bin/env python3
"""
Is a second striker a different job from an attacking midfielder?

The position taxonomy in `src/config.py` has to come from somewhere. Splitting
every position Transfermarkt records would give thirteen groups, several of
them too small to rank against; leaving them merged would measure players
against peers who do a different job. This script decides the question with
evidence instead of taste.

For each candidate pair it trains a cross-validated logistic classifier on that
position group's own model features and reports the **balanced accuracy** with
which it can tell the two apart - the mean of the two per-class recalls, so a
classifier that just guesses the more common position scores 0.50 rather than
being flattered by the imbalance. 0.50 is a coin flip: the two positions are
statistically the same job, and splitting them would halve the peer group for
no information. Toward 1.00 they are different jobs and deserve separate models.

    python scripts/position_separability.py
    python scripts/position_separability.py --source fbref_big5 --min-minutes 900

Controls (centre-back against defensive midfield, defensive against attacking
midfield) are included every run: if those do not come back near 0.95, the
measurement itself is broken and nothing else on the page should be believed.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_SOURCES, DEFAULT_SOURCE  # noqa: E402
from src.pipeline import build_features, build_platform  # noqa: E402
from src.validation import position_separability  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--seasons", nargs="*", default=None)
    parser.add_argument("--min-minutes", type=int, default=None)
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    features, report, used = build_features(args.source)
    # Default to the seasons the source opens on, so the numbers here describe
    # the taxonomy the app actually uses rather than some other pool.
    seasons = args.seasons or list(DATA_SOURCES[used].default_seasons) or None
    platform = build_platform(
        features, report, source=used,
        seasons=seasons,
        min_minutes=args.min_minutes,
    )
    print(f"source: {used} | pool {len(platform.pool):,} player-seasons")
    print(f"seasons: {', '.join(platform.seasons) or 'all'}\n")

    table = position_separability(platform)
    if table.empty:
        print("This source does not record positions specific enough to test.")
        return 1

    print(table.to_string(index=False))
    print()
    for _, row in table[table["kind"] == "candidate"].iterrows():
        print(f"  {row['pair']:<28} {row['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

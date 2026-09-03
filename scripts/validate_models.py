#!/usr/bin/env python3
"""
Fit every position model and write models/validation_report.md.

    python scripts/validate_models.py
    python scripts/validate_models.py --min-minutes 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_MIN_MINUTES, DEFAULT_SOURCE, VALIDATION_REPORT  # noqa: E402
from src.pipeline import build_features, build_platform  # noqa: E402
from src.validation import (  # noqa: E402
    clustering_diagnostics, similarity_role_agreement, write_validation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-minutes", type=int, default=DEFAULT_MIN_MINUTES)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="statsbomb or simulated")
    parser.add_argument(
        "--groups", nargs="*", default=["W", "CB", "DM"],
        help="position groups to run the sensitivity tests on",
    )
    args = parser.parse_args()

    features, report, used = build_features(args.source)
    print(f"source: {used} | cleaned {report.rows_out:,} of {report.rows_in:,} rows")

    platform = build_platform(features, report, min_minutes=args.min_minutes, source=used)
    print(f"pool: {len(platform.pool):,} player-seasons, {len(platform.models)} position models\n")

    print(clustering_diagnostics(platform).to_string(index=False))
    print()
    agreement = similarity_role_agreement(platform, sample=120)
    if not agreement.empty:
        print(agreement.to_string(index=False))
        print()

    text = write_validation_report(platform)
    print(f"written: {VALIDATION_REPORT}  ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

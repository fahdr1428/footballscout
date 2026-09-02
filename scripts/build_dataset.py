#!/usr/bin/env python3
"""
Rebuild the reference dataset and the processed feature table.

    python scripts/build_dataset.py               # regenerate everything
    python scripts/build_dataset.py --seed 11     # a different simulated universe
    python scripts/build_dataset.py --no-regenerate  # reuse raw, rebuild features only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, PROCESSED_PLAYERS_CSV, RAW_PLAYERS_CSV  # noqa: E402
from src.data_processing import clean_players, load_raw_players  # noqa: E402
from src.feature_engineering import build_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="simulation seed")
    parser.add_argument(
        "--no-regenerate", action="store_true", help="reuse the existing raw file"
    )
    args = parser.parse_args()

    raw = load_raw_players(regenerate=not args.no_regenerate, seed=args.seed)
    print(f"raw          {raw.shape[0]:>7,} rows x {raw.shape[1]} columns  -> {RAW_PLAYERS_CSV}")

    clean, report = clean_players(raw)
    for label, value in report.as_rows():
        print(f"  {label:<44} {value:>8}")
    for line in report.notes:
        print(f"  note: {line}")

    features = build_features(clean)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_PLAYERS_CSV, index=False)
    print(
        f"processed    {features.shape[0]:>7,} rows x {features.shape[1]} columns "
        f"-> {PROCESSED_PLAYERS_CSV}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

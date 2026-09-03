#!/usr/bin/env python3
"""
Build the Premier League dataset (2016/17 - 2025/26) from public season data.

    python scripts/fetch_premier_league.py                    # clone + build
    python scripts/fetch_premier_league.py --repo /path/to/clone
    python scripts/fetch_premier_league.py --seasons 2024-25 2025-26

Sources: the Fantasy Premier League season and gameweek exports, and Understat
per-player match logs, both mirrored by
https://github.com/vaastav/Fantasy-Premier-League (MIT licensed).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PREMIER_LEAGUE_CSV  # noqa: E402
from src.premier_league import ATTRIBUTION, SEASONS, build_dataset  # noqa: E402

MIRROR = "https://github.com/vaastav/Fantasy-Premier-League"
DEFAULT_CLONE = Path("/tmp/fantasy-premier-league")


def ensure_repo(path: Path) -> Path:
    """Clone the mirror if it is not already on disk."""
    if (path / "data").is_dir():
        return path
    print(f"cloning {MIRROR} -> {path} (about 360 MB, one-off)")
    environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(
        ["git", "clone", "--depth", "1", MIRROR, str(path)],
        check=True, env=environment,
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_CLONE,
                        help="existing clone of the mirror (cloned here if absent)")
    parser.add_argument("--seasons", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=PREMIER_LEAGUE_CSV)
    args = parser.parse_args()

    print(ATTRIBUTION)
    repo = ensure_repo(args.repo)
    started = time.time()

    frame = build_dataset(repo, seasons=args.seasons or SEASONS)
    if frame.empty:
        print("no data built")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"\nelapsed {time.time() - started:.0f}s")
    print(f"{len(frame):,} player-seasons x {frame.shape[1]} columns -> {args.out}")

    summary = (
        frame[frame["minutes"] >= 900]
        .groupby("season")
        .agg(
            players=("player", "size"),
            clubs=("team", "nunique"),
            with_age=("age", lambda s: f"{s.notna().mean():.0%}"),
            with_xg=("xg", lambda s: f"{s.notna().mean():.0%}"),
            with_lineup_pos=("position_source",
                             lambda s: f"{s.str.startswith('Understat').mean():.0%}"),
        )
    )
    print("\nplayers above 900 minutes:")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

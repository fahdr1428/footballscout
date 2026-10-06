#!/usr/bin/env python3
"""
Refuse to publish a site build that has lost data.

    python scripts/check_site.py                       # against the last commit
    python scripts/check_site.py --before old/index.html

Run by the weekly refresh before it commits: a fetch that half-failed upstream
would otherwise publish a site with seasons missing. Fails (exit 1) if a
source or season disappeared, if any season lost more than a tenth of its
players, or if a season file the page points to does not exist.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "static"
MAX_LOSS = 0.10
# A season being played re-sets its minutes floor every round, so its pool
# breathes; only a collapse there means something broke.
MAX_LOSS_IN_PROGRESS = 0.40


def catalog(html: str) -> dict:
    match = re.search(r"<script>window\.__SCOUT__=(.*?);</script>", html, re.S)
    if not match:
        raise SystemExit("no catalogue in the page")
    return json.loads(match.group(1).replace("<\\/", "</"))["catalog"]


def seasons(cat) -> dict[str, tuple[int, bool]]:
    return {f"{c['key']}:{s['season']}": (s["players"], bool(s.get("inProgress")))
            for c in cat for s in c["seasons"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", type=Path, default=None,
                        help="the previous index.html (default: the one in HEAD)")
    args = parser.parse_args()

    after_html = (SITE / "index.html").read_text(encoding="utf-8")
    if args.before:
        before_html = args.before.read_text(encoding="utf-8")
    else:
        before_html = subprocess.run(["git", "show", "HEAD:static/index.html"], cwd=ROOT,
                                     capture_output=True, text=True, check=True).stdout
    before, after = seasons(catalog(before_html)), seasons(catalog(after_html))

    problems = []
    for key, (players, live) in before.items():
        if key not in after:
            problems.append(f"{key} is gone")
            continue
        allowed = MAX_LOSS_IN_PROGRESS if live else MAX_LOSS
        if after[key][0] < players * (1 - allowed):
            problems.append(f"{key} fell from {players:,} to {after[key][0]:,} players")
    for c in catalog(after_html):
        for path in [s["file"] for s in c["seasons"]] + ([c["all"]] if c.get("all") else []):
            if not (SITE / path).exists():
                problems.append(f"{path} is referenced but missing")

    for key in sorted(set(after) - set(before)):
        print(f"new: {key} ({after[key][0]:,} players)")
    for key in sorted(set(after) & set(before)):
        if after[key][0] != before[key][0]:
            print(f"changed: {key} {before[key][0]:,} -> {after[key][0]:,}")
    if problems:
        print("NOT publishing:\n  " + "\n  ".join(problems))
        return 1
    print(f"ok: {len(after)} seasons, {sum(n for n, _ in after.values()):,} player-seasons")
    return 0


if __name__ == "__main__":
    sys.exit(main())

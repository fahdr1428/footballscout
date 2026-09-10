#!/usr/bin/env python3
"""
Export the platform as one self-contained HTML file.

    python scripts/export_static.py                       # -> static/index.html
    python scripts/export_static.py --season 2020-21
    python scripts/export_static.py --out /tmp/demo.html

The Streamlit app needs a Python process. This does not: it precomputes one
season's percentiles, category scores and nearest neighbours, inlines them as
JSON, and emits a single file that runs anywhere a browser can open it -
GitHub Pages, Netlify, Cloudflare Pages, an email attachment, a USB stick.

It is a **demo, not the platform**. Search, the player read-out and similarity
survive the trip. The recruitment finder, squad analysis, archetype maps and
the whole validation suite do not, because they refit models against whatever
pool you select and there is no Python at the other end to do it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    DEFAULT_SOURCE, METRIC_LABELS, POSITION_GROUP_NAMES, ROOT_DIR, categories_for,
)
from src.pipeline import build_features, build_platform  # noqa: E402

TEMPLATE_DIR = ROOT_DIR / "static"
# Named index.html so the folder can be served as-is by GitHub Pages,
# Netlify or Cloudflare Pages without renaming anything.
DEFAULT_OUT = TEMPLATE_DIR / "index.html"

TOP_STRENGTHS = 8
WEAKNESSES = 5
NEIGHBOURS = 6


def _num(value, places: int = 1):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return None if not np.isfinite(value) else round(value, places)


def export(platform, season: str) -> dict:
    """Everything the page needs, keyed short because it ships over the wire."""
    pool, categories, percentiles = platform.pool, platform.categories, platform.percentiles
    order = list(pool.index)
    position = {index: i for i, index in enumerate(order)}

    players = []
    for index in order:
        row = pool.loc[index]
        group = row["position_group"]
        names = list(categories_for(group))

        ranked = []
        for metric in (platform.models[group].features if group in platform.models else []):
            value = percentiles.loc[index].get(f"pct_{metric}")
            if pd.notna(value):
                ranked.append([METRIC_LABELS.get(metric, metric), round(float(value))])
        ranked.sort(key=lambda pair: -pair[1])

        neighbours = []
        if group in platform.models:
            for _, match in platform.similar(index, n=NEIGHBOURS).iterrows():
                seat = position.get(match.name)
                if seat is not None:
                    neighbours.append([seat, round(float(match["similarity"]))])

        players.append({
            "n": row["player"], "t": row["team"], "l": row["league"],
            "p": row["position"], "g": group,
            "fl": row.get("flank") if pd.notna(row.get("flank")) else None,
            "fs": row.get("footed_side") if pd.notna(row.get("footed_side")) else None,
            "ft": row.get("foot") if pd.notna(row.get("foot")) else None,
            "a": _num(row.get("age")), "m": int(row["minutes"]),
            "h": _num(row.get("height_cm"), 0),
            "v": None if pd.isna(row.get("market_value_eur")) else int(row["market_value_eur"]),
            "ar": row.get("archetype"),
            "cn": names,
            "cv": [_num(categories.loc[index].get(f"cat_{c}")) for c in names],
            "top": ranked[:TOP_STRENGTHS],
            "bot": ranked[-WEAKNESSES:][::-1],
            "s": neighbours,
        })

    return {
        "meta": {
            "season": season,
            "minMinutes": int(platform.min_minutes),
            "leagues": sorted(pool["league"].unique()),
            "groupNames": {g: POSITION_GROUP_NAMES[g]
                           for g in sorted(pool["position_group"].unique())},
            # Named honestly: on a one-season pool the thin groups fold into
            # their parent, and the page says which rather than quietly hiding it.
            "collapsed": {g: info["parent"]
                          for g, info in platform.collapsed_groups.items()},
            "counts": {g: int(n) for g, n in pool["position_group"].value_counts().items()},
        },
        "players": players,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--season", default="2021-22",
                        help="one season: every player appears once, which a demo wants")
    parser.add_argument("--min-minutes", type=int, default=900)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    features, report, used = build_features(args.source)
    platform = build_platform(features, report, source=used,
                              seasons=[args.season], min_minutes=args.min_minutes)
    if platform.pool.empty:
        print(f"no players in {args.season} for source {used}")
        return 1

    data = export(platform, args.season)
    head = (TEMPLATE_DIR / "_head.html").read_text(encoding="utf-8")
    body = (TEMPLATE_DIR / "_body.html").read_text(encoding="utf-8")
    payload = "<script>window.__SCOUT__=" + json.dumps(data, separators=(",", ":")) + ";</script>"

    # Standalone: the page has to carry its own charset, which the artifact
    # host would otherwise supply. Without it every euro sign renders as mojibake.
    page = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"{head}\n</head>\n<body>\n{payload}\n{body}\n</body>\n</html>\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")

    size = args.out.stat().st_size / 1e6
    print(f"{len(data['players']):,} players, {args.season} -> {args.out} ({size:.2f} MB)")
    print("open it directly, or drop it on any static host - no server needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

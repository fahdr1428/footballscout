#!/usr/bin/env python3
"""
Export the platform as a static site: one page, plus one data file per season.

    python scripts/export_static.py                          # -> static/
    python scripts/export_static.py --sources fbref_big5     # one source (quick rebuild)
    python scripts/export_static.py --seasons 2023-24        # one season of each source

The Streamlit app needs a Python process. This does not: it precomputes every
season's percentiles, category scores and model vectors, writes each season to
`static/data/<source>__<season>.json`, and emits `static/index.html` with a
catalogue of them and the default season inlined. Serve the folder from any
static host - GitHub Pages, Netlify, Cloudflare Pages, Vercel. Opened straight
from disk the page still works for the inlined season; the others need to be
served, because a browser will not fetch files for a page opened from disk.

Similarity is computed **in the browser**: each player ships as the z-vector
the model uses and each position as its metric weights (repeatability squared),
and `100 x cosine` over the two, weighted, is the same number the Streamlit app
reports. That lets the page rank every player in
a position group live, refilter and reweight the ranking, and show which
metrics pulled a pair together - none of which a precomputed list could do.

It is a **demo, not the platform**. Squad analysis, archetype maps and the
validation suite do not survive the trip, because they refit models against
whatever pool you select and there is no Python at the other end to do it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    COUNTING_STATS, DATA_SOURCES, LOWER_IS_BETTER, METRIC_LABELS, PERCENT_METRICS,
    DEFAULT_WEIGHTS, POSITION_GROUP_NAMES, RAW_DIR, ROOT_DIR, TEAM_CONTEXT_METRICS,
    categories_for,
)
from src.feature_engineering import metrics_for_percentiles  # noqa: E402
from src.pipeline import build_features, build_platform  # noqa: E402

SITE_DIR = ROOT_DIR / "static"
DATA_DIR = SITE_DIR / "data"
MIN_MINUTES = 900

# What the public site ships, and in what order the switcher lists it. Each
# source contributes every complete season it has; a fragment of a season
# (a mirror that stopped a few rounds in) is left to the Streamlit app, where it
# can be selected deliberately, rather than offered next to whole seasons.
SITE_SOURCES = [
    {
        "key": "understat_big6", "name": "Top leagues", "via": "Understat",
        "skip": set(),
        "blurb": "The most current: the big five plus Russia, 2014/15 to 2025/26 and the season "
                 "in progress, every player priced by Transfermarkt. Attacking output only - "
                 "xG, xA, xGChain, xGBuildup - so no defending, and goalkeepers have no model.",
    },
    {
        "key": "fbref_big5", "name": "Big five, in depth", "via": "FBref",
        "skip": {"2025-26"},
        "updated": "Season stats: FBref, via a public mirror last updated 18 Sep 2025",
        "blurb": "The deepest: up to 44 metrics and ten specific positions, 2017/18 to 2024/25, "
                 "priced by Transfermarkt. 2023/24 is the last season every block measured in "
                 "full - the public mirror of FBref stopped in September 2025.",
    },
    {
        "key": "premier_league", "name": "Premier League", "via": "FPL + Understat",
        "skip": set(),
        "updated": "Season stats: Fantasy Premier League, complete to 2025/26",
        "blurb": "Ten complete seasons to 2025/26: FPL's ICT and defensive-contribution numbers "
                 "with Understat's xG, real Transfermarkt prices beside the fantasy price. One "
                 "league and a summary feed - no progressive passes, duels or pass completion.",
    },
]
DEFAULT_DATASET = "understat_big6:2025-26"

# A season still being played cannot use the 900-minute floor - nobody has
# played it yet. Each league's floor is half the minutes its busiest player has
# had so far (never under two full games), so a league four rounds in and one
# eight rounds in are both measured on regular starters.
IN_PROGRESS_MAX_MINUTES = 1800
IN_PROGRESS_SHARE = 0.5
IN_PROGRESS_MIN_FLOOR = 180


def _num(value, places: int = 1):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return None if not np.isfinite(value) else round(value, places)


def _label(column: str) -> str:
    return METRIC_LABELS.get(column) or COUNTING_STATS.get(column) or column


def export(platform, season: str) -> dict:
    """Everything the page needs for one season, keyed short because it ships over the wire.

    The important choice is shipping each player's **z-vector** and each
    position's metric weights rather than a precomputed list of his nearest
    neighbours. Similarity in this
    project is `100 x cosine` over that vector, which is a few lines of
    JavaScript - so the page can rank every player in a group live, refilter and
    reweight the ranking without a round trip, and show which metrics pulled two
    players together.
    """
    pool, categories, percentiles = platform.pool, platform.categories, platform.percentiles

    groups: dict[str, dict] = {}
    labels: dict[str, str] = {}
    for group, model in platform.models.items():
        display = list(metrics_for_percentiles(group, list(pool.columns)))
        basket = categories_for(group)
        names = list(basket)
        # Which categories each model feature belongs to, so the page can let a
        # scout say "weight passing more" and reweight the cosine the same way
        # the engine does: every component scaled by sqrt(weight).
        membership = [[i for i, c in enumerate(names) if f in basket[c]] for f in model.features]
        # The engine's repeatability weights, rescaled to average 1 so a page
        # multiplying them by a scout's category weights keeps both readable.
        # Plain z plus these reproduces the engine's cosine exactly; shipping
        # z already multiplied by them would lose precision to rounding and
        # turn every z-gap on the page into something other than SDs.
        weights = model.engine.weights / model.engine.weights.mean()
        # The position's default category weights - the same ones the app's
        # role fit uses - so the page's rating is that number, shown working.
        default = DEFAULT_WEIGHTS.get(group, {})
        groups[group] = {
            "f": list(model.features), "d": display, "c": names, "w": membership,
            "rw": [int(default.get(c, 0)) for c in names],
            "r": [round(float(w), 3) for w in weights],
            "rr": [model.reliability.get(f) for f in model.features],
        }
        for metric in set(model.features) | set(display):
            labels[metric] = _label(metric)

    players = []
    for index in pool.index:
        row = pool.loc[index]
        group = row["position_group"]
        spec = groups.get(group)

        zrow, display = [], []
        if spec is not None:
            model = platform.models[group]
            seat = model.z.index.get_loc(index)
            zrow = [round(float(v), 2) for v in model.z.iloc[seat]]
            for metric in spec["d"]:
                value = row.get(metric)
                pct = percentiles.loc[index].get(f"pct_{metric}")
                display.append([_num(value, 2), None if pd.isna(pct) else round(float(pct))])

        players.append({
            "id": str(row["player_id"]),
            "n": row["player"], "t": row["team"], "l": row["league"],
            "p": row["position"], "g": group,
            "fl": row.get("flank") if pd.notna(row.get("flank")) else None,
            "fs": row.get("footed_side") if pd.notna(row.get("footed_side")) else None,
            "ft": row.get("foot") if pd.notna(row.get("foot")) else None,
            "a": _num(row.get("age")), "m": int(row["minutes"]),
            "h": _num(row.get("height_cm"), 0),
            "v": None if pd.isna(row.get("market_value_eur")) else int(row["market_value_eur"]),
            # The fantasy price, where a source has that instead of a valuation.
            # It is a popularity signal, not a fee, and the page labels it so.
            "pr": _num(row.get("price_m"), 1),
            # His most recent Transfermarkt valuation and its date - today's
            # price, where "v" is what he was worth as this season closed.
            "lv": None if pd.isna(row.get("latest_value_eur")) else int(row["latest_value_eur"]),
            "ld": row.get("latest_value_date") if pd.notna(row.get("latest_value_date")) else None,
            "ar": row.get("archetype") if pd.notna(row.get("archetype")) else None,
            "cv": [_num(categories.loc[index].get(f"cat_{c}"))
                   for c in (spec["c"] if spec else [])],
            "z": zrow,
            "d": display,
        })

    # A metric can be listed for a position and still be empty in this season -
    # the Premier League feed has an aerial-duel column with nothing in it. Those
    # would offer a leaderboard that ranks nobody and a comparison row of dashes,
    # so they are pruned here, per group, once the values are known.
    _prune_empty_metrics(groups, players)

    # What this season did not measure that the source does elsewhere, straight
    # from the cleaning report - so the page can say so instead of leaving a
    # scout to wonder why a metric vanished.
    thin = sorted(c for c, seasons in platform.cleaning.partial_columns.items() if season in seasons)
    count = lambda column: int(pool[column].notna().sum()) if column in pool.columns else 0

    return {
        "meta": {
            "season": season,
            "minMinutes": int(platform.min_minutes),
            "leagues": sorted(pool["league"].unique()),
            # Folded and unmodelled groups are named on the page too, so they
            # need a name here even though no player in the pool carries them.
            "groupNames": {g: POSITION_GROUP_NAMES.get(g, g) for g in sorted(
                set(pool["position_group"]) | set(platform.collapsed_groups)
                | set(platform.unmodelled_groups))},
            "groups": groups,
            "labels": labels,
            # Rendering hints: a percentage needs a % sign, and for a
            # lower-is-better metric the smaller number is the better one.
            "pct": sorted(PERCENT_METRICS & set(labels)),
            "lower": sorted(LOWER_IS_BETTER & set(labels)),
            "club": sorted(TEAM_CONTEXT_METRICS & set(labels)),
            "collapsed": {g: info["parent"] for g, info in platform.collapsed_groups.items()},
            "counts": {g: int(n) for g, n in pool["position_group"].value_counts().items()},
            "unmodelled": {g: int(n) for g, n in platform.unmodelled_groups.items()},
            "lacks": [_label(c) for c in thin],
            # Which filters mean anything here. A value slider on a season with no
            # valuations would filter nobody while looking like it worked.
            "valuesAsOf": (str(pool["latest_value_date"].dropna().max())
                           if "latest_value_date" in pool.columns
                           and pool["latest_value_date"].notna().any() else None),
            "has": {
                "age": count("age") > 0, "value": count("market_value_eur") > 0,
                "price": count("price_m") > 0, "height": count("height_cm") > 0,
                "foot": count("foot") > 0,
            },
        },
        "players": players,
    }


def _prune_empty_metrics(groups: dict, players: list) -> None:
    """Drop display metrics that no player in their group actually has."""
    for group, spec in groups.items():
        members = [p for p in players if p["g"] == group]
        if not members:
            continue
        keep = [
            i for i in range(len(spec["d"]))
            if any(p["d"][i][0] is not None for p in members if i < len(p["d"]))
        ]
        if len(keep) == len(spec["d"]):
            continue
        spec["d"] = [spec["d"][i] for i in keep]
        for player in members:
            player["d"] = [player["d"][i] for i in keep if i < len(player["d"])]


def build_source(spec: dict, only_seasons: set[str] | None, progress=print) -> dict[str, dict]:
    """Every complete season of one source, fitted and exported."""
    features, report, used = build_features(spec["key"])
    if used != spec["key"]:
        progress(f"  {spec['key']} is not built - skipped")
        return {}
    seasons = [s for s in sorted(features["season"].unique()) if s not in spec["skip"]]
    if only_seasons:
        seasons = [s for s in seasons if s in only_seasons]

    payloads: dict[str, dict] = {}
    for season in seasons:
        rows = features[features["season"] == season]
        in_progress = rows["minutes"].max() < IN_PROGRESS_MAX_MINUTES
        floors = {}
        source_rows = features
        if in_progress:
            for league, block in rows.groupby("league"):
                floor = int(block["minutes"].max() * IN_PROGRESS_SHARE // 90 * 90)
                floors[league] = max(IN_PROGRESS_MIN_FLOOR, floor)
            keep = (features["season"] != season) | (
                features["minutes"] >= features["league"].map(floors).fillna(MIN_MINUTES))
            source_rows = features[keep]
        platform = build_platform(source_rows, report, source=used, seasons=[season],
                                  min_minutes=min(floors.values()) if floors else MIN_MINUTES)
        if platform.pool.empty or not platform.models:
            progress(f"  {season}: nothing to model - skipped")
            continue
        payloads[season] = export(platform, season)
        if in_progress:
            payloads[season]["meta"].update({
                "inProgress": True, "floors": floors,
                "asOf": _scraped_on(spec["key"]),
                "rounds": {lg: int(b["matches"].max()) for lg, b in rows.groupby("league")
                           if "matches" in b and b["matches"].notna().any()},
            })
        progress(f"  {season}: {len(payloads[season]['players']):,} players"
                 + (f" (in progress, floors {floors})" if in_progress else ""))

    # Which other seasons each player appears in, so the page can follow him.
    appears: dict[str, list[str]] = defaultdict(list)
    for season, payload in payloads.items():
        for player in payload["players"]:
            appears[player["id"]].append(season)
    for season, payload in payloads.items():
        for player in payload["players"]:
            player["os"] = [s for s in appears[player["id"]] if s != season]
    return payloads


def _scraped_on(source: str) -> str | None:
    """The day the season in progress was last fetched, for the page to state."""
    meta = RAW_DIR / "live" / "refresh_meta.json"
    if source != "understat_big6" or not meta.exists():
        return None
    stamp = json.loads(meta.read_text()).get("understat", {}).get("fetched_at")
    return str(stamp)[:10] if stamp else None


def all_seasons(payloads: dict[str, dict]) -> dict:
    """Every player-season of a source in one compact file, for comparing across years.

    Only what a cross-season search needs: who, where, when, and the z-vector
    with its feature names and weights. Each z is standardised inside its own
    season, so comparing two seasons compares each player *against his own
    peers* - which is what makes a 2018 season and a 2025 one comparable at
    all. The page aligns two vectors on the metrics both seasons measured.
    """
    seasons = sorted(payloads)
    groups = {s: {g: {"f": spec["f"], "r": spec["r"]} for g, spec in payloads[s]["meta"]["groups"].items()}
              for s in seasons}
    rows = []
    for i, season in enumerate(seasons):
        for p in payloads[season]["players"]:
            if p["z"]:
                rows.append([p["id"], p["n"], p["t"], p["l"], i, p["g"], p["a"], p["m"], p["v"], p["z"],
                             p["h"], p["ft"], p["fl"]])
    return {"seasons": seasons, "groups": groups, "players": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="*", default=None,
                        help="limit to these sources (default: every site source)")
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="limit to these seasons, e.g. 2023-24")
    parser.add_argument("--out", type=Path, default=SITE_DIR)
    args = parser.parse_args()

    wanted = [s for s in SITE_SOURCES if not args.sources or s["key"] in args.sources]
    only = set(args.seasons) if args.seasons else None
    data_dir = args.out / "data"
    # Everything under data/ is generated here, so a stale season from a
    # previous build must not linger for the page to find.
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    catalog, inline = [], {}
    for spec in wanted:
        print(f"building {spec['name']} ({spec['key']}) …")
        payloads = build_source(spec, only)
        if not payloads:
            continue
        entries = []
        for season, payload in payloads.items():
            key = f"{spec['key']}:{season}"
            payload["meta"].update({
                "key": key, "source": spec["key"],
                "label": f"{spec['name']} {season.replace('-', '/')}",
                "attribution": DATA_SOURCES[spec["key"]].attribution,
            })
            path = f"data/{spec['key']}__{season}.json"
            (args.out / path).write_text(
                json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
            entries.append({
                "season": season, "file": path, "players": len(payload["players"]),
                "lacks": len(payload["meta"]["lacks"]),
            })
            if payload["meta"].get("inProgress"):
                entries[-1]["inProgress"] = True
            if key == DEFAULT_DATASET:
                inline[key] = payload
        everything = f"data/{spec['key']}__all.json"
        (args.out / everything).write_text(json.dumps(
            all_seasons(payloads), separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        scraped = _scraped_on(spec["key"])
        updated = spec.get("updated") or (
            f"Season stats: Understat - 2025/26 on fetched from understat.com on {scraped}"
            if scraped else "Season stats: Understat")
        catalog.append({"key": spec["key"], "name": spec["name"], "via": spec["via"],
                        "blurb": spec["blurb"], "updated": updated,
                        "attribution": DATA_SOURCES[spec["key"]].attribution,
                        "seasons": entries, "all": everything})

    if not catalog:
        print("nothing built")
        return 1
    default = DEFAULT_DATASET if inline else f"{catalog[0]['key']}:{catalog[0]['seasons'][-1]['season']}"
    if not inline:
        source, _, season = default.partition(":")
        path = args.out / f"data/{source}__{season}.json"
        inline[default] = json.loads(path.read_text(encoding="utf-8"))

    data = {"catalog": catalog, "default": default, "inline": inline}
    head = (SITE_DIR / "_head.html").read_text(encoding="utf-8")
    body = (SITE_DIR / "_body.html").read_text(encoding="utf-8")
    payload = "<script>window.__SCOUT__=" + json.dumps(
        data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/") + ";</script>"

    # Standalone: the page carries its own charset, which a host would otherwise
    # supply. Without it every euro sign renders as mojibake.
    page = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"{head}\n</head>\n<body>\n{payload}\n{body}\n</body>\n</html>\n"
    )
    out = args.out / "index.html"
    out.write_text(page, encoding="utf-8")

    files = sorted(f for f in data_dir.glob("*.json") if not f.name.endswith("__all.json"))
    total = sum(e["players"] for c in catalog for e in c["seasons"])
    size = sum(f.stat().st_size for f in files) / 1e6
    print(f"\n{len(files)} seasons across {len(catalog)} sources, {total:,} player-seasons")
    print(f"  {out} ({out.stat().st_size / 1e6:.2f} MB, default {default} inlined)")
    print(f"  {data_dir}/ ({size:.1f} MB, loaded on demand)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

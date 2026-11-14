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
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# One maths kernel and one thread on every machine, set before numpy loads.
# OpenBLAS picks its kernels by processor, and a K-Means fit on a laptop's Intel
# chip and on GitHub's AMD runners drifted far enough apart to give 1,542
# player-seasons a different playing style. Haswell's kernels run on both.
for _var, _value in (("OPENBLAS_CORETYPE", "Haswell"), ("OPENBLAS_NUM_THREADS", "1"),
                     ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_var, _value)


def _cpu_flags() -> set[str]:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as info:
            for line in info:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


# numpy chooses its own vector code by processor as well - AVX-512 on a recent
# Intel Xeon, AVX2 on GitHub's AMD runners - and a sum's last bits differ between
# them, enough to tip a near-tie in the choice of how many archetypes a position
# has. Where the processor can run it, numpy's AVX2 level (X86_V3) is used.
if {"avx2", "fma", "bmi1", "bmi2", "f16c", "movbe"} <= _cpu_flags() and not (
        os.environ.get("NPY_DISABLE_CPU_FEATURES") or os.environ.get("NPY_ENABLE_CPU_FEATURES")):
    os.environ["NPY_ENABLE_CPU_FEATURES"] = "X86_V3"

import numpy as np  # noqa: E402
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    COUNTING_STATS, DATA_SOURCES, LOWER_IS_BETTER, METRIC_LABELS, PERCENT_METRICS,
    DEFAULT_WEIGHTS, POSITION_GROUP_NAMES, RAW_DIR, ROOT_DIR, TEAM_CONTEXT_METRICS,
    categories_for,
)
from src.stat_sheet import (BUCKETS, DESCRIPTIONS, HEADLINE, NOT_RECORDED, ORDER, ROLE_NAMES, ROLE_OF, STYLE_CATEGORIES,
                            TEAM_READINGS, headline_for, order_for, role_sheet, sheet_for)  # noqa: E402
from src.feature_engineering import category_scores, compute_percentiles, metrics_for_percentiles  # noqa: E402
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
                 "xG, xA, xGChain, xGBuildup. Each player's stat sheet borrows his defending, passing and "
                 "keeping from the sources that record them, or says it is not possible. "
                 "Players are grouped as defenders, midfielders and forwards; the stat sheet also "
                 "ranks each against his exact Transfermarkt position, such as defensive midfielders.",
        # Said only once the keepers' matches have been fetched and modelled.
        "keepers": "Keepers are read from the matches they played in full: xG faced against goals "
                   "conceded, and clean sheets.",
        "no_keepers": "Goalkeepers have no model.",
        # What this source's sheet cannot hold, said under the category.
        "notes": {"Goalkeeping": "Understat records no saves or shots on target. These are the matches he played "
                                 "in full: the xG his team faced and the goals it let in, so they read the "
                                 "defence as well as him."},
    },
    {
        "key": "fbref_big5", "name": "Big five, in depth", "via": "FBref",
        "skip": {"2025-26"},
        "stops": "its public mirror of FBref is no longer updated",
        "updated": "Season stats: FBref, via a public mirror last updated 18 Sep 2025",
        "blurb": "The deepest: up to {metrics} metrics and ten specific positions, 2017/18 to 2024/25, "
                 "priced by Transfermarkt. 2023/24 is the last season every block measured in "
                 "full - the public mirror of FBref stopped in September 2025.",
    },
    {
        "key": "premier_league", "name": "Premier League", "via": "FPL + Understat",
        "skip": set(),
        "updated": "Season stats: Fantasy Premier League, every season to the one being played",
        "blurb": "Every season from 2016/17, the one being played included: FPL's ICT and defensive-contribution numbers "
                 "with Understat's xG, real Transfermarkt prices beside the fantasy price, and keepers' "
                 "xG faced against goals conceded. One league and a summary feed - no progressive passes, "
                 "duels or pass completion; tackles and recoveries from 2025/26. The stat sheet also ranks "
                 "each player against his exact Transfermarkt position.",
    },
    {
        "key": "statsbomb_2015_16", "name": "Event data", "via": "StatsBomb 2015/16",
        "skip": set(),
        "stops": "the one men's season StatsBomb has published openly for whole leagues",
        "updated": "Season stats: derived from StatsBomb's open event data, a complete season",
        "blurb": "The Premier League, La Liga, Serie A and Ligue 1 in 2015/16, every metric derived "
                 "from the raw events - the only tackles, pressures and aerial duels here before "
                 "2017/18. One season, so every metric weighs the same; Transfermarkt ages and "
                 "prices joined on.",
    },
]
SITE_SOURCES.append({
    "key": "statsbomb_tournaments", "name": "Tournaments", "via": "StatsBomb, 2018-2024",
    "skip": set(), "floor": 270, "squad": 180,
    "stops": "the tournaments StatsBomb has published openly",
    "updated": "Match events: StatsBomb's open data, every match of each tournament",
    "blurb": "The 2018 and 2022 World Cups, Euro 2020 and Euro 2024, the 2024 Copa America and the 2023 "
             "Africa Cup of Nations, every metric derived from the raw events: tackles by third, "
             "pressing and counter-pressing, ball won back in the opponent's half, dribblers stopped. "
             "Three to seven matches a player, so regulars are those with 270 minutes. Each tournament "
             "is filed under the club season it closes or interrupts.",
})
DEFAULT_DATASET = "understat_big6:2025-26"

# A season still being played cannot use the 900-minute floor - nobody has
# played it yet. Each league's floor is half the minutes its busiest player has
# had so far (never under two full games), so a league four rounds in and one
# eight rounds in are both measured on regular starters.
IN_PROGRESS_MAX_MINUTES = 1800
IN_PROGRESS_SHARE = 0.5
IN_PROGRESS_MIN_FLOOR = 180

# Squad players - short of the floor but with five full games behind them (three
# in a season being played) - are exported too, scored against the regulars:
# every model, percentile and count is fitted on the regulars alone, and a squad
# player is placed on that scale without moving it. The page lists regulars
# until a reader lowers the minutes slider, and tags a squad player wherever he
# appears, because a rate over 450 minutes swings more than one over 900.
SQUAD_MINUTES = 450
SQUAD_MINUTES_IN_PROGRESS = 270


def squad_players(features: pd.DataFrame, platform, season: str, floors: dict | None,
                  squad_floor: int | None = None) -> pd.DataFrame:
    """This season's players between the squad floor and the regulars' floor."""
    rows = features[features["season"] == season]
    if floors:
        regular = rows["league"].map(floors).fillna(MIN_MINUTES)
        squad = rows[(rows["minutes"] >= (squad_floor or SQUAD_MINUTES_IN_PROGRESS)) & (rows["minutes"] < regular)]
    else:
        squad = rows[(rows["minutes"] >= SQUAD_MINUTES) & (rows["minutes"] < platform.min_minutes)]
    squad = squad[squad["league"].isin(set(platform.pool["league"]))].copy()
    # A group folded into its parent for this pool is folded for its squad players too.
    parent = {g: info["parent"] for g, info in platform.collapsed_groups.items()}
    squad["position_group"] = squad["position_group"].map(lambda g: parent.get(g, g))
    return squad


def _peers(pool: pd.DataFrame) -> list[str]:
    """The pipeline's peer group: position, and gender where a pool mixes them."""
    return ["position_group"] + (["gender"] if "gender" in pool.columns and pool["gender"].nunique() > 1 else [])


def place_among(pool: pd.DataFrame, squad: pd.DataFrame, metrics: list[str], peers: list[str]) -> pd.DataFrame:
    """Where each squad player's figure would rank among the regulars sharing his peer columns.

    The same average-rank rule the regulars are ranked by, as if he were one
    more of them, flipped where lower is better - without moving the regulars.
    """
    columns: dict[str, pd.Series] = {}
    for metric in metrics:
        if metric not in squad.columns or metric not in pool.columns:
            continue
        out = pd.Series(np.nan, index=squad.index)
        for key, block in squad.groupby(peers):
            key = key if isinstance(key, tuple) else (key,)
            mask = np.logical_and.reduce([pool[c].eq(k) for c, k in zip(peers, key)])
            ref = np.sort(pool.loc[mask, metric].astype(float).dropna().to_numpy())
            values = block[metric].astype(float).to_numpy()
            if not len(ref):
                continue
            below = np.searchsorted(ref, values, side="left")
            ties = np.searchsorted(ref, values, side="right") - below
            rank = (below + ties / 2 + 1) / (len(ref) + 1) * 100
            if metric in LOWER_IS_BETTER:
                rank = 100 - rank
            rank[np.isnan(values)] = np.nan
            out.loc[block.index] = np.round(rank, 1)
        columns[f"pct_{metric}"] = out
    # Built in one go: a column at a time fragments the frame across ~100 metrics.
    return pd.DataFrame(columns, index=squad.index)


def project_squad(platform, squad: pd.DataFrame):
    """Score squad players on the regulars' scale, leaving the regulars untouched.

    * a percentile is where his figure would rank if he were one more member of
      his position's regulars - the same average-rank rule, flipped where lower
      is better - so a 90 means what it means for everyone else;
    * his z-vector uses the position model's own scaler, with gaps filled by the
      regulars' medians, as theirs were;
    * his playing style is the regulars' nearest archetype.
    """
    pool = platform.pool
    squad = squad.copy()
    squad.index = pd.RangeIndex(len(pool), len(pool) + len(squad))
    pct = place_among(pool, squad, [c[4:] for c in platform.percentiles.columns], _peers(pool))
    categories = category_scores(pct, squad["position_group"])
    z: dict[int, pd.Series] = {}
    squad["archetype"] = None
    for group, block in squad.groupby("position_group"):
        model = platform.models.get(group)
        if model is None:
            continue
        clean = lambda frame: frame[model.features].astype(float).replace([np.inf, -np.inf], np.nan)
        matrix = clean(block).fillna(clean(pool.loc[model.index]).median()).fillna(0.0)
        scaled = pd.DataFrame(model.scaler.transform(matrix), index=block.index, columns=model.features)
        z.update({i: scaled.loc[i] for i in block.index})
        labels = model.clusters.kmeans.predict(scaled)
        names = model.clusters.names
        squad.loc[block.index, "archetype"] = [names.get(int(l)) if hasattr(names, "get") else None
                                               for l in labels]
    return squad, pct, categories, z


def _num(value, places: int = 1):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return None if not np.isfinite(value) else round(value, places)


def _label(column: str) -> str:
    return METRIC_LABELS.get(column) or COUNTING_STATS.get(column) or column


def export(platform, season: str, squad: pd.DataFrame | None = None, notes: dict | None = None) -> dict:
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
    squad_z: dict = {}
    rows = pool
    if squad is not None and len(squad):
        squad, squad_pct, squad_cat, squad_z = project_squad(platform, squad)
        rows = pd.concat([pool, squad])
        percentiles = pd.concat([percentiles, squad_pct])
        categories = pd.concat([categories, squad_cat])

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
            # The same for every displayed metric: a category score is the mean
            # percentile of the displayed metrics in its basket, so a page that
            # combines seasons and re-ranks them can rebuild it exactly.
            "dw": [[i for i, c in enumerate(names) if metric in basket[c]] for metric in display],
            "rw": [int(default.get(c, 0)) for c in names],
            "r": [round(float(w), 3) for w in weights],
            "rr": [model.reliability.get(f) for f in model.features],
        }
        for metric in sorted(set(model.features) | set(display)):
            labels[metric] = _label(metric)

    # Each metric's percentile among his position in his own league as well, at
    # full precision and by the same rule, so the stat sheet can rank him
    # "against wingers in the Premier League" exactly as it ranks him against all.
    shown = sorted({m for spec in groups.values() for m in spec["d"]})
    league_peers = _peers(pool) + ["league"]
    league_pct = compute_percentiles(pool, shown, group_col=league_peers)
    if squad is not None and len(squad):
        league_pct = pd.concat([league_pct, place_among(pool, squad, shown, league_peers)])

    # A four-bucket feed also ranks each player against his exact position,
    # where Transfermarkt records one: a defensive midfielder among defensive
    # midfielders, by the same rule. A player without one has no such
    # percentile, rather than a guessed position.
    role_pct, role = None, pd.Series(np.nan, index=rows.index, dtype=object)
    if set(BUCKETS) & set(groups):
        where = rows["detailed_position"] if "detailed_position" in rows.columns else rows.get("position")
        if where is not None:
            role = where.map(ROLE_OF).where(rows["position_group"].isin(BUCKETS))
            role_peers = _peers(pool) + ["_role"]
            role_pct = compute_percentiles(pool.assign(_role=role.loc[pool.index]), shown, group_col=role_peers)
            if squad is not None and len(squad):
                role_pct = pd.concat([role_pct, place_among(pool.assign(_role=role.loc[pool.index]),
                                                            squad.assign(_role=role.loc[squad.index]),
                                                            shown, role_peers)])

    players = []
    for index in rows.index:
        row = rows.loc[index]
        group = row["position_group"]
        spec = groups.get(group)

        zrow, display = [], []
        if spec is not None:
            model = platform.models[group]
            vector = squad_z[index] if index in squad_z else model.z.iloc[model.z.index.get_loc(index)]
            zrow = [round(float(v), 2) for v in vector]
            mine, league = percentiles.loc[index], league_pct.loc[index]
            exact = role_pct.loc[index] if role_pct is not None else None
            whole = lambda x: None if x is None or pd.isna(x) else round(float(x))
            for metric in spec["d"]:
                value = row.get(metric)
                cell = [_num(value, 2), whole(mine.get(f"pct_{metric}")), whole(league.get(f"pct_{metric}"))]
                if exact is not None:
                    cell.append(whole(exact.get(f"pct_{metric}")))
                display.append(cell)

        players.append({
            "id": str(row["player_id"]),
            "n": row["player"], "t": row["team"], "l": row["league"],
            # His Transfermarkt id: the same player in every source, whatever each spells his name.
            **({"tm": int(row["tm_id"])} if pd.notna(row.get("tm_id")) else {}),
            "p": row["position"], "g": group,
            "fl": row.get("flank") if pd.notna(row.get("flank")) else None,
            "fs": row.get("footed_side") if pd.notna(row.get("footed_side")) else None,
            "ft": row.get("foot") if pd.notna(row.get("foot")) else None,
            "a": _num(row.get("age")), "m": int(row["minutes"]),
            # Context for the profile: nationality, appearances and starts, and
            # the contract Transfermarkt lists today, where the source links him.
            "nt": row.get("nationality") if pd.notna(row.get("nationality")) else None,
            "ap": None if pd.isna(row.get("matches")) else int(row["matches"]),
            "st": None if pd.isna(row.get("starts")) else int(row["starts"]),
            "ce": row.get("contract_expiry") if pd.notna(row.get("contract_expiry")) else None,
            # Transfermarkt's line-up record for the season: every position he started
            # in ("Right Winger:21|Left Winger:6"), or failing that the starts and share
            # of his main one, and games started as captain.
            **({"pp": row["positions_played"]} if isinstance(row.get("positions_played"), str) else
               {"psh": [int(row["position_starts"]), round(100 * float(row["position_share"]))]}
               if pd.notna(row.get("position_starts")) and pd.notna(row.get("position_share")) else {}),
            **({"cp": int(row["captain_starts"])} if pd.notna(row.get("captain_starts")) and row["captain_starts"] > 0 else {}),
            # His exact position in a four-bucket feed, where Transfermarkt has one.
            **({"r": role.loc[index]} if isinstance(role.loc[index], str) else {}),
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
            **({"sq": 1} if index in squad_z or (squad is not None and index in squad.index) else {}),
        })

    # A metric can be listed for a position and still be empty in this season -
    # the Premier League feed has an aerial-duel column with nothing in it. Those
    # would offer a leaderboard that ranks nobody and a comparison row of dashes,
    # so they are pruned here, per group, once the values are known.
    _prune_empty_metrics(groups, players)
    # The stat sheet for each position: its categories in reading order, each
    # with the metrics this season actually has.
    for group, spec in groups.items():
        spec["s"] = sheet_for(group, spec["d"])
        spec["k"] = headline_for(group, spec["d"])
        # Each exact position's reading order and headline numbers over the bucket's sheet.
        found = sorted({p["r"] for p in players if p["g"] == group and p.get("r")})
        if found:
            spec["rs"] = {r: role_sheet(r, spec["s"], spec["d"]) for r in found}
        # Every category the position reads in, so the page can name the ones
        # this source does not record at all.
        spec["so"] = order_for(group)

    # What this season did not measure that the source does elsewhere, straight
    # from the cleaning report - so the page can say so instead of leaving a
    # scout to wonder why a metric vanished.
    thin = sorted(c for c, seasons in platform.cleaning.partial_columns.items() if season in seasons)
    count = lambda column: int(pool[column].notna().sum()) if column in pool.columns else 0

    return {
        "meta": {
            "season": season,
            "minMinutes": int(platform.min_minutes),
            "squadPlayers": int(len(squad)) if squad is not None else 0,
            "leagues": sorted(pool["league"].unique()),
            # Folded and unmodelled groups are named on the page too, so they
            # need a name here even though no player in the pool carries them.
            "groupNames": {g: POSITION_GROUP_NAMES.get(g, g) for g in sorted(
                set(pool["position_group"]) | set(platform.collapsed_groups)
                | set(platform.unmodelled_groups))},
            "groups": groups,
            "labels": labels,
            # What each stat-sheet metric tells a scout, and what no open source records.
            "desc": {m: DESCRIPTIONS[m] for m in sorted(labels) if m in DESCRIPTIONS},
            "notRecorded": {**NOT_RECORDED, **(notes or {})},
            # Categories that say how a player plays rather than how well: not averaged.
            "style": list(STYLE_CATEGORIES),
            # Each position's reading order, so the page knows which category leads
            # even when this source does not record it.
            "orders": ORDER,
            # Each position's headline numbers in full, so the page can fill them from
            # another source's rows where this one lacks the position's lead category.
            "headline": HEADLINE,
            # Exact positions in a four-bucket feed: their names and how many regulars each has.
            "roleNames": ROLE_NAMES if role_pct is not None else {},
            "roleCounts": ({g: {r: int(n) for r, n in sorted(block.value_counts().items())}
                            for g, block in role.loc[pool.index].dropna().groupby(pool["position_group"])}
                           if role_pct is not None else {}),
            # Rendering hints: a percentage needs a % sign, and for a
            # lower-is-better metric the smaller number is the better one.
            "pct": sorted(PERCENT_METRICS & set(labels)),
            "lower": sorted(LOWER_IS_BETTER & set(labels)),
            "club": sorted(TEAM_CONTEXT_METRICS & set(labels)),
            # The stat sheet's team readings: shown, tagged, kept out of category averages.
            "team": sorted(TEAM_READINGS & set(labels)),
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


def _sheet_categories(meta: dict) -> list[str]:
    """Every stat-sheet category a season records for any position."""
    return sorted({category for group in meta["groups"].values() for category, _ in group.get("s", [])})


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
        # The category membership of each displayed metric goes with it.
        spec["dw"] = [spec["dw"][i] for i in keep]
        for player in members:
            player["d"] = [player["d"][i] for i in keep if i < len(player["d"])]


def link_transfermarkt(features: pd.DataFrame, progress=print) -> pd.DataFrame:
    """Each player's Transfermarkt id, where the source table does not carry one.

    The same matcher the Understat build uses (name, league-season and club,
    `tm_history.link_by_season`), run on any source: it is what lets the page
    find one player in every source when each spells him differently -
    "Martin Odegaard" in one, "Martin Ødegaard" in another.
    """
    from src import tm_history

    if "tm_id" in features.columns and features["tm_id"].notna().any():
        return features
    frame = features[["player_id", "player", "season", "league", "team", "minutes"]].copy()
    frame["player_id"] = frame["player_id"].astype(str)
    frame["key"] = frame["player"].map(tm_history.normalise_name)
    link = tm_history.link_by_season(frame)
    features = features.copy()
    features["tm_id"] = features["player_id"].astype(str).map(link)
    progress(f"  Transfermarkt ids for {features['tm_id'].notna().mean():.1%} of player-seasons")
    return features


def build_source(spec: dict, only_seasons: set[str] | None, progress=print) -> dict[str, dict]:
    """Every complete season of one source, fitted and exported."""
    features, report, used = build_features(spec["key"])
    if used != spec["key"]:
        progress(f"  {spec['key']} is not built - skipped")
        return {}
    features = link_transfermarkt(features, progress)
    seasons = [s for s in sorted(features["season"].unique()) if s not in spec["skip"]]
    if only_seasons:
        seasons = [s for s in seasons if s in only_seasons]

    payloads: dict[str, dict] = {}
    for season in seasons:
        rows = features[features["season"] == season]
        fixed = spec.get("floor")
        in_progress = not fixed and rows["minutes"].max() < IN_PROGRESS_MAX_MINUTES
        floors = {}
        source_rows = features
        if fixed:
            # A tournament is three to seven matches: its regulars are those with
            # three full matches, the same floor for every tournament.
            floors = {league: fixed for league in rows["league"].unique()}
            keep = (features["season"] != season) | (features["minutes"] >= fixed)
            source_rows = features[keep]
        elif in_progress:
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
        squad_floor = spec.get("squad") if fixed else SQUAD_MINUTES_IN_PROGRESS if in_progress else SQUAD_MINUTES
        squad = squad_players(features, platform, season, floors if (in_progress or fixed) else None, squad_floor)
        payloads[season] = export(platform, season, squad, notes=spec.get("notes"))
        payloads[season]["meta"]["squadMinutes"] = squad_floor if len(squad) else None
        if fixed:
            payloads[season]["meta"]["tournament"] = True
        if in_progress:
            payloads[season]["meta"].update({
                "inProgress": True, "floors": floors,
                "asOf": _scraped_on(spec["key"]),
                "rounds": {lg: int(b["matches"].max()) for lg, b in rows.groupby("league")
                           if "matches" in b and b["matches"].notna().any()},
            })
        progress(f"  {season}: {len(payloads[season]['players']):,} players, "
                 f"{payloads[season]['meta']['squadPlayers']:,} of them squad players"
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


def _published_catalog(site: Path) -> dict[str, dict]:
    """The catalogue entries of the page already built, keyed by source."""
    page = site / "index.html"
    if not page.exists():
        return {}
    import re

    match = re.search(r"<script>window\.__SCOUT__=(.*?);</script>", page.read_text("utf-8"), re.S)
    if not match:
        return {}
    data = json.loads(match.group(1).replace("<\\/", "</"))
    return {c["key"]: c for c in data["catalog"]
            if all((site / s["file"]).exists() for s in c["seasons"])}


def _scraped_on(source: str) -> str | None:
    """The day the season in progress was last fetched, for the page to state."""
    if source == "premier_league":
        meta = RAW_DIR / "live" / "premier_league_meta.json"
        return json.loads(meta.read_text()).get("mirror_updated") if meta.exists() else None
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
                             p["h"], p["ft"], p["fl"], p.get("sq", 0), p.get("tm")])
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
    # Rebuilding some sources keeps the others exactly as they were published:
    # their files stay and their catalogue entries are read back from the page.
    # A full rebuild clears data/ first, so a stale season cannot linger.
    kept = _published_catalog(args.out) if args.sources and not only else {}
    if not kept and data_dir.exists():
        shutil.rmtree(data_dir)
    for spec in wanted:
        for stale in data_dir.glob(f"{spec['key']}__*.json"):
            stale.unlink()
    data_dir.mkdir(parents=True, exist_ok=True)

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
                # Which stat-sheet categories the season records for anyone, so the
                # page only loads a season that can fill a category another lacks.
                "cats": _sheet_categories(payload["meta"]),
            })
            if payload["meta"].get("inProgress"):
                entries[-1]["inProgress"] = True
            if key == DEFAULT_DATASET:
                inline[key] = payload
        everything = f"data/{spec['key']}__all.json"
        (args.out / everything).write_text(json.dumps(
            all_seasons(payloads), separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        scraped = _scraped_on(spec["key"])
        when = (pd.Timestamp(scraped).strftime("%-d %b %Y") if scraped else None)
        updated = spec.get("updated") or (
            f"Season stats: Understat, with 2025/26 onward fetched from understat.com on {when}"
            if when else "Season stats: Understat")
        # The most metrics any position's stat sheet holds in this source, counted from the build.
        deepest = max((sum(len(rows) for _, rows in group["s"])
                       for p in payloads.values() for group in p["meta"]["groups"].values()), default=0)
        blurb = spec["blurb"].replace("{metrics}", str(deepest))
        if "keepers" in spec:
            keepers = any("GK" in p["meta"]["groups"] for p in payloads.values())
            blurb += " " + (spec["keepers"] if keepers else spec["no_keepers"])
        catalog.append({"key": spec["key"], "name": spec["name"], "via": spec["via"],
                        "blurb": blurb, "updated": updated,
                        **({"stops": spec["stops"]} if spec.get("stops") else {}),
                        **({"tournament": True} if spec.get("floor") else {}),
                        "attribution": DATA_SOURCES[spec["key"]].attribution,
                        "seasons": entries, "all": everything})

    if not catalog:
        print("nothing built")
        return 1
    built = {c["key"] for c in catalog}
    order = [s["key"] for s in SITE_SOURCES]
    for key, entry in kept.items():
        if key in built or key not in order:
            continue
        # A source kept from the published page: fill in what its entries lack.
        for season in entry["seasons"]:
            if "cats" not in season and (args.out / season["file"]).exists():
                season["cats"] = _sheet_categories(json.loads((args.out / season["file"]).read_text(encoding="utf-8"))["meta"])
        catalog.append(entry)
    catalog.sort(key=lambda c: order.index(c["key"]))
    source, _, season = DEFAULT_DATASET.partition(":")
    if not inline and (args.out / f"data/{source}__{season}.json").exists():
        inline[DEFAULT_DATASET] = json.loads(
            (args.out / f"data/{source}__{season}.json").read_text(encoding="utf-8"))
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

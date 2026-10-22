"""
Check the static page's similarity numbers against the Python engine's.

The page ships z-vectors and repeatability weights and computes `100 x cosine`
in the browser. This builds the same season with the engine, takes 40 players
per season, and compares the engine's ten closest matches - and their
percentages - with what the page computes for the same pairs in a real browser.

    pip install playwright          # once; plus a Chromium it can drive
    python scripts/export_static.py && python scripts/check_parity.py
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import random
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from export_static import MIN_MINUTES, SITE_DIR  # noqa: E402
from src.pipeline import build_features, build_platform  # noqa: E402

CASES = [("understat_big6", "2025-26"), ("fbref_big5", "2023-24"), ("premier_league", "2024-25")]
SUBJECTS = 40


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def serve(directory: Path) -> str:
    handler = functools.partial(_Quiet, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/index.html"


def main() -> int:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chromium", help="path to a Chromium executable, if Playwright has none")
    args = parser.parse_args()

    url = serve(SITE_DIR)
    gaps, shared, compared = [], 0, 0
    with sync_playwright() as p:
        browser = p.chromium.launch(**({"executable_path": args.chromium} if args.chromium else {}))
        page = browser.new_page()
        for source, season in CASES:
            features, report, used = build_features(source)
            platform = build_platform(features, report, source=used, seasons=[season],
                                      min_minutes=MIN_MINUTES)
            pool = platform.pool
            page.goto(f"{url}#d={source}:{season}", wait_until="networkidle")
            page.wait_for_function(f"typeof dsKey !== 'undefined' && dsKey === '{source}:{season}'")
            on_page = set(page.evaluate("P.map(p => p.id)"))
            candidates = [i for i in pool.index if platform.model_for(i) is not None
                          and str(pool.loc[i, "player_id"]) in on_page]
            for i in random.Random(7).sample(candidates, SUBJECTS):
                engine = platform.similar(i, n=10)
                subject = str(pool.loc[i, "player_id"])
                others = [str(pool.loc[j, "player_id"]) for j in engine.index]
                browser_sims = page.evaluate("""([sid, others]) => {
                    const s = byId.get(sid), scale = weightScale(P[s].g);
                    return others.map(o => byId.has(o) ? pctSim(P[s].z, P[byId.get(o)].z, scale) : null);
                }""", [subject, others])
                page.evaluate("sid => selectSubject(byId.get(sid))", subject)
                page_top = page.evaluate("ranked().slice(0, 10).map(r => P[r[0]].id)")
                shared += len(set(page_top) & set(others))
                compared += len(others)
                gaps += [abs(float(e) - b) for e, b in zip(engine["similarity"], browser_sims)
                         if b is not None]
            print(f"{source} {season}: {len(gaps):,} pairs so far")
        browser.close()

    result = {"pairs": len(gaps), "mean_gap": round(float(np.mean(gaps)), 3),
              "largest_gap": round(float(np.max(gaps)), 3),
              "top_ten_shared": round(shared / compared, 4)}
    print(json.dumps(result))
    return 0 if result["largest_gap"] < 1 and result["top_ten_shared"] > 0.98 else 1


if __name__ == "__main__":
    raise SystemExit(main())

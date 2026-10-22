"""
Open the built site in a real browser and check every part of it works.

The tests cover the Python; this covers the page people actually use. It
serves `static/`, then in Chromium:

- loads every season of every source and checks it has a subject, a ranked
  list and a rating wherever the position is modelled;
- opens every tool - profile search, leaderboard (including value for money),
  explorer (including rating against price), clubs, shortlist - and checks
  each draws something;
- searches every season, compares across seasons, reads the comparison
  summary, the cheaper and younger alternatives, and draws a share card;
- checks phones 360-414px wide never scroll sideways;
- fails on any script error or any request that does not load.

    pip install playwright && python -m playwright install chromium
    python scripts/export_static.py && python scripts/check_ui.py   # or --channel chrome
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_parity import serve  # noqa: E402

SITE = ROOT / "static"
PHONES = (360, 390, 414)
TABS = ("similar", "search", "board", "explore", "clubs", "shortlist")


def main() -> int:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chromium", help="path to a Chromium executable, if Playwright has none")
    parser.add_argument("--channel", help="use an installed browser instead, e.g. 'chrome' on GitHub's runners")
    parser.add_argument("--url", help="check a deployed site instead of the local build")
    args = parser.parse_args()
    url = args.url or serve(SITE)

    problems: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(("ok      " if ok else "PROBLEM ") + what)
        if not ok:
            problems.append(what)

    with sync_playwright() as p:
        browser = p.chromium.launch(**({"executable_path": args.chromium} if args.chromium
                                      else {"channel": args.channel} if args.channel else {}))
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        errors: list[str] = []
        failed: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        # Web fonts are a nicety the page falls back from; everything else must load.
        page.on("requestfailed", lambda r: None if "fonts." in r.url else failed.append(r.url))
        page.on("response", lambda r: failed.append(f"{r.status} {r.url}") if r.status >= 400 else None)

        def open_(fragment: str = "") -> None:
            page.goto("about:blank")
            page.goto(url + fragment, wait_until="networkidle")
            page.wait_for_function("typeof P !== 'undefined' && P && P.length > 0")
            page.wait_for_timeout(300)

        # 1. every season of every source
        open_()
        catalog = page.evaluate("BUNDLE.catalog.map(c => [c.key, c.seasons.map(s => s.season)])")
        for source, seasons in catalog:
            for season in seasons:
                key = f"{source}:{season}"
                open_(f"#d={key}")
                st = page.evaluate("""() => ({key: dsKey, n: P.length, modelled: !!G[P[subject].g],
                    rows: document.querySelectorAll('#matches .row').length,
                    rating: !!document.querySelector('#rating .rt-score b')})""")
                check(st["key"] == key and st["n"] > 0
                      and (not st["modelled"] or (st["rows"] > 0 and st["rating"])),
                      f"{key}: {st['n']} players, {st['rows']} matches, rating {st['rating']}")

        # 2. the default season, every tool
        open_()
        summary = page.locator(".cmp-sum")
        check(summary.count() == 1 and "separates them most" in summary.inner_text(), "comparison summary")
        check(page.locator("#alt .row").count() > 0, "cheaper and younger alternatives")
        page.click("#tab-search"); page.wait_for_timeout(300)
        check(page.locator("#results .row").count() > 0, "profile search")
        page.click("#tab-board"); page.wait_for_timeout(300)
        check(page.locator("#board .row").count() > 0, "leaderboard")
        page.select_option("#bmetric", "__vfm"); page.wait_for_timeout(300)
        check(page.locator("#board .row").count() > 0 and "least squares" in page.locator("#board-note").inner_text(),
              "leaderboard: value for money")
        page.click("#tab-explore"); page.wait_for_timeout(500)
        check(page.locator("#scatter circle.dot").count() > 50, "explorer")
        page.click("#ex-vfm"); page.wait_for_timeout(500)
        check(page.locator("#scatter line.trend").count() == 1, "explorer: rating against price, with its trend line")
        page.click("#tab-clubs"); page.wait_for_timeout(300)
        check(page.locator("#club tbody tr").count() > 5, "clubs")
        page.click("#tab-shortlist"); page.wait_for_timeout(300)
        check(page.locator("#shortlist").inner_text().strip() != "", "shortlist")

        # 3. every season: search, rank, compare across years
        page.click("#tab-similar"); page.wait_for_timeout(300)
        page.fill("#q", "kante"); page.wait_for_timeout(300)
        page.wait_for_function("document.querySelector('#picker [data-openseason]')", timeout=20000)
        check(True, "search reaches other seasons, accents ignored")
        page.keyboard.press("Escape")
        page.locator('#scope [data-scope="all"]').click()
        page.wait_for_function("document.querySelectorAll('#matches .row[data-xm]').length > 0", timeout=20000)
        cross = page.locator("#matches .row[data-xm]").nth(1)
        cross.click(); page.wait_for_timeout(1200)
        check(" vs " in page.locator("#compare-title").inner_text(), "comparison across seasons")

        # 4. the share card is a 1200 x 630 image
        page.click("#cardbtn")
        page.wait_for_function("document.getElementById('card-img').naturalWidth > 0", timeout=15000)
        size = page.evaluate("[document.getElementById('card-img').naturalWidth, document.getElementById('card-img').naturalHeight]")
        check(size == [1200, 630], f"share card {size[0]}x{size[1]}")
        page.click("#card-close")

        # 5. phones never scroll sideways
        for width in PHONES:
            phone = browser.new_page(viewport={"width": width, "height": 800}, device_scale_factor=2)
            phone.on("pageerror", lambda e: errors.append(f"{width}px: {e}"))
            phone.goto(url, wait_until="networkidle")
            phone.wait_for_function("typeof P !== 'undefined' && P && P.length > 0")
            for tab in TABS:
                phone.click(f"#tab-{tab}"); phone.wait_for_timeout(300)
                wide = phone.evaluate("document.documentElement.scrollWidth")
                check(wide <= width, f"{width}px {tab}: page {wide}px wide")
            phone.close()

        check(not errors, f"script errors: {errors[:3]}")
        check(not failed, f"failed requests: {failed[:3]}")
        browser.close()

    print(f"\n{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

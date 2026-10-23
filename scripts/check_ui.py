"""
Open the built site in a real browser and check every part of it works.

The tests cover the Python; this covers the page people actually use. It
serves `static/`, then in Chromium:

- loads every season of every source and checks it has a subject, a ranked
  list and a rating wherever the position is modelled;
- opens every page - Discover with its discovery cards, the head to head (the
  similarity ring, summary, headline tiles and grouped table), Explore's
  rankings (including value for money), charts (including rating against
  price), clubs and profile search, and the shortlist - and checks each draws
  something;
- searches every season (and the head-to-head search, best match first),
  compares across seasons, reads the comparison summary, the cheaper and
  younger alternatives, and draws a share card;
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
EXPLORE = {"search", "board", "explore", "clubs"}


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
                    rating: document.querySelectorAll('#rating .attr').length > 0})""")
                check(st["key"] == key and st["n"] > 0
                      and (not st["modelled"] or (st["rows"] > 0 and st["rating"])),
                      f"{key}: {st['n']} players, {st['rows']} matches, rating {st['rating']}")

        def go(tab: str, on=None) -> None:
            on = on or page
            if tab in EXPLORE:
                on.click("#nav-explore"); on.wait_for_timeout(200)
            on.click(f"#tab-{tab}"); on.wait_for_timeout(300)

        # 2. the default season, every page
        open_()
        check(page.locator("#discover-grid .dcard").count() >= 3, "discovery cards")
        check(page.locator("#vs .sim-ring").count() == 1, "head to head: similarity ring")
        summary = page.locator(".cmp-sum")
        check(summary.count() == 1 and "separates them most" in summary.inner_text(), "comparison summary")
        check(page.locator("#tiles .tile").count() >= 3, "head to head: headline metric tiles")
        check(page.locator("#compare tr.grp").count() >= 2, "head to head: table grouped by category")
        page.fill("#mfilter", "xg"); page.wait_for_timeout(200)
        shown = page.evaluate("[...document.querySelectorAll('#compare tr[data-label]')].filter(r => !r.hidden).length")
        check(0 < shown < page.locator("#compare tr[data-label]").count(), f"head to head: metric search ({shown} rows)")
        page.fill("#mfilter", "")
        # the head-to-head search: an exact name first, rows drawn like the main search, arrow keys
        name = page.evaluate("[...P].filter((p, i) => i !== subject).sort((a, b) => b.m - a.m)[3].n")
        page.fill("#q2", name); page.wait_for_timeout(600)
        first = page.locator("#picker2 .prow").first
        check(first.count() == 1 and name in first.inner_text() and first.locator(".avatar").count() == 1,
              f"compare search puts {name} first")
        page.keyboard.press("ArrowDown")
        check(page.locator('#picker2 .prow[aria-selected="true"]').count() == 1, "compare search: arrow keys move through the results")
        page.keyboard.press("Escape"); page.fill("#q2", "")
        check(page.locator("#alt .row").count() > 0, "cheaper and younger alternatives")
        go("search")
        check(page.locator("#results .row").count() > 0, "profile search")
        go("board")
        check(page.locator("#board .row").count() > 0, "leaderboard")
        page.select_option("#bmetric", "__vfm"); page.wait_for_timeout(300)
        check(page.locator("#board .row").count() > 0 and "least squares" in page.locator("#board-note").inner_text(),
              "leaderboard: value for money")
        go("explore"); page.wait_for_timeout(200)
        check(page.locator("#scatter circle.dot").count() > 50, "explorer")
        page.click("#ex-vfm"); page.wait_for_timeout(500)
        check(page.locator("#scatter line.trend").count() == 1, "explorer: rating against price, with its trend line")
        for colour in ("vfm", "age"):
            page.click(f'#xcolor [data-color="{colour}"]'); page.wait_for_timeout(300)
            painted = page.evaluate("document.querySelectorAll('#scatter circle.dot[style*=\"fill\"]').length")
            check(painted > 50, f"explorer: dots coloured by {colour} ({painted})")
        page.click('#xcolor [data-color="hl"]')
        go("clubs")
        check(page.locator("#club tbody tr").count() > 5, "clubs")
        go("shortlist")
        check(page.locator("#shortlist").inner_text().strip() != "", "shortlist")

        # 3. every season: search, rank, compare across years
        go("similar")
        page.click("#career-btn")
        page.wait_for_function("document.querySelector('#career .trend-wrap svg circle')", timeout=20000)
        check(True, "career: rating by season, drawn")
        page.fill("#q", "kante"); page.wait_for_timeout(300)
        page.wait_for_function("document.querySelector('#picker [data-openseason]')", timeout=20000)
        check(True, "search reaches other seasons, accents ignored")
        page.keyboard.press("ArrowDown"); page.wait_for_timeout(100)
        check(page.locator('#picker .prow[aria-selected="true"]').count() == 1, "search: arrow keys move through the results")
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
                go(tab, phone)
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

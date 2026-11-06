"""
Open the built site in a real browser and check every part of it works.

The tests cover the Python; this covers the page people actually use. It
serves `static/`, then in Chromium:

- loads every season of every source and checks it has a subject, a ranked
  list and a rating wherever the position is modelled;
- opens every page - Players (the start, with its standout lists, and a
  player's page with its Similar, Profile and Seasons tabs), Compare (the
  verdict, summary, headline tiles and grouped table), Rankings (leaderboard
  with value for money, chart with rating against price, find by profile),
  Clubs and the Shortlist - and checks each draws something;
- takes seasons together: the season picker, the combined numbers checked
  against the seasons they come from (minutes add up, per-90 figures are
  minutes-weighted, percentiles re-ranked), and Compare with each player over
  his own seasons;
- searches every season (and Compare's search, best match first), compares
  across seasons, reads the comparison summary, the cheaper and younger
  alternatives, and draws a share card;
- opens a link from before the pages were split, which must still land on the player;
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
PAGES = ("home", "similar", "compare", "board", "explore", "search", "clubs", "shortlist")
RANKINGS = {"board", "explore", "search"}

# What a player's page shows must all be about the player picked: his name in
# the band, only his position group in the lists, the cheaper and younger picks
# drawn from his own closest matches, and no panel left over from the player
# before when his position has no model.
CONSISTENT = """() => {
  const s = P[subject], out = [], shown = id => { const e = document.getElementById(id); return e && !e.hidden; };
  const modelled = !!(G[s.g] && s.z.length);
  const name = document.querySelector('#subject-line .pb-name');
  if (!name || name.textContent.trim() !== s.n) out.push(`profile shows ${name && name.textContent}`);
  for (const r of document.querySelectorAll('#matches .row[data-m], #alt .row[data-m]')){
    const o = P[+r.dataset.m];
    if (o.g !== s.g) out.push(`${o.n} (${o.g}) listed for a ${s.g}`);
  }
  for (const r of document.querySelectorAll('#matches .row[data-xm]')){
    const [key, id] = r.dataset.xm.split('|'), A = ALL[sourceOf(key)];
    const x = A && A.players.find(q => String(q[0]) === id && sourceOf(key) + ':' + A.seasons[q[4]] === key);
    if (x && x[5] !== s.g) out.push(`${x[1]} (${x[5]}, ${key}) listed for a ${s.g}`);
  }
  if (!modelled){
    if (shown('alt-panel') || document.getElementById('alt').textContent.trim()) out.push('cheaper/younger list left over');
  } else if (shown('alt-panel')){
    const own = new Set(ranked().slice(0, ALT_POOL).map(x => x[0]));
    for (const r of document.querySelectorAll('#alt .row[data-m]'))
      if (!own.has(+r.dataset.m)) out.push(`${P[+r.dataset.m].n} is not among his closest matches`);
  }
  return out;
}"""

# Seasons taken together, checked against the seasons themselves: his minutes
# are the sum, a per-90 figure the minutes-weighted mean, and each percentile
# his average rank among the regulars over the same seasons.
COMBINED = """async name => {
  const p = P.find(x => x.n === name), out = {name, problems: []};
  if (!p){ out.problems.push('not found'); return out; }
  const spec = G[p.g], parts = await Promise.all(M.seasons.map(s => getDataset(M.source + ':' + s)));
  const recs = parts.map(d => d.players.find(x => x.id === p.id && x.g === p.g)).filter(Boolean);
  const mins = recs.reduce((t, r) => t + r.m, 0);
  if (mins !== p.m) out.problems.push(`minutes ${p.m}, seasons add to ${mins}`);
  const k = spec.d.find(x => /_per90$/.test(x) && !WEIGHT_BY[x]), j = spec.d.indexOf(k);
  let num = 0, wt = 0;
  recs.forEach((r, i) => { const sp = parts[i].meta.groups[r.g], jj = sp.d.indexOf(k);
    if (jj >= 0 && r.d[jj] && r.d[jj][0] != null){ num += r.d[jj][0] * r.m; wt += r.m; } });
  if (Math.abs(p.d[j][0] - num / wt) > 0.006) out.problems.push(`${k} ${p.d[j][0]}, seasons give ${(num / wt).toFixed(3)}`);
  const ref = P.filter(x => x.g === p.g && !x.sq && x.d[j][0] != null).map(x => x.d[j][0]);
  const lt = ref.filter(v => v < p.d[j][0]).length, eq = ref.filter(v => v === p.d[j][0]).length;
  let pct = (lt + (eq + 1) / 2) / ref.length * 100; if (LOWER.has(k)) pct = 100 - pct;
  if (!p.sq && Math.abs(p.d[j][1] - pct) > 0.5) out.problems.push(`${k} percentile ${p.d[j][1]}, rank gives ${pct.toFixed(1)}`);
  out.metric = k; out.seasons = recs.length;
  return out;
}"""


# The stat sheet: the position's categories in its own order, every figure on
# it the player's own, each category's number the average of the percentiles
# listed under it, and - against his league - a higher figure never ranked
# below a lower one.
SHEET = """() => {
  const p = P[subject], spec = G[p.g], out = [];
  if (!spec || !spec.s) return ['no stat sheet'];
  const cats = [...document.querySelectorAll('#sheet .sheet-cat h3')].map(h => h.textContent);
  const order = sheetOf(p).sheet.map(x => x[0]);
  if (cats.join('|') !== order.join('|')) out.push(`categories ${cats.join(', ')}`);
  if (p.r && spec.rs && spec.rs[p.r] && cats[0] !== spec.rs[p.r].o[0]) out.push(`a ${p.r} should open on ${spec.rs[p.r].o[0]}`);
  for (const sec of document.querySelectorAll('#sheet .sheet-cat')){
    const fig = sec.querySelector('.sc-pct b');
    if (!fig) continue;   // role and style: not averaged
    const nums = [...sec.querySelectorAll('.sr')].filter(r => !r.querySelector('.tag[data-tip^="Reads his team"]'))
      .map(r => r.querySelector('.sr-p').textContent).filter(t => t !== '—').map(Number);
    if (nums.length && Math.round(nums.reduce((a, b) => a + b, 0) / nums.length) !== Number(fig.textContent))
      out.push(`${sec.querySelector('h3').textContent}: ${fig.textContent} is not the average of ${nums.join(', ')}`);
  }
  const scope = document.querySelector('#sheet-peers .on') ? document.querySelector('#sheet-peers .on').dataset.peers : 'all';
  const at = {all: 1, league: 2, role: 3}[scope];
  const peers = P.filter(o => o.g === p.g && !o.sq && (scope !== 'league' || o.l === p.l) && (scope !== 'role' || o.r === p.r));
  for (const k of spec.s.flatMap(x => x[1])){
    const j = spec.d.indexOf(k);
    const xs = peers.map(o => o.d[j]).filter(x => x && x[0] != null && x[at] != null).sort((a, b) => a[0] - b[0]);
    for (let i = 1; i < xs.length; i++) if (xs[i][0] > xs[i - 1][0] && (LOWER.has(k) ? xs[i][at] > xs[i - 1][at] : xs[i][at] < xs[i - 1][at])){
      out.push(`${k}: a higher figure ranks lower`); break; }
  }
  return out;
}"""


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
                open_(f"#d={key}&v=similar")
                st = page.evaluate("""() => ({key: dsKey, n: P.length, modelled: !!G[P[subject].g],
                    rows: document.querySelectorAll('#matches .row').length, rating: ratingOf(P[subject]) != null,
                    hidden: P.filter(p => p.m < F.mins && !p.sq).length})""")
                # a season opens with all its regulars in play: a minutes threshold carried
                # over from a finished season once hid all but 96 of a season in progress
                check(st["key"] == key and st["n"] > 0 and st["hidden"] == 0
                      and (not st["modelled"] or (st["rows"] > 0 and st["rating"])),
                      f"{key}: {st['n']} players ({st['hidden']} hidden), {st['rows']} matches, rating {st['rating']}")

        def go(view: str, on=None) -> None:
            on = on or page
            if view == "home":
                on.click("#brand")
            elif view == "similar":
                on.evaluate("chosen = true"); on.click("#nav-players"); on.wait_for_timeout(200); on.click("#pt-similar")
            elif view in RANKINGS:
                on.click("#nav-rankings"); on.wait_for_timeout(200); on.click(f"#tab-{view}")
            else:
                on.click(f"#nav-{view}")
            on.wait_for_timeout(350)

        def seasons(preset: str, anchor: str = "#seasons-btn") -> None:
            page.click(anchor); page.wait_for_timeout(200)
            page.click(f'#season-presets [data-preset]:has-text("{preset}")')
            page.click("#seasons-apply")

        # a reader switching from the default season to the one being played keeps every player in it
        open_("#v=similar")
        live = page.evaluate("(BUNDLE.catalog.find(c => c.key === sourceOf(dsKey)).seasons.find(s => s.inProgress) || {}).season")
        if live:
            seasons("This season so far")
            page.wait_for_function(f"dsKey.endsWith('{live}')", timeout=20000); page.wait_for_timeout(500)
            hidden = page.evaluate("P.filter(p => p.m < F.mins && !p.sq).length")
            check(hidden == 0, f"switching to {live}: {hidden} players hidden by the minutes filter")

        # 2. the default season, every page
        open_()
        check(page.evaluate("mode") == "home" and page.locator("#discover-grid .dcard").count() >= 3, "the start: search and standout lists")
        # squad players: off the lists by default, all of them in at the slider's floor
        open_("#v=similar")
        squad = page.evaluate("""() => ({n: P.filter(p => p.sq).length, floor: +document.getElementById('mins').min,
            regular: M.minMinutes, shown: P.filter(p => p.sq && p.m >= F.mins).length})""")
        check(squad["n"] > 0 and squad["shown"] == 0 and squad["floor"] < squad["regular"],
              f"squad players: {squad['n']} in the season, none listed by default")
        page.click("#filters-btn"); page.wait_for_timeout(200)
        page.evaluate("""() => { const s = document.getElementById('mins'); s.value = s.min;
            s.dispatchEvent(new Event('input')); }""")
        page.wait_for_timeout(400)
        reach = page.evaluate("P.filter(p => p.m < F.mins).length")
        check(reach == 0, f"squad players: lowering minutes to {squad['floor']} lists all of them ({reach} still hidden)")
        open_("#v=similar")
        check(page.locator("#alt .row").count() > 0, "cheaper and younger alternatives")
        page.click("#pt-profile"); page.wait_for_timeout(300)
        check(page.locator("#rating .attr").count() > 0 and page.locator("#sw .swr").count() > 0, "profile: rating and strengths")
        # the stat sheet, for a winger, a centre-back and a keeper, against all of them and his league
        fbref = [s for k, s in catalog if k == "fbref_big5"]
        if fbref:
            season = "2023-24" if "2023-24" in fbref[0] else fbref[0][-1]
            for who, first in (("Bukayo Saka", "Dribbling and carrying"), ("William Saliba", "Defending"),
                               ("David Raya", "Goalkeeping")):
                open_(f"#d=fbref_big5:{season}&t=profile&p={who}")
                if page.evaluate("P[subject].n") != who:
                    check(False, f"stat sheet: {who} not in Big five {season}")
                    continue
                heads = page.locator("#sheet .sheet-cat h3")
                check(heads.count() >= 4 and heads.first.inner_text() == first,
                      f"stat sheet: {who} opens on {heads.first.inner_text() if heads.count() else 'nothing'}")
                check(page.locator("#sheet-key .sk").count() >= 3, f"stat sheet: {who}'s headline numbers")
                wrong = page.evaluate(SHEET)
                check(not wrong, f"stat sheet: {who} against all: " + ("consistent" if not wrong else "; ".join(wrong[:3])))
                before = page.locator("#sheet .sr-p").all_inner_texts()
                page.click('#sheet-peers [data-peers="league"]'); page.wait_for_timeout(200)
                wrong = page.evaluate(SHEET)
                league = page.evaluate("P[subject].l")
                check(league in page.locator("#sheet-sub").inner_text() and before != page.locator("#sheet .sr-p").all_inner_texts()
                      and not wrong, f"stat sheet: {who} against his league: " + ("consistent" if not wrong else "; ".join(wrong[:3])))
                page.click('#sheet-peers [data-peers="all"]'); page.wait_for_timeout(100)
            # keepers lead on goals prevented and save percentage
            check(page.locator("#sheet-key .sk-n").first.inner_text().startswith("Goals prevented"), "stat sheet: a keeper's first number is goals prevented")
        go("board")
        # the rankings list follows the same categories
        groups = page.locator("#bmetric optgroup")
        check(groups.count() >= 3, f"rankings: metrics grouped by category ({groups.count()} groups)")
        # a four-bucket feed: a defensive midfielder ranked among defensive midfielders,
        # read in a defensive midfielder's order, and the leaderboard narrowed to them
        latest = [s for k, s in catalog if k == "understat_big6"]
        if latest:
            open_(f"#d=understat_big6:{latest[0][-2]}&t=profile")
            dm = page.evaluate("P.findIndex(p => p.r === 'DM' && !p.sq)")
            check(dm >= 0, "exact positions: Top leagues has defensive midfielders")
            if dm >= 0:
                page.evaluate(f"selectSubject({dm}); setTab('profile'); setMode('similar')"); page.wait_for_timeout(300)
                role = page.locator('#sheet-peers [data-peers="role"]')
                check(role.count() == 1 and "Defensive midfielders" in role.inner_text(), "exact positions: the stat sheet offers defensive midfielders")
                role.click(); page.wait_for_timeout(200)
                wrong = page.evaluate(SHEET)
                check("defensive midfielders" in page.locator("#sheet-sub").inner_text() and not wrong,
                      "exact positions: ranked among defensive midfielders: " + ("consistent" if not wrong else "; ".join(wrong[:3])))
                page.click('#sheet-peers [data-peers="all"]')
                page.click("#pt-similar"); page.wait_for_timeout(300)
                page.check("#samerole"); page.wait_for_timeout(300)
                listed = page.evaluate("[...document.querySelectorAll('#matches .row[data-m]')].map(r => P[+r.dataset.m].r)")
                check(listed and all(r == "DM" for r in listed), f"similar players: held to defensive midfielders ({len(listed)} listed)")
                page.uncheck("#samerole"); page.wait_for_timeout(100)
                go("board")
                page.select_option("#bgroup", "MID"); page.wait_for_timeout(200)
                page.select_option("#brole", "DM"); page.wait_for_timeout(300)
                listed = page.evaluate("[...document.querySelectorAll('#board .row[data-s]')].map(r => P[+r.dataset.s].r)")
                check(listed and all(r == "DM" for r in listed), f"rankings: narrowed to defensive midfielders ({len(listed)} listed)")
                page.select_option("#brole", ""); page.wait_for_timeout(100)
        open_("#v=similar")
        page.click("#pt-similar"); page.wait_for_timeout(200)
        # a match opens Compare, with the player and him side by side
        page.locator("#matches .row[data-m]").first.click(); page.wait_for_timeout(600)
        check(page.evaluate("mode") == "compare" and page.locator("#vs .vd-num").count() == 1, "compare: the verdict")
        check(page.locator("#slot-a-body .slot-name").inner_text() == page.evaluate("P[subject].n"), "compare: player A is the player picked")
        summary = page.locator(".cmp-sum")
        check(summary.count() == 1 and "separates them most" in summary.inner_text(), "comparison summary")
        check(page.locator("#tiles .tile").count() >= 3, "compare: headline metric tiles")
        check(page.locator("#compare tr.grp").count() >= 2, "compare: table grouped by category")
        page.fill("#mfilter", "xg"); page.wait_for_timeout(200)
        shown = page.evaluate("[...document.querySelectorAll('#compare tr[data-label]')].filter(r => !r.hidden).length")
        check(0 < shown < page.locator("#compare tr[data-label]").count(), f"compare: metric search ({shown} rows)")
        page.fill("#mfilter", "")
        # Compare's search: an exact name first, rows drawn like the main search, arrow keys
        name = page.evaluate("[...P].filter((p, i) => i !== subject).sort((a, b) => b.m - a.m)[3].n")
        page.fill("#q2", name); page.wait_for_timeout(600)
        first = page.locator("#picker2 .prow").first
        check(first.count() == 1 and name in first.inner_text() and first.locator(".avatar").count() == 1,
              f"compare search puts {name} first")
        page.keyboard.press("ArrowDown")
        check(page.locator('#picker2 .prow[aria-selected="true"]').count() == 1, "compare search: arrow keys move through the results")
        page.keyboard.press("Escape"); page.fill("#q2", "")
        go("search")
        check(page.locator("#results .row").count() > 0, "find by profile")
        go("board")
        check(page.locator("#board .row").count() > 0, "leaderboard")
        page.select_option("#bmetric", "__vfm"); page.wait_for_timeout(300)
        check(page.locator("#board .row").count() > 0 and "least squares" in page.locator("#board-note").inner_text(),
              "leaderboard: value for money")
        go("explore"); page.wait_for_timeout(200)
        check(page.locator("#scatter circle.dot").count() > 50, "chart")
        page.click("#ex-vfm"); page.wait_for_timeout(500)
        check(page.locator("#scatter line.trend").count() == 1, "chart: rating against price, with its trend line")
        for colour in ("vfm", "age"):
            page.click(f'#xcolor [data-color="{colour}"]'); page.wait_for_timeout(300)
            painted = page.evaluate("document.querySelectorAll('#scatter circle.dot[style*=\"fill\"]').length")
            check(painted > 50, f"chart: dots coloured by {colour} ({painted})")
        page.click('#xcolor [data-color="hl"]')
        go("clubs")
        check(page.locator("#club tbody tr").count() > 5, "clubs")
        go("shortlist")
        check(page.locator("#shortlist").inner_text().strip() != "", "shortlist")

        # 3. seasons taken together
        open_("#v=similar")
        seasons("Last 3 seasons")
        page.wait_for_function("isCombined(dsKey) && M && M.combined", timeout=60000); page.wait_for_timeout(500)
        st = page.evaluate("({n: P.length, label: document.getElementById('seasons-v').textContent, rows: document.querySelectorAll('#matches .row').length})")
        check(st["n"] > 0 and " to " in st["label"] and st["rows"] > 0, f"season picker: last 3 together ({st['label']}, {st['n']} players)")
        for name in page.evaluate("P.filter(p => G[p.g] && !p.sq && (p.ss || []).length > 1).sort((a, b) => b.m - a.m).slice(0, 3).map(p => p.n)"):
            res = page.evaluate(COMBINED, name)
            check(not res["problems"], f"combined: {name}, {res.get('seasons')} seasons, {res.get('metric')}: "
                                       + ("adds up" if not res["problems"] else "; ".join(res["problems"])))
        # Compare, each player over his own seasons
        page.locator("#matches .row[data-m]").first.click(); page.wait_for_timeout(600)
        seasons("Latest", "[data-seasons='b']")
        page.wait_for_function("matchRef() && matchRef().key !== dsKey", timeout=30000); page.wait_for_timeout(400)
        check(page.locator("#vs .vd-num").count() == 1 and page.evaluate("matchRef().other"),
              "compare: player B over different seasons from player A")
        a_key = page.evaluate("dsKey")
        seasons("Last 5 seasons", "[data-seasons='a']")
        page.wait_for_function(f"dsKey !== '{a_key}' && M && M.combined", timeout=60000); page.wait_for_timeout(400)
        check(page.evaluate("matchRef() && matchRef().key") != page.evaluate("dsKey"), "compare: changing A's seasons leaves B's alone")

        # 4. every season: search, rank, compare across years
        open_("#v=similar")
        page.click("#pt-seasons")
        page.wait_for_function("document.querySelector('#career .trend-wrap svg circle')", timeout=30000)
        check(True, "seasons tab: rating by season, drawn")
        go("home")
        page.fill("#q", "kante"); page.wait_for_timeout(300)
        page.wait_for_function("document.querySelector('#picker [data-sseason]')", timeout=20000)
        check(True, "search reaches other seasons, accents ignored")
        page.keyboard.press("ArrowDown"); page.wait_for_timeout(100)
        check(page.locator('#picker .prow[aria-selected="true"]').count() == 1, "search: arrow keys move through the results")
        page.keyboard.press("Escape")
        go("similar")
        page.locator('#scope [data-scope="all"]').click()
        page.wait_for_function("document.querySelectorAll('#matches .row[data-xm]').length > 0", timeout=20000)
        # a match from another season than the one in view
        page.locator("#matches .row[data-xm]").filter(has_not=page.locator(f"xpath=self::*[starts-with(@data-xm, '{page.evaluate('dsKey')}|')]")).first.click()
        page.wait_for_timeout(1200)
        check(page.evaluate("mode") == "compare" and page.evaluate("!!(matchRef() && matchRef().other)"), "comparison across seasons")

        # 5. the share card is a 1200 x 630 image
        go("similar")
        page.click("#cardbtn")
        page.wait_for_function("document.getElementById('card-img').naturalWidth > 0", timeout=15000)
        size = page.evaluate("[document.getElementById('card-img').naturalWidth, document.getElementById('card-img').naturalHeight]")
        check(size == [1200, 630], f"share card {size[0]}x{size[1]}")
        page.click("#card-close")

        # 6. a link shared before the pages were split still lands on the player
        who = page.evaluate("P[subject].n")
        open_(f"#d={page.evaluate('dsKey')}&p={who}")
        check(page.evaluate("mode") == "similar" and page.evaluate("P[subject].n") == who, f"an older link opens {who}'s page")

        # 7. every position in turn, modelled or not, in every source: whatever is
        # on the page belongs to the player picked, and only his position is listed
        for source, seasons_ in catalog:
            open_(f"#d={source}:{seasons_[-2] if len(seasons_) > 1 else seasons_[-1]}")
            picks = page.evaluate("""() => { const best = {};
                P.forEach((p, i) => { if (!best[p.g] || p.m > P[best[p.g]].m) best[p.g] = i; });
                return Object.values(best).map(i => P[i].n); }""")
            # a keeper (often unmodelled) between two outfield players, then the rest
            for scope in ("this", "all"):
                for name in picks + picks[:1]:
                    go("home")
                    page.fill("#q", name); page.wait_for_timeout(250)
                    page.keyboard.press("Enter"); page.wait_for_timeout(500)
                    if scope == "all":
                        page.locator('#scope [data-scope="all"]').click()
                        page.wait_for_function("!document.querySelector('#matches .skeleton')", timeout=30000)
                    wrong = page.evaluate(CONSISTENT)
                    check(not wrong, f"{source}: {name} ({scope} season{'s' if scope == 'all' else ''}): "
                                     + ("consistent" if not wrong else "; ".join(wrong[:3])))
                page.locator('#scope [data-scope="season"]').click(); page.wait_for_timeout(300)

        # 8. phones never scroll sideways
        for width in PHONES:
            phone = browser.new_page(viewport={"width": width, "height": 800}, device_scale_factor=2)
            phone.on("pageerror", lambda e: errors.append(f"{width}px: {e}"))
            phone.goto(url + "#v=similar", wait_until="networkidle")
            phone.wait_for_function("typeof P !== 'undefined' && P && P.length > 0")
            for view in PAGES:
                go(view, phone)
                wide = phone.evaluate("document.documentElement.scrollWidth")
                check(wide <= width, f"{width}px {view}: page {wide}px wide")
            phone.close()

        check(not errors, f"script errors: {errors[:3]}")
        check(not failed, f"failed requests: {failed[:3]}")
        browser.close()

    print(f"\n{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Every formula, assumption and limitation, in one place."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import (
    GK_CATEGORIES, LEAGUES, METRIC_LABELS, OUTFIELD_CATEGORIES, POSITION_FEATURES,
    POSITION_GROUP_NAMES, RATIO_METRICS,
)
from src.feature_engineering import PADJ_METRICS, XG_PER_SHOT_PRIOR_SHOTS
from src.ui import eyebrow, note, page_setup, sidebar_filters

page_setup("Methodology", "📖")
platform = sidebar_filters()

st.markdown("# Methodology")
note(
    "If a number appears anywhere in this app, its definition is on this page. Nothing is "
    "computed by a rule that is not written down here."
)

st.markdown("## 1. The data")
st.markdown(
    f"""
The platform runs on any of three datasets, selected in the sidebar. All of them flow through
**identical** cleaning, feature-engineering and modelling code - only ingestion differs. The pool
currently loaded is **{platform.spec.label}**.

The two real sources answer different questions. The Premier League feed is **current and broad**
- ten seasons to 2025/26, with age, price and ownership - but it is a summary feed, so there are
no progressive passes or duels in it. The StatsBomb feed is **deep but older**: every metric is
derived from raw event data, at the cost of the newest men's league season available being
2015/16. Neither is better; they are different instruments.
"""
)

premier, market, real, simulated = st.tabs([
    "Premier League 2016/17-2025/26 (real)",
    "Top six leagues (Transfermarkt)",
    "StatsBomb Open Data (real)",
    "Simulated reference universe",
])

with market:
    st.markdown(
        """
Every player in the **big five plus Liga Portugal** for one season, read through the open-source
[transfermarkt-api](https://github.com/felipeall/transfermarkt-api) service, which wraps
Transfermarkt in a small FastAPI app. `src/transfermarkt.py` walks competitions -> clubs ->
players and, optionally, each player's season statistics.

**What it adds that nothing else here has**

| | |
| --- | --- |
| **Market value in euros** | The only genuine valuation in this project. Everything else - FPL price, league exposure - is a proxy, and labelled as one. |
| **True positions** | "Centre-Back", "Left-Back", "Defensive Midfield" map straight onto this platform's detailed position groups. A fantasy feed knows only four buckets; this knows a full-back from a centre-back. |
| **The top leagues in one pool** | Six competitions in a single comparison pool, so a scout can rank across leagues rather than inside one. |
| **Biography** | Age, height, preferred foot, nationality, contract expiry, and the club a player signed from. |

**What it does not have.** Transfermarkt is a market and biographical database, not a performance
one. Its season statistics are appearances, goals, assists, cards and minutes: there is no xG, no
passing, no defending. Used **alone**, the similarity and archetype models have very few features
to work with and are correspondingly blunt - the app does not pretend otherwise.

**Its best use is enrichment.** Build the squad spine and every other dataset picks it up:

```bash
python scripts/fetch_transfermarkt.py --season 2025 --market-only
```

The Premier League source then gains real market values and real positions - its four buckets
become centre-backs, full-backs, holding midfielders and wingers. The join is on normalised name
plus season; **anything unmatched keeps exactly the position it had**, and the data-quality report
on the Home page states how many players matched. A position group left too small to rank against
is reported there too, rather than its players disappearing.

**Running it.** The service scrapes Transfermarkt, so it needs network access to that site:

```bash
docker run -d -p 8000:8000 --name transfermarkt-api \
    $(docker build -q https://github.com/felipeall/transfermarkt-api.git)
python scripts/fetch_transfermarkt.py --season 2025
```

The maintainer's hosted instance works too, rate limited to about two requests every three
seconds - pass `--rate 0.6`. Be considerate: it is one person's scraper.
"""
    )


with premier:
    st.markdown(
        """
Ten seasons of real Premier League players, through the **completed 2025/26 season**, built by
`src/premier_league.py` from two public feeds mirrored in
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League):

- the **official Fantasy Premier League** season and gameweek exports - minutes, starts, goals,
  assists, cards, saves, clean sheets, the Opta-derived Influence / Creativity / Threat indices,
  the bonus-point score, price and ownership;
- **Understat** per-player match logs - shots, key passes, non-penalty goals and xG, xGChain,
  xGBuildup, and the position a player actually lined up in each match.

**What each season carries.** A summary feed adds metrics over time, and this app never pretends
otherwise: a metric a season did not measure is left missing, not filled with zero, and any
feature missing from the selected pool is dropped from that position's model rather than imputed.

| Seasons | What arrives |
| --- | --- |
| 2016/17 - 2018/19 | Minutes, starts, goals, assists, cards, saves, clean sheets, ICT indices, bonus points, price, ownership |
| 2019/20 - 2021/22 | + Understat shots, key passes, npG, npxG, xA, xGChain, xGBuildup, and line-up positions |
| 2022/23 - 2024/25 | + Opta expected goals, expected assists and expected goals conceded from FPL itself |
| 2025/26 | + tackles, recoveries and clearances-blocks-interceptions, added when the game began scoring Defensive Contribution. Understat's mirror stops after 2024/25, so shots, key passes and xGChain are absent |

Pick a single season in the sidebar and the models use everything that season measured. Pick
several and only the metrics common to all of them survive the coverage check.

**Positions.** FPL publishes four buckets, so players are grouped **GK / DEF / MID / FWD**.
Understat's line-up position is carried as an attribute and can be filtered on, and where a
season predates the mirror the player's most recent known line-up position is carried forward and
labelled as such. It is never used to group: deriving a finer position from the same statistics
the models then read would be circular, and the circularity would show up as structure the data
does not have.

**Club of record** is the club a player played the most minutes for that season, taken from the
gameweek file - the season snapshot names his *current* club, which after a summer window is
somebody else's. Mid-season transfers are counted, not hidden.

**Price** is the fantasy game's own valuation, set by its operator and moved by transfers in and
out. It is a popularity and perceived-value signal - useful, and used for the value-for-money
component of the hidden-gem score - but it is **not a transfer fee or a wage**, and nothing here
is a market valuation.
"""
    )


with real:
    st.markdown(
        """
Real players, real matches, from the **StatsBomb Open Data** event feed
(<https://github.com/statsbomb/open-data>), which is free for public use. `src/statsbomb.py`
downloads the event files and derives the whole player-season schema from raw events - nothing
is taken from a provider's pre-computed summary table, so every definition below can be argued
with, and changed.

Only competition-seasons where the feed covers the **whole league** are ingested. Seasons where
the open data carries a single club - Barcelona's La Liga years, Leverkusen 2023/24 - are
excluded, because every other team would field a phantom squad.

**How each metric is derived**

| Metric | Definition used here |
| --- | --- |
| Minutes | Lineup position spells on the absolute match clock, rescaled so a full match is 90 minutes. StatsBomb's clock runs to ~95 with stoppage, which would depress every per-90 rate by ~5%. Spells are clamped so they cannot overlap, and a spell that starts after the player was substituted off is dropped as a recording artefact. |
| Position | The position the player spent the most minutes in, aggregated across the season. |
| Progressive pass / carry | Wyscout thresholds: the action must end at least **30m** closer to goal when it starts and ends in the player's own half, **15m** when it crosses halfway, **10m** inside the opposition half. |
| Passes into the final third / box | Completed passes ending beyond x=80 / inside the 18-yard box, having started outside it. |
| Long pass | `pass.length` of 30 yards or more. |
| xG | StatsBomb's own `shot.statsbomb_xg`, penalties separated out. |
| xA | Expected assists in the literal sense: the xG of the shot each key pass created, linked through `shot.key_pass_id`. |
| Shot-creating actions | The last two attacking actions - completed pass, completed take-on, foul won - by the shooting team inside the same possession. Goal-creating actions are the same for shots that were scored. |
| Touches | Occasions the player gained the ball: completed receipts, recoveries, interceptions, clearances and blocks. Touches in the box add carries that enter it. |
| Successful pressure | A pressure after which the pressing team is on the ball within five seconds. |
| Error | A miscontrol, dispossession or failed pass followed within five seconds by a shot for the opposition. |
| Aerials | Wins from the `aerial_won` flag on the winner's event; losses from `Duel: Aerial Lost`. |
| Passing under pressure | StatsBomb flags an event `under_pressure` when an opponent is actively closing the player down. Completion is computed on that subset, and the share of a player's passes that are pressed is reported beside it - that share is role and team context, not skill. |
| Open-play vs set-piece xG | Non-penalty xG is split by the shot's `play_pattern`: chances arriving from a corner, free kick, throw-in or keeper distribution are counted as set-piece, the rest as open play. A striker who feeds on corners is a different signing from one who creates from open play. |
| Team possession | Share of playing time in possession, from the gaps between consecutive events. Gaps over a minute are dropped as stoppages and period changes end an interval, so half-time counts for nobody. |
| Goalkeeping | Shots on target faced and goals conceded are read from the *opponent's* shots while that keeper was on the pitch; claims from keeper `Collected`/`Punch`/`Claim`/`Smother`; sweeper actions from keeper events outside the box; launches from keeper passes of 40+ yards. |

**What this feed cannot supply**

- **No birth dates**, so age is unavailable. Every age filter, the age column and the
  age-upside component of the hidden-gem score are switched off rather than filled with a guess.
- **No heights.**
- **No post-shot xG** - that is a paid StatsBomb feature - so goalkeepers are judged on save
  percentage and goals conceded rather than goals prevented. The platform drops any feature a
  source cannot populate for a position group instead of imputing it.
- League-strength coefficients are still assumptions, set in `src/statsbomb.py`.

Rebuild it with `python scripts/fetch_statsbomb.py`; match-level results are cached, so an
interrupted run resumes.
"""
    )

with simulated:
    st.markdown(
        """
Because the StatsBomb feed publishes no ages and no ground truth about playing roles, the
repository also ships a **simulated** universe - 14 leagues over two seasons, with every field
the schema supports.

1. Every player belongs to a **role profile** - "ball-playing centre-back", "poacher",
   "sweeper-keeper" - which defines a mean vector over latent traits (defending, aerial,
   progression, carrying, creation, finishing, pressing, ...).
2. Traits are perturbed per player and per season.
3. Each per-90 rate is `base_rate(position) x exp(loadings . traits)`, so metrics sharing a
   trait are genuinely correlated, as they are in real football.
4. Season **totals are sampled**: counts from a Poisson process over the player's actual
   minutes, success rates from a Binomial over their attempts. A 300-minute player therefore
   has a genuinely noisy per-90 profile.
5. Realistic defects - duplicate rows, missing optional columns, impossible values - are
   injected on purpose, so the cleaning pipeline has real work to do.

**These are not real players**, and nothing about a real footballer follows from them. Their
purpose is that the generative role of every player is *known*, which is what makes the
supervised checks on the Model Validation page possible: measuring whether K-Means and the
nearest-neighbour engine recover real structure is impossible on a feed with no ground truth.

**Bringing your own data.** `src.data_processing.load_external_csv(path, column_map)` reads any
player-season export into the same schema. Required columns are `player`, `position`, `team`,
`league`, `season`, `minutes`; anything else missing is created as `NaN` and the affected
metrics drop out of the models.
"""
    )

st.markdown("## 2. Cleaning rules")
st.markdown(
    """
| Rule | Action |
| --- | --- |
| Exact duplicate rows | Dropped |
| Duplicate player-season | Keep the row with the most minutes |
| Minutes ≤ 0 or > 3,420 | **Row dropped** - a bad minutes value corrupts every per-90 rate |
| Negative counting stat | Row dropped |
| Height outside 150-215 cm, age outside 15-45 | Set to missing, then imputed with the positional median |
| Successes > attempts | Clipped to the attempts |
| Missing advanced metric (xA, xG, PSxG, SCA...) | Imputed at the positional median **per 90**, rescaled to that player's minutes |
| Goalkeeping columns for outfielders | Left as missing, never zero |
| A column the source never supplies (age, height, PSxG) | **Not imputed at all** - reported as unavailable, and the features and controls that depend on it are switched off |
| A counting stat a **season** never measured | Left missing for that season and zero-filled only in the seasons that do measure it. "Made no tackles" and "tackles were not counted" are different claims |
"""
)
if platform.cleaning.unavailable:
    st.info(
        "Unavailable in the loaded dataset: "
        + ", ".join(f"`{c}`" for c in sorted(set(platform.cleaning.unavailable))),
        icon="ℹ️",
    )

st.markdown("## 3. Rates, ratios and adjustments")
st.markdown(
    f"""
**Per 90.** `metric_per90 = season_total ÷ minutes × 90`. Raw totals are never compared.

**Success percentages use empirical-Bayes shrinkage:**

```
rate = (successes + k × positional_pooled_rate) ÷ (attempts + k)
```

`k` is a prior weight expressed in attempts, listed per metric below. A player who won 3 of 3
tackles is pulled back towards the positional average; a player with 90 tackles is not. Without
this, small-sample percentages dominate every ranking.

**Non-penalty xG per shot** uses the same shrinkage with k = {XG_PER_SHOT_PRIOR_SHOTS} shots.
**Pass completion under pressure** uses the same formula on the pressed subset, with a smaller
prior weight because there are fewer of those passes.

**Possession adjustment** (optional columns, prefix `padj_`):

```
padj_x = x_per90 × 50 ÷ (100 − team_possession)
```

i.e. the rate the player would post if his side spent half the match out of possession. It is
applied only to defensive **volume** metrics - never to success percentages - because a defender
in a 65%-possession side simply gets fewer chances to make a tackle.
"""
)
st.dataframe(
    pd.DataFrame(
        [
            {"Metric": label, "Numerator": METRIC_LABELS.get(num, num),
             "Denominator": METRIC_LABELS.get(den, den), "Prior weight k": k}
            for _key, (num, den, k, label) in RATIO_METRICS.items()
        ]
    ),
    hide_index=True, height=390,
)
st.caption(
    "Possession-adjusted metrics available: " + ", ".join(METRIC_LABELS.get(m, m) for m in PADJ_METRICS)
)

st.markdown("## 4. Percentiles")
st.markdown(
    """
Percentiles are **rank-based within a position group, inside the current pool**:

```
percentile = rank(metric within position group) ÷ group size × 100
```

Ranks are used rather than a normal-distribution assumption because most football rates are
right-skewed. Metrics where less is better (goals conceded, miscontrols, errors, fouls) are
inverted, so 99 always means "one of the best in this position".

Changing the minimum-minutes filter changes the pool, and therefore changes every percentile in
the app. That is intentional: a percentile is a statement about a peer group, so the peer group
has to be visible.
"""
)

st.markdown("## 5. Attribute categories")
st.markdown(
    "A category score is the **mean of its metrics' positional percentiles**. Every radar axis, "
    "every recruitment weight and the hidden-gem performance component is one of these."
)
columns = st.columns(2)
with columns[0]:
    eyebrow("Outfield categories")
    st.dataframe(
        pd.DataFrame(
            [
                {"Category": category, "Metrics": ", ".join(METRIC_LABELS.get(m, m) for m in metrics)}
                for category, metrics in OUTFIELD_CATEGORIES.items()
            ]
        ),
        hide_index=True, height=330,
    )
with columns[1]:
    eyebrow("Goalkeeper categories")
    st.dataframe(
        pd.DataFrame(
            [
                {"Category": category, "Metrics": ", ".join(METRIC_LABELS.get(m, m) for m in metrics)}
                for category, metrics in GK_CATEGORIES.items()
            ]
        ),
        hide_index=True, height=330,
    )

st.markdown("## 6. Position-specific models")
st.markdown(
    "There is **no single model comparing every footballer to every other footballer**. Each "
    "position group gets its own feature set, its own scaler, its own similarity index and its "
    "own K-Means fit. A centre-back is never scored on touches in the opposition box."
)
group = st.selectbox(
    "Show the feature set for", list(POSITION_FEATURES),
    format_func=lambda g: f"{g} - {POSITION_GROUP_NAMES[g]}",
)
st.dataframe(
    pd.DataFrame(
        {"Feature": [METRIC_LABELS.get(m, m) for m in POSITION_FEATURES[group]],
         "Column": POSITION_FEATURES[group]}
    ),
    hide_index=True, height=340,
)

st.markdown("## 7. Similarity")
st.markdown(
    """
Features are z-scored **within the position group**, then:

**Cosine (default).** The cosine of the two standardised profile vectors, floored at zero and
shown as a percentage. Because the features are centred on the positional average, this asks
whether two players deviate from their peers *in the same direction* - the same style, whether or
not at the same intensity. That is usually what a scout wants, and it is why a lower-level player
can still read as a close match.

**Euclidean.** The root-mean-square z-difference, converted with

```
similarity % = 100 × (1 − rms_distance ÷ typical_pair_distance)
```

where `typical_pair_distance` is the **median RMS distance between two randomly chosen players in
the same position pool** (printed on the Similar Players page). 100% is an identical profile; 0%
means "as different as two random players in this position". This metric does care about
intensity, so it favours matches at a similar output level.

**Explanations.** For any pair the app reports, per feature, both raw values, both percentiles,
the z-gap, and that feature's share of the squared distance. Matches are features within 0.6 SD
(preferring those where both players are away from average); differences are features more than
0.6 SD apart.
"""
)

st.markdown("## 8. Archetypes")
st.markdown(
    """
K-Means per position group. `k` is chosen as the **largest k whose silhouette stays within 10% of
the best score**, cross-checked against the inertia elbow (computed numerically as the point
furthest from the line joining the first and last k, not eyeballed).

Names are generated from the cluster centroid. Each concept - the attribute categories plus
pressing, passing volume, long passing and crossing - gets a score equal to the mean centroid
z-score of its metrics, with "less is better" metrics sign-flipped. The top one or two concepts
above threshold become the adjectives and the position supplies the noun, producing labels like
*"progressive ball-playing centre-back"*. A cluster with no strength above threshold is named by
its largest deficit instead (*"heavily-worked goalkeeper"*, *"low-creativity winger"*).
"""
)

st.markdown("## 9. Scores")
st.markdown(
    """
**Recruitment fit score**

```
fit = Σ_c (weight_c ÷ 100) × category_percentile_c
```

A fit of 78 means: weighted across the things you said matter, this player sits at the 78th
percentile of his positional peers.

**Hidden gem score** - the weighted mean of five 0-100 components: performance (weighted
positional percentiles), age upside (100 at 19, 0 at 27), low exposure (inverse league-strength
coefficient), statistical uniqueness (percentile of mean distance to the 15 nearest peers) and
sample size (minutes ÷ 1,800, capped). **It is not a valuation** - there is no fee, wage or
contract data in this dataset, so nothing here can say a player is cheap.
"""
)

st.markdown("## 10. Leagues in the loaded pool")
st.markdown(
    "Strength coefficients are **assumptions** - set in `src/statsbomb.py` for the real feed and "
    "`src/config.py` for the simulated one. They are used for filtering and for the hidden-gem "
    "exposure component, and are never silently baked into a per-90 rate or a percentile."
)
st.dataframe(
    platform.league_table().rename(
        columns={"league": "League", "level": "Level", "strength": "Strength coefficient",
                 "players": "Players", "seasons": "Seasons", "clubs": "Clubs"}
    ),
    hide_index=True, height=400,
)

st.markdown("## 11. Known limitations")
st.markdown(
    """
- **Know which dataset you are reading.** The real one describes real footballers in the seasons
  ingested; the simulated one describes nobody. The sidebar and every page banner say which is
  loaded.
- **The real feed has no ages or heights**, so the age-based tools are switched off rather than
  filled in.
- **Small samples.** The minimum-minutes filter is the main defence. Below ~1,500 minutes,
  finishing and success-rate metrics in particular are noisy; the report generator raises this as
  a caveat automatically.
- **Position changes.** A player who switched role mid-season is compared against the peer group
  of his listed position, which will misrepresent him. The dataset carries one position per
  player-season.
- **Team context.** Possession is adjusted for on defensive volume, but team quality, tactical
  system and team-mate finishing are not controlled for. Assists in particular depend on other
  people.
- **League strength.** A coefficient, not a measurement. Cross-league comparisons carry that
  assumption.
- **Correlated features.** Reported on the Model Validation page rather than removed, because to
  a scout two correlated metrics can still be two different questions.
- **Finishing over-performance.** One season of goals-minus-xG is treated as descriptive only;
  it is not evidence of a repeatable finishing skill.
- **Clusters are a summary, not a taxonomy.** Silhouette scores around 0.1-0.2 say plainly that
  playing styles are a continuum.
"""
)

st.markdown("## 12. Project layout")
st.code(
    """
app.py                     Streamlit entry point (Home)
pages/                     One file per page - layout only, no modelling
src/
  config.py                Metric registry, position groups, feature sets, categories, theme
  premier_league.py        Summary-feed ETL: ten Premier League seasons to 2025/26
  transfermarkt.py         Client + ETL for a transfermarkt-api instance, and the
                           market-value / true-position enrichment layer
  statsbomb.py             Event-level ETL for the real StatsBomb Open Data feed
  data_generation.py       Latent-trait simulation of the reference dataset
  data_processing.py       Ingestion (real or simulated), cleaning, pool filtering
  feature_engineering.py   Per-90s, shrunk ratios, possession adjustment, percentiles, scaling
  similarity.py            Cosine / Euclidean nearest neighbours + explanations
  clustering.py            K-Means, k selection, centroid-derived archetype names, PCA
  recruitment.py           Fit scores, hidden-gem composite
  reporting.py             Scouting report generator
  validation.py            Diagnostics and sensitivity tests
  visualisation.py         Plotly builders
  pipeline.py              Orchestration + the ScoutingPlatform object every page reads
  ui.py                    Shared Streamlit helpers
scripts/                   fetch_premier_league.py, fetch_transfermarkt.py,
                           fetch_statsbomb.py, build_dataset.py, validate_models.py
tests/                     Unit tests for the analytics layer
""",
    language="text",
)

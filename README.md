# Football Player Scouting & Recruitment Intelligence Platform

A position-aware scouting and recruitment analytics platform running on **real match data**:
find statistically similar players, turn a recruitment brief into a ranked shortlist, read a
player's strengths and weaknesses against his positional peers, and work out who could replace
him — with the arithmetic behind every number on show.

**4,989 real player-seasons** derived from **2,384 matches** of StatsBomb Open Data: the complete
2015-16 Premier League, La Liga, Serie A and Ligue 1, four FA WSL seasons, two NWSL seasons,
Liga F, the Frauen Bundesliga, Serie A Women and the Indian Super League.

Built with **Python · pandas · NumPy · scikit-learn · Plotly · Streamlit**.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The real dataset is committed, so the app runs immediately — no downloads or API keys needed.

![Player profile](assets/screenshot-player-profile.png)

<details>
<summary>More screenshots</summary>

**Percentile profile — measured against the right peer group**

![Percentiles](assets/screenshot-percentiles.png)

**Squad analysis — Arsenal WFC, with archetypes generated from the data**

![Squad analysis](assets/screenshot-squad.png)

**Replacement finder — style versus quality, on a slider**

![Replacement finder](assets/screenshot-replacements.png)

**Similar players, with the reason behind the match**

![Similar players](assets/screenshot-similar-players.png)

**Recruitment finder — weights and shortlist**

![Recruitment finder](assets/screenshot-recruitment.png)

**Model validation — does the engine recognise the same player twice?**

![Validation](assets/screenshot-validation.png)

**Archetypes — how k was chosen**

![Archetypes](assets/screenshot-archetypes.png)

**Home — pool composition and the cleaning report**

![Home](assets/screenshot-home.png)

</details>

---

## The data: real players, from StatsBomb Open Data

The platform ships with a **real** player-season dataset built from the
[StatsBomb Open Data](https://github.com/statsbomb/open-data) event feed, which is free for
public use. `src/statsbomb.py` downloads the event files and derives the whole schema from raw
events — nothing is taken from a provider's pre-computed summary table, so every definition can
be argued with and changed.

| Competition | Seasons | Matches |
| --- | --- | --- |
| Premier League, La Liga, Serie A, Ligue 1 | 2015-16 (complete) | 1,517 |
| FA WSL | 2018-19, 2019-20, 2020-21, 2023-24 | 457 |
| NWSL | 2018, 2023 | 173 |
| Liga F, Frauen Bundesliga, Serie A Women | 2023-24 | 502 |
| Indian Super League | 2021-22 | 115 |

**2,384 matches → 4,989 player-seasons.** Only competition-seasons where the feed covers the
*whole* league are ingested; seasons carrying a single club (Barcelona's La Liga years,
Leverkusen 2023-24) are excluded, because every other team would field a phantom squad.

### Does it produce the right numbers?

Spot-checks against the historical record, straight out of the pipeline:

| Check | Platform output | Reality |
| --- | --- | --- |
| Premier League 2015-16 top scorer | Harry Kane 25 | Kane 25 (Golden Boot) |
| Next four | Vardy 24, Agüero 24, Lukaku 18, Mahrez 17 | exactly those figures |
| La Liga 2015-16 top scorer | Luis Suárez 40 | Suárez 40 (Pichichi) |
| Most tackles + interceptions, PL | N'Golo Kanté | Kanté led the league |
| Leicester City possession | 44.3% | ~44.8% — the lowest-possession champions in PL history |
| Total minutes, PL 2015-16 | 752,246 | 752,400 minus red cards |

Mahrez's 2015-16 (PFA Player of the Year) comes out at the 98th percentile for take-ons
attempted, 96th for completed, 93rd for xA, 91st for both goals and assists per 90.

### What this feed cannot supply

- **No birth dates**, so age is unavailable. Every age filter, the age column and the
  age-upside component of the hidden-gem score are **switched off** rather than filled with a
  guess. The platform reports what a source cannot supply instead of imputing it.
- **No heights.**
- **No post-shot xG** — a paid StatsBomb feature — so goalkeepers are judged on save percentage
  and goals conceded rather than goals prevented. That feature is dropped from the GK model,
  not zero-filled.

Rebuild it any time:

```bash
python scripts/fetch_statsbomb.py          # ~10 minutes; match results are cached and resumable
```

### The second dataset, and why it exists

The repository also ships a **simulated** universe — 14 leagues over two seasons, with every
field the schema supports including age and height. It is not padding: because each player is
generated from a known role profile, it is the only way to run a *supervised* check on an
unsupervised model. The Model Validation page uses it to measure whether K-Means and the
nearest-neighbour engine recover real structure — which is impossible on a feed with no ground
truth. Its players are invented and describe nobody.

Both datasets flow through **identical** cleaning, feature and modelling code; only ingestion
differs. Switch between them in the sidebar.

### Bringing your own data

```python
from src.data_processing import load_external_csv, clean_players
from src.feature_engineering import build_features

raw = load_external_csv("my_export.csv", column_map={"Gls": "goals", "Ast": "assists"})
clean, report = clean_players(raw)
features = build_features(clean)
```

Required columns: `player`, `position`, `team`, `league`, `season`, `minutes`. Any counting stat
in `src/config.COUNTING_STATS` that is missing is created as `NaN`, and the affected metrics
drop out of the models rather than being imputed.

## What it does

| Page | What it answers |
| --- | --- |
| 🏠 **Home** | Pool composition, the leagues loaded, and the full data-quality report from the cleaning step. |
| 👤 **Player Search** | Filter by position, league, nationality, minutes, archetype and raw metric thresholds. |
| 📊 **Player Profile** | Percentile radar, per-90 read-out, automatic strengths/weaknesses, archetype, season-by-season trajectory, similar players, and a generated scouting report. |
| 🔎 **Similar Players** | Nearest neighbours in the standardised position-specific space, with a feature-by-feature account of *why* — plus filters for a lower-league or younger equivalent. |
| 🆚 **Compare Players** | Two or three players side by side: info, per-90s, percentiles, radar, strengths, similarity. |
| 🎯 **Recruitment Finder** | Hard filters plus 100 points of weight across attribute categories → a ranked shortlist with the fit score broken down. |
| 💎 **Hidden Gems** | A transparent composite of output, minutes, league exposure and statistical rarity. |
| 📋 **Watchlist** | Everything flagged while browsing, with notes, a radar comparison and CSV export. |
| 🏟️ **Squad Analysis** | A club's squad, positional depth against the league, style profile, minutes reliance — and a **replacement finder** that ranks the rest of the pool on a style-versus-quality blend you control. |
| 🧬 **Player Archetypes** | K-Means per position: how *k* was chosen, what defines each cluster, a PCA map, and the most representative players. |
| 🌍 **League Explorer** | Leaderboards, breakout candidates, a scatter workbench and league style profiles. |
| 🔬 **Model Validation** | Cluster quality, self-season recall, team-mate bias, feature dominance and sensitivity tests, run live against the current pool. |
| 📖 **Methodology** | Every formula, assumption and limitation in one place. |

---

## How it works

### Peer groups, chosen deliberately

Every percentile is a claim about where a player stands among players he could actually be
compared with. The peer group is therefore the **position group**, and — on a dataset spanning
men's and women's competitions — the competition type as well. Comparing a Frauen Bundesliga
midfielder's output against Premier League men would not mean anything, so the platform does not
do it. The comparison pool itself (minimum minutes, leagues, seasons) is set in the sidebar and
every page states what it is.

### Position-specific models — not one model for every footballer

There is **no single model comparing every player to every other player.** Each of the eight
position groups (GK, CB, FB, DM, CM, AM, W, FW) gets its own feature set, scaler, similarity index
and K-Means fit. A centre-back is never scored on touches in the opposition box; a winger is never
scored on clearances. Feature sets live in `src/config.POSITION_FEATURES`.

### Rates, not totals

- `metric_per90 = season_total ÷ minutes × 90` — raw totals are never compared.
- Success percentages use **empirical-Bayes shrinkage**:
  `rate = (successes + k × positional_pooled_rate) ÷ (attempts + k)`, so a player who won 3 of 3
  tackles is pulled back towards the positional average while a player with 90 tackles is not.
- **Possession-adjusted** defensive volume: `padj_x = x_per90 × 50 ÷ (100 − team_possession)` —
  applied to volume metrics only, never to success rates.

### Percentiles against peers, always

Percentiles are rank-based **within a position group, inside the current pool**. Metrics where less
is better (goals conceded, miscontrols, errors) are inverted, so 99 always means "one of the best in
this position". Change the minimum-minutes filter and every percentile changes — that is intentional:
a percentile is a statement about a peer group, so the peer group has to be visible.

### Similarity, and why two players are alike

Features are z-scored within the position group, then:

- **Cosine (default)** — the cosine of the two standardised profile vectors. Because the features are
  centred on the positional average, this asks whether two players deviate from their peers *in the
  same direction*: the same style, whether or not at the same intensity. That is why a lower-level
  player can still read as a close match.
- **Euclidean** — the RMS z-difference, converted with
  `similarity % = 100 × (1 − rms ÷ typical_pair_distance)`, where the reference is the **median RMS
  distance between two randomly chosen players in the same position pool** (printed in the app).
  100% is an identical profile; 0% means "as different as two random players in this position".

Every match comes with an explanation: which metrics agree, which diverge, both players' raw values
and percentiles, and each metric's share of the squared distance. If one statistic is carrying a
match, the app shows you.

### Archetypes named by the data

K-Means per position. `k` is chosen as the **largest k whose silhouette stays within 10% of the best
score**, cross-checked against a numerically computed inertia elbow. Silhouette on football style
data almost always peaks at k=2 — styles are a continuum — and taking that argmax gives "two kinds of
winger", which is true and useless.

Cluster **names are generated from the centroid**: each concept scores the mean centroid z of its
metrics (sign-flipped where less is better), the top one or two above threshold become the
adjectives, and the position supplies the noun. A cluster with no strength above threshold is named
by its largest deficit. Real output from a run:

```
CB   Aerially dominant centre-back · High-pressing centre-back ·
     High-volume progressive centre-back
W    Goalscoring penalty-box winger · Crossing dribbling winger ·
     Creative crossing winger · High-pressing ball-winning winger
GK   Sweeper high-volume goalkeeper · Goal-preventing commanding goalkeeper ·
     Long-passing goalkeeper
```

Applied to the real data, Arsenal WFC's squad comes back as Miedema *penalty-box goalscoring
forward*, McCabe *creative high-volume full-back*, Williamson *high-volume progressive
centre-back*, Mead *creative crossing winger* — labels no one typed in.

### Scores

- **Recruitment fit** = `Σ (weight_c ÷ 100) × category_percentile_c`. A fit of 78 means: weighted
  across the things you said matter, this player sits at the 78th percentile of his peers.
- **Replacement score** = `w × similarity% + (1 − w) × role fit percentile`, with `w` on a slider.
  At `w = 1` it is a pure style match; at `w = 0` it is "best player available for the role",
  ignoring whether they play anything like the incumbent. The squad page also reports the fit
  delta against the player being replaced, so an upgrade is visible as a number.
- **Hidden gem** = the weighted mean of five 0–100 components (performance, age upside, low exposure,
  statistical uniqueness, sample size). **It is not a valuation** — the dataset has no fee, wage or
  contract data, so nothing here can say a player is cheap.

---

## Validation

`python scripts/validate_models.py --source statsbomb` writes
[`models/validation_report.md`](models/validation_report.md); `--source simulated` writes the
[simulated equivalent](models/validation_report_simulated.md).

### On real data: does the engine recognise the same player twice?

For every player with two seasons in the pool, where does his **own other season** rank among
his nearest neighbours? It is the one case where the right answer is known without any labels —
which makes it the check that still works on a feed with no ground truth about playing roles.

| Position | Player-seasons tested | Own season in top 10 | Chance | Lift | Median rank |
| --- | --- | --- | --- | --- | --- |
| GK | 76 | 24% | 4.8% | **5.0×** | 19 |
| CB | 100 | 47% | 1.9% | **24.5×** | 13 |
| FB | 76 | 26% | 2.0% | **13.2×** | 46 |
| DM | 24 | 71% | 3.0% | **23.9×** | 4 |
| CM | 24 | 46% | 4.5% | **10.3×** | 17 |
| AM | 14 | 86% | 8.5% | **10.1×** | 4 |
| W | 70 | 33% | 2.4% | **13.4×** | 18 |
| FW | 30 | 33% | 3.4% | **9.9×** | 44 |

The profile the engine builds is stable enough to pick the same footballer out of a 500-player
pool in a different season, with a different squad around him, at 5–25× chance.

The report also checks whether it is **matching on club rather than player** — team style leaks
into individual numbers, since a defender in a possession side passes more because of the side.
Team-mates take 1–4% of top-ten slots against a 0.1–0.6% baseline: a real effect, reported
rather than hidden.

### On simulated data: the supervised check real data cannot give

Because each simulated player is generated from a known role profile, the same models can be
scored against ground truth:

| Position | k | Silhouette | ARI vs true role | Cluster purity | Top-10 same role | Chance | Lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 5 | 0.127 | 0.127 | 0.49 | 0.50 | 0.25 | **2.0×** |
| CB | 5 | 0.120 | 0.248 | 0.61 | 0.60 | 0.25 | **2.4×** |
| FB | 3 | 0.144 | 0.166 | 0.50 | 0.64 | 0.25 | **2.6×** |
| DM | 4 | 0.124 | 0.304 | 0.66 | 0.60 | 0.26 | **2.3×** |
| CM | 4 | 0.151 | 0.210 | 0.58 | 0.62 | 0.25 | **2.4×** |
| AM | 6 | 0.149 | 0.258 | 0.69 | 0.71 | 0.25 | **2.8×** |
| W | 4 | 0.204 | 0.280 | 0.66 | 0.69 | 0.25 | **2.8×** |
| FW | 3 | 0.169 | 0.154 | 0.50 | 0.70 | 0.26 | **2.8×** |

**Reading this honestly.** Silhouette scores of 0.10–0.22 say plainly that playing styles form a
continuum, not well-separated groups; K-Means here is a useful summary of that continuum, not
evidence that discrete player types exist. What matters is the last columns: a player's ten
nearest neighbours share his generative role 2.0–2.8× more often than chance.

### Both datasets

- **Feature dominance** — the mean share of pairwise distance carried by each metric. No metric
  exceeds ~1.3× an even share, so no position's model rests on one statistic.
- **Drop-metric sensitivity** — remove one metric and 77–90% of the top ten survives.
- **Category re-weighting** — triple a category's weight and 60–75% survives: the weights do
  something without the ranking being at their mercy.
- **Correlation analysis** — feature pairs above |r| = 0.85 are reported rather than silently
  dropped, because to a scout two correlated metrics can still be two different questions.

---

## Project layout

```text
app.py                      Streamlit entry point (navigation only)
pages/                      One file per page — layout only, no modelling
src/
  config.py                 Metric registry, position groups, feature sets, categories, theme
  statsbomb.py              Event-level ETL: real data from StatsBomb Open Data
  data_generation.py        Latent-trait simulation of the second dataset
  data_processing.py        Ingestion (real or simulated), cleaning, pool filtering
  feature_engineering.py    Per-90s, shrunk ratios, possession adjustment, percentiles, scaling
  similarity.py             Cosine / Euclidean nearest neighbours + explanations
  clustering.py             K-Means, k selection, centroid-derived archetype names, PCA
  recruitment.py            Fit scores, hidden-gem composite
  reporting.py              Scouting report generator
  validation.py             Diagnostics and sensitivity tests
  visualisation.py          Plotly builders on one validated dark palette
  pipeline.py               Orchestration + the ScoutingPlatform every page reads
  ui.py                     Shared Streamlit helpers
scripts/
  fetch_statsbomb.py        Build the real dataset from the open-data feed
  build_dataset.py          Regenerate the simulated raw + processed data
  validate_models.py        Fit everything and write the validation report
tests/                      52 tests covering the analytics layer and the ETL
data/raw/                   statsbomb_players.csv.gz + players_raw.csv.gz (both committed)
data/processed/             Rebuilt on demand
models/                     validation_report.md (real) and _simulated.md
```

Machine-learning logic is kept entirely out of the Streamlit layer: pages read a cached
`ScoutingPlatform` object and never fit anything themselves.

```bash
python scripts/fetch_statsbomb.py         # rebuild the real dataset from the open-data feed
python scripts/build_dataset.py           # rebuild the simulated one (--seed for a new universe)
python scripts/validate_models.py         # refit and rewrite the validation report
python -m pytest tests/ -q                # 52 tests
```

---

## Known limitations

- **The bundled data is simulated.** Plausible values, invented players. No conclusion about a real
  footballer can be drawn from this app as shipped.
- **Small samples.** The minimum-minutes filter is the main defence. Below ~1,500 minutes, finishing
  and success-rate metrics are noisy; the report generator raises this automatically.
- **Position changes.** A player who switched role mid-season is compared against the peer group of
  his listed position, which will misrepresent him.
- **Team context.** Possession is adjusted for on defensive volume, but team quality, tactical system
  and team-mate finishing are not controlled for — assists in particular depend on other people.
- **League strength** coefficients (`src/config.LEAGUES`) are editable assumptions, not measurements.
  They are used for filtering and for the hidden-gem exposure component, never silently baked into a
  per-90 rate or a percentile.
- **Finishing over-performance.** One season of goals minus xG is treated as descriptive only.
- **No market data.** No fees, wages, contracts or injuries — so no output here is a valuation.

The app draws a hard line between **descriptive statistics** (per-90 rates, percentages,
percentiles — measurements of what happened) and **model outputs** (similarity, archetypes, fit and
hidden-gem scores — constructions that inherit the assumptions above). Every page says which it is
showing.

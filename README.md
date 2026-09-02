# Football Player Scouting & Recruitment Intelligence Platform

A position-aware scouting and recruitment analytics platform: find statistically similar players,
turn a recruitment brief into a ranked shortlist, and read a player's strengths and weaknesses
against his positional peers — with the arithmetic behind every number on show.

Built with **Python · pandas · NumPy · scikit-learn · Plotly · Streamlit**.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dataset builds itself on first run (a few seconds); no downloads or API keys are needed.

---

## Read this first: the data is simulated

Real season data (FBref, Opta, StatsBomb) cannot be redistributed with a public repository, so the
app ships with a **simulated player universe**. The names are invented and **nothing in this app
describes a real footballer.**

It is not random noise. Every player is drawn from a latent-trait model:

1. each player belongs to a **role profile** — "ball-playing centre-back", "poacher",
   "sweeper-keeper" — which defines a mean vector over interpretable traits (defending, aerial,
   progression, carrying, creation, finishing, pressing, …);
2. traits are perturbed per player and per season;
3. each per-90 rate is `base_rate(position) × exp(loadings · traits)`, so metrics that share a
   trait are genuinely correlated, as they are in real football;
4. **season totals are then sampled** — counts from a Poisson process over the player's actual
   minutes, success rates from a Binomial over their attempts.

Step 4 is the important one: a player with 300 minutes has a genuinely noisy per-90 profile, which
is exactly why the minimum-minutes filter matters. Realistic defects — duplicated rows, missing
optional columns, impossible values — are injected on purpose so the cleaning pipeline has real
work to do, and the app reports what it caught.

**Because the generative roles are known, the models can be validated properly** (see
[Validation](#validation)) — a supervised check that is simply not available on real data.

### Using real data instead

```python
from src.data_processing import load_external_csv, clean_players
from src.feature_engineering import build_features

raw = load_external_csv("my_fbref_export.csv", column_map={"Gls": "goals", "Ast": "assists"})
clean, report = clean_players(raw)
features = build_features(clean)
```

Required columns: `player`, `position`, `team`, `league`, `season`, `minutes`, `age`. Any counting
stat listed in `src/config.COUNTING_STATS` that is missing is created as `NaN` and the affected
metrics simply drop out of the models. Every page in the app then works unchanged.

---

## What it does

| Page | What it answers |
| --- | --- |
| 🏠 **Home** | Pool composition and the full data-quality report from the cleaning step. |
| 👤 **Player Search** | Filter by position, league, age, minutes, archetype and raw metric thresholds. |
| 📊 **Player Profile** | Percentile radar, per-90 read-out, automatic strengths/weaknesses, archetype, similar players, and a generated scouting report. |
| 🔎 **Similar Players** | Nearest neighbours in the standardised position-specific space, with a feature-by-feature account of *why* — plus filters for a younger or lower-league equivalent. |
| 🆚 **Compare Players** | Two or three players side by side: info, per-90s, percentiles, radar, strengths, similarity. |
| 🎯 **Recruitment Finder** | Hard filters plus 100 points of weight across attribute categories → a ranked shortlist with the fit score broken down. |
| 🧬 **Player Archetypes** | K-Means per position: how *k* was chosen, what defines each cluster, a PCA map, and the most representative players. |
| 💎 **Hidden Gems** | A transparent composite of output, age, minutes, league exposure and statistical rarity. |
| 🌍 **League Explorer** | Leaderboards, young breakouts, a scatter workbench and league style profiles. |
| 🔬 **Model Validation** | Cluster quality, role-recovery lift, feature dominance and sensitivity tests, run live against the current pool. |
| 📖 **Methodology** | Every formula, assumption and limitation in one place. |

---

## How it works

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
CB   Progressive high-volume centre-back · Aerially dominant centre-back ·
     High-pressing ball-winning centre-back · Long-passing ball-playing centre-back ·
     Ground-based centre-back
W    Dribbling progressive winger · Creative crossing winger ·
     Goalscoring penalty-box winger · Low-creativity winger
GK   Sweeper ball-playing goalkeeper · Shot-stopping goal-preventing goalkeeper ·
     Heavily-worked goalkeeper · High-volume long-passing goalkeeper · Long-passing goalkeeper
```

### Scores

- **Recruitment fit** = `Σ (weight_c ÷ 100) × category_percentile_c`. A fit of 78 means: weighted
  across the things you said matter, this player sits at the 78th percentile of his positional peers.
- **Hidden gem** = the weighted mean of five 0–100 components (performance, age upside, low exposure,
  statistical uniqueness, sample size). **It is not a valuation** — the dataset has no fee, wage or
  contract data, so nothing here can say a player is cheap.

---

## Validation

`python scripts/validate_models.py` writes [`models/validation_report.md`](models/validation_report.md).
Representative output at a 900-minute filter:

| Position | Players | k | Silhouette | ARI vs true role | Cluster purity | Top-10 same role | Chance | Lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 554 | 5 | 0.127 | 0.127 | 0.49 | 0.50 | 0.25 | **2.0×** |
| CB | 1090 | 5 | 0.120 | 0.248 | 0.61 | 0.60 | 0.25 | **2.4×** |
| FB | 813 | 3 | 0.144 | 0.166 | 0.50 | 0.64 | 0.25 | **2.6×** |
| DM | 543 | 4 | 0.124 | 0.304 | 0.66 | 0.60 | 0.26 | **2.3×** |
| CM | 798 | 4 | 0.151 | 0.210 | 0.58 | 0.62 | 0.25 | **2.4×** |
| AM | 552 | 6 | 0.149 | 0.258 | 0.69 | 0.71 | 0.25 | **2.8×** |
| W | 574 | 4 | 0.204 | 0.280 | 0.66 | 0.69 | 0.25 | **2.8×** |
| FW | 580 | 3 | 0.169 | 0.154 | 0.50 | 0.70 | 0.26 | **2.8×** |

**Reading this honestly.** Silhouette scores of 0.12–0.20 say plainly that playing styles form a
continuum, not well-separated groups; K-Means here is a useful summary of that continuum, not
evidence that discrete player types exist. The columns that matter are the last three: a player's ten
nearest neighbours share his generative role **2.0–2.8× more often than chance**, so the similarity
engine is recovering role rather than noise.

The report also covers:

- **Feature dominance** — the mean share of pairwise distance carried by each metric. No metric
  exceeds ~1.3× an even share, so no position's model rests on one statistic.
- **Drop-metric sensitivity** — remove one metric and 77–90% of the top ten survives.
- **Category re-weighting** — triple a category's weight and 60–75% survives, i.e. the weights do
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
  data_generation.py        Latent-trait simulation of the reference dataset
  data_processing.py        Ingestion (simulated or real), cleaning, pool filtering
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
  build_dataset.py          Regenerate raw + processed data
  validate_models.py        Fit everything and write the validation report
tests/                      35 tests covering the analytics layer
data/raw/                   players_raw.csv.gz (committed)
data/processed/             Rebuilt on demand
models/validation_report.md Latest validation output
```

Machine-learning logic is kept entirely out of the Streamlit layer: pages read a cached
`ScoutingPlatform` object and never fit anything themselves.

```bash
python scripts/build_dataset.py      # rebuild the dataset (--seed for a different universe)
python scripts/validate_models.py    # refit and rewrite the validation report
python -m pytest tests/ -q           # 35 tests
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

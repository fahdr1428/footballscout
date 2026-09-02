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
    """
The bundled dataset is **simulated**, because real season data (FBref, Opta, StatsBomb) cannot be
redistributed with a public repository. It is not random noise:

1. Every player belongs to a **role profile** - "ball-playing centre-back", "poacher",
   "sweeper-keeper" and so on - which defines a mean vector over interpretable latent traits
   (defending, aerial, progression, carrying, creation, finishing, pressing, ...).
2. Traits are perturbed per player and per season.
3. Each per-90 rate is `base_rate(position) × exp(loadings · traits)`, so metrics that share a
   trait are genuinely correlated - as they are in real football.
4. Season **totals are sampled**: counts from a Poisson process over the player's actual minutes,
   success rates from a Binomial over their attempts. A player with 300 minutes therefore has a
   genuinely noisy per-90 profile, which is exactly why the minimum-minutes filter matters.
5. Realistic defects are injected on purpose - duplicated rows, missing optional columns,
   impossible values - so the cleaning pipeline has real work to do. The Home page reports what
   it caught.

**Using real data instead.** `src.data_processing.load_external_csv(path, column_map)` reads a
player-season export into the same schema. Required columns are `player`, `position`, `team`,
`league`, `season`, `minutes`, `age`; any counting stat that is missing is created as `NaN` and
the affected metrics simply drop out of the models. Point it at an FBref scrape and every page
in this app works unchanged.
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
"""
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

st.markdown("## 10. League strength coefficients")
st.markdown(
    "These are **assumptions**, editable in `src/config.py`. They are used for filtering and for "
    "the hidden-gem exposure component. They are never silently baked into a per-90 rate or a "
    "percentile."
)
st.dataframe(
    pd.DataFrame(LEAGUES).rename(
        columns={"name": "League", "country": "Country", "tier": "Level",
                 "strength": "Strength coefficient", "teams": "Clubs"}
    ),
    hide_index=True, height=400,
)

st.markdown("## 11. Known limitations")
st.markdown(
    """
- **The bundled data is simulated.** Plausible values, invented players. No conclusion about a
  real footballer can be drawn from this app as shipped.
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
  data_generation.py       Latent-trait simulation of the reference dataset
  data_processing.py       Ingestion (simulated or real), cleaning, pool filtering
  feature_engineering.py   Per-90s, shrunk ratios, possession adjustment, percentiles, scaling
  similarity.py            Cosine / Euclidean nearest neighbours + explanations
  clustering.py            K-Means, k selection, centroid-derived archetype names, PCA
  recruitment.py           Fit scores, hidden-gem composite
  reporting.py             Scouting report generator
  validation.py            Diagnostics and sensitivity tests
  visualisation.py         Plotly builders
  pipeline.py              Orchestration + the ScoutingPlatform object every page reads
  ui.py                    Shared Streamlit helpers
scripts/                   build_dataset.py, validate_models.py
tests/                     Unit tests for the analytics layer
""",
    language="text",
)

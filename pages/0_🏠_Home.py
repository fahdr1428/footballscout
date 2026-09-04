"""
Football Player Scouting & Recruitment Intelligence Platform - home page.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import POSITION_GROUP_NAMES
from src.ui import badges, eyebrow, note, page_setup, sidebar_filters, source_banner, tiles

page_setup("Home", "🏠")
platform = sidebar_filters()

pool = platform.pool

st.markdown(
    """
<div class="sx-hero">
  <div class="sx-eyebrow">Recruitment analytics</div>
  <h1 style="margin:0.1rem 0 0.5rem 0;">Scouting Intelligence</h1>
  <div style="font-size:1.02rem; color:#c3c2b7; max-width:76ch;">
    Find statistically similar players, define a recruitment profile and get a ranked
    shortlist, read a player's strengths and weaknesses against his positional peers, and work
    out who could replace him - with the arithmetic behind every number on show.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")
tiles(
    [
        ("Player-seasons", f"{len(pool):,}", f"min {platform.min_minutes:,} minutes"),
        ("Players", f"{pool['player_id'].nunique():,}", "unique identities"),
        ("Leagues", f"{pool['league'].nunique()}", f"{pool['team'].nunique()} clubs"),
        ("Seasons", f"{pool['season'].nunique()}", ", ".join(platform.seasons)),
        ("Position models", f"{len(platform.models)}", "fitted independently"),
        (
            "Archetypes",
            f"{sum(m.clusters.k for m in platform.models.values())}",
            "generated from cluster centroids",
        ),
    ]
)

source_banner(platform)
for caveat in platform.spec.caveats:
    st.caption(f"- {caveat}")

left, right = st.columns([1.05, 1])

with left:
    st.markdown("## What it answers")
    st.markdown(
        """
- **Who resembles this player?** Nearest neighbours in a standardised, position-specific
  feature space - with a feature-by-feature account of *why*.
- **Which young player resembles an established one?** The same search, filtered by age and
  league level.
- **Who fits our brief?** Hard filters plus a weighted fit score across attribute categories.
- **What is this player good and bad at?** Percentiles against positional peers, never
  against the whole dataset.
- **What kind of player is he?** A K-Means archetype whose label is written from the cluster's
  own centroid.
- **Who is being overlooked?** A transparent composite of output, minutes, league exposure and
  statistical rarity.
- **Who could replace him?** Squad analysis ranks the rest of the pool on a blend of stylistic
  similarity and role quality that you control.
"""
    )

with right:
    st.markdown("## How to use it")
    st.markdown(
        """
1. Set the **minimum minutes** in the sidebar - it defines the pool every number is measured
   against, and the models refit when you change it.
2. **Player Search** to find someone, then **Player Profile** for the full read-out and a
   downloadable scouting report.
3. **Similar Players** for the nearest-neighbour engine and its explanations.
4. **Recruitment Finder** to turn a brief into a shortlist.
5. **Squad Analysis** for a club's make-up, style and replacement options; **Compare Players**,
   **Player Archetypes**, **Hidden Gems** and **League Explorer** for the wider population.
6. Flag anyone with **Add to watchlist**; the **Watchlist** page collects them with notes and a
   CSV export.
7. **Model Validation** for the sensitivity tests, and **Methodology** for every formula.
"""
    )

st.markdown("## Leagues in the pool")
st.dataframe(
    platform.league_table().rename(
        columns={"league": "League", "level": "Level", "strength": "Strength coefficient",
                 "players": "Players", "seasons": "Seasons", "clubs": "Clubs"}
    ),
    hide_index=True, height=320,
)

st.markdown("## Pool composition")
composition = (
    pool.groupby("position_group")
    .agg(
        **{
            "players": ("player_id", "nunique"),
            "seasons": ("player_id", "size"),
            "median_minutes": ("minutes", "median"),
            **({"median_age": ("age", "median")} if platform.has_age else {}),
        }
    )
    .reindex([g for g in POSITION_GROUP_NAMES if g in pool["position_group"].unique()])
    .reset_index()
)
composition["position_group"] = composition["position_group"].map(
    lambda g: f"{g} - {POSITION_GROUP_NAMES[g]}"
)
composition["archetypes"] = [
    platform.models[g].clusters.k if g in platform.models else 0
    for g in pool.groupby("position_group").size().reindex(
        [c.split(" - ")[0] for c in composition["position_group"]]
    ).index
]

st.dataframe(
    composition.rename(
        columns={
            "position_group": "Position group",
            "players": "Players",
            "seasons": "Player-seasons",
            "median_minutes": "Median minutes",
            "median_age": "Median age",  # only present when the source has ages
            "archetypes": "Archetypes (k)",
        }
    ),
    hide_index=True,
)

st.markdown("## Data quality report")
note(
    "Produced by <code>src/data_processing.clean_players</code> on every load. Rows are dropped "
    "only when a value makes the per-90 arithmetic impossible; everything else is repaired or "
    "imputed and counted here."
)
report = pd.DataFrame(platform.cleaning.as_rows(), columns=["Check", "Rows"])
st.dataframe(report, hide_index=True, height=380)
for line in platform.cleaning.notes:
    st.caption(line)
if platform.cleaning.unavailable_columns:
    with st.expander(
        f"{len(platform.cleaning.unavailable_columns)} columns this source does not supply"
    ):
        st.caption(
            "Left missing rather than zero-filled, and dropped from any model that needs them."
        )
        st.code(", ".join(platform.cleaning.unavailable_columns), language="text")
if platform.unmodelled_players:
    st.warning(
        f"**{platform.unmodelled_players} players are in the pool but have no model behind them** - "
        + ", ".join(f"{n} in {g}" for g, n in platform.unmodelled_groups.items())
        + ". A position group needs enough players to rank against before percentiles or "
        "archetypes mean anything. They still appear in search and tables.",
        icon="⚠️",
    )

st.markdown("## Where the models draw the line")
columns = st.columns(3)
with columns[0]:
    eyebrow("Descriptive")
    st.markdown(
        "Per-90 rates, success percentages, percentiles. These are **measurements** of what "
        "happened, subject only to sample size."
    )
with columns[1]:
    eyebrow("Model output")
    st.markdown(
        "Similarity scores, archetypes, fit scores, hidden-gem scores. These are **constructed** "
        "from the measurements and inherit the assumptions listed in Methodology."
    )
with columns[2]:
    eyebrow("Not available")
    st.markdown(
        "Transfer fees, wages, contract length, injury history. Nothing here is a valuation, "
        "and no score should be read as one."
    )

st.write("")
badges(
    [
        ("Python", ""), ("pandas", ""), ("NumPy", ""), ("scikit-learn", ""),
        ("Plotly", ""), ("Streamlit", ""),
    ]
)

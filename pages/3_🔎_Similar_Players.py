"""Nearest-neighbour search with a feature-level explanation of every match."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import LEAGUE_STRENGTH, LEAGUE_TIER
from src.pipeline import format_metric
from src.similarity import explanation_sentences
from src.ui import (
    add_to_watchlist, age_slider, chart, eyebrow, note, page_setup, player_header,
    player_selector, sidebar_filters, similarity_table, watchlist_sidebar,
)
from src.visualisation import contribution_chart, radar_chart

page_setup("Similar Players", "🔎")
platform = sidebar_filters()
pool = platform.pool

st.markdown("# Similar players")
index = player_selector(platform, "Reference player", key="similar")
if index is None:
    st.stop()

row = platform.row(index)
model = platform.model_for(index)
st.divider()
player_header(platform, index)

# ---- search controls -----------------------------------------------------
st.markdown("### Search settings")
controls = st.columns([1.3, 1, 1, 1])
with controls[0]:
    metric_choice = st.radio(
        "Similarity metric", ["cosine", "euclidean"], horizontal=True,
        format_func=lambda m: "Cosine (profile shape)" if m == "cosine" else "Euclidean (shape + level)",
        help="Cosine asks whether two players deviate from the positional average in the same "
             "direction. Euclidean also cares how far they deviate, so it favours players at a "
             "similar output level.",
    )
with controls[1]:
    count = st.slider("Results", 5, 25, 10)
with controls[2]:
    max_age = age_slider(platform, "Age range", (15.0, 40.0), key="sim_age")
    if max_age is None:
        st.caption("No ages in this dataset - filter by league level instead.")
with controls[3]:
    min_minutes = st.slider(
        "Minimum minutes", int(platform.min_minutes), int(pool["minutes"].max()),
        int(platform.min_minutes), 100,
    )

filters = st.columns([1.2, 1.2, 1.6])
with filters[0]:
    lower_leagues_only = st.checkbox(
        "Only lower-ranked leagues",
        help=f"Leagues with a strength coefficient below {row['league']}'s "
             f"({LEAGUE_STRENGTH.get(row['league'], 1.0):.2f}). Those coefficients are editable "
             "assumptions in config.py, not measurements.",
    )
with filters[1]:
    exclude_club = st.checkbox("Exclude his own club", value=True)
with filters[2]:
    league_filter = st.multiselect("Restrict to leagues", sorted(pool["league"].unique()))

candidates = pd.Series(True, index=pool.index)
if max_age is not None:
    candidates &= pool["age"].between(*max_age)
candidates &= pool["minutes"] >= min_minutes
if lower_leagues_only:
    reference_strength = LEAGUE_STRENGTH.get(row["league"], 1.0)
    candidates &= pool["league"].map(LEAGUE_STRENGTH).fillna(1.0) < reference_strength
if exclude_club:
    candidates &= pool["team"] != row["team"]
if league_filter:
    candidates &= pool["league"].isin(league_filter)

results = platform.similar(index, n=int(count), metric=metric_choice, candidate_mask=candidates)

if results.empty:
    st.info("No candidates left after those filters.")
    st.stop()

st.markdown("## Ranked matches")
similarity_table(results)

shortlist = st.multiselect(
    "Add matches to the watchlist", platform.player_labels.loc[results.index].tolist(),
    key="sim_watch",
)
if shortlist and st.button("Add to watchlist"):
    for label in shortlist:
        target = platform.index_for_label(label)
        if target is not None:
            add_to_watchlist(platform, target)
    st.rerun()
watchlist_sidebar()

if metric_choice == "euclidean":
    note(
        f"Euclidean similarity = 100 × (1 − RMS z-distance ÷ {model.engine.reference_distance:.2f}), "
        f"where {model.engine.reference_distance:.2f} is the median RMS distance between two "
        f"randomly chosen {model.group} seasons in this pool. 0% therefore means "
        "'as different as two random players in this position'."
    )
else:
    note(
        "Cosine similarity = the cosine of the two standardised profile vectors, floored at 0. "
        "It measures the direction of deviation from the positional average, so a lower-volume "
        "player with the same shape still scores highly - useful when looking down the leagues."
    )

# ---- explanation ---------------------------------------------------------
st.divider()
st.markdown("## Why these players are similar")
labels = platform.player_labels.loc[results.index]
chosen = st.selectbox("Explain a match", labels.tolist())
other = platform.index_for_label(chosen)

if other is not None:
    other_row = platform.row(other)
    similarity = float(results.loc[other, "similarity"])
    result = platform.explain_similarity(index, other, metric=metric_choice)
    matches, differences = explanation_sentences(result, row["player"], other_row["player"], pool)

    header = st.columns([1, 1, 1])
    with header[0]:
        st.metric("Similarity", f"{similarity:.1f}%")
    with header[1]:
        if platform.has_age:
            st.metric("Age gap", f"{other_row['age'] - row['age']:+.1f} years")
        else:
            st.metric("Minutes", f"{other_row['minutes']:,.0f}", f"{other_row['minutes'] - row['minutes']:+,.0f}")
    with header[2]:
        st.metric(
            "League level",
            f"{LEAGUE_TIER.get(other_row['league'], 1)} vs {LEAGUE_TIER.get(row['league'], 1)}",
        )

    columns = st.columns(2)
    with columns[0]:
        eyebrow("Where they match")
        for sentence in matches:
            st.markdown(f"- {sentence}")
        if not matches:
            st.caption("No metric is within 0.6 standard deviations - this is a weak match.")
    with columns[1]:
        eyebrow("Where they differ")
        for sentence in differences:
            st.markdown(f"- {sentence}")
        if not differences:
            st.caption("No metric differs by more than 0.6 standard deviations.")

    st.markdown("### Radar comparison")
    chart(
        radar_chart(
            [
                (row["player"], platform.category_scores(index)),
                (other_row["player"], platform.category_scores(other)),
            ],
            height=470,
        )
    )

    st.markdown("### What drives the distance between them")
    chart(contribution_chart(result.contributions, top=10))
    note(
        "Share of the squared z-distance carried by each metric. A single metric above ~25% means "
        "the match is being driven by one statistic - read the table below before trusting it."
    )

    table = result.contributions.copy()
    table["Value - " + row["player"]] = [
        format_metric(k, v) for k, v in zip(table["feature"], table["value_a"])
    ]
    table["Value - " + other_row["player"]] = [
        format_metric(k, v) for k, v in zip(table["feature"], table["value_b"])
    ]
    table["Percentile - " + row["player"]] = [
        platform.percentile(index, k) for k in table["feature"]
    ]
    table["Percentile - " + other_row["player"]] = [
        platform.percentile(other, k) for k in table["feature"]
    ]
    table["z gap"] = table["abs_z_gap"].round(2)
    st.dataframe(
        table[
            ["metric", "Value - " + row["player"], "Value - " + other_row["player"],
             "Percentile - " + row["player"], "Percentile - " + other_row["player"], "z gap"]
        ].rename(columns={"metric": "Metric"}),
        hide_index=True, height=430,
    )

st.divider()
with st.expander("How the search works", expanded=False):
    st.markdown(
        f"""
- The reference player is compared **only against {model.group} seasons** - {len(model.index):,}
  of them in the current pool - using {len(model.features)} position-specific features.
- Features are z-scored inside that group, so every metric enters the distance on the same scale.
- Cosine and Euclidean answer different questions; both are computed from the same standardised
  matrix, and neither applies any hidden weighting.
- Filters (age, minutes, league) are applied to the **candidates**, not to the scaling: removing
  players from the shortlist never changes anyone's percentile or z-score.
"""
    )

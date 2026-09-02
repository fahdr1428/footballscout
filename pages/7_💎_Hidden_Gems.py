"""Experimental composite for players who look under-exposed relative to their output."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import LEAGUE_TIER, POSITION_GROUP_NAMES, POSITION_GROUPS
from src.recruitment import (
    DEFAULT_GEM_WEIGHTS, GEM_AGE_CEILING, GEM_AGE_FLOOR, GEM_MINUTES_FULL, hidden_gem_scores,
)
from src.reporting import generate_report
from src.ui import chart, eyebrow, note, page_setup, sidebar_filters, tiles
from src.visualisation import component_bar, radar_chart, scatter

page_setup("Hidden Gems", "💎")
platform = sidebar_filters()
pool = platform.pool

st.markdown("# Hidden gems")
st.warning(
    "**This is an analytical model, not a valuation.** The dataset contains no transfer fees, "
    "wages or contract data, so nothing here can say a player is cheap or undervalued in a market "
    "sense. What it does say is: strong output for his position, young, enough minutes to trust, "
    "playing outside the strongest leagues, and with a statistical profile few others match.",
    icon="⚠️",
)

st.markdown("## Score components")
st.markdown(
    f"""
| Component | How it is computed |
| --- | --- |
| **Performance** | Weighted mean of the player's positional category percentiles, using the default position weights from `config.py`. |
| **Age upside** | 100 at {GEM_AGE_FLOOR:.0f} or younger, falling linearly to 0 at {GEM_AGE_CEILING:.0f}. |
| **Low exposure** | Rescaled inverse of the league-strength coefficient - highest for the weakest league in the dataset. An editable assumption, not a measurement. |
| **Statistical uniqueness** | Percentile of the mean distance to the 15 nearest peers in the standardised feature space. High = few close analogues. |
| **Sample size** | 100 × minutes ÷ {GEM_MINUTES_FULL:,}, capped at 100. Stops a 600-minute purple patch topping the list. |
"""
)

st.markdown("## Weights")
weight_columns = st.columns(len(DEFAULT_GEM_WEIGHTS))
weights: dict[str, float] = {}
for column, (component, default) in zip(weight_columns, DEFAULT_GEM_WEIGHTS.items()):
    with column:
        weights[component] = st.slider(component, 0, 60, int(default), 5, key=f"gem_{component}")
if sum(weights.values()) == 0:
    st.warning("Give at least one component a weight above zero.")
    st.stop()

st.markdown("## Filters")
filters = st.columns([1.3, 1.1, 1.1, 1.1])
with filters[0]:
    groups = st.multiselect(
        "Position group", POSITION_GROUPS,
        format_func=lambda g: f"{g} - {POSITION_GROUP_NAMES[g]}",
    )
with filters[1]:
    max_age = st.slider("Maximum age", 16.0, 34.0, 23.0, 0.5)
with filters[2]:
    min_minutes = st.number_input(
        "Minimum minutes", int(platform.min_minutes), 3400, max(900, int(platform.min_minutes)), 50
    )
with filters[3]:
    max_tier = st.selectbox(
        "League level", ["All", "Level 2 and below", "Level 3 only"], index=0,
        help="Level 1 here is the five strongest leagues in the dataset's own coefficient table.",
    )

scores = hidden_gem_scores(
    pool, platform.categories, {g: m.z for g, m in platform.models.items()}, weights=weights
)

candidates = pool.join(scores)
candidates = candidates[(candidates["age"] <= max_age) & (candidates["minutes"] >= min_minutes)]
if groups:
    candidates = candidates[candidates["position_group"].isin(groups)]
if max_tier == "Level 2 and below":
    candidates = candidates[candidates["league"].map(LEAGUE_TIER).fillna(1) >= 2]
elif max_tier == "Level 3 only":
    candidates = candidates[candidates["league"].map(LEAGUE_TIER).fillna(1) >= 3]

candidates = candidates.dropna(subset=["hidden_gem_score"]).sort_values(
    "hidden_gem_score", ascending=False
)

if candidates.empty:
    st.info("No players match those filters.")
    st.stop()

tiles(
    [
        ("Candidates", f"{len(candidates):,}", "after filters"),
        ("Top score", f"{candidates['hidden_gem_score'].max():.1f}", "weighted composite"),
        ("Median age", f"{candidates['age'].median():.1f}", "years"),
        ("Leagues", f"{candidates['league'].nunique()}", "represented"),
    ]
)

st.markdown("## Ranked list")
top = candidates.head(50)
view = pd.DataFrame(
    {
        "Score": top["hidden_gem_score"],
        "Player": top["player"],
        "Age": top["age"],
        "Pos": top["position"],
        "Club": top["team"],
        "League": top["league"],
        "Minutes": top["minutes"],
        "Performance": top["Performance"],
        "Age upside": top["Age upside"],
        "Low exposure": top["Low exposure"],
        "Uniqueness": top["Statistical uniqueness"],
        "Sample size": top["Sample size"],
        "Archetype": top["archetype"],
    }
)
st.dataframe(
    view, hide_index=True, height=440,
    column_config={
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
        "Minutes": st.column_config.NumberColumn(format="%d"),
        "Age": st.column_config.NumberColumn(format="%.1f"),
    },
)
note(
    "Every column right of 'Minutes' is one of the five components, each on a 0-100 scale. "
    "The score is their weighted mean - so a list can be re-pointed simply by moving the sliders."
)

st.markdown("## Performance against age")
chart(
    scatter(
        candidates, x="age", y="Performance", hover_name="player",
        hover_cols=["team", "league", "minutes", "hidden_gem_score"],
        x_title="Age", y_title="Performance component (weighted positional percentile)",
        highlight=candidates.index.isin(top.head(15).index)
        if len(candidates) > 15 else None,
        highlight_label="Top 15 by hidden-gem score", height=440,
    )
)

st.divider()
st.markdown("## Inspect a name")
labels = platform.player_labels.loc[top.index]
chosen = st.selectbox("Player", labels.tolist())
index = platform.index_for_label(chosen)

if index is not None:
    st.session_state["selected_player_label"] = chosen
    row = platform.row(index)
    components = {c: float(candidates.loc[index, c]) for c in DEFAULT_GEM_WEIGHTS}
    columns = st.columns([1, 1])
    with columns[0]:
        eyebrow("Component breakdown")
        chart(component_bar(components, height=290))
        st.caption(
            f"Weighted composite: **{candidates.loc[index, 'hidden_gem_score']:.1f}** "
            f"({' + '.join(f'{c} × {weights[c] / sum(weights.values()):.0%}' for c in components)})."
        )
    with columns[1]:
        eyebrow("Attribute profile")
        chart(radar_chart([(row["player"], platform.category_scores(index))], height=380))

    archetype, description = platform.archetype(index)
    st.markdown(f"**Archetype:** {archetype}")
    st.caption(description)

    if st.button("Generate scouting report", type="primary"):
        report = generate_report(platform, index)
        st.download_button(
            "Download as Markdown", report,
            file_name=f"hidden_gem_{row['player'].replace(' ', '_')}.md", mime="text/markdown",
        )
        with st.container(border=True):
            st.markdown(report)

"""Full player read-out: percentiles, radar, strengths, archetype, similar players, report."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import METRIC_LABELS
from src.pipeline import format_metric
from src.reporting import generate_report, headline_metrics, key_takeaways, ordinal
from src.ui import (
    chart, eyebrow, note, page_setup, player_header, player_selector, sidebar_filters,
    similarity_table, tiles, watchlist_button, watchlist_sidebar,
)
from src.visualisation import trajectory_chart
from src.visualisation import distribution_with_marker, percentile_bars, radar_chart

page_setup("Player Profile", "📊")
platform = sidebar_filters()

st.markdown("# Player profile")
index = player_selector(platform, "Player", key="profile")
if index is None:
    st.stop()

row = platform.row(index)
group = row["position_group"]
model = platform.model_for(index)
archetype, archetype_description = platform.archetype(index)

st.divider()
header_columns = st.columns([4, 1])
with header_columns[0]:
    player_header(platform, index)
with header_columns[1]:
    watchlist_button(platform, index, key=f"profile_{index}")
watchlist_sidebar()

# ---- headline numbers ----------------------------------------------------
if group == "GK":
    headline = [
        ("Save %", "gk_save_pct"), ("Goals prevented /90", "gk_psxg_minus_ga_per90"),
        ("Saves /90", "gk_saves_per90"), ("Conceded /90", "gk_goals_against_per90"),
        ("Cross claim %", "gk_cross_stop_pct"), ("Sweeper actions /90", "gk_def_actions_outside_box_per90"),
        ("Pass completion", "pass_pct"),
    ]
else:
    headline = [
        ("Goals /90", "goals_per90"), ("Non-pen goals /90", "np_goals_per90"),
        ("xG /90", "npxg_per90"), ("Assists /90", "assists_per90"), ("xA /90", "xa_per90"),
        ("Shots /90", "shots_per90"), ("Key passes /90", "key_passes_per90"),
        ("Progressive actions /90", "progressive_actions_per90"),
        ("Defensive actions /90", "defensive_actions_per90"),
    ]

st.write("")
tiles(
    [
        (
            label,
            format_metric(metric, row.get(metric)),
            f"{ordinal(platform.percentile(index, metric))} percentile"
            if pd.notna(platform.percentile(index, metric))
            else "no percentile",
        )
        for label, metric in headline
    ]
)
note(
    f"The small figure is the player's percentile against <b>{platform.peer_group_label}</b> "
    f"in the current pool - {len(platform.peers(index)):,} players - not against everyone in "
    "the dataset."
)

st.divider()

# ---- radar + percentile bars --------------------------------------------
left, right = st.columns([1, 1.15])

with left:
    st.markdown("### Attribute profile")
    categories = platform.category_scores(index)
    if categories:
        chart(radar_chart([(row["player"], categories)], height=430))
        st.caption(
            "Each axis is the mean of that category's metric percentiles. "
            "Category definitions are listed on the Methodology page."
        )
    st.markdown("### Archetype")
    st.markdown(
        f'<div class="sx-card"><div class="sx-eyebrow">K-Means archetype</div>'
        f'<div style="font-size:1.2rem;font-weight:650;margin:0.15rem 0 0.4rem 0;">{archetype}</div>'
        f'<div style="color:#c3c2b7;font-size:0.9rem;">{archetype_description}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Fitted on {len(model.index):,} {group} seasons only, k={model.clusters.k} "
        f"(silhouette {model.clusters.silhouette:.3f}). The label is written from the cluster "
        "centroid, not assigned by hand."
    )

with right:
    st.markdown("### Percentile profile")
    frame = platform.percentile_frame(index, model.features)
    chart(percentile_bars(frame, height=max(360, 24 * len(frame) + 80)))
    st.caption(
        f"Against {len(platform.peers(index)):,} {platform.peer_group_label} with at least "
        f"{platform.min_minutes:,} minutes."
    )

# ---- strengths / weaknesses ---------------------------------------------
st.divider()
st.markdown("## Strengths and weaknesses")
note(
    "Taken from the same percentile table: metrics at or above the 70th percentile are listed as "
    "strengths, at or below the 30th as weaknesses, out of the features this position is modelled "
    "on. No metric is cherry-picked."
)
columns = st.columns(2)
with columns[0]:
    eyebrow("Strengths")
    strengths = platform.strengths(index, n=6)
    if strengths.empty:
        st.info("No metric reaches the 70th percentile against positional peers.")
    for _, item in strengths.iterrows():
        st.markdown(
            f"**{item['metric']}** - {format_metric(item['key'], item['value'])} "
            f"· {ordinal(item['percentile'])} percentile"
        )
with columns[1]:
    eyebrow("Weaknesses")
    weaknesses = platform.weaknesses(index, n=6)
    if weaknesses.empty:
        st.info("No metric falls below the 30th percentile against positional peers.")
    for _, item in weaknesses.iterrows():
        st.markdown(
            f"**{item['metric']}** - {format_metric(item['key'], item['value'])} "
            f"· {ordinal(item['percentile'])} percentile"
        )

# ---- performance overview table -----------------------------------------
st.divider()
st.markdown("## Performance overview")
metrics = [m for m in headline_metrics(group) if m in platform.pool.columns]
overview = platform.percentile_frame(index, metrics)
overview["Season total"] = [
    format_metric(k, row.get(k.replace("_per90", ""), float("nan")))
    if k.endswith("_per90") and k.replace("_per90", "") in row.index else "-"
    for k in overview["key"]
]
overview["Per 90 / rate"] = [format_metric(k, v) for k, v in zip(overview["key"], overview["value"])]
st.dataframe(
    overview[["metric", "Season total", "Per 90 / rate", "percentile"]].rename(
        columns={"metric": "Metric", "percentile": "Positional percentile"}
    ),
    hide_index=True, height=380,
    column_config={
        "Positional percentile": st.column_config.ProgressColumn(
            "Positional percentile", min_value=0, max_value=100, format="%.0f"
        )
    },
)

# ---- distribution --------------------------------------------------------
with st.expander("Where he sits in the positional distribution", expanded=False):
    metric = st.selectbox(
        "Metric", model.features, format_func=lambda m: METRIC_LABELS.get(m, m), key="dist_metric"
    )
    peers = platform.pool.loc[model.index, metric]
    chart(
        distribution_with_marker(
            peers, float(row[metric]), row["player"], METRIC_LABELS.get(metric, metric)
        )
    )
    st.caption(
        f"{len(peers):,} {group} player-seasons in the pool. Percentile: "
        f"{ordinal(platform.percentile(index, metric))}."
    )

# ---- trajectory ----------------------------------------------------------
trajectory = platform.trajectory(index)
if not trajectory.empty:
    st.divider()
    st.markdown("## Season by season")
    note(
        "The same player's category percentiles across every season in the pool. Percentiles are "
        "recomputed against that season's peers, so a flat line means he held his level as the "
        "population changed around him - not that his raw numbers were identical."
    )
    columns = st.columns([1.5, 1])
    with columns[0]:
        chart(trajectory_chart(trajectory))
    with columns[1]:
        st.dataframe(
            trajectory[["season", "team", "league", "minutes", "position", "archetype"]].rename(
                columns={
                    "season": "Season", "team": "Club", "league": "League",
                    "minutes": "Minutes", "position": "Pos", "archetype": "Archetype",
                }
            ),
            hide_index=True,
            column_config={"Minutes": st.column_config.NumberColumn(format="%d")},
        )

# ---- similar players -----------------------------------------------------
st.divider()
st.markdown("## Similar players")
count = st.slider("How many", 5, 15, 8, key="profile_similar_n")
similar = platform.similar(index, n=int(count))
similarity_table(similar)
st.caption(
    "Cosine similarity of the standardised, position-specific feature vectors. Open "
    "**Similar Players** for the feature-by-feature explanation of each match."
)

# ---- report --------------------------------------------------------------
st.divider()
st.markdown("## Scouting report")
if st.button("Generate scouting report", type="primary"):
    report = generate_report(platform, index)
    st.session_state[f"report_{index}"] = report

report = st.session_state.get(f"report_{index}")
if report:
    st.download_button(
        "Download as Markdown",
        report,
        file_name=f"scouting_report_{row['player'].replace(' ', '_')}_{row['season']}.md",
        mime="text/markdown",
    )
    with st.container(border=True):
        st.markdown(report)
else:
    eyebrow("Key takeaways")
    for takeaway in key_takeaways(platform, index):
        st.markdown(f"- {takeaway}")

"""Search the pool by position, league, age, minutes, archetype and metric thresholds."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import METRIC_LABELS, POSITION_GROUP_NAMES, POSITION_GROUPS
from src.feature_engineering import metrics_for_percentiles
from src.pipeline import format_metric
from src.ui import age_slider, apply_age, chart, default_axis, eyebrow, note, page_setup, sidebar_filters
from src.visualisation import scatter

page_setup("Player Search", "👤")
platform = sidebar_filters()
pool = platform.pool

st.markdown("# Player search")
note(
    "Every column below is computed inside the current pool "
    f"(minimum {platform.min_minutes:,} minutes). Percentiles are always against players in the "
    "same position group."
)

row1 = st.columns([1.3, 1.2, 1, 1])
with row1[0]:
    groups = st.multiselect(
        "Position group", POSITION_GROUPS,
        format_func=lambda g: f"{g} - {POSITION_GROUP_NAMES[g]}",
    )
with row1[1]:
    leagues = st.multiselect("League", sorted(pool["league"].unique()))
with row1[2]:
    seasons = st.multiselect("Season", sorted(pool["season"].unique()))
with row1[3]:
    name_query = st.text_input("Name contains", "")

row2 = st.columns([1.4, 1.4, 1.6])
with row2[0]:
    age_range = age_slider(platform, "Age")
    if age_range is None:
        nationalities = st.multiselect("Nationality", sorted(pool.get("nationality", pd.Series(dtype=str)).dropna().unique()))
    else:
        nationalities = []
with row2[1]:
    minute_max = int(pool["minutes"].max())
    minutes_range = st.slider(
        "Minutes", int(platform.min_minutes), minute_max, (int(platform.min_minutes), minute_max), 50
    )
with row2[2]:
    archetypes = st.multiselect("Archetype", sorted(pool["archetype"].dropna().unique()))

filtered = apply_age(pool[pool["minutes"].between(*minutes_range)], age_range)
if nationalities:
    filtered = filtered[filtered["nationality"].isin(nationalities)]
if groups:
    filtered = filtered[filtered["position_group"].isin(groups)]
if leagues:
    filtered = filtered[filtered["league"].isin(leagues)]
if seasons:
    filtered = filtered[filtered["season"].isin(seasons)]
if archetypes:
    filtered = filtered[filtered["archetype"].isin(archetypes)]
if name_query.strip():
    filtered = filtered[filtered["player"].str.contains(name_query.strip(), case=False, na=False)]

# ---- metric thresholds ---------------------------------------------------
single_group = groups[0] if len(groups) == 1 else None
metric_choices = sorted(
    metrics_for_percentiles(single_group or "CM", list(pool.columns)),
    key=lambda m: METRIC_LABELS.get(m, m),
)

with st.expander("Metric thresholds", expanded=False):
    st.caption(
        "Add per-90 or percentage conditions. Values are the raw rates, not percentiles, so a "
        "threshold means the same thing in every league."
    )
    count = st.number_input("Number of conditions", 0, 5, 0, 1)
    conditions = []
    for i in range(int(count)):
        columns = st.columns([2.4, 0.9, 1])
        with columns[0]:
            metric = st.selectbox(
                "Metric", metric_choices, key=f"search_metric_{i}",
                format_func=lambda m: METRIC_LABELS.get(m, m),
            )
        with columns[1]:
            operator = st.selectbox("Operator", [">=", ">", "<=", "<"], key=f"search_op_{i}")
        with columns[2]:
            series = pool[metric].dropna()
            value = st.number_input(
                "Value", value=float(round(series.median(), 2)) if len(series) else 0.0,
                step=0.05, key=f"search_val_{i}", format="%.2f",
            )
        conditions.append((metric, operator, float(value)))

    for metric, operator, value in conditions:
        if metric not in filtered.columns:
            continue
        series = filtered[metric]
        mask = {
            ">=": series >= value, ">": series > value,
            "<=": series <= value, "<": series < value,
        }[operator]
        filtered = filtered[mask.fillna(False)]

# ---- ranking -------------------------------------------------------------
sort_columns = st.columns([2, 1, 1])
with sort_columns[0]:
    sort_metric = st.selectbox(
        "Rank by", (["minutes"] + (["age"] if platform.has_age else [])) + metric_choices,
        format_func=lambda m: METRIC_LABELS.get(m, m), index=0,
    )
with sort_columns[1]:
    ascending = st.selectbox("Order", ["Highest first", "Lowest first"]) == "Lowest first"
with sort_columns[2]:
    limit = st.number_input("Rows", 10, 500, 60, 10)

eyebrow(f"{len(filtered):,} player-seasons match")

if filtered.empty:
    st.info("No players match those filters. Loosen a threshold or widen the age range.")
    st.stop()

# `minutes` and `age` already have their own columns in the table below.
BASE_COLUMNS = {"minutes", "age"}
display_metrics = [
    m for m in [sort_metric, "np_goals_per90", "xa_per90", "progressive_actions_per90",
                "defensive_actions_per90"]
    if m not in BASE_COLUMNS and m in filtered.columns
]
display_metrics = list(dict.fromkeys(display_metrics))
table = filtered.sort_values(sort_metric, ascending=ascending).head(int(limit))
view = pd.DataFrame(
    {
        "Player": table["player"],
        "Pos": table["position"],
        "Club": table["team"],
        "League": table["league"],
        "Season": table["season"],
        "Minutes": table["minutes"],
        "Archetype": table["archetype"],
    }
)
for metric in display_metrics:
    view[METRIC_LABELS.get(metric, metric)] = [
        format_metric(metric, v) for v in table[metric]
    ]
    percentile_column = f"pct_{metric}"
    if percentile_column in platform.percentiles.columns:
        view[f"{METRIC_LABELS.get(metric, metric)} pct"] = platform.percentiles.loc[
            table.index, percentile_column
        ].round(0)

if platform.has_age:
    view.insert(1, "Age", table["age"].to_numpy())
if "nationality" in table.columns:
    view.insert(1, "Nation", table["nationality"].to_numpy())
st.dataframe(
    view, hide_index=True, height=430,
    column_config={
        "Minutes": st.column_config.NumberColumn(format="%d"),
        "Age": st.column_config.NumberColumn(format="%.1f"),
    },
)

st.markdown("## Open a player")
labels = platform.player_labels.loc[table.index]
chosen = st.selectbox("Select from the results", labels.tolist())
index = platform.index_for_label(chosen)
if index is not None:
    st.session_state["selected_player_label"] = chosen
    row = platform.row(index)
    archetype, description = platform.archetype(index)
    st.success(
        f"**{row['player']}** ({row['team']}, {row['season']}) selected - archetype "
        f"*{archetype}*. Open **Player Profile**, **Similar Players** or **Compare Players** "
        "and the selection carries across."
    )
    st.caption(description)
    links = st.columns(3)
    with links[0]:
        st.page_link("pages/2_📊_Player_Profile.py", label="Player profile", icon="📊")
    with links[1]:
        st.page_link("pages/3_🔎_Similar_Players.py", label="Similar players", icon="🔎")
    with links[2]:
        st.page_link("pages/5_🆚_Compare_Players.py", label="Compare", icon="🆚")

st.markdown("## Distribution of the ranking metric")
scatter_source = filtered.copy()
x_axis = default_axis(platform)
if x_axis == sort_metric:
    x_axis = "starts" if "starts" in scatter_source.columns else "matches"
chart(
    scatter(
        scatter_source, x=x_axis, y=sort_metric, hover_name="player",
        hover_cols=["team", "league", "minutes"],
        x_title=METRIC_LABELS.get(x_axis, x_axis.replace("_", " ").title()),
        y_title=METRIC_LABELS.get(sort_metric, sort_metric),
        highlight=pd.Series(scatter_source.index.isin(table.index), index=scatter_source.index)
        if len(table) < len(scatter_source) else None,
        highlight_label="In the table above", trend=True, height=430,
    )
)

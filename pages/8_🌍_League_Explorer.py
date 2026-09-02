"""League and population level exploration: leaderboards, age curves, style profiles."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import (
    LEAGUE_STRENGTH, LEAGUE_TIER, METRIC_LABELS, POSITION_GROUP_NAMES, POSITION_GROUPS,
    categories_for,
)
from src.pipeline import format_metric
from src.ui import chart, eyebrow, note, page_setup, sidebar_filters, tiles
from src.visualisation import scatter

page_setup("League Explorer", "🌍")
platform = sidebar_filters()
pool = platform.pool

st.markdown("# League explorer")
note(
    "Leaderboards here rank on raw per-90 rates within the selection. Percentiles shown elsewhere "
    "in the app are always positional; this page is about the population, not one player."
)

controls = st.columns([1.4, 1.2, 1.1, 1.1])
with controls[0]:
    leagues = st.multiselect("Leagues", sorted(pool["league"].unique()))
with controls[1]:
    group = st.selectbox(
        "Position group", ["All"] + POSITION_GROUPS,
        format_func=lambda g: g if g == "All" else f"{g} - {POSITION_GROUP_NAMES[g]}",
    )
with controls[2]:
    age_range = st.slider("Age", 15.0, 40.0, (15.0, 40.0), 0.5)
with controls[3]:
    min_minutes = st.number_input(
        "Minimum minutes", int(platform.min_minutes), 3400, int(platform.min_minutes), 50
    )

selection = pool[pool["age"].between(*age_range) & (pool["minutes"] >= min_minutes)]
if leagues:
    selection = selection[selection["league"].isin(leagues)]
if group != "All":
    selection = selection[selection["position_group"] == group]

if selection.empty:
    st.info("Nothing matches those filters.")
    st.stop()

tiles(
    [
        ("Player-seasons", f"{len(selection):,}", "in the selection"),
        ("Clubs", f"{selection['team'].nunique():,}", f"{selection['league'].nunique()} leagues"),
        ("Median age", f"{selection['age'].median():.1f}", "years"),
        ("Median minutes", f"{selection['minutes'].median():,.0f}", "per season"),
        ("U-23 share", f"{(selection['age'] < 23).mean():.0%}", "of the selection"),
    ]
)

# ---- leaderboards --------------------------------------------------------
st.markdown("## Leaderboards")
LEADERBOARDS = {
    "Best finishers": "np_goals_per90",
    "Best underlying shot volume": "npxg_per90",
    "Best creators": "xa_per90",
    "Most progressive": "progressive_actions_per90",
    "Best dribblers": "dribbles_completed_per90",
    "Most defensive activity": "defensive_actions_per90",
    "Best passers by volume": "passes_attempted_per90",
    "Most aerially dominant": "aerials_won_per90",
}
board_columns = st.columns(2)
for i, (title, metric) in enumerate(LEADERBOARDS.items()):
    if metric not in selection.columns:
        continue
    with board_columns[i % 2]:
        eyebrow(title)
        top = selection.nlargest(8, metric)
        st.dataframe(
            pd.DataFrame(
                {
                    "Player": top["player"],
                    "Club": top["team"],
                    "League": top["league"],
                    "Age": top["age"].round(1),
                    METRIC_LABELS.get(metric, metric): [format_metric(metric, v) for v in top[metric]],
                }
            ),
            hide_index=True, height=317,
        )

# ---- young breakouts -----------------------------------------------------
st.markdown("## Young breakout candidates")
young = selection[selection["age"] <= 22]
if young.empty:
    st.caption("No players aged 22 or under in this selection.")
else:
    category_columns = [c for c in platform.categories.columns if c.startswith("cat_")]
    composite = platform.categories.loc[young.index, category_columns].mean(axis=1)
    ranked = young.assign(profile_score=composite.round(1)).nlargest(12, "profile_score")
    st.dataframe(
        pd.DataFrame(
            {
                "Player": ranked["player"],
                "Age": ranked["age"].round(1),
                "Pos": ranked["position"],
                "Club": ranked["team"],
                "League": ranked["league"],
                "Minutes": ranked["minutes"],
                "Mean category percentile": ranked["profile_score"],
                "Archetype": ranked["archetype"],
            }
        ),
        hide_index=True,
        column_config={
            "Mean category percentile": st.column_config.ProgressColumn(
                "Mean category percentile", min_value=0, max_value=100, format="%.0f"
            ),
            "Minutes": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption(
        "Ranked on the unweighted mean of every attribute-category percentile for the player's "
        "own position - a deliberately blunt all-round measure, not a role-specific one."
    )

# ---- scatter workbench ---------------------------------------------------
st.markdown("## Scatter workbench")
numeric = [
    c for c in [
        "age", "minutes", "np_goals_per90", "npxg_per90", "xa_per90", "assists_per90",
        "key_passes_per90", "sca_per90", "progressive_passes_per90", "progressive_carries_per90",
        "progressive_actions_per90", "dribbles_completed_per90", "defensive_actions_per90",
        "ball_recoveries_per90", "pressures_per90", "aerials_won_per90", "pass_pct",
        "touches_att_pen_per90", "league_strength",
    ] if c in selection.columns
]
axis_columns = st.columns([1.3, 1.3, 1.3])
with axis_columns[0]:
    x_metric = st.selectbox(
        "X axis", numeric, index=numeric.index("age"),
        format_func=lambda m: METRIC_LABELS.get(m, m.replace("_", " ").title()),
    )
with axis_columns[1]:
    y_metric = st.selectbox(
        "Y axis", numeric,
        index=numeric.index("progressive_actions_per90") if "progressive_actions_per90" in numeric else 1,
        format_func=lambda m: METRIC_LABELS.get(m, m.replace("_", " ").title()),
    )
with axis_columns[2]:
    highlight_league = st.selectbox("Highlight a league", ["(none)"] + sorted(selection["league"].unique()))

highlight = (
    selection["league"].eq(highlight_league) if highlight_league != "(none)" else None
)
chart(
    scatter(
        selection, x=x_metric, y=y_metric, hover_name="player",
        hover_cols=["team", "league", "age", "minutes"],
        x_title=METRIC_LABELS.get(x_metric, x_metric.replace("_", " ").title()),
        y_title=METRIC_LABELS.get(y_metric, y_metric.replace("_", " ").title()),
        highlight=highlight, highlight_label=highlight_league, trend=True, height=520,
    )
)
note(
    "The dotted line is an ordinary least-squares fit through the selection - a description of "
    "the current sample, not a prediction. Watch for selection effects: the pool already excludes "
    f"anyone under {platform.min_minutes:,} minutes."
)

# ---- league comparison ---------------------------------------------------
st.markdown("## League profiles")
league_metrics = [
    "np_goals_per90", "npxg_per90", "xa_per90", "progressive_passes_per90",
    "progressive_carries_per90", "dribbles_completed_per90", "defensive_actions_per90",
    "pass_pct", "aerials_won_per90",
]
league_metrics = [m for m in league_metrics if m in selection.columns]
summary = (
    selection.groupby("league")
    .agg(
        players=("player_id", "nunique"),
        median_age=("age", "median"),
        **{m: (m, "mean") for m in league_metrics},
    )
    .round(2)
    .reset_index()
)
summary.insert(1, "level", summary["league"].map(LEAGUE_TIER))
summary.insert(2, "strength", summary["league"].map(LEAGUE_STRENGTH))
summary = summary.sort_values("strength", ascending=False)
st.dataframe(
    summary.rename(columns={**{m: METRIC_LABELS.get(m, m) for m in league_metrics},
                            "league": "League", "level": "Level", "strength": "Strength coefficient",
                            "players": "Players", "median_age": "Median age"}),
    hide_index=True, height=430,
)
note(
    "Averages per league across the current selection. The strength coefficient is an "
    "assumption stored in <code>src/config.py</code>; it is used for filtering and for the "
    "hidden-gem exposure component, never silently baked into a per-90 rate."
)

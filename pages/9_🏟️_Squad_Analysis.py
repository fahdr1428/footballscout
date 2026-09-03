"""Club-level view: squad make-up, style against the league, and replacements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import METRIC_LABELS, OUTFIELD_GROUPS, POSITION_GROUP_NAMES, POSITION_GROUPS
from src.pipeline import format_metric
from src.ui import (
    add_to_watchlist, chart, eyebrow, note, page_setup, sidebar_filters, similarity_table,
    tiles, watchlist_sidebar,
)
from src.visualisation import category_bars, percentile_bars, radar_chart, scatter

page_setup("Squad Analysis", "🏟️")
platform = sidebar_filters()
pool = platform.pool

st.markdown("# Squad analysis")
note(
    "A club-level read of the same models: who plays, what they are, where the squad is thin, "
    "and who in the rest of the pool could replace them."
)

controls = st.columns([1.3, 1.6, 1])
with controls[0]:
    league = st.selectbox("League", sorted(pool["league"].unique()))
league_pool = pool[pool["league"] == league]
with controls[1]:
    team = st.selectbox("Club", sorted(league_pool["team"].unique()))
with controls[2]:
    seasons = sorted(league_pool[league_pool["team"] == team]["season"].unique())
    season = st.selectbox("Season", ["All"] + seasons)

squad = platform.squad(team, None if season == "All" else season)
if squad.empty:
    st.info("No players from this club clear the minimum-minutes filter.")
    st.stop()

peers = league_pool if season == "All" else league_pool[league_pool["season"] == season]
category_columns = [c for c in platform.categories.columns if c.startswith("cat_")]
squad_profile = platform.categories.loc[squad.index, category_columns].mean(axis=1).round(1)

tiles(
    [
        ("Players", f"{squad['player_id'].nunique():,}", f"above {platform.min_minutes:,} minutes"),
        ("Minutes covered", f"{squad['minutes'].sum():,.0f}", "in the filtered pool"),
        ("Possession", f"{squad['team_possession'].mean():.1f}%", "share of on-ball involvements"),
        (
            "Mean profile",
            f"{squad_profile.mean():.0f}",
            "mean category percentile vs the league",
        ),
        (
            "Archetypes",
            f"{squad['archetype'].nunique()}",
            "distinct roles in the squad",
        ),
    ]
)

# ---- squad list ----------------------------------------------------------
st.markdown("## Squad")
view = pd.DataFrame(
    {
        "Player": squad["player"],
        "Pos": squad["position"],
        "Group": squad["position_group"],
        "Season": squad["season"],
        "Minutes": squad["minutes"],
        "Starts": squad["starts"],
        "Archetype": squad["archetype"],
        "Profile": squad_profile,
    }
)
if platform.has("price_m"):
    view["Price £m"] = squad["price_m"].to_numpy()
if "detailed_position" in squad.columns and squad["detailed_position"].notna().any():
    view.insert(2, "Line-up", squad["detailed_position"].to_numpy())
if platform.has_age:
    view.insert(1, "Age", squad["age"].to_numpy())
if "nationality" in squad.columns:
    view.insert(1, "Nation", squad["nationality"].to_numpy())
st.dataframe(
    view, hide_index=True, height=430,
    column_config={
        "Minutes": st.column_config.NumberColumn(format="%d"),
        "Profile": st.column_config.ProgressColumn(
            "Profile", min_value=0, max_value=100, format="%.0f",
            help="Unweighted mean of every attribute-category percentile, against positional peers.",
        ),
    },
)

# ---- positional depth ----------------------------------------------------
st.markdown("## Positional depth")
depth = (
    squad.groupby("position_group")
    .agg(players=("player_id", "nunique"), minutes=("minutes", "sum"))
    .reindex([g for g in POSITION_GROUPS])
    .fillna(0)
)
league_depth = (
    peers.groupby(["team", "position_group"])["minutes"].sum().unstack(fill_value=0)
    .reindex(columns=POSITION_GROUPS, fill_value=0)
)
depth["league_median_minutes"] = league_depth.median().reindex(depth.index).fillna(0)
depth["vs_league"] = (depth["minutes"] - depth["league_median_minutes"]).round(0)
depth = depth.reset_index()
depth["position_group"] = depth["position_group"].map(
    lambda g: f"{g} - {POSITION_GROUP_NAMES.get(g, g)}"
)
st.dataframe(
    depth.rename(
        columns={
            "position_group": "Position group", "players": "Players", "minutes": "Minutes",
            "league_median_minutes": "League median minutes", "vs_league": "Difference",
        }
    ),
    hide_index=True,
    column_config={
        "Minutes": st.column_config.NumberColumn(format="%d"),
        "League median minutes": st.column_config.NumberColumn(format="%d"),
        "Difference": st.column_config.NumberColumn(format="%d"),
    },
)
thin = depth[depth["Difference" if "Difference" in depth.columns else "vs_league"] < 0] \
    if "vs_league" in depth.columns or "Difference" in depth.columns else depth.iloc[0:0]
note(
    "Minutes are only counted for players above the sidebar's minimum-minutes filter, so a "
    "position covered by many short spells will look thin here. That is usually the point: "
    "it is the squad's reliance on trusted starters."
)

# ---- style profile -------------------------------------------------------
st.markdown("## Style against the league")
profile = platform.team_category_profile(
    team, None if season == "All" else season, position_groups=OUTFIELD_GROUPS
)
if not profile.empty:
    columns = st.columns([1, 1.05])
    with columns[0]:
        chart(
            radar_chart(
                [
                    (team, dict(zip(profile["category"], profile["club"]))),
                    (f"{league} median", dict(zip(profile["category"], profile["league_median"]))),
                ],
                height=440,
            )
        )
    with columns[1]:
        chart(
            category_bars(
                {
                    team: dict(zip(profile["category"], profile["club"])),
                    f"{league} median": dict(zip(profile["category"], profile["league_median"])),
                },
                height=440,
            )
        )
    st.caption(
        "Outfield players only, weighted by minutes played. Each axis is the club's average "
        "positional percentile in that category - so it describes the players the club fields, "
        "not the team's tactics directly."
    )

# ---- reliance ------------------------------------------------------------
st.markdown("## Minutes distribution")
ranked_minutes = squad.groupby("player")["minutes"].sum().sort_values(ascending=False)
share_top = ranked_minutes.head(11).sum() / max(ranked_minutes.sum(), 1)
columns = st.columns([1.4, 1])
with columns[0]:
    chart(
        percentile_bars(
            pd.DataFrame(
                {
                    "metric": ranked_minutes.head(18).index,
                    "percentile": (100 * ranked_minutes.head(18) / ranked_minutes.max()).round(1),
                    "value": ranked_minutes.head(18).to_numpy(),
                }
            ),
            title="Minutes, as a share of the most-used player",
        )
    )
with columns[1]:
    eyebrow("Reliance")
    st.metric("Top 11 share of minutes", f"{share_top:.0%}")
    st.caption(
        "A high share means a settled side - or a thin one. Compare it against the league "
        "table below before reading anything into it."
    )
    league_shares = []
    for other, block in peers.groupby("team"):
        totals = block.groupby("player")["minutes"].sum().sort_values(ascending=False)
        league_shares.append(totals.head(11).sum() / max(totals.sum(), 1))
    st.metric("League median", f"{np.median(league_shares):.0%}")

# ---- replacement finder --------------------------------------------------
st.divider()
st.markdown("## Replacement finder")
note(
    "Pick a squad member and the engine ranks the rest of the pool on a blend you control: "
    "how closely they play like the incumbent, and how well they perform in the role."
)
settings = st.columns([2, 1.2, 1, 1])
with settings[0]:
    labels = platform.player_labels.loc[squad.index]
    chosen = st.selectbox("Player to replace", labels.tolist())
    index = platform.index_for_label(chosen)
with settings[1]:
    similarity_weight = st.slider(
        "Style vs quality", 0.0, 1.0, 0.5, 0.05,
        help="1.0 ranks purely on stylistic similarity; 0.0 ranks purely on role fit, "
             "ignoring whether the replacement plays anything like the incumbent.",
    )
with settings[2]:
    same_league_only = st.checkbox("Same league only", value=False)
with settings[3]:
    count = st.number_input("Results", 5, 40, 12, 1)

if index is not None:
    incumbent = platform.row(index)
    candidates = pool["team"] != team
    if same_league_only:
        candidates &= pool["league"] == league
    if platform.has_age:
        age_cap = st.slider(
            "Maximum age", float(pool["age"].min()), float(pool["age"].max()),
            float(pool["age"].max()), 0.5,
        )
        candidates &= pool["age"] <= age_cap

    results = platform.replacements(
        index, n=int(count), similarity_weight=similarity_weight, candidate_mask=candidates
    )
    if results.empty:
        st.info("No candidates match those filters.")
    else:
        st.markdown(
            f"**Replacing {incumbent['player']}** - "
            f"{incumbent['position']}, {incumbent['minutes']:,.0f} minutes, "
            f"archetype *{platform.archetype(index)[0]}*."
        )
        similarity_table(
            results,
            extra={
                "replacement_score": "Replacement score",
                "role_fit": "Role fit",
                "fit_delta": "Fit vs incumbent",
            },
        )
        st.caption(
            "Role fit uses the default positional weights from `config.DEFAULT_WEIGHTS`; "
            "'fit vs incumbent' is the difference against the player being replaced, so a "
            "positive number is an upgrade on those weights."
        )
        chart(
            scatter(
                results, x="similarity", y="role_fit", hover_name="player",
                hover_cols=["team", "league", "minutes", "archetype"],
                x_title="Similarity to the incumbent (%)",
                y_title="Role fit percentile", height=430,
            )
        )
        note(
            "Top-right is a like-for-like upgrade. Top-left is a better player who would change "
            "how the side plays. Bottom-right is a cheaper impersonation."
        )
        shortlist = st.multiselect(
            "Add targets to the watchlist",
            platform.player_labels.loc[results.index].tolist(), key="squad_watch",
        )
        if shortlist and st.button("Add to watchlist"):
            for label in shortlist:
                target = platform.index_for_label(label)
                if target is not None:
                    add_to_watchlist(platform, target)
            st.rerun()
        watchlist_sidebar()

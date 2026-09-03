"""The scout's shortlist: everything flagged while browsing, in one place."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import METRIC_LABELS, POSITION_GROUP_NAMES
from src.pipeline import format_metric
from src.reporting import generate_report, headline_metrics, ordinal
from src.ui import (
    add_to_watchlist, chart, eyebrow, note, page_setup, remove_from_watchlist, sidebar_filters,
    watchlist, watchlist_indices,
)
from src.visualisation import category_bars, radar_chart

page_setup("Watchlist", "📋")
platform = sidebar_filters()

st.markdown("# Watchlist")
note(
    "Players flagged anywhere in the app collect here. The list is held per browser session "
    "and stores player identities, so changing the minimum-minutes filter or the dataset "
    "re-resolves it rather than scrambling it."
)

# ---- add ----------------------------------------------------------------
with st.expander("Add players", expanded=not watchlist()):
    labels = platform.player_labels.sort_values()
    picked = st.multiselect("Search the pool", labels.tolist(), key="wl_add_multi")
    if picked and st.button("Add selected", type="primary"):
        for label in picked:
            index = platform.index_for_label(label)
            if index is not None:
                add_to_watchlist(platform, index)
        st.rerun()

entries = watchlist()
indices = watchlist_indices(platform)

if not entries:
    st.info(
        "Nothing on the watchlist yet. Add players above, or use the **Add to watchlist** button "
        "on the Player Profile, Similar Players, Recruitment Finder and Squad Analysis pages."
    )
    st.stop()

missing = len(entries) - len(indices)
if missing:
    st.warning(
        f"{missing} watchlisted player-season(s) are not in the current pool - they are below the "
        "minimum-minutes filter, outside the selected seasons, or from the other dataset.",
        icon="⚠️",
    )

if not indices:
    st.stop()

# ---- table --------------------------------------------------------------
st.markdown("## Shortlist")
rows = platform.pool.loc[indices]
category_columns = [c for c in platform.categories.columns if c.startswith("cat_")]
profile = platform.categories.loc[indices, category_columns].mean(axis=1).round(1)

table = pd.DataFrame(
    {
        "Player": rows["player"],
        "Pos": rows["position"],
        "Club": rows["team"],
        "League": rows["league"],
        "Season": rows["season"],
        "Minutes": rows["minutes"],
        "Archetype": rows["archetype"],
        "Profile": profile,
    }
)
if platform.has_age:
    table.insert(1, "Age", rows["age"].to_numpy())
st.dataframe(
    table, hide_index=True,
    column_config={
        "Minutes": st.column_config.NumberColumn(format="%d"),
        "Profile": st.column_config.ProgressColumn(
            "Profile", min_value=0, max_value=100, format="%.0f"
        ),
    },
)

export = table.copy()
for metric in ["np_goals_per90", "xa_per90", "progressive_actions_per90", "defensive_actions_per90"]:
    if metric in rows.columns:
        export[METRIC_LABELS.get(metric, metric)] = rows[metric].round(3).to_numpy()
st.download_button(
    "Download the shortlist (CSV)", export.to_csv(index=False),
    file_name="watchlist.csv", mime="text/csv",
)

# ---- notes and removal ---------------------------------------------------
st.markdown("## Notes")
st.caption("Scouting notes are kept in this browser session and are included in the CSV export.")
note_rows = []
for index in indices:
    row = platform.row(index)
    columns = st.columns([2.4, 4, 1])
    with columns[0]:
        st.markdown(f"**{row['player']}**  \n{row['team']} · {row['season']}")
    with columns[1]:
        key = f"note_{row['player_id']}_{row['season']}"
        note_rows.append(st.text_input("Note", key=key, label_visibility="collapsed",
                                       placeholder="What are you watching for?"))
    with columns[2]:
        if st.button("Remove", key=f"rm_{row['player_id']}_{row['season']}"):
            remove_from_watchlist(str(row["player_id"]), str(row["season"]))
            st.rerun()

if any(note_rows):
    export_with_notes = export.copy()
    export_with_notes["Note"] = note_rows
    st.download_button(
        "Download with notes (CSV)", export_with_notes.to_csv(index=False),
        file_name="watchlist_with_notes.csv", mime="text/csv",
    )

# ---- comparison ----------------------------------------------------------
st.markdown("## Compare from the shortlist")
groups = rows["position_group"].unique()
comparable = st.multiselect(
    "Pick up to three", platform.player_labels.loc[indices].tolist(),
    default=platform.player_labels.loc[indices].tolist()[:2], max_selections=3,
)
selected = [platform.index_for_label(label) for label in comparable]
selected = [i for i in selected if i is not None]

if len(selected) >= 2:
    if len({platform.row(i)["position_group"] for i in selected}) > 1:
        st.warning(
            "These players are in different position groups, so each one's percentiles are "
            "measured against a different peer set.",
            icon="⚠️",
        )
    scores = {platform.row(i)["player"]: platform.category_scores(i) for i in selected}
    shared = [c for c in next(iter(scores.values())) if all(c in s for s in scores.values())]
    if shared:
        trimmed = {name: {c: values[c] for c in shared} for name, values in scores.items()}
        columns = st.columns([1, 1.05])
        with columns[0]:
            chart(radar_chart(list(trimmed.items()), height=450))
        with columns[1]:
            chart(category_bars(trimmed, height=450))

# ---- reports -------------------------------------------------------------
st.markdown("## Reports")
if st.button("Generate a report for every shortlisted player"):
    bundle = "\n\n---\n\n".join(generate_report(platform, index) for index in indices)
    st.download_button(
        "Download the bundle (Markdown)", bundle,
        file_name="watchlist_reports.md", mime="text/markdown",
    )
    with st.container(border=True):
        st.markdown(bundle)

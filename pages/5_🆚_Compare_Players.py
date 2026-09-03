"""Side-by-side comparison of two or three players."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import LEAGUE_TIER, METRIC_LABELS, POSITION_GROUP_NAMES
from src.pipeline import format_metric
from src.reporting import headline_metrics, ordinal
from src.similarity import explanation_sentences
from src.ui import chart, eyebrow, note, page_setup, sidebar_filters
from src.visualisation import category_bars, metric_comparison_bars, radar_chart

page_setup("Compare Players", "🆚")
platform = sidebar_filters()
pool = platform.pool

st.markdown("# Compare players")
note(
    "Percentiles are only comparable when players share a position group, so the picker defaults "
    "to the first player's group. Comparing across groups is allowed but the percentile columns "
    "then answer different questions and the app says so."
)

count = st.radio("How many players", [2, 3], horizontal=True)

stored = st.session_state.get("selected_player_label")
all_labels = platform.player_labels.sort_values()
default_index = int(np.where(all_labels.to_numpy() == stored)[0][0]) if stored in set(all_labels) else 0

selectors = st.columns(int(count))
indices = []
with selectors[0]:
    first = st.selectbox("Player 1", all_labels.tolist(), index=default_index, key="cmp_0")
    indices.append(platform.index_for_label(first))

base_group = platform.row(indices[0])["position_group"]
same_group_only = st.checkbox(f"Restrict the other picks to {base_group}", value=True)
peer_labels = (
    platform.player_labels.loc[pool["position_group"] == base_group].sort_values()
    if same_group_only else all_labels
)

for i in range(1, int(count)):
    with selectors[i]:
        options = [label for label in peer_labels.tolist() if label != first]
        choice = st.selectbox(f"Player {i + 1}", options, index=min(i * 7, len(options) - 1), key=f"cmp_{i}")
        indices.append(platform.index_for_label(choice))

indices = [i for i in indices if i is not None]
if len(indices) < 2:
    st.stop()

rows = [platform.row(i) for i in indices]
names = [r["player"] for r in rows]
groups = {r["position_group"] for r in rows}
if len(groups) > 1:
    st.warning(
        "These players are in different position groups. Each player's percentiles are still "
        "measured against **his own** peers, so a 90 for a centre-back and a 90 for a winger do "
        "not describe the same thing.",
        icon="⚠️",
    )

# ---- basic information ---------------------------------------------------
st.markdown("## Basic information")
def _profile_fields(index, row) -> list[tuple[str, str]]:
    fields = []
    if platform.has_age:
        fields.append(("Age", f"{row['age']:.1f}"))
    if "nationality" in row.index and pd.notna(row.get("nationality")):
        fields.append(("Nationality", str(row["nationality"])))
    fields += [
        ("Position", f"{row['position']} - {POSITION_GROUP_NAMES[row['position_group']]}"),
        ("Club", str(row["team"])),
        ("League", f"{row['league']} (level {int(row.get('league_tier', 1))})"),
        ("Season", str(row["season"])),
        ("Minutes", f"{row['minutes']:,.0f}"),
        ("Appearances", f"{row['matches']:,.0f}"),
        ("Starts", f"{row['starts']:,.0f}"),
    ]
    if platform.has("height_cm"):
        fields.append(("Height", f"{row['height_cm']:.0f} cm"))
    fields.append(("Archetype", platform.archetype(index)[0]))
    return fields


columns_of = [_profile_fields(i, r) for i, r in zip(indices, rows)]
info = pd.DataFrame(
    {
        "Field": [name for name, _ in columns_of[0]],
        **{r["player"]: [value for _, value in column]
           for r, column in zip(rows, columns_of)},
    }
)
st.dataframe(info, hide_index=True)

# ---- similarity ----------------------------------------------------------
if len(groups) == 1:
    st.markdown("## Similarity")
    pairs = [(a, b) for i, a in enumerate(indices) for b in indices[i + 1 :]]
    columns = st.columns(len(pairs))
    for column, (a, b) in zip(columns, pairs):
        with column:
            similarity = platform.pair_similarity(a, b)
            st.metric(
                f"{platform.row(a)['player']} vs {platform.row(b)['player']}",
                f"{similarity:.1f}%" if np.isfinite(similarity) else "-",
            )
    st.caption(
        "Cosine similarity of the standardised, position-specific profiles - the same number the "
        "Similar Players page ranks on."
    )

# ---- radar + category bars ----------------------------------------------
st.markdown("## Attribute profiles")
scores = {r["player"]: platform.category_scores(i) for i, r in zip(indices, rows)}
shared = [c for c in next(iter(scores.values())) if all(c in s for s in scores.values())]
if shared:
    trimmed = {name: {c: values[c] for c in shared} for name, values in scores.items()}
    columns = st.columns([1, 1.05])
    with columns[0]:
        chart(radar_chart(list(trimmed.items()), height=470))
    with columns[1]:
        chart(category_bars(trimmed, height=470))
    st.caption(
        "The radar shows the shape; the bars are the precise read. Both use the same numbers - "
        "the mean positional percentile of each category's metrics."
    )

# ---- per-90 comparison ---------------------------------------------------
st.markdown("## Per-90 statistics")
group = rows[0]["position_group"]
metrics = [m for m in headline_metrics(group) if m in pool.columns]
per90 = pd.DataFrame({"Metric": [METRIC_LABELS.get(m, m) for m in metrics]})
for i, r in zip(indices, rows):
    per90[r["player"]] = [format_metric(m, pool.loc[i, m]) for m in metrics]
for i, r in zip(indices, rows):
    per90[f"{r['player']} pct"] = [platform.percentile(i, m) for m in metrics]
st.dataframe(per90, hide_index=True, height=430)

st.markdown("### Visual comparison")
selected = st.multiselect(
    "Metrics", metrics, default=metrics[: min(8, len(metrics))],
    format_func=lambda m: METRIC_LABELS.get(m, m),
)
if selected:
    frame = pd.DataFrame({"Metric": [METRIC_LABELS.get(m, m) for m in selected]})
    for i, r in zip(indices, rows):
        frame[r["player"]] = [float(pool.loc[i, m]) for m in selected]
    chart(metric_comparison_bars(frame, "Metric", names))
    st.caption("Raw per-90 rates and percentages - not percentiles, so scales differ by metric.")

# ---- strengths and weaknesses -------------------------------------------
st.markdown("## Strengths and weaknesses")
columns = st.columns(len(indices))
for column, (i, r) in zip(columns, zip(indices, rows)):
    with column:
        st.markdown(f"### {r['player']}")
        eyebrow("Strengths")
        strengths = platform.strengths(i, n=4)
        if strengths.empty:
            st.caption("Nothing above the 70th percentile.")
        for _, item in strengths.iterrows():
            st.markdown(f"- {item['metric']} · {ordinal(item['percentile'])}")
        eyebrow("Weaknesses")
        weaknesses = platform.weaknesses(i, n=4)
        if weaknesses.empty:
            st.caption("Nothing below the 30th percentile.")
        for _, item in weaknesses.iterrows():
            st.markdown(f"- {item['metric']} · {ordinal(item['percentile'])}")

# ---- head-to-head explanation -------------------------------------------
if len(groups) == 1 and len(indices) >= 2:
    st.markdown("## Where the first pair matches and diverges")
    result = platform.explain_similarity(indices[0], indices[1])
    matches, differences = explanation_sentences(result, names[0], names[1], pool)
    columns = st.columns(2)
    with columns[0]:
        eyebrow("Matches")
        for sentence in matches:
            st.markdown(f"- {sentence}")
    with columns[1]:
        eyebrow("Differences")
        for sentence in differences:
            st.markdown(f"- {sentence}")

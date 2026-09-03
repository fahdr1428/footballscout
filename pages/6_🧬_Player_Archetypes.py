"""Explore the K-Means archetypes: how many, what defines them, who belongs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import METRIC_LABELS, POSITION_GROUP_NAMES, POSITION_GROUPS
from src.ui import chart, eyebrow, note, page_setup, sidebar_filters, tiles
from src.visualisation import cluster_facets, cluster_map, elbow_chart, silhouette_chart

page_setup("Player Archetypes", "🧬")
platform = sidebar_filters()
pool = platform.pool

st.markdown("# Player archetypes")
note(
    "K-Means is fitted separately for each position group on that group's own features. "
    "Cluster names are generated from the centroid - the concepts a cluster is strongest in "
    "become the adjectives - so the labels change if the data changes."
)

available = [g for g in POSITION_GROUPS if g in platform.models]
group = st.selectbox(
    "Position group", available, index=available.index("W") if "W" in available else 0,
    format_func=lambda g: f"{g} - {POSITION_GROUP_NAMES[g]}",
)
model = platform.models[group]
members = pool.loc[model.index]
archetypes = model.archetypes

tiles(
    [
        ("Players", f"{len(model.index):,}", f"{group} seasons in the pool"),
        ("Features", f"{len(model.features)}", "position-specific"),
        ("Clusters (k)", f"{model.clusters.k}", f"inertia elbow at k={model.clusters.elbow_k}"),
        ("Silhouette", f"{model.clusters.silhouette:.3f}", "mean over all players"),
        (
            "Map shows",
            f"{model.pca.explained_variance_ratio_[:2].sum():.0%}",
            "of the feature variance",
        ),
    ]
)

# ---- choosing k ----------------------------------------------------------
st.markdown("## How k was chosen")
columns = st.columns(2)
with columns[0]:
    chart(elbow_chart(model.clusters.evaluation, model.clusters.k))
with columns[1]:
    chart(silhouette_chart(model.clusters.evaluation, model.clusters.k))
note(
    "Silhouette on football style data almost always peaks at k=2, because styles form a "
    "continuum rather than separated groups. Taking that argmax gives 'two kinds of "
    f"{POSITION_GROUP_NAMES[group].lower()}', which is true and useless. The rule used here is the "
    "<b>largest k whose silhouette stays within 10% of the best score</b>, subject to every "
    "cluster keeping at least 20 players (or 4% of the group) - an archetype supported by nine "
    "players is a curiosity, not a role. The inertia elbow is shown alongside as a cross-check, "
    "and both curves are above so the choice can be argued with."
)
st.dataframe(
    model.clusters.evaluation.rename(
        columns={"k": "k", "inertia": "Inertia", "silhouette": "Silhouette",
                 "smallest_cluster": "Smallest cluster"}
    ).round({"Inertia": 1, "Silhouette": 3}),
    hide_index=True,
)

# ---- what defines each archetype ----------------------------------------
st.markdown("## What defines each archetype")
summary = pd.DataFrame(
    {
        "Archetype": [model.clusters.names[c] for c in sorted(model.clusters.names)],
        "Players": [int((model.clusters.labels == c).sum()) for c in sorted(model.clusters.names)],
        "Share": [
            f"{(model.clusters.labels == c).mean():.0%}" for c in sorted(model.clusters.names)
        ],
        "Median minutes": [
            int(members.loc[model.clusters.labels == c, "minutes"].median())
            for c in sorted(model.clusters.names)
        ],
    }
)
if platform.has_age:
    summary.insert(
        4, "Median age",
        [round(float(members.loc[model.clusters.labels == c, "age"].median()), 1)
         for c in sorted(model.clusters.names)],
    )
st.dataframe(summary, hide_index=True)
if model.dropped_features:
    st.caption(
        "Not modelled for this position in this dataset (the source does not supply them): "
        + ", ".join(METRIC_LABELS.get(f, f) for f in model.dropped_features)
    )

for cluster in sorted(model.clusters.names):
    with st.expander(
        f"{model.clusters.names[cluster]}  -  {int((model.clusters.labels == cluster).sum())} players",
        expanded=cluster == 0,
    ):
        st.markdown(f"**{model.clusters.descriptions[cluster]}**")
        centroid = model.clusters.centroids.loc[cluster].sort_values(ascending=False)
        frame = pd.DataFrame(
            {
                "Metric": [METRIC_LABELS.get(m, m) for m in centroid.index],
                "Centroid (SD from positional average)": centroid.to_numpy().round(2),
            }
        )
        columns = st.columns([1.1, 1])
        with columns[0]:
            st.dataframe(frame, hide_index=True, height=320)
        with columns[1]:
            eyebrow("Concept scores")
            concepts = model.clusters.concept_scores.loc[cluster].sort_values(ascending=False)
            st.dataframe(
                pd.DataFrame({"Concept": concepts.index, "SD vs average": concepts.to_numpy()}),
                hide_index=True, height=320,
            )
            st.caption(
                "Concept scores are the mean centroid z-score of that concept's metrics, with "
                "'less is better' metrics sign-flipped. The top two above the threshold become "
                "the archetype's name."
            )

        # closest players to the centroid
        distances = np.linalg.norm(
            model.z.to_numpy() - model.clusters.kmeans.cluster_centers_[cluster], axis=1
        )
        ranked = pd.Series(distances, index=model.z.index).sort_values()
        ranked = ranked[model.clusters.labels.reindex(ranked.index) == cluster].head(6)
        eyebrow("Most representative players (closest to the centroid)")
        st.dataframe(
            pd.DataFrame(
                {
                    "Player": members.loc[ranked.index, "player"],
                    "Club": members.loc[ranked.index, "team"],
                    "League": members.loc[ranked.index, "league"],
                    "Minutes": members.loc[ranked.index, "minutes"],
                    "Distance to centroid": ranked.round(2),
                }
            ),
            hide_index=True,
        )

# ---- the map -------------------------------------------------------------
st.markdown("## Archetype map")
view = st.radio(
    "View", ["Highlight one archetype", "Small multiples"], horizontal=True,
    help="Six archetypes cannot be told apart by colour alone at this density, so the map "
         "highlights one at a time or splits into panels rather than using six hues.",
)
meta_columns = ["player", "team", "league", "minutes"] + (["age"] if platform.has_age else [])
meta = members[meta_columns].join(archetypes.rename("archetype"))

if view == "Highlight one archetype":
    columns = st.columns([1.4, 1.4])
    with columns[0]:
        highlight = st.selectbox("Archetype", sorted(model.clusters.names.values()))
    with columns[1]:
        focus_labels = ["(none)"] + platform.player_labels.loc[model.index].sort_values().tolist()
        focus_choice = st.selectbox("Pin a player", focus_labels)
    focus = platform.index_for_label(focus_choice) if focus_choice != "(none)" else None
    chart(
        cluster_map(
            model.coords, meta, highlight=highlight, focus_index=focus,
            explained=(
                float(model.pca.explained_variance_ratio_[0]),
                float(model.pca.explained_variance_ratio_[1]),
            ),
        )
    )
else:
    chart(cluster_facets(model.coords, meta))

note(
    "The map is a PCA projection: it compresses "
    f"{len(model.features)} features into two axes and shows "
    f"{model.pca.explained_variance_ratio_[:2].sum():.0%} of the variance. Two dots sitting "
    "together are not necessarily close in the full feature space - the similarity table is the "
    "authority, the map is orientation."
)

with st.expander("What the two axes are made of", expanded=False):
    for component, loadings in model.loadings.items():
        st.markdown(f"**{component.upper()}** - strongest loadings")
        st.dataframe(
            pd.DataFrame(loadings, columns=["Metric", "Loading"]), hide_index=True,
        )

# ---- browse members ------------------------------------------------------
st.markdown("## Browse an archetype")
chosen = st.selectbox("Archetype", sorted(model.clusters.names.values()), key="browse_archetype")
selection = members[archetypes.reindex(members.index) == chosen]
st.caption(f"{len(selection):,} player-seasons.")
st.dataframe(
    pd.DataFrame(
        {
            "Player": selection["player"],
            "Club": selection["team"],
            "League": selection["league"],
            "Season": selection["season"],
            "Minutes": selection["minutes"],
        }
    ).sort_values("Minutes", ascending=False),
    hide_index=True, height=400,
    column_config={
        "Minutes": st.column_config.NumberColumn(format="%d"),
        "Age": st.column_config.NumberColumn(format="%.1f"),
    },
)

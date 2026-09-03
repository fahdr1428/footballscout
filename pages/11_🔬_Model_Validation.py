"""Model diagnostics and sensitivity testing, run against the live pool."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import METRIC_LABELS, POSITION_GROUP_NAMES
from src.ui import chart, eyebrow, note, page_setup, sidebar_filters
from src.validation import (
    build_validation_report, clustering_diagnostics, correlation_summary, drop_metric_sensitivity,
    feature_dominance, pca_variance, reweight_sensitivity, self_season_recall,
    similarity_role_agreement, team_mate_bias,
)
from src.visualisation import correlation_heatmap

page_setup("Model Validation", "🔬")
platform = sidebar_filters()

st.markdown("# Model validation")
note(
    "Everything on this page is recomputed against the pool currently selected in the sidebar. "
    "Where a result is weak it is shown and explained rather than smoothed over."
)

# ---- clustering ----------------------------------------------------------
st.markdown("## 1. Clustering quality")
diagnostics = clustering_diagnostics(platform)
st.dataframe(diagnostics, hide_index=True)
st.markdown(
    """
Silhouette scores of **0.10-0.25 are normal for football style data**, and should be read
honestly: playing styles are a continuum, not well-separated groups. K-Means is a useful summary
of that continuum, not proof that discrete player types exist.

`k` is chosen as the largest k whose silhouette stays within 10% of the best score **and whose
smallest cluster still holds enough players to mean anything** (at least 20, or 4% of the position
group), with the inertia elbow reported as a cross-check. Both curves are plotted on the Player
Archetypes page.
"""
)
if "adjusted_rand_vs_true_role" in diagnostics.columns:
    st.info(
        "**`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters against the role "
        "profiles used to generate the simulated dataset.** They are only computable because the "
        "sample data is simulated - on a real feed there is no ground truth and these columns "
        "would be absent. A purity around 0.5-0.7 against a 0.25 chance baseline means K-Means is "
        "recovering real structure without perfectly reproducing it, which is the expected result "
        "when roles overlap.",
        icon="🧪",
    )

st.markdown("### How much the 2-D archetype map actually shows")
st.dataframe(pca_variance(platform), hide_index=True)
st.caption(
    "Two principal components carry a minority of the variance in every position. The map is for "
    "orientation; the similarity table is the authority."
)

# ---- role recovery -------------------------------------------------------
agreement = similarity_role_agreement(platform, sample=120)
if agreement.empty:
    st.caption(
        "The generative-role check below is only available on the simulated dataset, where the "
        "true role of every player is known by construction. On real data there is no such "
        "ground truth - the label-free checks that follow are what stand in its place."
    )
if not agreement.empty:
    st.markdown("## 2. Does similarity find players who do the same job?")
    st.dataframe(agreement, hide_index=True)
    st.markdown(
        """
`top10_same_role` is the share of a player's ten nearest neighbours drawn from the **same
generative role**; `chance_baseline` is what random picking would give, and `lift` is the ratio.
A lift above 1 means the engine is recovering role rather than noise. This is the closest thing
to a supervised test available for an unsupervised model - and again, only possible because the
reference dataset is simulated.
"""
    )

# ---- self-season recall (works on real data) -----------------------------
recall = self_season_recall(platform)
if recall.empty:
    st.markdown("## 2b. Does the engine recognise the same player twice?")
    st.info(
        "This check needs a pool spanning **at least two seasons** - it asks where a player's own "
        "other season ranks among his nearest neighbours. Add a season in the sidebar to run it.",
        icon="ℹ️",
    )
if not recall.empty:
    st.markdown("## 2b. Does the engine recognise the same player twice?")
    st.dataframe(recall, hide_index=True)
    st.markdown(
        """
For every player with two seasons in the pool, this asks where his **own other season** ranks
among his nearest neighbours. It is the one case where the right answer is known without any
labels - which makes it the check that still works on real data, where no ground truth about
playing roles exists.

`chance` is what random ordering would produce. A lift well above 1 means the profile the engine
builds is stable enough to recognise the same footballer in a different season, with a different
squad around him.
"""
    )

bias = team_mate_bias(platform, sample=100)
if not bias.empty:
    st.markdown("## 2c. Is it matching on the club rather than the player?")
    st.dataframe(bias, hide_index=True)
    st.caption(
        "Team style leaks into individual numbers - a defender in a possession side passes more "
        "because of the side. Some over-representation of team-mates is expected and correct; a "
        "large lift would mean the model is partly clustering clubs rather than players."
    )

# ---- dominance -----------------------------------------------------------
st.markdown("## 3. Is one metric steering the model?")
group = st.selectbox(
    "Position group", sorted(platform.models),
    format_func=lambda g: f"{g} - {POSITION_GROUP_NAMES[g]}",
)
model = platform.models[group]
dominance = feature_dominance(platform, group)
even = 1 / len(model.features)
st.caption(
    f"With {len(model.features)} equally weighted features, an even share would be "
    f"{even:.3f} per metric."
)
st.dataframe(
    dominance.rename(
        columns={
            "metric": "Metric", "mean_distance_share": "Mean share of pairwise distance",
            "vs_even_share": "× even share",
        }
    )[["Metric", "Mean share of pairwise distance", "× even share"]],
    hide_index=True, height=420,
)
worst = dominance.iloc[0]
if worst["vs_even_share"] >= 2:
    st.warning(
        f"**{worst['metric']}** carries {worst['mean_distance_share']:.1%} of the average pairwise "
        f"distance - {worst['vs_even_share']:.1f}× an even share. Similarity results for "
        f"{group}s lean on it, which is worth knowing before trusting a close match.",
        icon="⚠️",
    )
else:
    st.success(
        f"No metric carries more than {worst['vs_even_share']:.1f}× an even share of the distance "
        f"for {group}s - the model is not resting on a single statistic.",
        icon="✅",
    )

# ---- sensitivity ---------------------------------------------------------
st.markdown("## 4. Sensitivity of the rankings")
st.caption(
    "These tests re-run the nearest-neighbour search for a sample of players under a changed "
    "model and measure how much of the top ten survives. They take a few seconds."
)
if st.button("Run sensitivity tests", type="primary"):
    with st.spinner("Re-running the similarity search under altered models..."):
        drops = drop_metric_sensitivity(platform, group, model.features[:10], sample=60)
        reweights = reweight_sensitivity(platform, group, factor=3.0, sample=60)
    st.session_state[f"sens_{group}"] = (drops, reweights)

cached = st.session_state.get(f"sens_{group}")
if cached:
    drops, reweights = cached
    columns = st.columns(2)
    with columns[0]:
        eyebrow("Removing one metric")
        st.dataframe(
            drops.rename(
                columns={"removed": "Metric removed", "top_k_overlap": "Top-10 overlap",
                         "rank_correlation": "Rank correlation"}
            ),
            hide_index=True, height=380,
        )
    with columns[1]:
        eyebrow("Tripling the weight on one category")
        st.dataframe(
            reweights.rename(
                columns={"category_weighted_x3": "Category", "top_k_overlap": "Top-10 overlap",
                         "rank_correlation": "Rank correlation"}
            ),
            hide_index=True, height=380,
        )
    st.markdown(
        """
**Reading these tables.** `Top-10 overlap` is the Jaccard overlap of the shortlist before and
after the change; `rank correlation` is the Spearman correlation of the survivors' ordering.

- Overlap near 1.0 - the metric or weight barely matters; the ranking is driven by the rest.
- Overlap around 0.5-0.7 - a normal, healthy amount of influence.
- Overlap below ~0.4 - that single metric is effectively steering the position's model, and a
  scout should know which one before acting on a match.
"""
    )

# ---- correlations --------------------------------------------------------
st.markdown("## 5. Correlated features")
corr, pairs = correlation_summary(platform, group, threshold=0.85)
chart(correlation_heatmap(corr, [METRIC_LABELS.get(f, f) for f in model.features]))
if pairs:
    st.dataframe(
        pd.DataFrame(
            [
                {"Metric A": METRIC_LABELS.get(a, a), "Metric B": METRIC_LABELS.get(b, b), "r": r}
                for a, b, r in pairs
            ]
        ),
        hide_index=True,
    )
    st.caption(
        "Pairs above |r| = 0.85. Correlated features double-count one idea inside a distance "
        "calculation. They are reported rather than silently dropped, because to a scout "
        "'progressive passes' and 'passes into the final third' are different questions even when "
        "they move together."
    )
else:
    st.success(f"No pair of {group} features exceeds |r| = 0.85.", icon="✅")

# ---- download ------------------------------------------------------------
st.divider()
st.markdown("## Full written report")
if st.button("Build the full validation report"):
    with st.spinner("Running every diagnostic..."):
        report = build_validation_report(platform)
    st.session_state["validation_report"] = report

report = st.session_state.get("validation_report")
if report:
    st.download_button(
        "Download validation_report.md", report, file_name="validation_report.md",
        mime="text/markdown",
    )
    with st.container(border=True):
        st.markdown(report)

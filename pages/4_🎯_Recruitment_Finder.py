"""Turn a recruitment brief into a weighted, ranked shortlist."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_ROLE, DEFAULT_WEIGHTS, LEAGUE_STRENGTH, METRIC_LABELS, POSITION_GROUP_NAMES,
    POSITION_GROUPS, categories_for, role_weights, roles_for,
)
from src.feature_engineering import metrics_for_percentiles
from src.pipeline import format_metric
from src.recruitment import RecruitmentBrief, search, threshold_summary
from src.reporting import generate_report
from src.ui import (
    add_to_watchlist, age_slider, chart, default_axis, eyebrow, note, page_setup,
    position_filters, sidebar_filters, tiles, watchlist_sidebar,
)
from src.visualisation import component_bar, radar_chart, scatter

page_setup("Recruitment Finder", "🎯")
platform = sidebar_filters()
pool = platform.pool

st.markdown("# Recruitment finder")
note(
    "Hard filters decide who is eligible; the weights decide the order. The fit score is the "
    "weighted mean of the player's positional percentiles - nothing else goes into it."
)

# ---- the brief -----------------------------------------------------------
st.markdown("## 1. Define the profile")
row1 = st.columns([1.1, 1.2, 1.1, 1.1])
with row1[0]:
    groups_present = [g for g in POSITION_GROUPS if g in set(pool["position_group"].unique())]
    default_group = "W" if "W" in groups_present else groups_present[min(2, len(groups_present) - 1)]
    group = st.selectbox(
        "Position", groups_present, index=groups_present.index(default_group),
        format_func=lambda g: f"{g} - {POSITION_GROUP_NAMES[g]}",
    )
with row1[1]:
    age_range = age_slider(platform, "Age range", (18.0, 23.0), key="rec_age") or (0.0, 99.0)
with row1[2]:
    min_minutes = st.number_input(
        "Minimum minutes", int(platform.min_minutes), 3400, max(900, int(platform.min_minutes)), 50
    )
with row1[3]:
    max_strength = st.slider(
        "Maximum league strength", 0.5, 1.0, 1.0, 0.02,
        help="Filters to leagues at or below this coefficient. The coefficients live in "
             "src/config.py and are assumptions, not measurements.",
    )

filter_row = st.columns([2, 1.3, 1.2])
with filter_row[0]:
    leagues = st.multiselect("Restrict to leagues (optional)",
                             sorted(pool["league"].unique()))

# A budget and a preferred foot are two of the first things a real brief
# specifies, so both appear wherever the dataset can answer them.
max_market_value = None
with filter_row[1]:
    if platform.has("market_value_eur"):
        values = pd.to_numeric(pool["market_value_eur"], errors="coerce").dropna()
        ceiling = float(values.max() / 1e6) if len(values) else 100.0
        budget = st.slider("Budget (market value, EURm)", 0.0, ceiling, ceiling,
                           max(0.5, round(ceiling / 100, 1)),
                           help="Transfermarkt market value for that season - an estimate of "
                                "what the player is worth, not a fee that was paid. Players "
                                "with no recorded value are kept and flagged.")
        if budget < ceiling:
            max_market_value = budget * 1e6
    else:
        st.caption("No market values in this dataset.")
with filter_row[2]:
    feet: list[str] = []
    if platform.has("foot"):
        options = sorted(pool["foot"].dropna().astype(str).str.lower().unique())
        feet = st.multiselect("Preferred foot", options,
                              help="A left-footed centre-back or right-back is a specific "
                                   "recruitment brief, not a preference.")
    else:
        st.caption("No footedness in this dataset.")

# Side of the pitch and inverted-foot are filters, not models: a left-back and a
# right-back share a peer set because the data says they do the same job, but a
# shortlist for a left-back should only contain left-backs.
group_pool = pool[pool["position_group"] == group]
_, _, position_choice = position_filters(platform, group_pool, "rec")

metric_choices = sorted(
    metrics_for_percentiles(group, list(pool.columns)), key=lambda m: METRIC_LABELS.get(m, m)
)
with st.expander("Desired characteristics (hard thresholds on raw rates)", expanded=True):
    st.caption(
        "Example brief for a winger: goals/90 > 0.25, xA/90 > 0.15, progressive carries/90 > 5, "
        "successful dribbles/90 > 2, key passes/90 > 1.5."
    )
    count = st.number_input("Number of conditions", 0, 6, 3, 1)
    thresholds: list[tuple[str, str, float]] = []
    subset = pool[pool["position_group"] == group]
    suggested = {
        "W": ["np_goals_per90", "xa_per90", "progressive_carries_per90"],
        "FW": ["np_goals_per90", "npxg_per90", "touches_att_pen_per90"],
        "CB": ["aerial_win_pct", "progressive_passes_per90", "interceptions_per90"],
        "FB": ["progressive_carries_per90", "xa_per90", "tackles_per90"],
        "DM": ["interceptions_per90", "progressive_passes_per90", "pass_pct"],
        "CM": ["progressive_passes_per90", "xa_per90", "ball_recoveries_per90"],
        "AM": ["xa_per90", "key_passes_per90", "npxg_per90"],
        "GK": ["gk_save_pct", "gk_psxg_minus_ga_per90", "pass_pct"],
        # the four-bucket taxonomy used by summary feeds
        "DEF": ["cbi_per90", "ball_recoveries_per90", "xa_per90"],
        "MID": ["xa_per90", "xg_per90", "threat_per90"],
        "FWD": ["xg_per90", "goals_per90", "threat_per90"],
    }.get(group, [])
    suggested = [m for m in suggested if m in metric_choices] or metric_choices[:3]
    for i in range(int(count)):
        columns = st.columns([2.4, 0.8, 1, 1.2])
        default_metric = suggested[i] if i < len(suggested) else metric_choices[0]
        with columns[0]:
            metric = st.selectbox(
                "Metric", metric_choices,
                index=metric_choices.index(default_metric) if default_metric in metric_choices else 0,
                key=f"rec_metric_{i}", format_func=lambda m: METRIC_LABELS.get(m, m),
            )
        with columns[1]:
            operator = st.selectbox("Op", [">=", ">", "<=", "<"], key=f"rec_op_{i}")
        series = subset[metric].dropna()
        # Default to the positional median so the example brief returns a usable
        # shortlist; the caption shows how many players clear each condition.
        default_value = float(round(series.quantile(0.55), 2)) if len(series) else 0.0
        with columns[2]:
            value = st.number_input(
                "Value", value=default_value, step=0.05, format="%.2f", key=f"rec_val_{i}"
            )
        with columns[3]:
            if len(series):
                share = float((series >= value).mean() if operator in {">=", ">"} else (series <= value).mean())
                st.caption(
                    f"Positional median {series.median():.2f} · "
                    f"{share:.0%} of {group}s clear this"
                )
        thresholds.append((metric, operator, float(value)))

# ---- weights -------------------------------------------------------------
st.markdown("## 2. Weight what matters")
category_names = list(categories_for(group))

# A position is not a job: two centre-backs can be recruited against opposite
# briefs. Picking a role reseeds the sliders, which stay editable afterwards.
role = st.selectbox(
    "Role", roles_for(group), key=f"role_{group}",
    help="Each role is a named starting set of weights, listed in src/config.ROLE_TEMPLATES. "
         "They are assumptions about what a role asks for, not measurements - move the "
         "sliders and the shortlist follows.",
)
seed_key = f"_seeded_{group}"
if st.session_state.get(seed_key) != role:
    st.session_state[seed_key] = role
    template = role_weights(group, role)
    for category in category_names:
        st.session_state[f"weight_{group}_{category}"] = int(template.get(category, 0))

weight_columns = st.columns(min(4, len(category_names)))
weights: dict[str, float] = {}
for i, category in enumerate(category_names):
    with weight_columns[i % len(weight_columns)]:
        weights[category] = st.slider(
            category, 0, 60, step=5, key=f"weight_{group}_{category}"
        )
if role != DEFAULT_ROLE:
    st.caption(f"Starting weights for **{role}**. Adjust any slider to depart from the template.")

total = sum(weights.values())
if total == 0:
    st.warning("Give at least one category a weight above zero.")
    st.stop()
normalised = {k: 100 * v / total for k, v in weights.items() if v > 0}
badge_text = " · ".join(f"{k} {v:.0f}%" for k, v in sorted(normalised.items(), key=lambda kv: -kv[1]))
st.caption(f"Normalised weights: {badge_text}")

brief = RecruitmentBrief(
    position_group=group,
    min_minutes=int(min_minutes),
    age_range=age_range,
    leagues=leagues,
    max_league_strength=max_strength if max_strength < 1.0 else None,
    thresholds=thresholds,
    weights=weights,
    role=None if role == DEFAULT_ROLE else role,
    max_market_value=max_market_value,
    feet=feet,
    positions=position_choice["positions"],
    flanks=position_choice["flanks"],
    footed_sides=position_choice["footed_sides"],
)

# ---- results -------------------------------------------------------------
st.markdown("## 3. Shortlist")
results = search(pool, platform.categories, brief, top_n=60)
eligible = int(len(results))

if results.empty or "fit_score" not in results.columns or results["fit_score"].isna().all():
    st.info(
        "No player clears every condition. Relax a threshold, widen the age range, or lower the "
        "minimum minutes."
    )
    st.stop()

tiles(
    [
        ("Shortlisted", f"{eligible:,}", f"of {int((pool['position_group'] == group).sum()):,} {group} seasons"),
        ("Best fit", f"{results['fit_score'].max():.0f}", "weighted percentile score"),
        (
            ("Median age", f"{results['age'].median():.1f}", "years")
            if platform.has_age
            else ("Median minutes", f"{results['minutes'].median():,.0f}", "per season")
        ),
        ("Leagues", f"{results['league'].nunique()}", "represented"),
    ]
)

view = pd.DataFrame(
    {
        "Rank": results["rank"],
        "Fit score": results["fit_score"],
        "Player": results["player"],
        "Club": results["team"],
        "League": results["league"],
        "Season": results["season"],
        "Minutes": results["minutes"],
        "Archetype": results["archetype"],
    }
)
if platform.has_age:
    view.insert(3, "Age", results["age"].to_numpy())
for category in normalised:
    column = f"cat_{category}"
    if column in platform.categories.columns:
        view[category] = platform.categories.loc[results.index, column].round(0)

st.dataframe(
    view, hide_index=True, height=430,
    column_config={
        "Fit score": st.column_config.ProgressColumn(
            "Fit score", min_value=0, max_value=100, format="%.1f"
        ),
        "Minutes": st.column_config.NumberColumn(format="%d"),
        "Age": st.column_config.NumberColumn(format="%.1f"),
    },
)
note(
    "Category columns are the player's mean positional percentile inside that category. "
    "Fit score = sum(weight × category percentile) ÷ 100."
)

shortlist = st.multiselect(
    "Add to the watchlist", platform.player_labels.loc[results.index].tolist(), key="rec_watch"
)
if shortlist and st.button("Add selected to watchlist"):
    for label in shortlist:
        target = platform.index_for_label(label)
        if target is not None:
            add_to_watchlist(platform, target)
    st.rerun()
watchlist_sidebar()

st.markdown("### Fit against age" if platform.has_age else "### Fit against minutes played")
chart(
    scatter(
        results, x=default_axis(platform), y="fit_score", hover_name="player",
        hover_cols=["team", "league", "minutes", "archetype"],
        x_title="Age" if platform.has_age else "Minutes played",
        y_title="Recruitment fit score", height=420,
    )
)

# ---- candidate detail ----------------------------------------------------
st.divider()
st.markdown("## 4. Inspect a candidate")
labels = platform.player_labels.loc[results.index]
chosen = st.selectbox("Candidate", labels.tolist())
index = platform.index_for_label(chosen)

if index is not None:
    st.session_state["selected_player_label"] = chosen
    candidate = platform.row(index)
    fit = float(results.loc[index, "fit_score"])
    detail = st.columns([1, 1.1])

    with detail[0]:
        eyebrow("Fit breakdown")
        st.markdown(f"### {candidate['player']} - fit score {fit:.1f}")
        breakdown = pd.DataFrame(
            [
                {
                    "Category": category,
                    "Weight %": round(weight, 1),
                    "Percentile": platform.categories.loc[index, f"cat_{category}"],
                    "Points": round(results.loc[index, f"contrib_{category}"], 1),
                }
                for category, weight in normalised.items()
                if f"contrib_{category}" in results.columns
            ]
        ).sort_values("Points", ascending=False)
        st.dataframe(breakdown, hide_index=True)
        st.caption(
            f"The points column sums to the fit score of {fit:.1f}. A high weight on a category "
            "the player is weak in shows up here as a small contribution."
        )
        eyebrow("Brief")
        for line in threshold_summary(brief):
            st.markdown(f"- {line}")

    with detail[1]:
        chart(radar_chart([(candidate["player"], platform.category_scores(index))], height=430))
        strengths = platform.strengths(index, n=4)
        eyebrow("Top metrics")
        for _, item in strengths.iterrows():
            st.markdown(
                f"- **{item['metric']}** {format_metric(item['key'], item['value'])} "
                f"({item['percentile']:.0f}th pct)"
            )

    if st.button("Generate scouting report for this candidate", type="primary"):
        report = generate_report(platform, index, brief=brief)
        st.download_button(
            "Download as Markdown", report,
            file_name=f"recruitment_report_{candidate['player'].replace(' ', '_')}.md",
            mime="text/markdown",
        )
        with st.container(border=True):
            st.markdown(report)

"""
Shared Streamlit helpers: caching, page chrome, filters and reusable blocks.

The rule this module enforces is that pages contain layout only. Anything that
loads, fits or scores lives in the analytics modules and arrives here through a
cached `ScoutingPlatform`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .config import (
    LEAGUE_TIER,
    LEAGUES,
    METRIC_LABELS,
    MINUTES_PRESETS,
    POSITION_GROUP_NAMES,
    POSITION_GROUPS,
    SEASONS,
    THEME,
)
from .pipeline import ScoutingPlatform, build_features, build_platform, format_metric

DATA_NOTICE = (
    "Simulated dataset - these are not real players. The pipeline reads a real "
    "FBref-style export just as happily; see the Methodology page."
)

CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
  h1, h2, h3 {{ letter-spacing: -0.015em; }}
  h1 {{ font-size: 2.05rem !important; font-weight: 700 !important; }}
  h2 {{ font-size: 1.32rem !important; font-weight: 650 !important; margin-top: 1.6rem !important; }}
  h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}

  .sx-eyebrow {{
    text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.7rem;
    color: {THEME['ink_muted']}; font-weight: 600; margin-bottom: 0.25rem;
  }}
  .sx-card {{
    background: {THEME['panel']}; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 1rem 1.15rem; height: 100%;
  }}
  .sx-tiles {{ display: flex; gap: 0.6rem; flex-wrap: wrap; }}
  .sx-tile {{
    background: {THEME['panel']}; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 0.7rem 0.95rem; min-width: 118px; max-width: 230px;
    flex: 1 1 118px;
  }}
  .sx-tile .k {{ font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.08em;
                 color: {THEME['ink_muted']}; font-weight: 600; min-height: 2.1em; }}
  .sx-tile .v {{ font-size: 1.42rem; font-weight: 680; color: {THEME['ink']}; line-height: 1.35; }}
  .sx-tile .s {{ font-size: 0.76rem; color: {THEME['ink_secondary']}; }}

  .sx-badge {{
    display: inline-block; padding: 0.18rem 0.6rem; border-radius: 999px;
    font-size: 0.74rem; font-weight: 600; margin-right: 0.35rem; margin-bottom: 0.3rem;
    border: 1px solid rgba(255,255,255,0.12); color: {THEME['ink_secondary']};
  }}
  .sx-badge.accent {{ background: rgba(57,135,229,0.16); border-color: rgba(57,135,229,0.45);
                      color: #cfe1fb; }}
  .sx-badge.good {{ background: rgba(12,163,12,0.14); border-color: rgba(12,163,12,0.42); color: #b9e8b9; }}
  .sx-badge.warn {{ background: rgba(250,178,25,0.14); border-color: rgba(250,178,25,0.42); color: #f6dda6; }}

  .sx-note {{
    border-left: 2px solid {THEME['baseline']}; padding: 0.15rem 0 0.15rem 0.75rem;
    color: {THEME['ink_muted']}; font-size: 0.82rem; margin: 0.5rem 0 0.9rem 0;
  }}
  .sx-hero {{
    background: linear-gradient(135deg, rgba(57,135,229,0.14), rgba(25,158,112,0.08));
    border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 1.4rem 1.6rem;
  }}
  [data-testid="stSidebar"] {{ border-right: 1px solid rgba(255,255,255,0.07); }}
  [data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
  div[data-testid="stDataFrame"] {{ border-radius: 8px; }}
</style>
"""


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_features():
    features, report = build_features()
    return features, report


@st.cache_resource(show_spinner=False)
def get_platform(min_minutes: int, seasons: tuple[str, ...]) -> ScoutingPlatform:
    features, report = _cached_features()
    return build_platform(features, report, min_minutes=min_minutes, seasons=list(seasons))


# --------------------------------------------------------------------------
# Page chrome
# --------------------------------------------------------------------------

def page_setup(title: str, icon: str = "⚽") -> None:
    """Page config (once per run) plus the shared stylesheet."""
    try:
        st.set_page_config(
            page_title=f"{title} - Scouting Intelligence", page_icon=icon, layout="wide"
        )
    except Exception:
        # app.py already configured the page when running under st.navigation.
        pass
    st.markdown(CSS, unsafe_allow_html=True)


def chart(fig, **kwargs) -> None:
    """Render a Plotly figure with our own template (Streamlit's would override it)."""
    st.plotly_chart(fig, theme=None, **kwargs)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="sx-eyebrow">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="sx-note">{text}</div>', unsafe_allow_html=True)


def badges(items: list[tuple[str, str]]) -> None:
    html = "".join(f'<span class="sx-badge {tone}">{label}</span>' for label, tone in items)
    st.markdown(html, unsafe_allow_html=True)


def tiles(items: list[tuple[str, str, str]]) -> None:
    """Compact stat tiles: (label, value, sub-label)."""
    html = '<div class="sx-tiles">' + "".join(
        f'<div class="sx-tile"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="s">{s}</div></div>'
        for k, v, s in items
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def sidebar_filters(default_minutes: int | None = None) -> ScoutingPlatform:
    """Pool-defining controls, shared by every page."""
    with st.sidebar:
        st.markdown("### Comparison pool")
        st.caption(
            "These settings define who every percentile, cluster and similarity score is "
            "measured against."
        )
        options = MINUTES_PRESETS + ["Custom"]
        stored = st.session_state.get("min_minutes", default_minutes or 900)
        default_choice = options.index(stored) if stored in options else len(options) - 1
        choice = st.radio(
            "Minimum minutes played", options, index=default_choice, horizontal=True,
            format_func=lambda v: f"{v:,}" if isinstance(v, int) else v,
            help="Per-90 rates from small samples are unreliable. Raising this filter removes "
                 "fringe players from every percentile and every similarity search.",
        )
        minutes = (
            st.slider("Custom minimum", 200, 3000, int(stored) if isinstance(stored, int) else 900, 100)
            if choice == "Custom" else int(choice)
        )
        seasons = st.multiselect("Seasons", SEASONS, default=st.session_state.get("seasons", SEASONS))
        if not seasons:
            seasons = list(SEASONS)
        st.session_state["min_minutes"] = minutes
        st.session_state["seasons"] = seasons

    with st.spinner("Fitting position models..."):
        platform = get_platform(minutes, tuple(sorted(seasons)))

    with st.sidebar:
        st.markdown(
            f'<div class="sx-note">Pool: <b>{len(platform.pool):,}</b> player-seasons across '
            f'{platform.pool["league"].nunique()} leagues.<br>{DATA_NOTICE}</div>',
            unsafe_allow_html=True,
        )
    return platform


# --------------------------------------------------------------------------
# Player selection
# --------------------------------------------------------------------------

def player_selector(
    platform: ScoutingPlatform,
    label: str = "Player",
    key: str = "player",
    restrict_group: str | None = None,
    use_session: bool = True,
    compact: bool = False,
):
    """Position/league narrowed player picker. Returns a pool index or None."""
    pool = platform.pool
    if restrict_group:
        pool = pool[pool["position_group"] == restrict_group]

    columns = st.columns([1, 1, 2] if not compact else [1, 2])
    if not compact:
        with columns[0]:
            group = st.selectbox(
                "Position", ["All"] + POSITION_GROUPS, key=f"{key}_group",
                format_func=lambda g: g if g == "All" else f"{g} - {POSITION_GROUP_NAMES[g]}",
                disabled=restrict_group is not None,
                index=0 if not restrict_group else POSITION_GROUPS.index(restrict_group) + 1,
            )
        with columns[1]:
            leagues = st.multiselect("League", sorted(pool["league"].unique()), key=f"{key}_leagues")
        target = columns[2]
    else:
        with columns[0]:
            group, leagues = "All", []
            leagues = st.multiselect("League", sorted(pool["league"].unique()), key=f"{key}_leagues")
        target = columns[1]

    if group != "All":
        pool = pool[pool["position_group"] == group]
    if leagues:
        pool = pool[pool["league"].isin(leagues)]
    if pool.empty:
        st.warning("No players match those filters.")
        return None

    labels = platform.player_labels.loc[pool.index].sort_values()
    stored = st.session_state.get("selected_player_label") if use_session else None
    default = int(np.where(labels.to_numpy() == stored)[0][0]) if stored in set(labels) else 0
    with target:
        chosen = st.selectbox(label, labels.tolist(), index=default, key=f"{key}_select")
    index = platform.index_for_label(chosen)
    if use_session and index is not None:
        st.session_state["selected_player_label"] = chosen
    return index


def player_header(platform: ScoutingPlatform, index, show_archetype: bool = True) -> None:
    row = platform.row(index)
    archetype, _ = platform.archetype(index) if show_archetype else ("", "")
    st.markdown(f"## {row['player']}")
    items = [
        (f"{row['position']} - {POSITION_GROUP_NAMES[row['position_group']]}", "accent"),
        (f"{row['team']}", ""),
        (f"{row['league']} (level {LEAGUE_TIER.get(row['league'], 1)})", ""),
        (f"{row['season']}", ""),
        (f"Age {row['age']:.1f}", ""),
        (f"{row['height_cm']:.0f} cm", ""),
        (f"{row['minutes']:,.0f} min", "good" if row["minutes"] >= 1500 else "warn"),
    ]
    if show_archetype:
        items.insert(1, (archetype, "accent"))
    badges(items)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

SIMILARITY_COLUMNS = {
    "similarity": "Similarity %",
    "player": "Player",
    "age": "Age",
    "position": "Pos",
    "team": "Club",
    "league": "League",
    "minutes": "Minutes",
    "archetype": "Archetype",
}


def similarity_table(results: pd.DataFrame, extra: dict[str, str] | None = None) -> None:
    columns = {**SIMILARITY_COLUMNS, **(extra or {})}
    available = [c for c in columns if c in results.columns]
    frame = results[available].rename(columns=columns)
    st.dataframe(
        frame,
        hide_index=True,
        column_config={
            "Similarity %": st.column_config.ProgressColumn(
                "Similarity %", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Minutes": st.column_config.NumberColumn("Minutes", format="%d"),
            "Age": st.column_config.NumberColumn("Age", format="%.1f"),
        },
    )


def metric_table(platform: ScoutingPlatform, index, metrics: list[str]) -> pd.DataFrame:
    frame = platform.percentile_frame(index, metrics)
    frame["display"] = [format_metric(k, v) for k, v in zip(frame["key"], frame["value"])]
    return frame


def league_options() -> list[str]:
    return [league["name"] for league in LEAGUES]


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)

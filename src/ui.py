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
    DATA_SOURCES,
    DEFAULT_SOURCE,
    LEAGUE_TIER,
    LEAGUES,
    METRIC_LABELS,
    MINUTES_PRESETS,
    POSITION_GROUP_NAMES,
    POSITION_GROUPS,
    THEME,
)
from .pipeline import ScoutingPlatform, build_features, build_platform, format_metric

SOURCE_HELP = (
    "Both datasets run through identical cleaning, feature and modelling code. "
    "Switching refits every position model."
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
def _cached_features(source: str):
    features, report, used = build_features(source)
    return features, report, used


@st.cache_resource(show_spinner=False)
def get_platform(
    source: str, min_minutes: int, seasons: tuple[str, ...], leagues: tuple[str, ...]
) -> ScoutingPlatform:
    features, report, used = _cached_features(source)
    return build_platform(
        features, report, min_minutes=min_minutes, seasons=list(seasons),
        leagues=list(leagues), source=used,
    )


@st.cache_data(show_spinner=False)
def source_scope(source: str) -> pd.DataFrame:
    """Seasons, leagues and (where present) competition gender for a source."""
    features, _report, _used = _cached_features(source)
    columns = ["league", "season"] + (["gender"] if "gender" in features.columns else [])
    return features[columns].drop_duplicates().reset_index(drop=True)


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
    """Dataset choice and the pool-defining controls, shared by every page."""
    with st.sidebar:
        st.markdown("### Dataset")
        keys = list(DATA_SOURCES)
        stored_source = st.session_state.get("source", DEFAULT_SOURCE)
        source = st.radio(
            "Source", keys,
            index=keys.index(stored_source) if stored_source in keys else 0,
            format_func=lambda k: DATA_SOURCES[k].label,
            help=SOURCE_HELP,
        )
        st.session_state["source"] = source

        scope = source_scope(source)
        seasons_available = sorted(scope["season"].unique())

        st.markdown("### Comparison pool")
        st.caption(
            "These settings define who every percentile, cluster and similarity score is "
            "measured against."
        )

        leagues_available = sorted(scope["league"].unique())
        if "gender" in scope.columns and scope["gender"].nunique() > 1:
            options = ["All competitions", "Men's football", "Women's football"]
            choice = st.radio(
                "Competitions", options,
                index=options.index(st.session_state.get("scope_%s" % source, options[0])),
                help="This dataset spans men's and women's competitions. A percentile is a "
                     "statement about a peer group, so scope the pool before reading one.",
            )
            st.session_state["scope_%s" % source] = choice
            if choice != "All competitions":
                gender = "male" if choice.startswith("Men") else "female"
                leagues_available = sorted(scope.loc[scope["gender"] == gender, "league"].unique())
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
        spec_defaults = [s for s in DATA_SOURCES[source].default_seasons if s in seasons_available]
        picked_leagues = st.session_state.get("leagues_%s" % source, leagues_available)
        picked_leagues = [lg for lg in picked_leagues if lg in leagues_available] or leagues_available
        leagues = st.multiselect("Leagues in the pool", leagues_available, default=picked_leagues)
        if not leagues:
            leagues = list(leagues_available)

        seasons_available = sorted(scope.loc[scope["league"].isin(leagues), "season"].unique())
        picked = st.session_state.get(
            "seasons_%s" % source, spec_defaults or seasons_available
        )
        picked = [s for s in picked if s in seasons_available] or (
            spec_defaults or seasons_available
        )
        seasons = st.multiselect(
            "Seasons", seasons_available, default=picked,
            help="Metric coverage varies by season on a summary feed. Picking a single season "
                 "lets the models use everything that season measured.",
        )
        if not seasons:
            seasons = list(seasons_available)
        st.session_state["min_minutes"] = minutes
        st.session_state["seasons_%s" % source] = seasons
        st.session_state["leagues_%s" % source] = leagues

    with st.spinner("Fitting position models..."):
        platform = get_platform(
            source, minutes, tuple(sorted(seasons)), tuple(sorted(leagues))
        )

    with st.sidebar:
        if platform.source != source:
            st.warning(
                f"{DATA_SOURCES[source].label} has not been built yet - run "
                "`python scripts/fetch_statsbomb.py`. Showing "
                f"{DATA_SOURCES[platform.source].label} instead.",
                icon="⚠️",
            )
        spec = platform.spec
        tone = "good" if platform.is_real else "warn"
        st.markdown(
            f'<div class="sx-note"><span class="sx-badge {tone}">'
            f'{"Real data" if platform.is_real else "Simulated data"}</span><br>'
            f'Pool: <b>{len(platform.pool):,}</b> player-seasons · '
            f'{platform.pool["league"].nunique()} leagues · '
            f'{platform.pool["team"].nunique()} clubs.<br>{spec.attribution}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("What this dataset can and cannot say"):
            for caveat in spec.caveats:
                st.markdown(f"- {caveat}")
    return platform


def source_banner(platform: ScoutingPlatform) -> None:
    """The standing statement about what the numbers on the page are."""
    spec = platform.spec
    if platform.is_real:
        st.info(f"**{spec.label}.** {spec.summary} {spec.attribution}", icon="✅")
    else:
        st.warning(f"**{spec.label}.** {spec.summary}", icon="⚠️")


# --------------------------------------------------------------------------
# Optional-column helpers
# --------------------------------------------------------------------------
# A real feed may not publish everything the schema supports (StatsBomb open
# data has no birth dates). Rather than invent values, pages ask the platform
# what it has and drop the affected control or column.

AGE_MISSING_NOTE = (
    "Age is not published in this dataset, so age filters, the age-upside score component "
    "and the age columns are switched off."
)


def age_slider(platform, label: str = "Age", default=(15.0, 40.0), key: str | None = None):
    """Age range control, or None when the dataset has no ages."""
    if not platform.has_age:
        return None
    low = float(np.floor(platform.pool["age"].min()))
    high = float(np.ceil(platform.pool["age"].max()))
    default = (max(default[0], low), min(default[1], high))
    return st.slider(label, low, high, default, 0.5, key=key)


def apply_age(pool: pd.DataFrame, age_range) -> pd.DataFrame:
    if age_range is None or "age" not in pool.columns:
        return pool
    return pool[pool["age"].between(*age_range)]


def age_columns(platform, frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Insert an Age column only when the dataset supplies one."""
    if platform.has_age and "age" in source.columns:
        frame.insert(min(2, len(frame.columns)), "Age", source["age"].to_numpy())
    return frame


def default_axis(platform, preferred: str = "age", fallback: str = "minutes") -> str:
    return preferred if platform.has(preferred) else fallback


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


def format_market_value(value) -> str:
    """Transfermarkt values in the units a scout reads them in."""
    if pd.isna(value):
        return "-"
    value = float(value)
    if value >= 1e6:
        return f"EUR {value / 1e6:.1f}m"
    if value >= 1e3:
        return f"EUR {value / 1e3:.0f}k"
    return f"EUR {value:.0f}"


def player_header(platform: ScoutingPlatform, index, show_archetype: bool = True) -> None:
    row = platform.row(index)
    archetype, _ = platform.archetype(index) if show_archetype else ("", "")
    st.markdown(f"## {row['player']}")
    items = [
        (f"{row['position']} - {POSITION_GROUP_NAMES[row['position_group']]}", "accent"),
        (f"{row['team']}", ""),
        (f"{row['league']} (level {int(row.get('league_tier', 1))})", ""),
        (f"{row['season']}", ""),
    ]
    # Only show what the dataset actually supplies.
    if "nationality" in row.index and pd.notna(row.get("nationality")):
        items.append((str(row["nationality"]), ""))
    if platform.has_age and pd.notna(row.get("age")):
        items.append((f"Age {row['age']:.1f}", ""))
    if platform.has("height_cm") and pd.notna(row.get("height_cm")):
        items.append((f"{row['height_cm']:.0f} cm", ""))
    if platform.has("foot") and pd.notna(row.get("foot")):
        items.append((f"{str(row['foot']).capitalize()}-footed", ""))
    if platform.has("market_value_eur") and pd.notna(row.get("market_value_eur")):
        items.append((format_market_value(row["market_value_eur"]), "accent"))
    if platform.has("price_m") and pd.notna(row.get("price_m")):
        items.append((f"£{row['price_m']:.1f}m", ""))
    items.append((f"{row['minutes']:,.0f} min", "good" if row["minutes"] >= 1500 else "warn"))
    if show_archetype:
        items.insert(1, (archetype, "accent"))
    badges(items)
    if "position_source" in row.index and pd.notna(row.get("position_source")):
        source = str(row["position_source"])
        detail = row.get("detailed_position")
        if pd.notna(detail):
            st.caption(
                f"Line-up position: **{detail}** - {source}. "
                "Models group by the source's own positional buckets."
            )
        elif source.startswith("Transfermarkt") and "broad" not in source:
            st.caption(
                f"Position **{row['position']}** recorded by {source}, independently of the "
                "statistics the models read - so grouping by position is not circular."
            )
        else:
            st.caption(f"Position from {source}: only a broad grouping was available.")


# --------------------------------------------------------------------------
# Watchlist
# --------------------------------------------------------------------------
# A scout's shortlist, held in the session so it survives moving between pages.
# Entries are stored by player identity rather than row position, so changing
# the minimum-minutes filter or the dataset does not scramble them.

WATCHLIST_KEY = "watchlist"


def _entry(platform: ScoutingPlatform, index) -> dict:
    row = platform.row(index)
    return {
        "player_id": str(row["player_id"]),
        "season": str(row["season"]),
        "player": str(row["player"]),
        "team": str(row["team"]),
        "league": str(row["league"]),
        "source": platform.source,
    }


def watchlist() -> list[dict]:
    return st.session_state.setdefault(WATCHLIST_KEY, [])


def in_watchlist(platform: ScoutingPlatform, index) -> bool:
    entry = _entry(platform, index)
    return any(
        e["player_id"] == entry["player_id"] and e["season"] == entry["season"]
        for e in watchlist()
    )


def add_to_watchlist(platform: ScoutingPlatform, index) -> bool:
    if in_watchlist(platform, index):
        return False
    watchlist().append(_entry(platform, index))
    return True


def remove_from_watchlist(player_id: str, season: str) -> None:
    st.session_state[WATCHLIST_KEY] = [
        e for e in watchlist() if not (e["player_id"] == player_id and e["season"] == season)
    ]


def watchlist_indices(platform: ScoutingPlatform) -> list:
    """Resolve stored entries to rows in the current pool, dropping any that fell out."""
    lookup = {
        (str(r.player_id), str(r.season)): i
        for i, r in zip(platform.pool.index, platform.pool.itertuples())
    }
    return [
        lookup[(e["player_id"], e["season"])]
        for e in watchlist()
        if (e["player_id"], e["season"]) in lookup
    ]


def watchlist_button(platform: ScoutingPlatform, index, key: str) -> None:
    """A single add/remove control, used from any page."""
    entry = _entry(platform, index)
    if in_watchlist(platform, index):
        if st.button("Remove from watchlist", key=f"wl_del_{key}"):
            remove_from_watchlist(entry["player_id"], entry["season"])
            st.rerun()
    elif st.button("Add to watchlist", key=f"wl_add_{key}"):
        add_to_watchlist(platform, index)
        st.rerun()


def watchlist_sidebar() -> None:
    entries = watchlist()
    if entries:
        with st.sidebar:
            st.markdown(f"### Watchlist · {len(entries)}")
            st.caption(", ".join(e["player"] for e in entries[:6]) + ("..." if len(entries) > 6 else ""))
            st.page_link("pages/10_📋_Watchlist.py", label="Open watchlist", icon="📋")


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

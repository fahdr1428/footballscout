"""
Central configuration for the scouting platform.

Everything that defines *what* the models look at lives here:
the metric registry, the position groups, the per-position feature sets,
the attribute categories used by radars / recruitment weights, and the
league reference table.

Keeping this in one module means the Streamlit UI never hard-codes a
metric name, and the analytics layer never has to guess which columns
are relevant for a centre-back versus a winger.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

RAW_PLAYERS_CSV = RAW_DIR / "players_raw.csv.gz"
PROCESSED_PLAYERS_CSV = PROCESSED_DIR / "players_processed.csv.gz"
STATSBOMB_PLAYERS_CSV = RAW_DIR / "statsbomb_players.csv.gz"
VALIDATION_REPORT = MODELS_DIR / "validation_report.md"

# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------

POSITION_GROUPS = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "FW"]

POSITION_GROUP_NAMES = {
    "GK": "Goalkeeper",
    "CB": "Centre-back",
    "FB": "Full-back / Wing-back",
    "DM": "Defensive midfielder",
    "CM": "Central midfielder",
    "AM": "Attacking midfielder",
    "W": "Winger",
    "FW": "Forward",
}

# Detailed positions that map into each analysis group. The similarity and
# clustering models never compare across groups.
DETAILED_POSITIONS = {
    "GK": ["GK"],
    "CB": ["CB", "LCB", "RCB"],
    "FB": ["LB", "RB", "LWB", "RWB"],
    "DM": ["DM"],
    "CM": ["CM", "B2B"],
    "AM": ["AM", "SS"],
    "W": ["LW", "RW", "LM", "RM"],
    "FW": ["CF", "ST"],
}

POSITION_TO_GROUP = {
    pos: group for group, positions in DETAILED_POSITIONS.items() for pos in positions
}

OUTFIELD_GROUPS = [g for g in POSITION_GROUPS if g != "GK"]

# --------------------------------------------------------------------------
# Metric registry
# --------------------------------------------------------------------------
# COUNTING_STATS are raw season totals in the source data. Each is turned into
# a per-90 rate by the feature engineering step (never compared as a total).

COUNTING_STATS = {
    # --- attacking ---
    "goals": "Goals",
    "np_goals": "Non-penalty goals",
    "pens_scored": "Penalties scored",
    "pens_taken": "Penalties taken",
    "xg": "xG",
    "npxg": "Non-penalty xG",
    "assists": "Assists",
    "xa": "xA (expected assists)",
    "shots": "Shots",
    "shots_on_target": "Shots on target",
    "sca": "Shot-creating actions",
    "gca": "Goal-creating actions",
    "key_passes": "Key passes",
    "touches_att_pen": "Touches in opposition box",
    "progressive_carries": "Progressive carries",
    "carries_into_final_third": "Carries into final third",
    "carries_into_pen_area": "Carries into penalty area",
    "dribbles_completed": "Successful dribbles",
    "dribbles_attempted": "Dribble attempts",
    # --- passing & possession ---
    "passes_attempted": "Passes attempted",
    "passes_completed": "Passes completed",
    "progressive_passes": "Progressive passes",
    "passes_into_final_third": "Passes into final third",
    "passes_into_pen_area": "Passes into penalty area",
    "through_balls": "Through balls",
    "crosses": "Crosses",
    "switches": "Switches of play",
    "long_passes_attempted": "Long passes attempted",
    "long_passes_completed": "Long passes completed",
    "progressive_receptions": "Progressive passes received",
    "touches": "Touches",
    "miscontrols": "Miscontrols",
    "dispossessed": "Times dispossessed",
    # --- defensive ---
    "tackles": "Tackles",
    "tackles_won": "Tackles won",
    "interceptions": "Interceptions",
    "blocks": "Blocks",
    "clearances": "Clearances",
    "ball_recoveries": "Ball recoveries",
    "pressures": "Pressures",
    "pressures_successful": "Successful pressures",
    "aerials_won": "Aerial duels won",
    "aerials_lost": "Aerial duels lost",
    "fouls_committed": "Fouls committed",
    "errors": "Errors leading to shot",
    # --- goalkeeping ---
    "gk_shots_on_target_against": "Shots on target faced",
    "gk_saves": "Saves",
    "gk_goals_against": "Goals conceded",
    "gk_psxg": "Post-shot xG faced",
    "gk_crosses_faced": "Crosses faced",
    "gk_crosses_stopped": "Crosses claimed",
    "gk_def_actions_outside_box": "Defensive actions outside the box",
    "gk_launches_attempted": "Long goal-kicks / launches",
    "gk_launches_completed": "Long launches completed",
}

# Ratio metrics: (numerator, denominator, minimum denominator per 90 to be
# considered reliable, label). Percentages are recomputed from the totals so
# that a player with three attempts never shows a "100%" success rate.
RATIO_METRICS = {
    "pass_pct": ("passes_completed", "passes_attempted", 100, "Pass completion %"),
    "long_pass_pct": ("long_passes_completed", "long_passes_attempted", 30, "Long-pass completion %"),
    "tackle_win_pct": ("tackles_won", "tackles", 20, "Tackle success %"),
    "pressure_success_pct": ("pressures_successful", "pressures", 50, "Pressure success %"),
    "dribble_success_pct": ("dribbles_completed", "dribbles_attempted", 15, "Dribble success %"),
    "shot_accuracy_pct": ("shots_on_target", "shots", 15, "Shot accuracy %"),
    "aerial_win_pct": ("aerials_won", "_aerials_total", 20, "Aerial duel success %"),
    "gk_save_pct": ("gk_saves", "gk_shots_on_target_against", 30, "Save %"),
    "gk_cross_stop_pct": ("gk_crosses_stopped", "gk_crosses_faced", 30, "Cross claim %"),
    "gk_launch_pct": ("gk_launches_completed", "gk_launches_attempted", 30, "Launch completion %"),
}

# Derived metrics that are neither a plain per-90 nor a ratio.
DERIVED_METRICS = {
    "npxg_per_shot": "Non-penalty xG per shot",
    "np_goals_minus_npxg_per90": "Non-penalty goals - xG per 90",
    "gk_psxg_minus_ga_per90": "Goals prevented vs post-shot xG per 90",
    "aerials_contested_per90": "Aerial duels contested per 90",
    "defensive_actions_per90": "Defensive actions per 90",
    "progressive_actions_per90": "Progressive actions per 90",
}

# Metrics where a lower value is better (used by percentile calculations).
LOWER_IS_BETTER = {
    "miscontrols_per90",
    "dispossessed_per90",
    "fouls_committed_per90",
    "errors_per90",
    "gk_goals_against_per90",
    "aerials_lost_per90",
    "pens_taken_per90",
}

# Counting stats that should never be turned into a per-90 rate.
NO_PER90 = {"pens_scored", "pens_taken"}


def per90(stat: str) -> str:
    """Column name of the per-90 version of a counting stat."""
    return f"{stat}_per90"


METRIC_LABELS: dict[str, str] = {}
for _stat, _label in COUNTING_STATS.items():
    METRIC_LABELS[_stat] = _label
    METRIC_LABELS[per90(_stat)] = f"{_label} per 90"
for _key, (_n, _d, _m, _label) in RATIO_METRICS.items():
    METRIC_LABELS[_key] = _label
METRIC_LABELS.update(DERIVED_METRICS)
METRIC_LABELS.update(
    {
        "minutes": "Minutes played",
        "starts": "Starts",
        "matches": "Appearances",
        "age": "Age",
        "height_cm": "Height (cm)",
        "team_possession": "Team possession %",
    }
)

# Percentage-style metrics are displayed with a % suffix and one decimal.
PERCENT_METRICS = set(RATIO_METRICS) | {"team_possession"}

# --------------------------------------------------------------------------
# Attribute categories
# --------------------------------------------------------------------------
# Each category is a small basket of metrics. A player's category score is the
# mean of their positional percentiles across the basket, so every score shown
# in the app traces back to raw per-90 numbers.

OUTFIELD_CATEGORIES = {
    "Finishing": [
        "np_goals_per90",
        "npxg_per90",
        "shots_per90",
        "shot_accuracy_pct",
        "npxg_per_shot",
    ],
    "Box Threat": [
        "touches_att_pen_per90",
        "carries_into_pen_area_per90",
        "shots_per90",
    ],
    "Chance Creation": [
        "assists_per90",
        "xa_per90",
        "key_passes_per90",
        "sca_per90",
        "passes_into_pen_area_per90",
        "through_balls_per90",
    ],
    "Passing": [
        "pass_pct",
        "passes_attempted_per90",
        "long_pass_pct",
        "switches_per90",
    ],
    "Ball Progression": [
        "progressive_passes_per90",
        "progressive_carries_per90",
        "passes_into_final_third_per90",
        "carries_into_final_third_per90",
        "progressive_receptions_per90",
    ],
    "Dribbling": [
        "dribbles_completed_per90",
        "dribbles_attempted_per90",
        "dribble_success_pct",
        "progressive_carries_per90",
    ],
    "Defending": [
        "tackles_per90",
        "interceptions_per90",
        "blocks_per90",
        "clearances_per90",
        "ball_recoveries_per90",
        "pressures_per90",
        "tackle_win_pct",
    ],
    "Aerial": [
        "aerials_won_per90",
        "aerial_win_pct",
        "aerials_contested_per90",
    ],
}

GK_CATEGORIES = {
    "Shot Stopping": ["gk_save_pct", "gk_psxg_minus_ga_per90", "gk_saves_per90"],
    "Goal Prevention": ["gk_goals_against_per90", "gk_psxg_minus_ga_per90"],
    "Claiming Crosses": ["gk_cross_stop_pct", "gk_crosses_stopped_per90"],
    "Sweeping": ["gk_def_actions_outside_box_per90"],
    "Distribution": ["pass_pct", "passes_attempted_per90", "progressive_passes_per90"],
    "Long Distribution": ["gk_launch_pct", "long_pass_pct", "long_passes_attempted_per90"],
}


def categories_for(position_group: str) -> dict[str, list[str]]:
    """Attribute categories (radar axes / recruitment weights) for a group."""
    return GK_CATEGORIES if position_group == "GK" else OUTFIELD_CATEGORIES


# --------------------------------------------------------------------------
# Position-specific model features
# --------------------------------------------------------------------------
# These are the columns fed to the scaler, the similarity engine and K-Means.
# They are deliberately different per position: comparing a centre-back on
# "touches in the box" would add noise, not signal.

_CORE_PROGRESSION = [
    "progressive_passes_per90",
    "progressive_carries_per90",
    "passes_into_final_third_per90",
]

POSITION_FEATURES = {
    "GK": [
        "gk_save_pct",
        "gk_psxg_minus_ga_per90",
        "gk_saves_per90",
        "gk_goals_against_per90",
        "gk_cross_stop_pct",
        "gk_def_actions_outside_box_per90",
        "gk_launch_pct",
        "gk_launches_attempted_per90",
        "pass_pct",
        "passes_attempted_per90",
        "progressive_passes_per90",
        "long_passes_attempted_per90",
    ],
    "CB": [
        "tackles_per90",
        "tackle_win_pct",
        "interceptions_per90",
        "blocks_per90",
        "clearances_per90",
        "ball_recoveries_per90",
        "pressures_per90",
        "aerials_won_per90",
        "aerial_win_pct",
        "aerials_contested_per90",
        "pass_pct",
        "passes_attempted_per90",
        "long_passes_attempted_per90",
        "long_pass_pct",
        "progressive_passes_per90",
        "progressive_carries_per90",
        "passes_into_final_third_per90",
        "errors_per90",
        "fouls_committed_per90",
    ],
    "FB": [
        "tackles_per90",
        "tackle_win_pct",
        "interceptions_per90",
        "blocks_per90",
        "ball_recoveries_per90",
        "pressures_per90",
        "aerial_win_pct",
        "pass_pct",
        "passes_attempted_per90",
        "crosses_per90",
        "key_passes_per90",
        "xa_per90",
        "sca_per90",
        "passes_into_final_third_per90",
        "passes_into_pen_area_per90",
        "progressive_passes_per90",
        "progressive_carries_per90",
        "progressive_receptions_per90",
        "dribbles_completed_per90",
        "touches_att_pen_per90",
    ],
    "DM": [
        "tackles_per90",
        "tackle_win_pct",
        "interceptions_per90",
        "blocks_per90",
        "clearances_per90",
        "ball_recoveries_per90",
        "pressures_per90",
        "pressure_success_pct",
        "aerials_won_per90",
        "aerial_win_pct",
        "pass_pct",
        "passes_attempted_per90",
        "long_passes_attempted_per90",
        "long_pass_pct",
        "switches_per90",
        "progressive_passes_per90",
        "progressive_carries_per90",
        "progressive_receptions_per90",
        "passes_into_final_third_per90",
        "fouls_committed_per90",
    ],
    "CM": [
        "tackles_per90",
        "interceptions_per90",
        "ball_recoveries_per90",
        "pressures_per90",
        "pressure_success_pct",
        "aerial_win_pct",
        "pass_pct",
        "passes_attempted_per90",
        "progressive_passes_per90",
        "progressive_carries_per90",
        "progressive_receptions_per90",
        "passes_into_final_third_per90",
        "passes_into_pen_area_per90",
        "key_passes_per90",
        "xa_per90",
        "sca_per90",
        "dribbles_completed_per90",
        "npxg_per90",
        "shots_per90",
        "touches_att_pen_per90",
    ],
    "AM": [
        "np_goals_per90",
        "npxg_per90",
        "shots_per90",
        "touches_att_pen_per90",
        "assists_per90",
        "xa_per90",
        "key_passes_per90",
        "sca_per90",
        "through_balls_per90",
        "passes_into_pen_area_per90",
        "pass_pct",
        "passes_attempted_per90",
        "progressive_passes_per90",
        "progressive_carries_per90",
        "progressive_receptions_per90",
        "dribbles_completed_per90",
        "dribble_success_pct",
        "pressures_per90",
        "ball_recoveries_per90",
        "tackles_per90",
    ],
    "W": [
        "np_goals_per90",
        "npxg_per90",
        "npxg_per_shot",
        "shots_per90",
        "touches_att_pen_per90",
        "assists_per90",
        "xa_per90",
        "key_passes_per90",
        "sca_per90",
        "crosses_per90",
        "passes_into_pen_area_per90",
        "progressive_carries_per90",
        "carries_into_final_third_per90",
        "carries_into_pen_area_per90",
        "dribbles_completed_per90",
        "dribbles_attempted_per90",
        "dribble_success_pct",
        "progressive_receptions_per90",
        "progressive_passes_per90",
        "pass_pct",
        "pressures_per90",
        "tackles_per90",
    ],
    "FW": [
        "np_goals_per90",
        "npxg_per90",
        "npxg_per_shot",
        "shots_per90",
        "shot_accuracy_pct",
        "touches_att_pen_per90",
        "assists_per90",
        "xa_per90",
        "key_passes_per90",
        "sca_per90",
        "aerials_won_per90",
        "aerial_win_pct",
        "pass_pct",
        "passes_attempted_per90",
        "progressive_passes_per90",
        "progressive_carries_per90",
        "progressive_receptions_per90",
        "dribbles_completed_per90",
        "dribble_success_pct",
        "pressures_per90",
        "ball_recoveries_per90",
    ],
}

# Default recruitment weightings per position (percentages summing to 100).
# The scout can override every one of these in the UI.
DEFAULT_WEIGHTS = {
    "CB": {"Defending": 30, "Aerial": 25, "Passing": 20, "Ball Progression": 20, "Dribbling": 5},
    "FB": {
        "Ball Progression": 25,
        "Chance Creation": 20,
        "Defending": 20,
        "Dribbling": 15,
        "Passing": 15,
        "Aerial": 5,
    },
    "DM": {
        "Defending": 30,
        "Passing": 25,
        "Ball Progression": 25,
        "Aerial": 10,
        "Chance Creation": 10,
    },
    "CM": {
        "Ball Progression": 25,
        "Passing": 20,
        "Defending": 20,
        "Chance Creation": 20,
        "Dribbling": 10,
        "Finishing": 5,
    },
    "AM": {
        "Chance Creation": 30,
        "Finishing": 20,
        "Ball Progression": 20,
        "Dribbling": 15,
        "Passing": 10,
        "Defending": 5,
    },
    "W": {
        "Dribbling": 25,
        "Chance Creation": 20,
        "Finishing": 20,
        "Ball Progression": 20,
        "Passing": 10,
        "Defending": 5,
    },
    "FW": {
        "Finishing": 35,
        "Box Threat": 20,
        "Chance Creation": 20,
        "Aerial": 10,
        "Ball Progression": 10,
        "Passing": 5,
    },
    "GK": {
        "Shot Stopping": 40,
        "Goal Prevention": 15,
        "Claiming Crosses": 15,
        "Sweeping": 10,
        "Distribution": 15,
        "Long Distribution": 5,
    },
}

# --------------------------------------------------------------------------
# Leagues
# --------------------------------------------------------------------------
# `strength` is a transparent, editable coefficient used only where the app
# explicitly says it is adjusting for league level. It is an assumption, not a
# measurement, and the UI states that wherever it is applied.

LEAGUES = [
    {"name": "Premier League", "country": "England", "tier": 1, "strength": 1.00, "teams": 20},
    {"name": "La Liga", "country": "Spain", "tier": 1, "strength": 0.96, "teams": 20},
    {"name": "Serie A", "country": "Italy", "tier": 1, "strength": 0.95, "teams": 20},
    {"name": "Bundesliga", "country": "Germany", "tier": 1, "strength": 0.94, "teams": 18},
    {"name": "Ligue 1", "country": "France", "tier": 1, "strength": 0.90, "teams": 18},
    {"name": "Eredivisie", "country": "Netherlands", "tier": 2, "strength": 0.78, "teams": 18},
    {"name": "Primeira Liga", "country": "Portugal", "tier": 2, "strength": 0.78, "teams": 18},
    {"name": "Jupiler Pro League", "country": "Belgium", "tier": 2, "strength": 0.74, "teams": 16},
    {"name": "Championship", "country": "England", "tier": 2, "strength": 0.74, "teams": 24},
    {"name": "Süper Lig", "country": "Turkey", "tier": 3, "strength": 0.70, "teams": 20},
    {"name": "Austrian Bundesliga", "country": "Austria", "tier": 3, "strength": 0.66, "teams": 12},
    {"name": "Danish Superliga", "country": "Denmark", "tier": 3, "strength": 0.65, "teams": 12},
    {"name": "Swiss Super League", "country": "Switzerland", "tier": 3, "strength": 0.64, "teams": 12},
    {"name": "Ekstraklasa", "country": "Poland", "tier": 3, "strength": 0.60, "teams": 18},
]

LEAGUE_STRENGTH = {lg["name"]: lg["strength"] for lg in LEAGUES}
LEAGUE_TIER = {lg["name"]: lg["tier"] for lg in LEAGUES}

SEASONS = ["2023-24", "2024-25"]
CURRENT_SEASON = SEASONS[-1]

# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------
# The platform runs on either a real event-derived dataset or the simulated
# reference universe. Both flow through exactly the same cleaning, feature and
# modelling code; only the ingestion differs.


@dataclass(frozen=True)
class DataSource:
    key: str
    label: str
    path: Path
    kind: str                      # "real" or "simulated"
    summary: str
    attribution: str
    caveats: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()  # columns this source cannot supply


DATA_SOURCES: dict[str, DataSource] = {
    "statsbomb": DataSource(
        key="statsbomb",
        label="StatsBomb Open Data (real players)",
        path=STATSBOMB_PLAYERS_CSV,
        kind="real",
        summary=(
            "Real players and real matches. Every metric is computed here from raw event "
            "data - see the Methodology page for each definition."
        ),
        attribution=(
            "Data provided by StatsBomb Open Data "
            "(https://github.com/statsbomb/open-data), free for public use."
        ),
        caveats=(
            "The open-data feed carries no birth dates, so **age is unavailable** and every "
            "age-based filter and score component is switched off.",
            "Height is not published either.",
            "Post-shot xG is a paid StatsBomb feature, so goalkeeper shot-stopping is measured "
            "on save percentage and goals conceded rather than goals prevented.",
            "Only competition-seasons with complete league coverage are ingested; seasons where "
            "the feed carries a single club are excluded.",
        ),
        missing=("age", "height_cm", "gk_psxg"),
    ),
    "simulated": DataSource(
        key="simulated",
        label="Simulated reference universe",
        path=RAW_PLAYERS_CSV,
        kind="simulated",
        summary=(
            "A latent-trait simulation of 14 leagues over two seasons. Invented players, "
            "plausible numbers, and every field the schema supports - including age and height."
        ),
        attribution="Generated by src/data_generation.py; no real footballer is described.",
        caveats=(
            "**These are not real players.** Nothing here supports a conclusion about a real "
            "footballer.",
            "Its value is that the generative role of every player is known, which is what makes "
            "the supervised checks on the Model Validation page possible.",
        ),
    ),
}

DEFAULT_SOURCE = "statsbomb"

# Minimum-minutes presets offered in the sidebar.
MINUTES_PRESETS = [500, 900, 1500]
DEFAULT_MIN_MINUTES = 900

# --------------------------------------------------------------------------
# Chart theme
# --------------------------------------------------------------------------
# Dark-surface palette, validated for colour-vision deficiency separation.
# The three categorical slots are the only hues used to distinguish players or
# series; anything needing more categories highlights one against grey context
# rather than inventing a fourth hue.

THEME = {
    "surface": "#1a1a19",       # chart surface
    "page": "#0d0d0d",          # page plane
    "panel": "#161615",
    "ink": "#ffffff",           # primary text
    "ink_secondary": "#c3c2b7",
    "ink_muted": "#898781",
    "grid": "#2c2c2a",
    "baseline": "#383835",
    "context": "#4a4a47",       # unhighlighted marks
    # categorical slots (validated all-pairs on the dark surface)
    "series_1": "#3987e5",
    "series_2": "#d95926",
    "series_3": "#199e70",
    # diverging poles for "above / below positional average"
    "diverging_high": "#3987e5",
    "diverging_low": "#e66767",
    "diverging_mid": "#383835",
    # status (reserved - never used as a series colour)
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}

SERIES_COLORS = [THEME["series_1"], THEME["series_2"], THEME["series_3"]]

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

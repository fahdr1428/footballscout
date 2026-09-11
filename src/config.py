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
PREMIER_LEAGUE_CSV = RAW_DIR / "premier_league.csv.gz"
FBREF_BIG5_CSV = RAW_DIR / "fbref_big5.csv.gz"
UNDERSTAT_CSV = RAW_DIR / "understat_big6.csv.gz"
TRANSFERMARKT_CSV = RAW_DIR / "transfermarkt.csv.gz"
TRANSFERMARKT_MARKET_CSV = RAW_DIR / "transfermarkt_market.csv.gz"
VALIDATION_REPORT = MODELS_DIR / "validation_report.md"

# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------

# Two taxonomies. Event data resolves a player's position from lineup data, so
# it supports the detailed groups. A summary feed such as the Fantasy Premier
# League API knows only four buckets, and inventing a finer position from the
# same statistics the models then read would be circular - so those buckets are
# first-class groups rather than a guess dressed up as detail.
# Ten groups, and the split between them is measured rather than assumed.
# `scripts/position_separability.py` trains a cross-validated classifier to tell
# each candidate pair apart on their own model features, scored by balanced
# accuracy so 0.50 is a coin flip whatever the class imbalance. Two splits earn
# their place - a second striker is not an attacking midfielder (0.81) and a
# wide midfielder is not a winger (0.76) - against controls at 0.96 and 0.98.
# Two do not: left-back against right-back reads 0.62 and left against right
# wing 0.60, because they are the same job mirrored. So side is carried as a
# filter (see FLANKS) instead of halving those peer groups for no information.
DETAILED_GROUPS = ["GK", "CB", "FB", "DM", "CM", "AM", "SS", "WM", "W", "FW"]
BUCKET_GROUPS = ["GK", "DEF", "MID", "FWD"]
POSITION_GROUPS = DETAILED_GROUPS + ["DEF", "MID", "FWD"]

POSITION_GROUP_NAMES = {
    "GK": "Goalkeeper",
    "SS": "Second striker",
    "WM": "Wide midfielder",
    "DEF": "Defender",
    "MID": "Midfielder",
    "FWD": "Forward",
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
    "AM": ["AM"],
    "SS": ["SS"],
    "WM": ["LM", "RM"],
    "W": ["LW", "RW"],
    "FW": ["CF", "ST"],
}

# When a pool is too thin to model a group on its own, its players are measured
# against the next group up rather than dropped. The parent is the group the
# split came out of, so the fallback is the taxonomy this project used before
# the separability test justified going finer.
POSITION_PARENT = {"SS": "AM", "WM": "W"}

# Side of the pitch. Not a separate model - the classifier says a left-back and a
# right-back do the same job - but a club recruiting a left-back does not want
# right-backs in the shortlist, so it is a first-class filter.
FLANKS = {
    "LB": "Left", "LW": "Left", "LM": "Left",
    "RB": "Right", "RW": "Right", "RM": "Right",
}
DEFAULT_FLANK = "Central"

# Groups where being on the "wrong" foot for your flank is a real distinction:
# a right-footed left winger cuts inside, a left-footed one goes outside.
WIDE_GROUPS = {"FB", "W", "WM"}

POSITION_TO_GROUP = {
    pos: group for group, positions in DETAILED_POSITIONS.items() for pos in positions
}

OUTFIELD_GROUPS = [g for g in POSITION_GROUPS if g != "GK"]

# Free-text positions the Understat lineup feed reports, mapped to a readable
# label. They are carried as an attribute, never used to group.
UNDERSTAT_POSITIONS = {
    "GK": "Goalkeeper", "DC": "Centre-back", "DR": "Right-back", "DL": "Left-back",
    "DMC": "Defensive midfield", "DMR": "Defensive midfield (right)",
    "DML": "Defensive midfield (left)", "MC": "Central midfield",
    "MR": "Right midfield", "ML": "Left midfield", "AMC": "Attacking midfield",
    "AMR": "Right attacking midfield", "AML": "Left attacking midfield",
    "FW": "Forward", "FWR": "Forward (right)", "FWL": "Forward (left)", "Sub": "Substitute",
}

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
    "npxg_open_play": "Non-penalty xG from open play",
    "npxg_set_piece": "Non-penalty xG from set pieces",
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
    "passes_under_pressure": "Passes attempted under pressure",
    "passes_completed_under_pressure": "Passes completed under pressure",
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
    "yellow_cards": "Yellow cards",
    "red_cards": "Red cards",
    # --- available from summary feeds (FPL / Understat) rather than events ---
    "cbi": "Clearances, blocks and interceptions",
    "defensive_contribution": "Defensive contribution actions",
    "clean_sheets": "Clean sheets",
    "xgc": "Expected goals conceded (team, while on pitch)",
    "xg_chain": "xGChain (possessions ending in a shot)",
    "xg_buildup": "xGBuildup (xGChain excluding shots and key passes)",
    "influence": "Influence (Opta index)",
    "creativity": "Creativity (Opta index)",
    "threat": "Threat (Opta index)",
    "bps": "Bonus points system score",
    "ict": "ICT index (influence + creativity + threat)",
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
    "gk_pens_faced": "Penalties faced",
    "gk_pens_saved": "Penalties saved",
    # --- available from a full match-data feed (FBref) ---
    # Where a player passes, tackles and touches the ball, rather than only how
    # often. Zone counts are what separate a full-back who defends his own box
    # from one who spends the game in the opposition half.
    "short_passes_attempted": "Short passes attempted (5-15y)",
    "short_passes_completed": "Short passes completed (5-15y)",
    "medium_passes_attempted": "Medium passes attempted (15-30y)",
    "medium_passes_completed": "Medium passes completed (15-30y)",
    "progressive_pass_distance": "Progressive passing distance (yards)",
    "crosses_into_pen_area": "Crosses into the penalty area",
    "corners_taken": "Corners taken",
    "free_kick_passes": "Free-kicks taken",
    "free_kick_shots": "Direct free-kick shots",
    "sca_from_open_play_pass": "Shot-creating actions from open-play passes",
    "sca_from_set_piece": "Shot-creating actions from set pieces",
    "sca_from_dribble": "Shot-creating actions from dribbles",
    "sca_from_defensive_action": "Shot-creating actions from defensive actions",
    "tackles_def_third": "Tackles in the defensive third",
    "tackles_mid_third": "Tackles in the middle third",
    "tackles_att_third": "Tackles in the attacking third",
    "dribblers_challenged": "Dribblers challenged",
    "dribblers_tackled": "Dribblers dispossessed",
    "pressures_att_third": "Pressures in the attacking third",
    "shots_blocked": "Shots blocked",
    "touches_def_pen": "Touches in own box",
    "touches_def_third": "Touches in the defensive third",
    "touches_mid_third": "Touches in the middle third",
    "touches_att_third": "Touches in the attacking third",
    "carries": "Carries",
    "progressive_carry_distance": "Progressive carrying distance (yards)",
    "fouls_won": "Fouls won",
    "offsides": "Offsides",
    "pens_won": "Penalties won",
    "pens_conceded": "Penalties conceded",
}

# Ratio metrics: (numerator, denominator, minimum denominator per 90 to be
# considered reliable, label). Percentages are recomputed from the totals so
# that a player with three attempts never shows a "100%" success rate.
RATIO_METRICS = {
    "pass_pct": ("passes_completed", "passes_attempted", 100, "Pass completion %"),
    "pass_pct_under_pressure": (
        "passes_completed_under_pressure", "passes_under_pressure", 40,
        "Pass completion under pressure %",
    ),
    "long_pass_pct": ("long_passes_completed", "long_passes_attempted", 30, "Long-pass completion %"),
    "tackle_win_pct": ("tackles_won", "tackles", 20, "Tackle success %"),
    "pressure_success_pct": ("pressures_successful", "pressures", 50, "Pressure success %"),
    "dribble_success_pct": ("dribbles_completed", "dribbles_attempted", 15, "Dribble success %"),
    "shot_accuracy_pct": ("shots_on_target", "shots", 15, "Shot accuracy %"),
    "aerial_win_pct": ("aerials_won", "_aerials_total", 20, "Aerial duel success %"),
    "gk_save_pct": ("gk_saves", "gk_shots_on_target_against", 30, "Save %"),
    "save_rate": ("gk_saves", "_saves_plus_conceded", 25, "Save rate (saves / shots faced) %"),
    "gk_cross_stop_pct": ("gk_crosses_stopped", "gk_crosses_faced", 30, "Cross claim %"),
    "gk_launch_pct": ("gk_launches_completed", "gk_launches_attempted", 30, "Launch completion %"),
    "gk_pen_save_pct": ("gk_pens_saved", "gk_pens_faced", 3, "Penalty save %"),
    "short_pass_pct": ("short_passes_completed", "short_passes_attempted", 50, "Short-pass completion %"),
    "medium_pass_pct": ("medium_passes_completed", "medium_passes_attempted", 40, "Medium-pass completion %"),
    # One-against-one defending: the share of dribblers taken on who were
    # actually dispossessed. Volume of tackles says nothing about this.
    "dribbler_stop_pct": ("dribblers_tackled", "dribblers_challenged", 20, "Dribblers stopped %"),
}

# Derived metrics that are neither a plain per-90 nor a ratio.
DERIVED_METRICS = {
    "npxg_per_shot": "Non-penalty xG per shot",
    "clean_sheet_rate": "Clean sheets per appearance %",
    "starts_share": "Share of appearances that were starts %",
    "goal_involvements_per90": "Goals and assists per 90",
    "xgi_per90": "Expected goal involvements per 90",
    "pressured_pass_share": "Share of passes made under pressure %",
    "open_play_npxg_share": "Share of non-penalty xG from open play %",
    "np_goals_minus_npxg_per90": "Non-penalty goals - xG per 90",
    "gk_psxg_minus_ga_per90": "Goals prevented vs post-shot xG per 90",
    "aerials_contested_per90": "Aerial duels contested per 90",
    "defensive_actions_per90": "Defensive actions per 90",
    "progressive_actions_per90": "Progressive actions per 90",
    # Positional signature: not how much a player does, but where he does it.
    "att_third_touch_share": "Share of touches in the attacking third %",
    "box_touch_share": "Share of touches in the opposition box %",
    "def_third_touch_share": "Share of touches in the defensive third %",
    "att_third_tackle_share": "Share of tackles in the attacking third %",
    "pass_progress_per_pass": "Progressive distance per pass attempted (yards)",
    "carry_progress_per_carry": "Progressive distance per carry (yards)",
    "set_piece_sca_share": "Share of chances created from set pieces %",
}

# Metrics that describe the *team* more than the player. A summary feed
# attributes them to whoever was on the pitch, so three of them in one feature
# set is really one signal - the club - counted three times, and the similarity
# engine starts clustering clubs instead of players. They stay on the radar and
# in the recruitment weights, where a scout can read them as context, but for
# outfield model features only the most informative one is kept.
TEAM_CONTEXT_METRICS = {"clean_sheet_rate", "goals_conceded_per90", "xgc_per90"}

# Metrics where a lower value is better (used by percentile calculations).
LOWER_IS_BETTER = {
    "goals_conceded_per90",
    "xgc_per90",
    "yellow_cards_per90",
    "red_cards_per90",
    "miscontrols_per90",
    "dispossessed_per90",
    "fouls_committed_per90",
    "errors_per90",
    "gk_goals_against_per90",
    "aerials_lost_per90",
    "pens_taken_per90",
    "offsides_per90",
    "pens_conceded_per90",
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
        "price_m": "Price (fantasy valuation, GBPm)",
        "ownership_pct": "Ownership %",
        "total_points": "Fantasy points",
        "clubs_in_season": "Clubs played for",
        "detailed_position": "Line-up position",
        "position_source": "Position source",
        "market_value_eur": "Market value (Transfermarkt, EUR)",
        "foot": "Preferred foot",
        "nationality": "Nationality",
        "date_of_birth": "Date of birth",
        "xa_definition": "Expected-assist definition",
        "team_points_per_match": "Team points per match while on the pitch",
        "gk_avg_pass_length": "Average pass length (yards)",
        "gk_avg_sweeper_distance": "Average sweeper distance from goal (yards)",
    }
)

# Percentage-style metrics are displayed with a % suffix and one decimal.
PERCENT_METRICS = set(RATIO_METRICS) | {
    "team_possession", "pressured_pass_share", "open_play_npxg_share",
    "clean_sheet_rate", "starts_share", "ownership_pct",
    "att_third_touch_share", "box_touch_share", "def_third_touch_share",
    "att_third_tackle_share", "set_piece_sca_share",
}

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
        "npxg_open_play_per90",
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
        "pass_pct_under_pressure",
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


# Categories for the four-bucket taxonomy. They are built only from metrics a
# summary feed actually reports, so nothing on a radar is an empty axis.
BUCKET_CATEGORIES = {
    "Goal Threat": ["npxg_per90", "xg_per90", "goals_per90", "threat_per90", "shots_per90"],
    "Chance Creation": ["xa_per90", "assists_per90", "creativity_per90", "key_passes_per90"],
    "Build-up Involvement": ["xg_chain_per90", "xg_buildup_per90", "influence_per90"],
    "Defensive Work": [
        "tackles_per90", "ball_recoveries_per90", "cbi_per90", "defensive_contribution_per90",
    ],
    "Defensive Solidity": ["clean_sheet_rate", "goals_conceded_per90", "xgc_per90"],
    "Overall Rating": ["bps_per90", "ict_per90"],
    "Availability": ["starts_share", "minutes"],
}

BUCKET_GK_CATEGORIES = {
    "Shot Stopping": ["save_rate", "gk_saves_per90"],
    "Goal Prevention": ["goals_conceded_per90", "xgc_per90", "clean_sheet_rate"],
    "Overall Rating": ["bps_per90", "influence_per90"],
    "Availability": ["starts_share", "minutes"],
}


def categories_for(position_group: str, taxonomy: str = "detailed") -> dict[str, list[str]]:
    """Attribute categories (radar axes / recruitment weights) for a group."""
    if taxonomy == "bucket":
        return BUCKET_GK_CATEGORIES if position_group == "GK" else BUCKET_CATEGORIES
    if position_group in {"DEF", "MID", "FWD"}:
        return BUCKET_CATEGORIES
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
        "save_rate",
        "goals_conceded_per90",
        "xgc_per90",
        "clean_sheet_rate",
        "bps_per90",
        "influence_per90",
        "starts_share",
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
    # The three bucket groups are only populated by a summary feed; on an
    # event dataset every one of these columns is absent and the group is empty.
    "DEF": [
        "cbi_per90", "tackles_per90", "ball_recoveries_per90", "defensive_contribution_per90",
        "xgc_per90",
        "xg_per90", "npxg_per90", "xa_per90", "goals_per90", "np_goals_per90",
        "assists_per90", "xgi_per90",
        "threat_per90", "creativity_per90", "influence_per90", "bps_per90",
        "key_passes_per90", "shots_per90", "xg_chain_per90", "xg_buildup_per90",
        "yellow_cards_per90", "starts_share",
    ],
    "MID": [
        "xg_per90", "npxg_per90", "xa_per90", "goals_per90", "np_goals_per90",
        "assists_per90", "xgi_per90", "npxg_per_shot", "np_goals_minus_npxg_per90",
        "shots_per90", "key_passes_per90", "threat_per90", "creativity_per90",
        "influence_per90", "bps_per90", "xg_chain_per90", "xg_buildup_per90",
        "tackles_per90", "ball_recoveries_per90", "cbi_per90",
        "defensive_contribution_per90", "xgc_per90",
        "yellow_cards_per90", "starts_share",
    ],
    "FWD": [
        "xg_per90", "npxg_per90", "goals_per90", "np_goals_per90", "shots_per90",
        "npxg_per_shot", "np_goals_minus_npxg_per90", "xa_per90", "assists_per90",
        "xgi_per90", "key_passes_per90",
        "threat_per90", "creativity_per90", "influence_per90", "bps_per90",
        "xg_chain_per90", "xg_buildup_per90", "tackles_per90",
        "ball_recoveries_per90", "defensive_contribution_per90", "starts_share",
    ],
    "CB": [
        "pass_pct_under_pressure",
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
        "pass_pct_under_pressure",
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
        "pass_pct_under_pressure",
        "pressured_pass_share",
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
        "pass_pct_under_pressure",
        "pressured_pass_share",
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
        "pass_pct_under_pressure",
        "npxg_open_play_per90",
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
    # A second striker plays off a centre-forward: the classifier separates him
    # from an attacking midfielder mainly on where he receives the ball and how
    # little defensive work he does, so both are in his feature set.
    "SS": [
        "npxg_per90",
        "np_goals_per90",
        "npxg_per_shot",
        "shots_per90",
        "shot_accuracy_pct",
        "touches_att_pen_per90",
        "box_touch_share",
        "att_third_touch_share",
        "assists_per90",
        "xa_per90",
        "key_passes_per90",
        "sca_per90",
        "progressive_receptions_per90",
        "progressive_carries_per90",
        "dribbles_completed_per90",
        "dribble_success_pct",
        "pass_pct",
        "passes_attempted_per90",
        "aerials_won_per90",
        "aerial_win_pct",
        "pressures_per90",
        "ball_recoveries_per90",
        "offsides_per90",
    ],
    # A wide midfielder in a flat four is not a winger: less dribbling, more
    # defending, and he receives the ball far less often in a progressive spot.
    "WM": [
        "crosses_per90",
        "crosses_into_pen_area_per90",
        "key_passes_per90",
        "xa_per90",
        "sca_per90",
        "passes_into_final_third_per90",
        "passes_into_pen_area_per90",
        "pass_pct",
        "passes_attempted_per90",
        "progressive_passes_per90",
        "progressive_carries_per90",
        "progressive_receptions_per90",
        "dribbles_completed_per90",
        "dribble_success_pct",
        "npxg_per90",
        "np_goals_per90",
        "shots_per90",
        "touches_att_pen_per90",
        "tackles_per90",
        "interceptions_per90",
        "pressures_per90",
        "att_third_tackle_share",
        "ball_recoveries_per90",
        "aerial_win_pct",
    ],
    "W": [
        "npxg_open_play_per90",
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
        "npxg_open_play_per90",
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

# --------------------------------------------------------------------------
# Role templates
# --------------------------------------------------------------------------
# A position is not a job. Two centre-backs in the same squad can be recruited
# against opposite briefs - one to carry the ball out, one to head it away - and
# a search that ranks both on one "centre-back score" is answering a question
# nobody asked.
#
# Each template is simply a named set of category weights: the same arithmetic
# as the default weighting, starting from a different place. They are editable
# in the app, and they are **assumptions, not measurements** - a reasonable
# reading of what each role asks for, offered as a starting point for a scout
# who will disagree with some of them.

ROLE_TEMPLATES: dict[str, dict[str, dict[str, int]]] = {
    "GK": {
        "Sweeper keeper": {"Sweeping": 30, "Distribution": 25, "Shot Stopping": 25,
                           "Goal Prevention": 10, "Claiming Crosses": 10},
        "Pure shot-stopper": {"Shot Stopping": 45, "Goal Prevention": 30,
                              "Claiming Crosses": 15, "Distribution": 5, "Sweeping": 5},
        "Commanding in the air": {"Claiming Crosses": 35, "Shot Stopping": 30,
                                  "Goal Prevention": 15, "Long Distribution": 10,
                                  "Sweeping": 10},
    },
    "CB": {
        "Ball-playing centre-back": {"Passing": 30, "Ball Progression": 30,
                                     "Defending": 20, "Aerial": 15, "Dribbling": 5},
        "Aerial stopper": {"Aerial": 40, "Defending": 35, "Passing": 15,
                           "Ball Progression": 10},
        "Covering defender": {"Defending": 45, "Ball Progression": 20,
                              "Passing": 20, "Aerial": 15},
        "Wide centre-back (back three)": {"Ball Progression": 30, "Defending": 25,
                                          "Passing": 20, "Dribbling": 15, "Aerial": 10},
    },
    "FB": {
        "Attacking full-back": {"Chance Creation": 30, "Ball Progression": 25,
                                "Dribbling": 20, "Passing": 15, "Defending": 10},
        "Defensive full-back": {"Defending": 40, "Aerial": 20, "Passing": 20,
                                "Ball Progression": 15, "Dribbling": 5},
        "Inverted full-back": {"Passing": 35, "Ball Progression": 25, "Defending": 20,
                               "Dribbling": 10, "Chance Creation": 10},
        "Wing-back": {"Ball Progression": 25, "Chance Creation": 25, "Dribbling": 20,
                      "Defending": 20, "Aerial": 10},
    },
    "DM": {
        "Ball-winning holder": {"Defending": 45, "Aerial": 20, "Passing": 20,
                                "Ball Progression": 15},
        "Deep-lying playmaker": {"Passing": 35, "Ball Progression": 30,
                                 "Defending": 20, "Chance Creation": 15},
        "Anchor": {"Defending": 35, "Aerial": 25, "Passing": 25, "Ball Progression": 15},
    },
    "CM": {
        "Box-to-box": {"Ball Progression": 25, "Defending": 20, "Chance Creation": 20,
                       "Passing": 15, "Box Threat": 10, "Finishing": 10},
        "Deep playmaker": {"Passing": 35, "Ball Progression": 30,
                           "Chance Creation": 20, "Defending": 15},
        "Ball-winner": {"Defending": 40, "Passing": 25, "Ball Progression": 20,
                        "Aerial": 15},
        "Arriving midfielder": {"Box Threat": 30, "Finishing": 25,
                                "Ball Progression": 20, "Chance Creation": 15,
                                "Passing": 10},
    },
    "AM": {
        "Classic number 10": {"Chance Creation": 40, "Passing": 20, "Dribbling": 15,
                              "Ball Progression": 15, "Finishing": 10},
        "Goalscoring 10": {"Finishing": 30, "Box Threat": 25, "Chance Creation": 25,
                           "Ball Progression": 10, "Dribbling": 10},
        "Pressing 10": {"Defending": 30, "Chance Creation": 25, "Ball Progression": 20,
                        "Finishing": 15, "Dribbling": 10},
    },
    "SS": {
        "Poacher off the striker": {"Finishing": 40, "Box Threat": 35, "Chance Creation": 10,
                                    "Ball Progression": 10, "Aerial": 5},
        "Creative second striker": {"Chance Creation": 35, "Ball Progression": 25,
                                    "Finishing": 20, "Dribbling": 10, "Passing": 10},
        "Pressing second striker": {"Defending": 30, "Finishing": 25, "Box Threat": 20,
                                    "Ball Progression": 15, "Chance Creation": 10},
    },
    "WM": {
        "Crossing wide midfielder": {"Chance Creation": 40, "Passing": 20,
                                     "Ball Progression": 20, "Defending": 15, "Dribbling": 5},
        "Defensive wide midfielder": {"Defending": 40, "Ball Progression": 20, "Passing": 20,
                                      "Chance Creation": 15, "Aerial": 5},
        "Inside-moving wide midfielder": {"Ball Progression": 30, "Finishing": 20,
                                          "Chance Creation": 20, "Dribbling": 20,
                                          "Passing": 10},
    },
    "W": {
        "Touchline dribbler": {"Dribbling": 35, "Ball Progression": 25,
                               "Chance Creation": 20, "Finishing": 10, "Passing": 10},
        "Inverted goalscorer": {"Finishing": 30, "Box Threat": 25, "Dribbling": 20,
                                "Chance Creation": 15, "Ball Progression": 10},
        "Creator and crosser": {"Chance Creation": 40, "Passing": 20,
                                "Ball Progression": 15, "Dribbling": 15, "Finishing": 10},
        "Hard-working wide man": {"Defending": 30, "Ball Progression": 20,
                                  "Chance Creation": 20, "Dribbling": 15, "Finishing": 15},
    },
    "FW": {
        "Penalty-box poacher": {"Finishing": 45, "Box Threat": 30, "Aerial": 10,
                                "Chance Creation": 10, "Passing": 5},
        "Target man": {"Aerial": 35, "Box Threat": 20, "Finishing": 20, "Passing": 15,
                       "Chance Creation": 10},
        "Complete forward": {"Finishing": 25, "Chance Creation": 25, "Box Threat": 20,
                             "Ball Progression": 15, "Aerial": 10, "Passing": 5},
        "Pressing forward": {"Defending": 30, "Finishing": 25, "Box Threat": 20,
                             "Ball Progression": 15, "Chance Creation": 10},
    },
    # The four-bucket taxonomy, for summary-feed sources.
    "DEF": {
        "Progressive defender": {"Build-up Involvement": 35, "Chance Creation": 25,
                                 "Defensive Work": 25, "Defensive Solidity": 15},
        "Stay-at-home defender": {"Defensive Solidity": 40, "Defensive Work": 35,
                                  "Build-up Involvement": 15, "Goal Threat": 10},
    },
    "MID": {
        "Creator": {"Chance Creation": 45, "Build-up Involvement": 25,
                    "Goal Threat": 20, "Defensive Work": 10},
        "Destroyer": {"Defensive Work": 50, "Build-up Involvement": 25,
                      "Chance Creation": 15, "Goal Threat": 10},
        "Goalscoring midfielder": {"Goal Threat": 45, "Chance Creation": 25,
                                   "Build-up Involvement": 20, "Defensive Work": 10},
    },
    "FWD": {
        "Goalscorer": {"Goal Threat": 55, "Chance Creation": 20,
                       "Build-up Involvement": 15, "Overall Rating": 10},
        "Link forward": {"Build-up Involvement": 35, "Chance Creation": 35,
                         "Goal Threat": 20, "Overall Rating": 10},
    },
}

DEFAULT_ROLE = "Balanced (position default)"


def role_weights(position_group: str, role: str | None = None) -> dict[str, int]:
    """Category weights for a role, falling back to the position default."""
    if role and role != DEFAULT_ROLE:
        template = ROLE_TEMPLATES.get(position_group, {}).get(role)
        if template:
            return dict(template)
    return dict(DEFAULT_WEIGHTS.get(position_group, {}))


def roles_for(position_group: str) -> list[str]:
    """Every selectable role for a position group, default first."""
    return [DEFAULT_ROLE] + sorted(ROLE_TEMPLATES.get(position_group, {}))


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
    "SS": {
        "Finishing": 30,
        "Box Threat": 25,
        "Chance Creation": 20,
        "Ball Progression": 15,
        "Dribbling": 5,
        "Aerial": 5,
    },
    "WM": {
        "Chance Creation": 25,
        "Defending": 20,
        "Ball Progression": 20,
        "Passing": 15,
        "Dribbling": 10,
        "Finishing": 10,
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
    "DEF": {
        "Defensive Work": 30, "Defensive Solidity": 25, "Build-up Involvement": 20,
        "Chance Creation": 15, "Goal Threat": 10,
    },
    "MID": {
        "Chance Creation": 30, "Goal Threat": 25, "Build-up Involvement": 20,
        "Defensive Work": 20, "Overall Rating": 5,
    },
    "FWD": {
        "Goal Threat": 45, "Chance Creation": 25, "Build-up Involvement": 15,
        "Overall Rating": 10, "Defensive Work": 5,
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
    # Understat covers the Russian top flight alongside the big five. It sat
    # around Süper Lig level before 2022; the European ban and the departure of
    # most foreign players since have cost it, hence tier 3. Like every other
    # coefficient here this is an editable assumption, not a measurement.
    {"name": "Russian Premier League", "country": "Russia", "tier": 3, "strength": 0.66, "teams": 16},
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
    default_seasons: tuple[str, ...] = ()   # what the pool opens on
    taxonomy: str = "detailed"              # "detailed" or "bucket" position groups


DATA_SOURCES: dict[str, DataSource] = {
    "understat_big6": DataSource(
        key="understat_big6",
        label="Six leagues 2014/15-2024/25 (Understat)",
        path=UNDERSTAT_CSV,
        kind="real",
        summary=(
            "Eleven complete seasons to 2024/25 across the big five plus the Russian Premier "
            "League - 34,159 player-seasons, 10,541 players. The longest, most recent and "
            "widest run of real data here, and the only source carrying 2022/23 onwards."
        ),
        attribution=(
            "Understat per-player season aggregates, mirrored by the open-source "
            "understat_players_aggregated repository "
            "(https://github.com/vibedatascience/understat_players_aggregated)."
        ),
        caveats=(
            "**This source does not measure defending. At all.** Understat models shots, not "
            "the rest of the game, so there are no tackles, interceptions, clearances, duels, "
            "pressures or blocks. Just under half the pool are defenders, and here they are "
            "ranked purely on what they contribute going forward. For defending, switch to the "
            "big-five (FBref) source, which carries 44 metrics including all of those.",
            "**Four positional buckets, not ten.** The season aggregate records only GK / D / "
            "M / F - no centre-back against full-back, no left against right wing. Deriving a "
            "finer position from the same statistics the models then read would be circular, "
            "so this source uses the four-bucket taxonomy.",
            "**What it is unmatched at**: recency and reach. Eleven seasons is enough to follow "
            "a career, and **xGChain** and **xGBuildup** credit every player in a move that "
            "ended in a shot - xGBuildup excluding the shot and the assist, which is the closest "
            "thing in open data to contribution without finishing.",
            "**2025/26 is a fragment.** The mirror stopped updating in September 2025, leaving "
            "about ten rounds. It is excluded by default; `--include-partial` adds it, and the "
            "app will then rank those players at the bottom of every volume metric for a reason "
            "that has nothing to do with them.",
            "**Similarity percentages read high here, and mean less.** Every metric this source "
            "has is measuring attacking output, so they move together and two players look "
            "alike easily: the median closest match scores **95.5%**, against **76.3%** on the "
            "big-five source with its 15-24 more varied metrics. Read the ranking, not the "
            "number - and do not compare a percentage here with a percentage there.",
            "**Goalkeepers have no model.** None of the 17 metrics a goalkeeper is ranked on "
            "exist in this feed, so they stay in the pool, are listed everywhere, and carry no "
            "similarity score or archetype. The app says how many that is.",
            "**Ages, heights, feet, nationalities and market values are joined on from "
            "Transfermarkt**, because Understat publishes none of them - and without an age, "
            "the whole youth side of scouting is unavailable. The two feeds share no id, so "
            "players are matched on name and **only where the name is unique on both sides**: "
            "about 73% match, covering 79% of the minutes played. A name held by two players is "
            "left unmatched rather than guessed at, and where the joined date of birth implies "
            "an impossible age the match is treated as wrong and withdrawn entirely.",
            "**Market value is read as at that season**, not scraped once and applied to every "
            "year: Transfermarkt revalues players a few times annually, so each row takes the "
            "most recent valuation on or before 1 January inside its season.",
        ),
        missing=("team_possession", "starts",
                 "tackles", "interceptions", "clearances", "blocks", "pressures",
                 "aerials_won", "ball_recoveries", "passes_attempted", "progressive_passes"),
        default_seasons=("2022-23", "2023-24", "2024-25"),
        taxonomy="bucket",
    ),
    "fbref_big5": DataSource(
        key="fbref_big5",
        label="Big five leagues 2017/18-2021/22 (FBref + Transfermarkt)",
        path=FBREF_BIG5_CSV,
        kind="real",
        summary=(
            "13,230 player-seasons and 5,309 players across the Premier League, La Liga, "
            "Serie A, Bundesliga and Ligue 1, with full match-data metrics, a true position "
            "and a market value in euros. The deepest and widest dataset here, and the only "
            "one that supports all ten position groups."
        ),
        attribution=(
            "FBref season statistics and Transfermarkt squad records, mirrored by the "
            "open-source worldfootballR_data repository "
            "(https://github.com/JaseZiv/worldfootballR_data)."
        ),
        caveats=(
            "**Positions come from Transfermarkt, not from the statistics.** 99.8% of "
            "player-seasons carry a specific position - centre-back, left-back, defensive "
            "midfield - taken from an independent source and joined through a curated "
            "URL mapping, so grouping players by position is not circular.",
            "**Ten position groups, and the splits are measured.** A classifier that tells a "
            "centre-back from a defensive midfielder at 0.96 balanced accuracy separates a "
            "second striker from an attacking midfielder at 0.81, and a wide midfielder from a "
            "winger at 0.76 - so both get their own model. It reads left-back against "
            "right-back at 0.62 and left wing against right wing at 0.60, near the 0.50 coin "
            "flip, so side is a **filter** rather than a separate group. Run it yourself with "
            "`python scripts/position_separability.py`.",
            "**Market values are real and move season by season** (Messi runs 180 -> 150 -> "
            "112 -> 80 -> 50 million euro across these five seasons). 97.5% of player-seasons "
            "carry one. This is the only source here where 'undervalued' means anything.",
            "**No contract data.** Transfermarkt records contract expiry as at the time the "
            "page was read, so every one of Harry Kane's five seasons reads 2024-06-30. A "
            "contract filter is exactly what a recruitment tool gets used for, which is why a "
            "wrong one is worse than none - the columns are dropped rather than shown.",
            "**It is not current.** It ends with 2021/22. FBref changed data provider in "
            "October 2022 and the upstream mirror was archived in September 2025; the 2022/23 "
            "snapshot stops after about 13 rounds, so pooling it with whole seasons would put "
            "every 2022/23 player at the bottom of every volume metric. For the current "
            "season, use the Premier League source.",
            "Players who moved mid-season are one row: totals summed, club and league of "
            "record taken from wherever they played the most minutes.",
        ),
        missing=("npxg_open_play", "npxg_set_piece", "passes_completed_under_pressure",
                 "xg_chain", "xg_buildup", "influence", "creativity", "threat", "bps", "ict",
                 "xgc", "cbi", "defensive_contribution"),
        # Three seasons, not one. A single season leaves only ~20 second
        # strikers and ~20 wide midfielders in the pool - too few to rank
        # against, so both would fold back into attacking midfield and the
        # wingers. Three gives every one of the ten groups its own peer set and
        # triples the pool, which is also how a scout reads recent form.
        default_seasons=("2019-20", "2020-21", "2021-22"),
        taxonomy="detailed",
    ),
    "premier_league": DataSource(
        key="premier_league",
        label="Premier League 2016/17-2025/26 (FPL + Understat)",
        path=PREMIER_LEAGUE_CSV,
        kind="real",
        summary=(
            "Ten seasons of real Premier League players, through the completed 2025/26 "
            "season. Current, and the only source here carrying age, price and ownership."
        ),
        attribution=(
            "Fantasy Premier League and Understat data, mirrored by the open-source "
            "Fantasy-Premier-League repository (https://github.com/vaastav/Fantasy-Premier-League)."
        ),
        caveats=(
            "This is a **summary feed, not event data**. There are no progressive passes, no "
            "pass completion, no dribbles and no aerial duels - the Methodology page lists "
            "exactly what each season does carry.",
            "FPL publishes only four positional buckets, so players are grouped **GK / DEF / "
            "MID / FWD**. Understat's line-up position rides along as an attribute you can "
            "filter on, but it is never used to group: inferring a finer position from the same "
            "statistics the models then read would be circular.",
            "Tackles, recoveries and clearances-blocks-interceptions exist only from 2025/26, "
            "when the game began scoring them. Shots, key passes and xGChain come from Understat "
            "and stop after 2024/25. Select one season in the sidebar to model on everything "
            "that season carries.",
            "**Price is the fantasy game's own valuation**, set by its operator and moved by "
            "transfers in and out. It is a popularity and perceived-value signal, not a transfer "
            "fee or a wage.",
        ),
        missing=("height_cm", "team_possession"),
        default_seasons=("2025-26",),
        taxonomy="bucket",
    ),
    "transfermarkt": DataSource(
        key="transfermarkt",
        label="Top six leagues (Transfermarkt)",
        path=TRANSFERMARKT_CSV,
        kind="real",
        summary=(
            "Every player in the big five plus Liga Portugal, one season, with real market "
            "values, true positions, ages, heights and nationalities."
        ),
        attribution=(
            "Transfermarkt data, read through the open-source transfermarkt-api service "
            "(https://github.com/felipeall/transfermarkt-api)."
        ),
        caveats=(
            "Transfermarkt is a **market and biographical database, not a performance one**. "
            "Its season statistics are appearances, goals, assists, cards and minutes - there is "
            "no xG, no passing, no defending. The similarity and archetype models therefore have "
            "very few features to work with here, and are correspondingly blunt.",
            "Its real strength is the other columns: **market value in euros** - the only genuine "
            "valuation in this project - plus true positions (centre-back and full-back, not "
            "'defender'), age, height, preferred foot, nationality and contract expiry.",
            "Best used to **enrich** a performance dataset rather than alone: build it with "
            "`--market-only` and the Premier League source gains real values and real positions.",
            "Requires network access to Transfermarkt through a running transfermarkt-api "
            "instance; see `scripts/fetch_transfermarkt.py`.",
        ),
        missing=("team_possession",),
    ),
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

DEFAULT_SOURCE = "understat_big6"

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

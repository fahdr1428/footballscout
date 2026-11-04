"""
The stat sheet: every player's numbers grouped the way a scout reads them.

The similarity engine and the rating work on a few chosen metrics per
position. The stat sheet is wider - every metric a source actually records,
put into plain categories and ordered by what matters for the position: a
keeper's page opens on goalkeeping and distribution, a centre-back's on
defending, a winger's on dribbling, a striker's on attacking.

Nothing here is a new model. Each metric is a rate or a percentage worked out
from the source's own counts, each is ranked against players in the same
position (percentiles, flipped where lower is better), and a category's
percentile on the page is the plain average of the percentiles listed under
it. A metric a source does not record is not shown - never estimated - and the
page names what is missing, so "no tackles" never reads as "made no tackles".

Physical tracking data - top speed, distance covered, sprints,
accelerations - is not published by any open dataset, so it is not here.
"""

from __future__ import annotations

# Each category: (name, [(metric, what it tells you), ...]). A metric belongs to
# one category only, so nothing is counted twice in a category average.
SHEET: list[tuple[str, list[tuple[str, str]]]] = [
    ("Attacking", [
        ("goals_per90", "Overall scoring output"),
        ("np_goals_per90", "Scoring without penalties"),
        ("goal_involvements_per90", "Direct goal output of any kind, penalties included"),
        ("np_goal_contributions_per90", "Non-penalty goals plus assists: open-play output"),
        ("xg_per90", "Quality of the chances taken"),
        ("npxg_per90", "Quality of the chances taken, penalties left out"),
        ("npxg_open_play_per90", "Chance quality from open play only"),
        ("np_goals_minus_npxg_per90", "Finishing above or below what the chances were worth"),
        ("contributions_above_expected_per90", "Goals and assists above what the chances and passes were worth"),
        ("shots_per90", "How often the player shoots"),
        ("shots_on_target_per90", "Shots that test the keeper"),
        ("shot_accuracy_pct", "Share of shots on target"),
        ("npxg_per_shot", "Average quality of the chances shot from"),
        ("sca_per90", "Actions that lead directly to a shot"),
        ("gca_per90", "Actions that lead directly to a goal"),
        ("touches_att_pen_per90", "Presence in the opposition box"),
        ("pens_won_per90", "Fouls drawn in the box that earned a penalty"),
        ("threat_per90", "FPL's threat index: how likely the touches are to lead to goals"),
    ]),
    ("Playmaking", [
        ("assists_per90", "Direct creative output"),
        ("xa_per90", "Expected assists: the quality of the chances created"),
        ("key_passes_per90", "Passes that lead to a shot"),
        ("progressive_passes_per90", "Passes that move the ball significantly towards goal"),
        ("through_balls_per90", "Passes that break a defensive line"),
        ("passes_into_final_third_per90", "Passes that take play into the final third"),
        ("passes_into_pen_area_per90", "Passes into the penalty area"),
        ("crosses_per90", "How often the player crosses, completed or not"),
        ("crosses_into_pen_area_per90", "Crosses completed into the box"),
        ("switches_per90", "Changing the direction of an attack"),
        ("sca_from_open_play_pass_per90", "Open-play passes that lead to a shot"),
        ("xg_chain_per90", "xG of every move the player is part of that ends in a shot"),
        ("xg_buildup_per90", "The same, leaving out the shot and the assist: build-up work"),
        ("creativity_per90", "FPL's creativity index"),
    ]),
    ("Passing and possession", [
        ("pass_pct", "Passing accuracy"),
        ("passes_attempted_per90", "How much of the ball the player sees and moves on"),
        ("short_pass_pct", "Accuracy over 5 to 15 yards"),
        ("medium_pass_pct", "Accuracy over 15 to 30 yards"),
        ("long_pass_pct", "Accuracy over 30 yards: range of distribution"),
        ("long_passes_attempted_per90", "How often the player goes long"),
        ("progressive_pass_distance_per90", "Yards the ball is moved towards goal with passes"),
        ("progressive_actions_per90", "Ball progression: progressive passes plus progressive carries"),
        ("passes_under_pressure_per90", "Passes made with an opponent closing in"),
        ("pass_pct_under_pressure", "Accuracy with an opponent closing in: composure"),
        ("progressive_receptions_per90", "Getting on the ball in advanced spots"),
        ("touches_per90", "Involvement in play"),
        ("carries_into_final_third_per90", "Carrying the ball into the final third"),
        ("possession_lost_per90", "Miscontrols plus times dispossessed: ball security"),
        ("miscontrols_per90", "First-touch reliability"),
        ("dispossessed_per90", "Losing the ball to a tackle"),
    ]),
    ("Dribbling and carrying", [
        ("dribbles_completed_per90", "Successful take-ons"),
        ("dribble_success_pct", "Share of take-ons that beat the man"),
        ("dribbles_attempted_per90", "How often the player takes someone on"),
        ("progressive_carries_per90", "Carries that move the ball significantly towards goal"),
        ("progressive_carry_distance_per90", "Yards the ball is carried towards goal"),
        ("carries_into_pen_area_per90", "Carrying the ball into the box"),
        ("sca_from_dribble_per90", "Take-ons and carries that lead to a shot"),
        ("carries_per90", "How often the player carries the ball"),
        ("fouls_won_per90", "Fouls suffered: how often opponents can only stop the player illegally"),
    ]),
    ("Defending", [
        ("tackles_per90", "Every tackle attempted, won or lost"),
        ("tackles_won_per90", "Tackles that win the ball"),
        ("tackle_win_pct", "Share of tackles that win the ball"),
        ("dribbler_stop_pct", "Share of opponents who tried to dribble past and were stopped"),
        ("interceptions_per90", "Reading the game"),
        ("blocks_per90", "Blocking shots and passes"),
        ("shots_blocked_per90", "Getting in the way of opposition shots"),
        ("clearances_per90", "Clearing danger"),
        ("ball_recoveries_per90", "Winning back loose balls"),
        ("pressures_per90", "Pressing an opponent on the ball"),
        ("pressure_success_pct", "Pressures after which the team wins the ball back within five seconds"),
        ("pressures_att_third_per90", "Pressing high up the pitch"),
        ("tackles_att_third_per90", "Winning the ball high up the pitch"),
        ("defensive_actions_per90", "Tackles, interceptions, blocks and clearances together"),
        ("padj_tackles_per90", "Tackles adjusted for how much of the ball the team has"),
        ("padj_interceptions_per90", "Interceptions adjusted for how much of the ball the team has"),
        ("errors_per90", "Errors that lead to an opposition shot"),
        ("cbi_per90", "FPL's count of defensive interventions"),
        ("defensive_contribution_per90", "FPL's defensive contribution count"),
        ("goals_conceded_per90", "Goals conceded while on the pitch: the whole defence's reading"),
        ("xgc_per90", "Expected goals conceded while on the pitch: the whole defence's reading"),
    ]),
    ("Goalkeeping", [
        ("gk_psxg_minus_ga_per90", "Goals prevented: post-shot xG faced minus goals let in"),
        ("gk_save_pct", "Share of shots on target saved"),
        ("save_rate", "Saves as a share of saves plus goals conceded"),
        ("gk_saves_per90", "Shot-stopping activity"),
        ("gk_psxg_per90", "Quality of the shots faced, after they are struck"),
        ("gk_shots_on_target_against_per90", "How many shots on target the keeper faces"),
        ("gk_goals_against_per90", "Goals let in: the defence in front of the keeper shapes it as much as the keeper"),
        ("clean_sheet_rate", "Share of matches without conceding"),
        ("gk_cross_stop_pct", "Crosses claimed: command of the area"),
        ("gk_def_actions_outside_box_per90", "Sweeping behind the defence"),
        ("gk_avg_sweeper_distance", "How far from goal the keeper defends"),
        ("gk_pen_save_pct", "Share of penalties faced that were saved"),
    ]),
    ("Distribution", [
        ("gk_launch_pct", "Launches completed: long kicks that find a team-mate"),
        ("gk_avg_pass_length", "Short build-up or long balls: a style reading more than a quality one"),
    ]),
    ("Physical and availability", [
        ("aerials_won_per90", "Headers and jumps won against an opponent"),
        ("aerial_win_pct", "Share of aerial duels won: physical dominance"),
        ("aerials_contested_per90", "How often the player goes up for a header: a role reading"),
        ("minutes", "Availability and workload over the period"),
        ("starts_share", "Share of appearances started"),
    ]),
    ("Discipline", [
        ("yellow_cards_per90", "Bookings"),
        ("red_cards_per90", "Sendings-off"),
        ("fouls_committed_per90", "Fouls given away"),
        ("pens_conceded_per90", "Penalties given away"),
        ("offsides_per90", "Timing of runs against the line"),
    ]),
]

# For keepers, distribution is their passing: the passing metrics move to it.
GK_DISTRIBUTION = ["pass_pct", "passes_attempted_per90", "long_pass_pct", "long_passes_attempted_per90",
                   "progressive_passes_per90"]

# The order a scout reads each position in. Keepers see keeping; outfield
# players never see a keeper's category.
GK_ORDER = ["Goalkeeping", "Distribution", "Physical and availability", "Discipline"]
ORDER = {
    "GK": GK_ORDER,
    "CB": ["Defending", "Passing and possession", "Physical and availability", "Playmaking",
           "Dribbling and carrying", "Attacking", "Discipline"],
    "FB": ["Defending", "Playmaking", "Dribbling and carrying", "Passing and possession", "Attacking",
           "Physical and availability", "Discipline"],
    "DEF": ["Defending", "Passing and possession", "Playmaking", "Attacking", "Physical and availability", "Discipline"],
    "DM": ["Defending", "Passing and possession", "Playmaking", "Dribbling and carrying", "Attacking",
           "Physical and availability", "Discipline"],
    "CM": ["Passing and possession", "Playmaking", "Defending", "Dribbling and carrying", "Attacking",
           "Physical and availability", "Discipline"],
    "MID": ["Playmaking", "Passing and possession", "Attacking", "Defending", "Dribbling and carrying",
            "Physical and availability", "Discipline"],
    "WM": ["Playmaking", "Dribbling and carrying", "Attacking", "Passing and possession", "Defending",
           "Physical and availability", "Discipline"],
    "AM": ["Playmaking", "Attacking", "Dribbling and carrying", "Passing and possession", "Defending",
           "Physical and availability", "Discipline"],
    "W": ["Dribbling and carrying", "Attacking", "Playmaking", "Passing and possession", "Defending",
          "Physical and availability", "Discipline"],
    "SS": ["Attacking", "Playmaking", "Dribbling and carrying", "Passing and possession", "Defending",
           "Physical and availability", "Discipline"],
    "FW": ["Attacking", "Playmaking", "Dribbling and carrying", "Passing and possession", "Defending",
           "Physical and availability", "Discipline"],
    "FWD": ["Attacking", "Playmaking", "Dribbling and carrying", "Passing and possession", "Defending",
            "Physical and availability", "Discipline"],
}
OUTFIELD_ORDER = ORDER["CM"]
# Rows that mean nothing for a keeper even where a feed carries them.
OUTFIELD_ONLY = {"offsides_per90"}

# What the categories cannot hold, because no open source records it - said on
# the page rather than left for a reader to wonder about.
NOT_RECORDED = {
    "Attacking": "Big chances missed and off-ball runs are not in any open dataset.",
    "Passing and possession": "Forward-pass share and total passes received are not in these feeds.",
    "Physical and availability": "Top speed, distance covered, sprints and accelerations come from tracking "
                                 "data that no open dataset publishes, so they are not here.",
}

# The handful of numbers each position is judged on first, shown large above
# the sheet: for a keeper, goals prevented, save percentage and distribution.
# The first four the source has are shown; this is a reading order, not a model.
HEADLINE: dict[str, list[str]] = {
    "GK": ["gk_psxg_minus_ga_per90", "gk_save_pct", "gk_launch_pct", "pass_pct", "save_rate",
           "gk_cross_stop_pct", "clean_sheet_rate"],
    "CB": ["defensive_actions_per90", "aerial_win_pct", "progressive_passes_per90", "pass_pct",
           "cbi_per90", "defensive_contribution_per90", "clearances_per90", "xg_buildup_per90"],
    "FB": ["progressive_carries_per90", "crosses_into_pen_area_per90", "tackles_won_per90", "xa_per90",
           "defensive_actions_per90", "creativity_per90", "cbi_per90", "key_passes_per90"],
    "DEF": ["defensive_contribution_per90", "cbi_per90", "creativity_per90", "threat_per90", "xa_per90"],
    "DM": ["interceptions_per90", "tackles_won_per90", "progressive_passes_per90", "pass_pct",
           "defensive_actions_per90", "xg_buildup_per90", "key_passes_per90"],
    "CM": ["progressive_passes_per90", "pass_pct", "xa_per90", "ball_recoveries_per90",
           "key_passes_per90", "xg_buildup_per90", "np_goal_contributions_per90"],
    "MID": ["np_goal_contributions_per90", "xa_per90", "creativity_per90", "threat_per90", "defensive_contribution_per90"],
    "WM": ["dribbles_completed_per90", "progressive_carries_per90", "xa_per90", "npxg_per90",
           "key_passes_per90", "np_goal_contributions_per90"],
    "AM": ["xa_per90", "key_passes_per90", "np_goal_contributions_per90", "progressive_receptions_per90",
           "npxg_per90", "xg_chain_per90"],
    "W": ["dribbles_completed_per90", "progressive_carries_per90", "npxg_per90", "xa_per90",
          "key_passes_per90", "np_goal_contributions_per90", "xg_chain_per90"],
    "SS": ["np_goals_per90", "npxg_per90", "xa_per90", "np_goals_minus_npxg_per90", "key_passes_per90"],
    "FW": ["np_goals_per90", "npxg_per90", "np_goals_minus_npxg_per90", "touches_att_pen_per90",
           "npxg_per_shot", "shots_per90", "xa_per90"],
    "FWD": ["np_goals_per90", "npxg_per90", "threat_per90", "np_goals_minus_npxg_per90", "xa_per90"],
}

# Numbers that read the team as much as the player: how many shots a keeper
# faces is his defence's doing, as are goals conceded and clean sheets. The
# sheet shows them, tagged, and leaves them out of the category's average.
TEAM_READINGS = frozenset({
    "clean_sheet_rate", "goals_conceded_per90", "xgc_per90", "gk_goals_against_per90", "gk_saves_per90",
    "gk_psxg_per90", "gk_shots_on_target_against_per90",
})

DESCRIPTIONS: dict[str, str] = {m: d for _, rows in SHEET for m, d in rows}
DESCRIPTIONS.update({
    "pass_pct": "Passing accuracy",
    "passes_attempted_per90": "How much of the ball the player sees and moves on",
})
CATEGORY_OF: dict[str, str] = {m: c for c, rows in SHEET for m, _ in rows}


def order_for(group: str) -> list[str]:
    """The stat sheet categories for a position, in reading order."""
    return ORDER.get(group, OUTFIELD_ORDER)


def metrics_for(group: str) -> list[str]:
    """Every stat-sheet metric a position can show, before checking the source has it."""
    out: list[str] = []
    for category in order_for(group):
        rows = dict(SHEET)[category]
        out += [m for m, _ in rows]
    if group == "GK":
        out += GK_DISTRIBUTION
    return list(dict.fromkeys(out))


def sheet_for(group: str, available: list[str]) -> list[list]:
    """[[category, [metric, ...]], ...] for a position, only metrics the source has."""
    have = set(available)
    out = []
    for category in order_for(group):
        rows = [m for m, _ in dict(SHEET)[category]]
        if group == "GK" and category == "Distribution":
            rows = GK_DISTRIBUTION + rows
        rows = [m for m in rows if m in have and not (group == "GK" and m in OUTFIELD_ONLY)]
        if rows:
            out.append([category, rows])
    return out


def headline_for(group: str, available: list[str], n: int = 4) -> list[str]:
    """The first n headline metrics for a position that the source has."""
    have = set(available)
    return [m for m in HEADLINE.get(group, []) if m in have][:n]

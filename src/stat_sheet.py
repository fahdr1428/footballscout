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

Two kinds of row are shown but kept out of a category's average: team
readings (goals conceded, shots a keeper faces - the defence's doing as much as
the player's) and the "Role and style" category (where on the pitch a player
touches the ball, set-piece duty, how often a keeper goes long), which says how
a player plays rather than how well.

Physical tracking data - top speed, distance covered, sprints,
accelerations - is not published by any open dataset, so it is not here.
"""

from __future__ import annotations

# Each category: (name, [(metric, what it tells you), ...]). A metric belongs to
# one category only, so nothing is counted twice in a category average. Within
# a category the rows run from what matters most to least.
SHEET: list[tuple[str, list[tuple[str, str]]]] = [
    ("Attacking", [
        ("goals_per90", "Overall scoring output"),
        ("np_goals_per90", "Scoring without penalties"),
        ("goal_involvements_per90", "Direct goal output of any kind, penalties included"),
        ("np_goal_contributions_per90", "Non-penalty goals plus assists: open-play output"),
        ("xg_per90", "Quality of the chances taken"),
        ("npxg_per90", "Quality of the chances taken, penalties left out"),
        ("xgi_per90", "xG plus xA: the goal threat a player gets and creates"),
        ("npxg_open_play_per90", "Chance quality from open play only"),
        ("npxg_set_piece_per90", "Chance quality from corners and free-kicks"),
        ("np_goals_minus_npxg_per90", "Finishing above or below what the chances were worth"),
        ("contributions_above_expected_per90", "Goals and assists above what the chances and passes were worth"),
        ("shots_per90", "How often the player shoots"),
        ("shots_on_target_per90", "Shots that test the keeper"),
        ("shot_accuracy_pct", "Share of shots on target"),
        ("npxg_per_shot", "Average quality of the chances shot from"),
        ("shot_conversion_pct", "Share of shots that go in (penalties left out where the source counts them apart)"),
        ("npxg_xa_per90", "Chances taken plus chances created: all-round attacking threat without penalties"),
        ("sca_per90", "Actions that lead directly to a shot"),
        ("gca_per90", "Actions that lead directly to a goal"),
        ("touches_att_pen_per90", "Presence in the opposition box"),
        ("pens_won_per90", "Fouls drawn in the box that earned a penalty"),
        ("free_kick_shots_per90", "Shots straight from a free-kick"),
        ("threat_per90", "FPL's threat index: how likely the touches are to lead to goals"),
    ]),
    ("Playmaking", [
        ("assists_per90", "Direct creative output"),
        ("xa_per90", "Expected assists: the quality of the chances created"),
        ("key_passes_per90", "Passes that lead to a shot"),
        ("xa_per_key_pass", "Average quality of the chances created: a cut-back in the box against a pass to a 30-yard shot"),
        ("assists_minus_xa_per90", "Assists above or below what the chances created were worth: mostly team-mates' finishing"),
        ("progressive_passes_per90", "Passes that move the ball significantly towards goal"),
        ("through_balls_per90", "Passes that break a defensive line"),
        ("passes_into_final_third_per90", "Passes that take play into the final third"),
        ("passes_into_pen_area_per90", "Passes into the penalty area"),
        ("crosses_per90", "How often the player crosses, completed or not"),
        ("crosses_into_pen_area_per90", "Crosses completed into the box"),
        ("cross_box_pct", "Share of crosses that find a team-mate in the box"),
        ("switches_per90", "Changing the direction of an attack"),
        ("sca_from_open_play_pass_per90", "Open-play passes that lead to a shot"),
        ("sca_from_set_piece_per90", "Set pieces delivered that lead to a shot"),
        ("xg_chain_per90", "xG of every move the player is part of that ends in a shot"),
        ("xg_buildup_per90", "xG of the moves the player is part of, leaving out the shot and the assist: build-up work"),
        ("creativity_per90", "FPL's creativity index"),
    ]),
    ("Passing and possession", [
        ("pass_pct", "Passing accuracy"),
        ("passes_attempted_per90", "How much of the ball the player sees and moves on"),
        ("passes_completed_per90", "Passes that find a team-mate: volume and security together"),
        ("short_pass_pct", "Accuracy over 5 to 15 yards"),
        ("short_passes_completed_per90", "Short passes that find a team-mate"),
        ("medium_pass_pct", "Accuracy over 15 to 30 yards"),
        ("medium_passes_completed_per90", "Medium passes that find a team-mate"),
        ("long_pass_pct", "Accuracy over 30 yards: range of distribution"),
        ("long_passes_completed_per90", "Long passes that find a team-mate"),
        ("progressive_pass_distance_per90", "Yards the ball is moved towards goal with passes"),
        ("progressive_pass_rate", "How often a completed pass moves play on: forward-thinking, not just busy"),
        ("pass_progress_per_pass", "Yards gained towards goal per pass attempted"),
        ("progressive_actions_per90", "Ball progression: progressive passes plus progressive carries"),
        ("passes_under_pressure_per90", "Passes made with an opponent closing in"),
        ("passes_completed_under_pressure_per90", "Passes completed with an opponent closing in"),
        ("pass_pct_under_pressure", "Accuracy with an opponent closing in: composure"),
        ("progressive_receptions_per90", "Getting on the ball in advanced spots"),
        ("touches_per90", "Involvement in play"),
        ("carries_into_final_third_per90", "Carrying the ball into the final third"),
        ("possession_lost_per90", "Miscontrols plus times dispossessed: ball security"),
        ("turnover_rate", "Ball security per touch, so a busy player is not marked down for seeing more of it"),
        ("miscontrols_per90", "First-touch reliability"),
        ("dispossessed_per90", "Losing the ball to a tackle"),
        ("influence_per90", "FPL's influence index: involvement in the moments that decide matches"),
    ]),
    ("Dribbling and carrying", [
        ("dribbles_completed_per90", "Successful take-ons"),
        ("dribble_success_pct", "Share of take-ons that beat the man"),
        ("dribbles_attempted_per90", "How often the player takes someone on"),
        ("progressive_carries_per90", "Carries that move the ball significantly towards goal"),
        ("progressive_carry_distance_per90", "Yards the ball is carried towards goal"),
        ("progressive_carry_rate", "How often a carry moves play on"),
        ("carry_progress_per_carry", "Yards gained towards goal per carry"),
        ("carries_into_pen_area_per90", "Carrying the ball into the box"),
        ("sca_from_dribble_per90", "Take-ons and carries that lead to a shot"),
        ("carries_per90", "How often the player carries the ball"),
        ("fouls_won_per90", "Fouls suffered: how often opponents can only stop the player illegally"),
    ]),
    ("Defending", [
        ("tackles_per90", "Every tackle attempted, won or lost"),
        ("tackles_won_per90", "Tackles that win the ball"),
        ("tackle_win_pct", "Share of tackles that win the ball"),
        ("tackles_def_third_per90", "Last-line defending"),
        ("tackles_mid_third_per90", "Screening in front of the defence"),
        ("tackles_att_third_per90", "Winning the ball high up the pitch"),
        ("dribbler_stop_pct", "Share of opponents who tried to dribble past and were stopped"),
        ("dribblers_tackled_per90", "Dribblers stopped with a tackle"),
        ("interceptions_per90", "Reading the game"),
        ("tackles_interceptions_per90", "The two ways of winning the ball back, together"),
        ("dribbled_past_per90", "Dribblers faced who got past: one-against-one defending lost"),
        ("blocks_per90", "Blocking shots and passes"),
        ("shots_blocked_per90", "Getting in the way of opposition shots"),
        ("passes_blocked_per90", "Getting in the way of opposition passes"),
        ("clearances_per90", "Clearing danger"),
        ("ball_recoveries_per90", "Winning back loose balls"),
        ("pressures_per90", "Pressing an opponent on the ball"),
        ("pressures_successful_per90", "Pressures after which the team wins the ball back"),
        ("pressure_success_pct", "Pressures after which the team wins the ball back within five seconds"),
        ("pressures_att_third_per90", "Pressing high up the pitch"),
        ("defensive_actions_per90", "Tackles, interceptions, blocks and clearances together"),
        ("sca_from_defensive_action_per90", "Winning the ball in a way that leads straight to a shot"),
        ("padj_tackles_per90", "Tackles adjusted for how much of the ball the team has"),
        ("padj_interceptions_per90", "Interceptions adjusted for how much of the ball the team has"),
        ("padj_blocks_per90", "Blocks adjusted for how much of the ball the team has"),
        ("padj_clearances_per90", "Clearances adjusted for how much of the ball the team has"),
        ("padj_ball_recoveries_per90", "Recoveries adjusted for how much of the ball the team has"),
        ("padj_pressures_per90", "Pressures adjusted for how much of the ball the team has"),
        ("padj_defensive_actions_per90", "All defensive actions adjusted for how much of the ball the team has"),
        ("errors_per90", "Errors that lead to an opposition shot"),
        ("cbi_per90", "FPL's count of clearances, blocks and interceptions"),
        ("defensive_contribution_per90", "FPL's defensive contribution count"),
        ("goals_conceded_per90", "Goals conceded while on the pitch: the whole defence's reading"),
        ("xgc_per90", "Expected goals conceded while on the pitch: the whole defence's reading"),
    ]),
    ("Goalkeeping", [
        ("gk_psxg_minus_ga_per90", "Goals prevented: post-shot xG faced minus goals let in"),
        ("gk_xgc_minus_ga_per90", "xG of the shots faced while in goal minus goals let in. Pre-shot xG, so it "
                                  "does not see where shots were placed, and it reads the defence too"),
        ("gk_save_pct", "Share of shots on target saved"),
        ("save_rate", "Saves as a share of saves plus goals conceded"),
        ("gk_saves_per90", "Shot-stopping activity"),
        ("gk_psxg_per90", "Quality of the shots faced, after they are struck"),
        ("gk_psxg_per_sot", "How hard the average shot on target faced was to save"),
        ("gk_shots_on_target_against_per90", "How many shots on target the keeper faces"),
        ("gk_goals_against_per90", "Goals let in: the defence in front of the keeper shapes it as much as the keeper"),
        ("clean_sheet_rate", "Share of matches without conceding"),
        ("gk_cross_stop_pct", "Crosses claimed: command of the area"),
        ("gk_crosses_stopped_per90", "Crosses claimed"),
        ("gk_crosses_faced_per90", "Crosses into the box faced: how often the area is tested"),
        ("gk_def_actions_outside_box_per90", "Sweeping behind the defence"),
        ("gk_pen_save_pct", "Share of penalties faced that were saved"),
        ("gk_pens_saved_per90", "Penalties saved"),
        ("gk_pens_faced_per90", "Penalties faced: given away by the team"),
    ]),
    ("Distribution", [
        ("gk_launch_pct", "Launches completed: long kicks that find a team-mate"),
        ("gk_launches_completed_per90", "Long kicks that find a team-mate"),
    ]),
    ("Physical and availability", [
        ("aerials_won_per90", "Headers and jumps won against an opponent"),
        ("aerial_win_pct", "Share of aerial duels won: physical dominance"),
        ("aerials_lost_per90", "Aerial duels lost"),
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
    ("Overall indexes", [
        ("bps_per90", "FPL's bonus points system score: its own all-round match rating"),
        ("ict_per90", "FPL's influence, creativity and threat added together"),
    ]),
    # How a player plays rather than how well: shown, drawn neutral and not averaged.
    ("Role and style", [
        ("team_shot_share", "His shots per 90 as a share of the team's: how much of the shooting is his"),
        ("team_npxg_share", "His chances per 90 as a share of the team's"),
        ("team_key_pass_share", "His key passes per 90 as a share of the team's: how much of the creating is his"),
        ("team_xg_chain_share", "xG of the moves he is part of, per 90, against the team's xG per 90"),
        ("team_touch_share", "His touches per 90 as a share of the team's: how much play runs through him"),
        ("team_progressive_pass_share", "His progressive passes per 90 as a share of the team's"),
        ("team_defensive_action_share", "His tackles and interceptions per 90 as a share of the team's"),
        ("buildup_share", "How much of his involvement is build-up rather than the shot or the final pass"),
        ("short_pass_share", "Passes of 5 to 15 yards as a share of his passing"),
        ("medium_pass_share", "Passes of 15 to 30 yards as a share of his passing"),
        ("long_pass_share", "Passes over 30 yards as a share of his passing"),
        ("take_on_rate", "How often a touch becomes a take-on"),
        ("touches_def_pen_per90", "Touches in the player's own box"),
        ("touches_def_third_per90", "Touches in the defensive third"),
        ("touches_mid_third_per90", "Touches in the middle third"),
        ("touches_att_third_per90", "Touches in the attacking third"),
        ("def_third_touch_share", "Share of touches in the defensive third: how deep the player operates"),
        ("mid_third_touch_share", "Share of touches in the middle third"),
        ("att_third_touch_share", "Share of touches in the attacking third: how high the player operates"),
        ("box_touch_share", "Share of touches in the opposition box"),
        ("att_third_tackle_share", "Share of tackles made in the attacking third: where the ball is won"),
        ("dribblers_challenged_per90", "How often opponents try to dribble past the player"),
        ("aerials_contested_per90", "How often the player goes up for a header"),
        ("long_passes_attempted_per90", "How often the player goes long"),
        ("pressured_pass_share", "Share of passes made under pressure: how tight the spaces are"),
        ("open_play_npxg_share", "Share of chance quality from open play rather than set pieces"),
        ("set_piece_sca_share", "Share of chances created from set pieces"),
        ("corners_taken_per90", "Corner duty"),
        ("free_kick_passes_per90", "Free-kick duty"),
        ("gk_launches_attempted_per90", "How often the keeper kicks long"),
        ("gk_avg_pass_length", "Short build-up or long balls"),
        ("gk_avg_sweeper_distance", "How far from goal the keeper defends"),
    ]),
]

# For keepers, distribution is their passing: the passing metrics move to it,
# and what the team gives away in front of them sits with their goalkeeping.
GK_DISTRIBUTION = ["pass_pct", "passes_attempted_per90", "long_pass_pct", "long_passes_completed_per90",
                   "progressive_passes_per90", "xg_buildup_per90"]
GK_KEEPING = ["xgc_per90"]

# Categories that are not averaged and whose rows are drawn neutral.
STYLE_CATEGORIES = ("Role and style",)
# A source's own composite indexes: shown where it has them, never a gap where it does not.
OPTIONAL_CATEGORIES = ("Overall indexes",)
_TAIL = ["Physical and availability", "Discipline", "Overall indexes", "Role and style"]

# The order a scout reads each position in. Keepers see keeping; outfield
# players never see a keeper's category.
GK_ORDER = ["Goalkeeping", "Distribution"] + _TAIL
ORDER = {
    "GK": GK_ORDER,
    "CB": ["Defending", "Passing and possession", "Playmaking", "Dribbling and carrying", "Attacking"] + _TAIL,
    "FB": ["Defending", "Playmaking", "Dribbling and carrying", "Passing and possession", "Attacking"] + _TAIL,
    "DEF": ["Defending", "Passing and possession", "Playmaking", "Dribbling and carrying", "Attacking"] + _TAIL,
    "DM": ["Defending", "Passing and possession", "Playmaking", "Dribbling and carrying", "Attacking"] + _TAIL,
    "CM": ["Passing and possession", "Playmaking", "Defending", "Dribbling and carrying", "Attacking"] + _TAIL,
    "MID": ["Playmaking", "Passing and possession", "Attacking", "Defending", "Dribbling and carrying"] + _TAIL,
    "WM": ["Playmaking", "Dribbling and carrying", "Attacking", "Passing and possession", "Defending"] + _TAIL,
    "AM": ["Playmaking", "Attacking", "Dribbling and carrying", "Passing and possession", "Defending"] + _TAIL,
    "W": ["Dribbling and carrying", "Attacking", "Playmaking", "Passing and possession", "Defending"] + _TAIL,
    "SS": ["Attacking", "Playmaking", "Dribbling and carrying", "Passing and possession", "Defending"] + _TAIL,
    "FW": ["Attacking", "Playmaking", "Dribbling and carrying", "Passing and possession", "Defending"] + _TAIL,
    "FWD": ["Attacking", "Playmaking", "Dribbling and carrying", "Passing and possession", "Defending"] + _TAIL,
}
OUTFIELD_ORDER = ORDER["CM"]
# Rows that mean nothing for a keeper even where a feed carries them.
OUTFIELD_ONLY = {"offsides_per90", "dribblers_challenged_per90", "pressured_pass_share",
                 "team_shot_share", "team_npxg_share", "team_key_pass_share", "team_xg_chain_share",
                 "team_progressive_pass_share", "team_defensive_action_share", "buildup_share", "take_on_rate",
                 "mid_third_touch_share",
                 "open_play_npxg_share", "set_piece_sca_share", "corners_taken_per90", "free_kick_passes_per90",
                 "att_third_tackle_share", "box_touch_share", "att_third_touch_share", "touches_att_third_per90"}
# Rows that mean nothing for an outfield player.
KEEPER_ONLY = {"gk_launches_attempted_per90", "gk_avg_pass_length", "gk_avg_sweeper_distance"}

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
# Each list runs on past what the richest source has, so a thinner one (Top
# leagues, the Premier League feed) still finds four numbers that fit the role.
HEADLINE: dict[str, list[str]] = {
    "GK": ["gk_psxg_minus_ga_per90", "gk_xgc_minus_ga_per90", "gk_save_pct", "gk_launch_pct", "pass_pct",
           "save_rate", "gk_cross_stop_pct", "clean_sheet_rate", "xgc_per90", "gk_goals_against_per90"],
    "CB": ["defensive_actions_per90", "aerial_win_pct", "progressive_passes_per90", "pass_pct",
           "cbi_per90", "defensive_contribution_per90", "tackles_per90", "ball_recoveries_per90",
           "xg_buildup_per90", "xg_chain_per90", "xa_per90", "npxg_per90"],
    "FB": ["progressive_carries_per90", "crosses_into_pen_area_per90", "tackles_won_per90", "xa_per90",
           "tackles_per90", "defensive_contribution_per90", "cbi_per90", "key_passes_per90",
           "creativity_per90", "xg_buildup_per90", "xg_chain_per90"],
    "DEF": ["defensive_contribution_per90", "cbi_per90", "tackles_per90", "ball_recoveries_per90",
            "creativity_per90", "threat_per90", "xa_per90", "xg_buildup_per90", "xg_chain_per90"],
    "DM": ["tackles_interceptions_per90", "tackles_won_per90", "progressive_passes_per90", "pass_pct",
           "interceptions_per90", "tackles_mid_third_per90",
           "tackles_per90", "ball_recoveries_per90", "defensive_contribution_per90", "cbi_per90",
           "xg_buildup_per90", "xg_chain_per90", "key_passes_per90", "xa_per90"],
    "CM": ["progressive_passes_per90", "pass_pct", "xa_per90", "ball_recoveries_per90",
           "tackles_per90", "key_passes_per90", "xg_buildup_per90", "xg_chain_per90",
           "np_goal_contributions_per90", "defensive_contribution_per90", "creativity_per90"],
    "MID": ["np_goal_contributions_per90", "xa_per90", "creativity_per90", "threat_per90",
            "defensive_contribution_per90", "xg_chain_per90", "key_passes_per90", "npxg_per90"],
    "WM": ["dribbles_completed_per90", "progressive_carries_per90", "xa_per90", "npxg_per90",
           "key_passes_per90", "np_goal_contributions_per90", "xg_chain_per90", "creativity_per90"],
    "AM": ["xa_per90", "key_passes_per90", "np_goal_contributions_per90", "progressive_receptions_per90",
           "npxg_per90", "xg_chain_per90", "creativity_per90", "threat_per90"],
    "W": ["dribbles_completed_per90", "progressive_carries_per90", "npxg_per90", "xa_per90",
          "key_passes_per90", "np_goal_contributions_per90", "xg_chain_per90", "threat_per90"],
    "SS": ["np_goals_per90", "npxg_per90", "xa_per90", "np_goals_minus_npxg_per90", "key_passes_per90",
           "xg_chain_per90"],
    "FW": ["np_goals_per90", "npxg_per90", "np_goals_minus_npxg_per90", "touches_att_pen_per90",
           "npxg_per_shot", "shots_per90", "xa_per90", "threat_per90"],
    "FWD": ["np_goals_per90", "npxg_per90", "threat_per90", "np_goals_minus_npxg_per90", "xa_per90",
            "npxg_per_shot", "shots_per90"],
}

# Numbers that read the team as much as the player: how many shots a keeper
# faces is his defence's doing, as are goals conceded and clean sheets. The
# sheet shows them, tagged, and leaves them out of the category's average.
TEAM_READINGS = frozenset({
    "clean_sheet_rate", "goals_conceded_per90", "xgc_per90", "gk_goals_against_per90", "gk_saves_per90",
    "gk_psxg_per90", "gk_shots_on_target_against_per90", "gk_crosses_faced_per90", "gk_pens_faced_per90",
    "gk_psxg_per_sot",
})

DESCRIPTIONS: dict[str, str] = {m: d for _, rows in SHEET for m, d in rows}
DESCRIPTIONS.update({
    "pass_pct": "Passing accuracy",
    "passes_attempted_per90": "How much of the ball the player sees and moves on",
})
CATEGORY_OF: dict[str, str] = {m: c for c, rows in SHEET for m, _ in rows}
STYLE_METRICS = frozenset(m for c, rows in SHEET if c in STYLE_CATEGORIES for m, _ in rows)


def order_for(group: str) -> list[str]:
    """The stat sheet categories for a position, in reading order."""
    return ORDER.get(group, OUTFIELD_ORDER)


def _rows(group: str, category: str) -> list[str]:
    rows = [m for m, _ in dict(SHEET)[category]]
    if group == "GK":
        if category == "Distribution":
            rows = GK_DISTRIBUTION + rows
        if category == "Goalkeeping":
            rows = rows + GK_KEEPING
        rows = [m for m in rows if m not in OUTFIELD_ONLY]
    else:
        rows = [m for m in rows if m not in KEEPER_ONLY]
    return rows


def metrics_for(group: str) -> list[str]:
    """Every stat-sheet metric a position can show, before checking the source has it."""
    out: list[str] = []
    for category in order_for(group):
        out += _rows(group, category)
    return list(dict.fromkeys(out))


def sheet_for(group: str, available: list[str]) -> list[list]:
    """[[category, [metric, ...]], ...] for a position, only metrics the source has.

    A metric appears once: a keeper's long passes sit under distribution, not
    again under role and style.
    """
    have, seen = set(available), set()
    out = []
    for category in order_for(group):
        rows = [m for m in _rows(group, category) if m in have and m not in seen]
        seen.update(rows)
        if rows:
            out.append([category, rows])
    return out


def headline_for(group: str, available: list[str], n: int = 4) -> list[str]:
    """The first n headline metrics for a position that the source has."""
    have = set(available)
    return [m for m in HEADLINE.get(group, []) if m in have and m not in STYLE_METRICS][:n]


# ---- exact positions in a four-bucket feed --------------------------------
# Top leagues and the Premier League feed rank players in four buckets
# (keepers, defenders, midfielders, forwards), because that is all Understat and
# the Fantasy Premier League record. Transfermarkt, a separate source, says where
# most of them actually played that season, so the sheet can also rank a
# defensive midfielder against defensive midfielders only - and read his
# categories in a defensive midfielder's order. The models stay on the buckets.
ROLE_OF = {
    "Centre-Back": "CB", "CB": "CB", "LCB": "CB", "RCB": "CB",
    "Left-Back": "FB", "Right-Back": "FB", "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "Defensive Midfield": "DM", "DM": "DM",
    "Central Midfield": "CM", "CM": "CM",
    "Attacking Midfield": "AM", "AM": "AM",
    "Left Midfield": "WM", "Right Midfield": "WM", "LM": "WM", "RM": "WM",
    "Left Winger": "W", "Right Winger": "W", "LW": "W", "RW": "W",
    "Centre-Forward": "FW", "CF": "FW", "ST": "FW",
    "Second Striker": "SS", "SS": "SS",
}
ROLE_NAMES = {
    "CB": "Centre-back", "FB": "Full-back", "DM": "Defensive midfielder", "CM": "Central midfielder",
    "AM": "Attacking midfielder", "WM": "Wide midfielder", "W": "Winger", "FW": "Centre-forward",
    "SS": "Second striker",
}
BUCKETS = ("DEF", "MID", "FWD")


def role_sheet(role: str, sheet: list[list], available: list[str]) -> dict:
    """A role's reading order over a bucket's sheet, and its headline numbers."""
    present = [c for c, _ in sheet]
    order = [c for c in order_for(role) if c in present]
    return {"o": order + [c for c in present if c not in order], "k": headline_for(role, available)}

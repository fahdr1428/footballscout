"""The stat sheet: plain categories, ordered by position, every metric labelled and described."""

from __future__ import annotations

from src.config import COUNTING_STATS, LOWER_IS_BETTER, METRIC_LABELS
from src.stat_sheet import (CATEGORY_OF, DESCRIPTIONS, HEADLINE, ORDER, SHEET, TEAM_READINGS, headline_for,
                            metrics_for, order_for, sheet_for)

CATEGORIES = [c for c, _ in SHEET]


def test_every_metric_sits_in_one_category():
    metrics = [m for _, rows in SHEET for m, _ in rows]
    assert len(metrics) == len(set(metrics))
    assert set(CATEGORY_OF) == set(metrics)


def test_every_metric_is_labelled_and_described():
    for metric in DESCRIPTIONS:
        assert metric in METRIC_LABELS or metric in COUNTING_STATS, metric
        assert DESCRIPTIONS[metric].strip(), metric


def test_orders_name_real_categories_and_keep_keeping_to_keepers():
    for group, order in ORDER.items():
        assert set(order) <= set(CATEGORIES), group
        assert len(order) == len(set(order)), group
        if group != "GK":
            assert "Goalkeeping" not in order and "Distribution" not in order, group
    assert order_for("GK")[:2] == ["Goalkeeping", "Distribution"]
    assert order_for("CB")[0] == "Defending"
    assert order_for("W")[0] == "Dribbling and carrying"
    assert order_for("FW")[0] == "Attacking"
    assert order_for("unknown") == order_for("CM")


def test_keeper_sheet_leads_on_goals_prevented_and_moves_passing_to_distribution():
    sheet = dict(sheet_for("GK", metrics_for("GK")))
    assert list(sheet)[:2] == ["Goalkeeping", "Distribution"]
    assert sheet["Goalkeeping"][:3] == ["gk_psxg_minus_ga_per90", "gk_xgc_minus_ga_per90", "gk_save_pct"]
    assert "xgc_per90" in sheet["Goalkeeping"]
    assert sheet["Distribution"][0] == "pass_pct"
    assert "offsides_per90" not in sheet["Discipline"]
    assert not {"Defending", "Attacking", "Playmaking"} & set(sheet)
    # A metric is listed once, even where a keeper's passing moves to distribution.
    listed = [m for rows in sheet.values() for m in rows]
    assert len(listed) == len(set(listed))
    # FBref's keepers: goals prevented, save percentage, then distribution.
    fbref = ["gk_psxg_minus_ga_per90", "gk_save_pct", "save_rate", "gk_launch_pct", "pass_pct"]
    assert headline_for("GK", fbref) == ["gk_psxg_minus_ga_per90", "gk_save_pct", "gk_launch_pct", "pass_pct"]
    # The Premier League feed's keepers: xG faced against goals conceded leads.
    fpl = ["gk_xgc_minus_ga_per90", "save_rate", "clean_sheet_rate", "xgc_per90"]
    assert headline_for("GK", fpl)[0] == "gk_xgc_minus_ga_per90"


def test_style_rows_are_never_headline_or_on_the_wrong_side():
    from src.stat_sheet import KEEPER_ONLY, OUTFIELD_ONLY, STYLE_METRICS
    for group in ORDER:
        assert not set(headline_for(group, metrics_for(group))) & STYLE_METRICS, group
        if group == "GK":
            assert not set(metrics_for(group)) & OUTFIELD_ONLY
        else:
            assert not set(metrics_for(group)) & KEEPER_ONLY, group


def test_exact_positions_read_in_their_own_order():
    from src.stat_sheet import ROLE_NAMES, ROLE_OF, role_sheet
    assert set(ROLE_OF.values()) == set(ROLE_NAMES)
    assert ROLE_OF["Defensive Midfield"] == ROLE_OF["DM"] == "DM"
    available = ["xa_per90", "tackles_per90", "ball_recoveries_per90", "xg_chain_per90", "npxg_per90"]
    sheet = sheet_for("MID", available)
    dm = role_sheet("DM", sheet, available)
    assert dm["o"][0] == "Defending" and sorted(dm["o"]) == sorted(c for c, _ in sheet)
    assert dm["k"][:2] == ["tackles_per90", "ball_recoveries_per90"]


def test_sheet_keeps_only_what_the_source_has():
    sheet = sheet_for("FW", ["goals_per90", "xa_per90", "tackles_per90", "not_a_metric"])
    assert sheet == [["Attacking", ["goals_per90"]], ["Playmaking", ["xa_per90"]], ["Defending", ["tackles_per90"]]]
    assert sheet_for("FW", []) == []


def test_headlines_and_team_readings_are_sheet_metrics():
    for group, metrics in HEADLINE.items():
        assert set(metrics) <= set(metrics_for(group)), group
    assert TEAM_READINGS <= set(DESCRIPTIONS)
    assert "possession_lost_per90" in LOWER_IS_BETTER

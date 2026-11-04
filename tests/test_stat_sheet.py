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
    assert sheet["Goalkeeping"][:2] == ["gk_psxg_minus_ga_per90", "gk_save_pct"]
    assert sheet["Distribution"][0] == "pass_pct"
    assert "offsides_per90" not in sheet["Discipline"]
    assert headline_for("GK", metrics_for("GK"))[:2] == ["gk_psxg_minus_ga_per90", "gk_save_pct"]


def test_sheet_keeps_only_what_the_source_has():
    sheet = sheet_for("FW", ["goals_per90", "xa_per90", "tackles_per90", "not_a_metric"])
    assert sheet == [["Attacking", ["goals_per90"]], ["Playmaking", ["xa_per90"]], ["Defending", ["tackles_per90"]]]
    assert sheet_for("FW", []) == []


def test_headlines_and_team_readings_are_sheet_metrics():
    for group, metrics in HEADLINE.items():
        assert set(metrics) <= set(metrics_for(group)), group
    assert TEAM_READINGS <= set(DESCRIPTIONS)
    assert "possession_lost_per90" in LOWER_IS_BETTER

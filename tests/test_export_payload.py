"""The built season files: each displayed metric's category membership lines up with it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import categories_for

DATA = Path(__file__).resolve().parents[1] / "static" / "data"
SEASONS = sorted(p for p in DATA.glob("*__*.json") if not p.name.endswith("__all.json"))


@pytest.mark.skipif(not SEASONS, reason="the site has not been exported")
@pytest.mark.parametrize("path", SEASONS, ids=[p.stem for p in SEASONS])
def test_category_membership_matches_the_baskets(path):
    meta = json.loads(path.read_text())["meta"]
    for group, spec in meta["groups"].items():
        basket = categories_for(group)
        assert len(spec["dw"]) == len(spec["d"]), group
        for metric, member in zip(spec["d"], spec["dw"]):
            assert member == [i for i, c in enumerate(spec["c"]) if metric in basket[c]], (group, metric)


@pytest.mark.skipif(not SEASONS, reason="the site has not been exported")
@pytest.mark.parametrize("path", SEASONS, ids=[p.stem for p in SEASONS])
def test_stat_sheet_only_lists_metrics_the_season_has(path):
    payload = json.loads(path.read_text())
    meta = payload["meta"]
    for group, spec in meta["groups"].items():
        listed = [m for _, metrics in spec["s"] for m in metrics]
        assert len(listed) == len(set(listed)), group          # one category per metric
        assert set(listed) <= set(spec["d"]), group
        assert set(spec["k"]) <= set(spec["d"]), group
        assert [c for c, _ in spec["s"]] == [c for c in spec["so"] if c in {c for c, _ in spec["s"]}], group
        for metric in listed:
            assert metric in meta["labels"] and metric in meta["desc"], (group, metric)
            # Every listed metric has a value for somebody in the position.
            j = spec["d"].index(metric)
            assert any(p["d"][j][0] is not None for p in payload["players"] if p["g"] == group), (group, metric)


@pytest.mark.skipif(not SEASONS, reason="the site has not been exported")
@pytest.mark.parametrize("path", SEASONS, ids=[p.stem for p in SEASONS])
def test_league_percentiles_follow_the_figures(path):
    """Within a position and league, a higher figure never ranks below a lower one."""
    payload = json.loads(path.read_text())
    lower = set(payload["meta"]["lower"])
    for group, spec in payload["meta"]["groups"].items():
        for league in payload["meta"]["leagues"]:
            peers = [p for p in payload["players"] if p["g"] == group and p["l"] == league and not p.get("sq")]
            for j, metric in enumerate(spec["d"]):
                rows = sorted((p["d"][j][0], p["d"][j][2]) for p in peers
                              if len(p["d"][j]) > 2 and p["d"][j][0] is not None and p["d"][j][2] is not None)
                for (v0, p0), (v1, p1) in zip(rows, rows[1:]):
                    if v1 > v0:
                        assert (p1 <= p0) if metric in lower else (p1 >= p0), (group, league, metric, v0, v1)


@pytest.mark.skipif(not SEASONS, reason="the site has not been exported")
@pytest.mark.parametrize("path", SEASONS, ids=[p.stem for p in SEASONS])
def test_exact_position_percentiles_follow_the_figures(path):
    """In a four-bucket feed, among one exact position a higher figure never ranks lower."""
    payload = json.loads(path.read_text())
    meta = payload["meta"]
    lower = set(meta["lower"])
    roles = {p.get("r") for p in payload["players"]} - {None}
    if not meta.get("roleNames"):
        assert not roles and all(len(c) <= 3 for p in payload["players"] for c in p["d"])
        return
    assert roles <= set(meta["roleNames"])
    for group, spec in meta["groups"].items():
        assert set(spec.get("rs", {})) == {p["r"] for p in payload["players"] if p["g"] == group and p.get("r")}
        for role in spec.get("rs", {}):
            peers = [p for p in payload["players"] if p["g"] == group and p.get("r") == role and not p.get("sq")]
            for j, metric in enumerate(spec["d"]):
                rows = sorted((p["d"][j][0], p["d"][j][3]) for p in peers
                              if p["d"][j][0] is not None and p["d"][j][3] is not None)
                for (v0, p0), (v1, p1) in zip(rows, rows[1:]):
                    if v1 > v0:
                        assert (p1 <= p0) if metric in lower else (p1 >= p0), (group, role, metric)

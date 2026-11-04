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

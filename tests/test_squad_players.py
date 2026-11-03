"""
Squad players - short of the 900-minute floor - are exported scored against the
regulars, and must leave every regular's number exactly where it was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_static import SQUAD_MINUTES, export, project_squad, squad_players  # noqa: E402


@pytest.fixture(scope="module")
def season_and_squad(features, platform):
    season = platform.seasons[-1]
    squad = squad_players(features, platform, season, None)
    return season, squad


def test_squad_players_sit_between_the_two_floors(season_and_squad, platform):
    _, squad = season_and_squad
    assert len(squad) > 0
    assert squad["minutes"].between(SQUAD_MINUTES, platform.min_minutes - 1e-9).all()


def test_regulars_are_unchanged_by_adding_squad_players(season_and_squad, platform):
    season, squad = season_and_squad
    alone = export(platform, season)["players"]
    together = export(platform, season, squad)["players"]
    # the regulars come first, in the same order, every field identical
    assert together[:len(alone)] == alone                   # z, percentiles, categories, all of it
    added = together[len(alone):]
    assert len(added) == len(squad) and all(p.get("sq") == 1 for p in added)


def test_a_squad_percentile_is_his_rank_among_the_regulars(season_and_squad, platform):
    _, squad = season_and_squad
    projected, pct, _, z = project_squad(platform, squad)
    column = next(c for c in pct.columns if pct[c].notna().any() and c[4:] not in ("minutes",))
    metric = column[4:]
    i = pct[column].first_valid_index()
    group = projected.loc[i, "position_group"]
    ref = platform.pool.loc[platform.pool["position_group"] == group, metric].astype(float).dropna()
    value = float(projected.loc[i, metric])
    expected = ((ref < value).sum() + (ref == value).sum() / 2 + 1) / (len(ref) + 1) * 100
    from src.config import LOWER_IS_BETTER
    if metric in LOWER_IS_BETTER:
        expected = 100 - expected
    assert pct.loc[i, column] == pytest.approx(round(expected, 1))


def test_a_squad_z_vector_uses_the_regulars_scaler(season_and_squad, platform):
    _, squad = season_and_squad
    projected, _, _, z = project_squad(platform, squad)
    i = next(iter(z))
    model = platform.models[projected.loc[i, "position_group"]]
    raw = projected.loc[i, model.features].astype(float)
    filled = raw.fillna(platform.pool.loc[model.index, model.features].astype(float).median())
    expected = (filled.to_numpy() - model.scaler.mean_) / model.scaler.scale_
    assert np.allclose(z[i].to_numpy(), expected)

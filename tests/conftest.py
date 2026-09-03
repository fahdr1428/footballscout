"""Shared fixtures. The platform is expensive to fit, so it is session-scoped."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_generation import generate_dataset  # noqa: E402
from src.data_processing import clean_players  # noqa: E402
from src.feature_engineering import build_features  # noqa: E402
from src.pipeline import build_platform  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    # Two seasons so the multi-season tools (trajectory, self-season recall)
    # have something to work with.
    return generate_dataset(seed=3, seasons=["2023-24", "2024-25"])


@pytest.fixture(scope="session")
def cleaned(raw):
    return clean_players(raw)


@pytest.fixture(scope="session")
def features(cleaned):
    clean, _report = cleaned
    return build_features(clean)


@pytest.fixture(scope="session")
def platform(features, cleaned):
    _clean, report = cleaned
    return build_platform(features, report, min_minutes=900)

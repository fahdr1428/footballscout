"""The committed feeds agree with each other, and the check itself never guesses."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cross_check  # noqa: E402

CHECKS = cross_check.checks()


def feed(rows):
    return pd.DataFrame(rows, columns=["key", "season", "league", "minutes", "goals"])


def test_a_name_twice_on_either_side_is_left_out_rather_than_guessed():
    left = feed([("gabriel", "2017-18", "La Liga", 2584, 0),
                 ("gabriel", "2017-18", "La Liga", 2377, 5),
                 ("pedri", "2017-18", "La Liga", 2000, 3)])
    right = feed([("gabriel", "2017-18", "La Liga", 2377, 5),
                  ("pedri", "2017-18", "La Liga", 2010, 3)])
    out = cross_check.agreement(left, right, ["key", "season", "league"])
    assert out["pairs"] == 1
    assert out["goals_same"] == 1.0


def test_short_seasons_are_not_counted():
    left = feed([("a", "2020-21", "Serie A", 400, 1), ("b", "2020-21", "Serie A", 2000, 7)])
    right = feed([("a", "2020-21", "Serie A", 400, 2), ("b", "2020-21", "Serie A", 2000, 8)])
    out = cross_check.agreement(left, right, ["key", "season", "league"])
    assert out["pairs"] == 1
    assert out["goals_same"] == 0.0 and out["goals_within_one"] == 1.0


@pytest.mark.parametrize("pair", ["Understat vs FBref", "Understat vs Transfermarkt",
                                  "FBref vs Transfermarkt", "Understat vs FPL", "FBref vs FPL"])
def test_independent_feeds_count_the_same_goals(pair):
    # A broken scrape or a mismatched join shows up here first: the feeds count
    # the same matches, so a goal tally is either right or someone is wrong.
    row = CHECKS[pair]
    assert row["pairs"] >= 2000
    assert row["goals_same"] >= 0.98
    assert row["minutes_within_5pct"] >= 0.97


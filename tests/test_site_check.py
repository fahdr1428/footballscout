"""The weekly refresh must not publish a site that lost data."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_site  # noqa: E402


def page(seasons: dict[str, tuple[int, bool]], root: Path, make_files: bool = True) -> str:
    catalog = []
    for key in sorted({k.split(":")[0] for k in seasons}):
        entries = []
        for full, (players, live) in seasons.items():
            source, season = full.split(":")
            if source != key:
                continue
            path = f"data/{source}__{season}.json"
            if make_files:
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text("{}")
            entries.append({"season": season, "file": path, "players": players,
                            **({"inProgress": True} if live else {})})
        catalog.append({"key": key, "seasons": entries})
    data = json.dumps({"catalog": catalog, "default": "", "inline": {}})
    return f"<html><script>window.__SCOUT__={data};</script></html>"


def run(tmp_path, monkeypatch, before, after, make_files=True) -> int:
    site = tmp_path / "static"
    site.mkdir()
    (site / "index.html").write_text(page(after, site, make_files))
    old = tmp_path / "before.html"
    old.write_text(page(before, tmp_path / "unused", make_files=False))
    monkeypatch.setattr(check_site, "SITE", site)
    monkeypatch.setattr(sys, "argv", ["check_site", "--before", str(old)])
    return check_site.main()


def test_a_normal_refresh_passes(tmp_path, monkeypatch):
    before = {"u:2025-26": (1800, False), "u:2026-27": (1200, True)}
    after = {"u:2025-26": (1810, False), "u:2026-27": (1300, True), "u:2027-28": (500, True)}
    assert run(tmp_path, monkeypatch, before, after) == 0


def test_a_vanished_season_blocks_publishing(tmp_path, monkeypatch):
    before = {"u:2024-25": (1800, False), "u:2025-26": (1800, False)}
    after = {"u:2025-26": (1800, False)}
    assert run(tmp_path, monkeypatch, before, after) == 1


@pytest.mark.parametrize("live, after_players, expected", [
    (False, 1500, 1),   # a complete season losing 17% is a fault
    (False, 1700, 0),   # 6% is noise
    (True, 1200, 0),    # a season in progress breathes as its floor rises
    (True, 500, 1),     # ...but not by more than 40%
])
def test_shrinking_is_judged_by_whether_the_season_is_finished(
        tmp_path, monkeypatch, live, after_players, expected):
    before = {"u:2025-26": (1800, live)}
    after = {"u:2025-26": (after_players, live)}
    assert run(tmp_path, monkeypatch, before, after) == expected


def test_a_missing_season_file_blocks_publishing(tmp_path, monkeypatch):
    seasons = {"u:2025-26": (1800, False)}
    assert run(tmp_path, monkeypatch, seasons, seasons, make_files=False) == 1

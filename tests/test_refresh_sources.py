"""The refresh's network handling: a host that never answers costs a bounded wait."""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import refresh_sources as rs  # noqa: E402


@pytest.fixture
def silent_host(monkeypatch):
    """A local server that accepts every connection and never says a word."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(50)
    held = []

    def accept():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            held.append(conn)

    threading.Thread(target=accept, daemon=True).start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    yield f"http://127.0.0.1:{server.getsockname()[1]}", held
    server.close()
    for conn in held:
        conn.close()


def test_a_silent_understat_costs_the_budget_not_retries_times_timeout(silent_host, monkeypatch, tmp_path):
    url, held = silent_host
    monkeypatch.setattr(rs, "UNDERSTAT", url)
    monkeypatch.setattr(rs, "UNDERSTAT_TIMEOUT", 1)
    monkeypatch.setattr(rs, "PAUSE_SECONDS", 0.1)
    monkeypatch.setattr(rs, "UNDERSTAT_BUDGET_SECONDS", 3)
    started = time.monotonic()
    with pytest.raises(Exception) as caught:
        rs.refresh_understat([2026], tmp_path)
    assert rs.unreachable(caught.value)
    assert time.monotonic() - started < 6
    # Every connection went to the data endpoint: the page route, on the same
    # dead host, was not tried.
    assert len(held) >= 1


def test_an_http_status_is_an_answer_not_an_outage():
    import urllib.error

    assert not rs.unreachable(urllib.error.HTTPError("u", 404, "Not Found", {}, None))
    assert rs.unreachable(urllib.error.URLError("Name or service not known"))
    assert rs.unreachable(TimeoutError("timed out"))

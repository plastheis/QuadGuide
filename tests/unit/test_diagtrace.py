import json
import os

import pytest

from quadguide.core.diagtrace import DiagTrace, resolve_trace_dir


class TestResolveTraceDir:
    def test_creates_timestamped_dir_under_base(self, tmp_path):
        d = resolve_trace_dir(str(tmp_path))
        assert os.path.isdir(d)
        assert str(tmp_path) in d
        assert os.path.basename(os.path.dirname(d)) == "trace"

    def test_falls_back_when_base_unwritable(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = resolve_trace_dir("/proc/cannot-create-here")
        assert os.path.isdir(d)
        assert "quadguide-trace" in d


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestDisabled:
    def test_disabled_is_noop_and_writes_nothing(self, tmp_path):
        t = DiagTrace("tracker", enabled=False, dir=str(tmp_path))
        t.latency(100, 90, 90)
        t.state(101, armed=True)
        t.health(102, "ok")
        t.flush()
        assert len(t) == 0
        assert list(tmp_path.iterdir()) == []

    def test_enabled_but_no_dir_is_noop(self):
        t = DiagTrace("tracker", enabled=True, dir=None)
        t.latency(100, 90, 90)
        t.flush()
        assert t.enabled is False
        assert len(t) == 0


class TestEnabled:
    def test_records_and_flushes_jsonl(self, tmp_path):
        t = DiagTrace("guidance", enabled=True, dir=str(tmp_path))
        t.latency(now_ns=200, input_ts_ns=150, origin_ns=120)
        t.state(now_ns=201, method="pure_pursuit", publishing=True)
        t.health(now_ns=202, state="ok", detail="ANGLE")
        assert len(t) == 3
        t.flush()

        rows = _read_jsonl(tmp_path / "guidance.jsonl")
        assert rows[0] == {"t": 200, "p": "guidance", "k": "lat", "in": 150, "org": 120}
        assert rows[1] == {"t": 201, "p": "guidance", "k": "state",
                           "method": "pure_pursuit", "publishing": True}
        assert rows[2] == {"t": 202, "p": "guidance", "k": "health",
                           "state": "ok", "detail": "ANGLE"}

    def test_latency_allows_none_input(self, tmp_path):
        t = DiagTrace("control", enabled=True, dir=str(tmp_path))
        t.latency(now_ns=300, input_ts_ns=None, origin_ns=0)
        t.flush()
        rows = _read_jsonl(tmp_path / "control.jsonl")
        assert rows[0]["in"] is None
        assert rows[0]["org"] == 0

    def test_flush_is_idempotent(self, tmp_path):
        t = DiagTrace("link", enabled=True, dir=str(tmp_path))
        t.latency(1, 1, 1)
        t.flush()
        # second flush must not duplicate or raise even if more rows were added
        t.latency(2, 2, 2)
        t.flush()
        rows = _read_jsonl(tmp_path / "link.jsonl")
        assert len(rows) == 1

    def test_max_rows_caps_with_ring(self, tmp_path):
        t = DiagTrace("tracker", enabled=True, dir=str(tmp_path), max_rows=3)
        for i in range(10):
            t.latency(i, i, i)
        assert len(t) == 3
        t.flush()
        rows = _read_jsonl(tmp_path / "tracker.jsonl")
        assert [r["t"] for r in rows] == [7, 8, 9]  # oldest dropped

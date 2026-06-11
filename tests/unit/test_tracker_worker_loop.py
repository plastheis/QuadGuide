"""Tests for TrackerWorker — no cv2, no real bus/SHM."""
import time

import numpy as np
import pytest

from quadguide.core.messages import (
    BoundingBox, LockOnCmd, TrackerEstimate, TrackerHealth,
)
from quadguide.perception.tracker_worker import TrackerWorker


class _StubBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []
        self.latest_map: dict[str, object] = {}

    def publish(self, topic, msg):
        self.published.append((topic, msg))

    def latest(self, topic):
        return self.latest_map.get(topic)

    def detach(self):
        pass


class _StubFrameBuffer:
    def __init__(self, frame=None, ts: int = 0) -> None:
        self.frame = frame
        self.ts = ts

    def read_latest(self):
        return self.frame, self.ts


class _StubBBox:
    def __init__(self, x, y, w, h):
        self.x = x; self.y = y; self.w = w; self.h = h


class _StubTrackerOutput:
    def __init__(self, x, y, w, h, confidence, health):
        self.bbox = _StubBBox(x, y, w, h)
        self.confidence = confidence
        self.health = health


class _StubTracker:
    def __init__(self, output: _StubTrackerOutput | None = None) -> None:
        self._output = output or _StubTrackerOutput(0.1, 0.2, 0.3, 0.4, 0.9, "nominal")
        self.init_calls: list = []
        self.update_calls: int = 0
        self.reset_calls: int = 0
        self.close_calls: int = 0

    def name(self): return "stub"
    def init(self, frame, bbox): self.init_calls.append((frame, bbox))
    def update(self, frame):
        self.update_calls += 1
        return self._output
    def reset(self): self.reset_calls += 1
    def close(self): self.close_calls += 1


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _run_one_iteration(worker: TrackerWorker) -> None:
    original_publish = worker._bus.publish
    def _publish_then_stop(topic, msg):
        original_publish(topic, msg)
        worker._stop = True
    worker._bus.publish = _publish_then_stop
    worker.run()


class TestTrackerWorkerLockonFlow:
    def test_new_seq_triggers_init(self):
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=_frame(), ts=time.monotonic_ns())
        tracker = _StubTracker()
        bus.latest_map["lockon/cmd"] = LockOnCmd(
            timestamp_ns=time.monotonic_ns(), seq=7,
            bbox=BoundingBox(0.1, 0.1, 0.2, 0.2),
        )
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        worker._check_lockon()
        assert len(tracker.init_calls) == 1
        assert worker._last_seq == 7

    def test_same_seq_does_not_reinit(self):
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=_frame(), ts=time.monotonic_ns())
        tracker = _StubTracker()
        bus.latest_map["lockon/cmd"] = LockOnCmd(
            time.monotonic_ns(), 3, BoundingBox(0.1, 0.1, 0.2, 0.2),
        )
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        worker._check_lockon()
        worker._check_lockon()
        assert len(tracker.init_calls) == 1

    def test_zero_size_bbox_triggers_reset(self):
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=_frame(), ts=time.monotonic_ns())
        tracker = _StubTracker()
        bus.latest_map["lockon/cmd"] = LockOnCmd(
            timestamp_ns=time.monotonic_ns(), seq=1,
            bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
        )
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        worker._check_lockon()
        assert tracker.reset_calls == 1
        assert len(tracker.init_calls) == 0


class TestTrackerWorkerPublish:
    def test_publish_translates_output_to_tracker_estimate(self):
        bus = _StubBus()
        ts = time.monotonic_ns() - 1_000_000
        fb = _StubFrameBuffer(frame=_frame(), ts=ts)
        tracker = _StubTracker(_StubTrackerOutput(0.1, 0.2, 0.3, 0.4, 0.75, "nominal"))
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        _run_one_iteration(worker)
        published = [(t, m) for (t, m) in bus.published if t == "target/estimate"]
        assert len(published) >= 1
        _, est = published[0]
        assert isinstance(est, TrackerEstimate)
        assert est.bbox.x == pytest.approx(0.1)
        assert est.bbox.y == pytest.approx(0.2)
        assert est.bbox.w == pytest.approx(0.3)
        assert est.bbox.h == pytest.approx(0.4)
        assert est.confidence == pytest.approx(0.75)
        assert est.tracker_health == TrackerHealth.NOMINAL
        assert est.origin_ns > 0  # = frame_ts (capture timestamp lineage)


def _run_n_loops(worker: TrackerWorker, n: int) -> None:
    """Drive the loop for n passes using _check_lockon (called once per pass) as a
    counter, then stop. Works with the new-frame gate (which sleeps + continues)."""
    calls = {"n": 0}
    original = worker._check_lockon
    def _counting():
        calls["n"] += 1
        if calls["n"] >= n:
            worker._stop = True
        original()
    worker._check_lockon = _counting
    worker.run()


class TestNewFrameGate:
    def test_repeated_frame_processed_once(self):
        # Constant frame_ts → only the first pass should run the tracker; the rest
        # must skip (no reprocessing of a stale frame).
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=_frame(), ts=12_345)
        tracker = _StubTracker()
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        _run_n_loops(worker, 5)
        assert tracker.update_calls == 1
        assert len([m for t, m in bus.published if t == "target/estimate"]) == 1

    def test_new_frame_each_pass_is_processed(self):
        # A fresh frame_ts every read → every pass processes.
        class _IncrementingFB:
            def __init__(self): self.ts = 1_000
            def read_latest(self):
                self.ts += 1_000
                return _frame(), self.ts
        bus = _StubBus()
        tracker = _StubTracker()
        worker = TrackerWorker(tracker, bus, _IncrementingFB(), cpu_core=None, config={})
        _run_n_loops(worker, 5)
        assert tracker.update_calls == 5


class TestTrackerWorkerLifecycle:
    def test_close_called_after_loop_exits(self):
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=_frame(), ts=time.monotonic_ns())
        tracker = _StubTracker()
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        _run_one_iteration(worker)
        assert tracker.close_calls == 1

    def test_proc_name_uses_tracker_name(self):
        bus = _StubBus()
        fb = _StubFrameBuffer()
        tracker = _StubTracker()
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        assert worker._proc_name == "tracker_stub"

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

    def test_tracker_origin_ns_preferred_over_frame_ts(self):
        # An async tracker (AcquireTrack via the adapter) supplies its own
        # source-frame origin_ns; the worker must forward it, not frame_ts.
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=_frame(), ts=999_000)
        out = _StubTrackerOutput(0.1, 0.2, 0.3, 0.4, 0.9, "nominal")
        out.origin_ns = 555_000  # tracker-provided lineage
        tracker = _StubTracker(out)
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        _run_one_iteration(worker)
        _, est = [(t, m) for t, m in bus.published if t == "target/estimate"][0]
        assert est.origin_ns == 555_000


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


class TestDropLockOnLost:
    """drop_lock_on_lost: the first sustained LOST releases the lock for good.

    Bare single-object trackers re-lock onto background and report high
    confidence on the wrong box, so per-frame confidence cannot detect loss.
    Latching the drop means a loss only has to be caught once.
    """

    _CFG = {"tracker": {"drop_lock_on_lost": True, "drop_lock_frames": 2}}

    def _worker(self, health, cfg=None):
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=_frame(), ts=time.monotonic_ns())
        tracker = _StubTracker(_StubTrackerOutput(0.1, 0.2, 0.3, 0.4, 0.1, health))
        w = TrackerWorker(tracker, bus, fb, cpu_core=None,
                          config=cfg if cfg is not None else self._CFG)
        return w, tracker, bus

    def test_disabled_by_default(self):
        w, _, _ = self._worker("lost", cfg={})
        assert w._drop_on_lost is False

    def test_drops_after_configured_consecutive_lost_frames(self):
        w, tracker, _ = self._worker("lost")
        # simulate the loop's drop logic over 2 LOST frames
        for _ in range(2):
            assert not w._dropped
            out = tracker.update(None)
            if str(out.health) == "lost":
                w._lost_run += 1
                if w._lost_run >= w._drop_after:
                    tracker.reset()
                    w._dropped = True
        assert w._dropped is True
        assert tracker.reset_calls == 1

    def test_single_lost_frame_does_not_drop(self):
        w, tracker, _ = self._worker("lost")
        out = tracker.update(None)
        if str(out.health) == "lost":
            w._lost_run += 1
            if w._lost_run >= w._drop_after:
                w._dropped = True
        assert w._dropped is False

    def test_dropped_state_reports_lost_not_no_lock(self):
        """The failsafe latch trips on LOST — reporting no_lock would disarm it."""
        from quadguide.perception.tracker_worker import _DROPPED
        assert _DROPPED.health == "lost"
        assert TrackerHealth(_DROPPED.health) is TrackerHealth.LOST
        assert (_DROPPED.bbox.w, _DROPPED.bbox.h) == (0.0, 0.0)

    def test_relock_clears_the_drop_latch(self):
        w, tracker, bus = self._worker("lost")
        w._dropped = True
        w._lost_run = 5
        bus.latest_map["lockon/cmd"] = LockOnCmd(
            timestamp_ns=time.monotonic_ns(), seq=42,
            bbox=BoundingBox(0.1, 0.1, 0.2, 0.2),
        )
        w._check_lockon()
        assert w._dropped is False
        assert w._lost_run == 0
        assert len(tracker.init_calls) == 1

    def test_operator_reset_also_clears_the_drop_latch(self):
        w, tracker, bus = self._worker("lost")
        w._dropped = True
        bus.latest_map["lockon/cmd"] = LockOnCmd(
            timestamp_ns=time.monotonic_ns(), seq=9,
            bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
        )
        w._check_lockon()
        assert w._dropped is False
        assert tracker.reset_calls == 1

    def test_loop_stops_updating_tracker_once_dropped(self):
        """The whole point: a dropped tracker must not keep running, because
        updating is exactly what lets it re-lock onto background clutter."""
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=_frame(), ts=1)
        tracker = _StubTracker(_StubTrackerOutput(0.1, 0.2, 0.3, 0.4, 0.1, "lost"))
        w = TrackerWorker(tracker, bus, fb, cpu_core=None, config=self._CFG)

        # Advance the frame timestamp each publish so the new-frame gate passes,
        # and stop after 6 estimates.
        n = {"c": 0}
        orig = bus.publish
        def _pub(topic, msg):
            orig(topic, msg)
            if topic == "target/estimate":
                n["c"] += 1
                fb.ts += 1
                if n["c"] >= 6:
                    w._stop = True
        bus.publish = _pub
        w.run()

        ests = [m for t, m in bus.published if t == "target/estimate"]
        assert len(ests) == 6
        # drop_lock_frames=2 → update() called twice, then never again
        assert tracker.update_calls == 2, tracker.update_calls
        assert tracker.reset_calls == 1
        assert w._dropped is True
        # every estimate reports LOST, so the failsafe latch keeps its trip
        assert all(e.tracker_health is TrackerHealth.LOST for e in ests)
        assert all(e.bbox.w == 0.0 for e in ests[2:])

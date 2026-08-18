"""Tests for VerifiedAcquireTrack — concurrent YOLO verification during LOCKED.

Spec: docs/superpowers/specs/2026-06-16-acquire-track-lock-verification-design.md

Reuses the AcquireTrack co-located harness (spawn_workers=False; real state
machine + real worker step() driven by fakes, no NPU). Verification runs YOLO
during LOCKED (Mode.BOTH), so the harness's YOLO worker now also fires while
locked, fed by the FakeYoloDetector script.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.runtime.shm.control_channel import Mode

from tests.edgecv._acquire_stubs import FakeNano, FakeYoloDetector, make_frame

_A = BoundingBox(0.5, 0.5, 0.1, 0.1)   # locked target region
_B = BoundingBox(0.1, 0.1, 0.1, 0.1)   # drifted-away region (no overlap with A)


def _build(**kw):
    from edgecv.trackers.hybrid.verified_acquire_track import (
        State,
        VerifiedAcquireTrack,
    )

    defaults = dict(
        acquire_crop=0.5, lock_pad=1.0, lock_min_score=0.3,
        drop_score=0.35, drop_frames=3,
        lost_timeout_frames=6, search_timeout_frames=5,
        verify=True, verify_min_iou=0.2, verify_min_score=0.3, verify_miss_frames=3,
        max_h=48, max_w=64, max_c=3, frame_slots=4,
    )
    defaults.update(kw)
    return VerifiedAcquireTrack(spawn_workers=False, **defaults), State


class MovingNano:
    """NanoTrack fake whose box follows a scripted list (cycling on the last),
    always high-confidence — to exercise drift independent of confidence."""

    def __init__(self, boxes, conf=0.95):
        self._boxes = list(boxes)
        self._conf = conf
        self.calls = 0
        self.inits: list[BoundingBox] = []
        self.closed = False

    def init(self, frame, bbox):
        self.inits.append(bbox)
        self.calls = 0

    def update(self, frame):
        b = self._boxes[min(self.calls, len(self._boxes) - 1)]
        self.calls += 1
        return TrackResult(bbox=b, confidence=self._conf,
                           status=TrackStatus.LOCKED, timestamp=time.monotonic(),
                           seq=self.calls)

    def close(self):
        self.closed = True


class _Harness:
    def __init__(self, tracker, det, nano):
        from edgecv.trackers.hybrid.acquire_workers import NanoWorker, YoloWorker

        self.t = tracker
        self.yolo_worker = YoloWorker(det, tracker._frame_ring, tracker._control,
                                      tracker._yolo_result)
        self.nano_worker = NanoWorker(nano, tracker._frame_ring, tracker._control,
                                      tracker._nano_result)

    def tick(self, frame=None):
        self.yolo_worker.step()
        self.nano_worker.step()
        return self.t.update(frame if frame is not None else make_frame(48, 64))

    def lock(self, bbox=_A):
        self.tick(); self.tick()          # acquire a candidate
        self.t.init(make_frame(48, 64), bbox)


# ── unit: geometry / support ────────────────────────────────────────────────

class TestSupport:
    def test_max_iou(self):
        from edgecv.trackers.hybrid.verified_acquire_track import VerifiedAcquireTrack
        boxes = np.array([[0.5, 0.5, 0.1, 0.1], [0.0, 0.0, 0.05, 0.05]], np.float32)
        assert VerifiedAcquireTrack._max_iou(_A, boxes) == pytest.approx(1.0)
        assert VerifiedAcquireTrack._max_iou(_B, boxes) == pytest.approx(0.0)
        assert VerifiedAcquireTrack._max_iou(_A, np.empty((0, 4), np.float32)) == 0.0

    def test_supported(self):
        t, _ = _build()
        try:
            sc_hi = np.array([0.9], np.float32)
            assert t._supported(_A, np.array([[0.5, 0.5, 0.1, 0.1]], np.float32), sc_hi)
            # overlapping but below verify_min_score → unsupported
            assert not t._supported(_A, np.array([[0.5, 0.5, 0.1, 0.1]], np.float32),
                                    np.array([0.1], np.float32))
            # detection far away → unsupported
            assert not t._supported(_A, np.array([[0.0, 0.0, 0.05, 0.05]], np.float32), sc_hi)
            # no detections → unsupported
            assert not t._supported(_A, np.empty((0, 4), np.float32),
                                    np.empty((0,), np.float32))
        finally:
            t.close()


# ── control mode ────────────────────────────────────────────────────────────

class TestControlMode:
    def test_locked_publishes_both_when_verify(self):
        t, State = _build(verify=True)
        try:
            t._state = State.LOCKED
            t._publish_control()
            assert t._control.read_latest().mode == Mode.BOTH
        finally:
            t.close()

    def test_locked_publishes_nano_when_not_verify(self):
        t, State = _build(verify=False)
        try:
            t._state = State.LOCKED
            t._publish_control()
            assert t._control.read_latest().mode == Mode.NANO
        finally:
            t.close()


# ── integration: drift detection ─────────────────────────────────────────────

class TestVerification:
    def test_supported_stays_locked(self):
        # YOLO keeps detecting the target where NanoTrack reports it → no drift.
        t, State = _build()
        det = FakeYoloDetector(script=[(np.array([[0.5, 0.5, 0.1, 0.1]], np.float32),
                                        np.array([0.9], np.float32))])
        nano = FakeNano(scores=[0.95])     # returns the locked box (A) each update
        h = _Harness(t, det, nano)
        try:
            h.lock(_A)
            assert t._state == State.LOCKED
            for _ in range(10):
                h.tick()
            assert t._state == State.LOCKED
            assert t._verify_miss == 0
        finally:
            t.close()

    def test_unsupported_triggers_reacq_anchored_on_last_good(self):
        # NanoTrack drifts A→B while YOLO only ever sees A. Drift must fire and
        # re-acquire around A (last verified-good), not B (the drift).
        t, State = _build(verify_miss_frames=3)
        det = FakeYoloDetector(script=[(np.array([[0.5, 0.5, 0.1, 0.1]], np.float32),
                                        np.array([0.9], np.float32))])
        nano = MovingNano(boxes=[_A, _A, _B, _B, _B, _B, _B, _B])
        h = _Harness(t, det, nano)
        try:
            h.lock(_A)
            assert t._state == State.LOCKED
            for _ in range(12):
                h.tick()
                if t._state != State.LOCKED:
                    break
            assert t._state == State.REACQ
            assert t._last_bbox.x == pytest.approx(_A.x)   # anchored on A, not B
            assert t._last_bbox.y == pytest.approx(_A.y)
        finally:
            t.close()

    def test_no_detections_triggers_drift(self):
        # Confident NanoTrack box but YOLO sees nothing (occlusion/drift) → re-acq.
        t, State = _build(verify_miss_frames=3)
        det = FakeYoloDetector()           # always empty
        nano = FakeNano(scores=[0.95])
        h = _Harness(t, det, nano)
        try:
            h.lock(_A)
            assert t._state == State.LOCKED
            for _ in range(12):
                h.tick()
                if t._state != State.LOCKED:
                    break
            assert t._state == State.REACQ
        finally:
            t.close()

    def test_verify_false_never_drifts(self):
        # verify=False → legacy behaviour: unsupported box stays LOCKED forever.
        t, State = _build(verify=False)
        det = FakeYoloDetector()           # empty → would be unsupported if checked
        nano = FakeNano(scores=[0.95])
        h = _Harness(t, det, nano)
        try:
            h.lock(_A)
            for _ in range(12):
                h.tick()
            assert t._state == State.LOCKED
        finally:
            t.close()

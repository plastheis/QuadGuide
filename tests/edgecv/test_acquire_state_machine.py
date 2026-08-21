"""State-machine tests for AcquireTrack (Tasks 6-8).

Runs the REAL state machine and the REAL worker step() logic co-located in one
process (no spawning): the tracker is built with spawn_workers=False, and the
test drives the YOLO/NanoTrack workers against the tracker's own channels with
the FakeYoloDetector/FakeNano stubs. This models the async parent↔worker timing
(a worker result read in update() reflects the previous tick) without an NPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.runtime.shm.control_channel import Mode
from tests.edgecv._acquire_stubs import FakeNano, FakeYoloDetector, make_frame


def _build(**kw):
    from edgecv.trackers.hybrid.acquire_track import AcquireTrack, State

    defaults = dict(
        acquire_crop=0.5, lock_pad=1.0, lock_min_score=0.3,
        drop_score=0.35, drop_frames=3,
        lost_timeout_frames=6, search_timeout_frames=5,
        max_h=48, max_w=64, max_c=3, frame_slots=4,
    )
    defaults.update(kw)
    return AcquireTrack(spawn_workers=False, **defaults), State


class _Harness:
    """Couples the tracker to in-process workers driven by stubs."""

    def __init__(self, tracker, det: FakeYoloDetector, nano: FakeNano):
        from edgecv.trackers.hybrid.acquire_workers import NanoWorker, YoloWorker

        self.t = tracker
        self.det = det
        self.nano = nano
        self.yolo_worker = YoloWorker(det, tracker._frame_ring, tracker._control,
                                      tracker._yolo_result)
        self.nano_worker = NanoWorker(nano, tracker._frame_ring, tracker._control,
                                      tracker._nano_result)

    def tick(self, frame=None):
        """One full cycle: workers run on the prior control, then the parent."""
        # Workers act on whatever control the parent last published.
        self.yolo_worker.step()
        self.nano_worker.step()
        return self.t.update(frame if frame is not None else make_frame(48, 64))


class TestAcquireSkeleton:
    def test_starts_in_acquire_with_central_crop(self):
        t, State = _build()
        try:
            assert t.name() == "AcquireTrack"
            assert t._state == State.ACQUIRE
            snap = t._control.read_latest()
            assert snap.mode == Mode.YOLO
            # central square crop: side 0.5*min(W,H)=0.5*48=24px → w=24/64, h=24/48
            assert snap.crop.w == pytest.approx(24 / 64)
            assert snap.crop.h == pytest.approx(24 / 48)
            assert snap.crop.x == pytest.approx((1 - 24 / 64) / 2)
        finally:
            t.close()

    def test_context_manager_closes_clean(self):
        t, _ = _build()
        with t:
            pass  # __exit__ → close(); must not raise BufferError

    def test_acquire_reports_best_candidate(self):
        t, State = _build()
        det = FakeYoloDetector(script=[
            (np.array([[0.45, 0.45, 0.08, 0.08], [0.5, 0.5, 0.08, 0.08]], np.float32),
             np.array([0.4, 0.9], np.float32))])
        h = _Harness(t, det, FakeNano())
        try:
            r0 = h.tick()                  # no result yet
            assert r0.status == TrackStatus.INITIALIZING
            r1 = h.tick()                  # yolo result now available
            assert r1.status == TrackStatus.INITIALIZING
            assert r1.bbox is not None
            # best (score 0.9) is the second box, centred ~ (0.54,0.54)
            assert r1.confidence == pytest.approx(0.9)
            assert r1.bbox.x == pytest.approx(0.5)
        finally:
            t.close()

    def test_acquire_none_when_no_detection(self):
        t, _ = _build()
        h = _Harness(t, FakeYoloDetector(), FakeNano())
        try:
            h.tick()
            r = h.tick()
            assert r.bbox is None
        finally:
            t.close()


class TestLockAndDrop:
    def test_init_locks_best_candidate(self):
        t, State = _build()
        det = FakeYoloDetector(script=[
            (np.array([[0.5, 0.5, 0.1, 0.1]], np.float32), np.array([0.9], np.float32))])
        h = _Harness(t, det, FakeNano(scores=[0.9]))
        try:
            h.tick()
            h.tick()  # acquire a candidate
            t.init(make_frame(48, 64), BoundingBox(0.25, 0.25, 0.5, 0.5))
            assert t._state == State.LOCKED
            snap = t._control.read_latest()
            assert snap.mode == Mode.NANO
            assert snap.lock_gen == 1
            assert snap.lock_bbox.x == pytest.approx(0.5)  # candidate, not the crop box
        finally:
            t.close()

    def test_init_seeds_from_crop_when_no_candidate(self):
        t, State = _build()
        h = _Harness(t, FakeYoloDetector(), FakeNano())  # never detects
        try:
            h.tick()
            h.tick()
            crop = BoundingBox(0.3, 0.3, 0.2, 0.2)
            t.init(make_frame(48, 64), crop)
            assert t._state == State.LOCKED
            snap = t._control.read_latest()
            assert snap.lock_bbox.x == pytest.approx(0.3)  # seeded from crop box
        finally:
            t.close()

    def test_zero_size_init_resets_to_acquire(self):
        t, State = _build()
        h = _Harness(t, FakeYoloDetector(
            script=[(np.array([[0.5, 0.5, 0.1, 0.1]], np.float32),
                     np.array([0.9], np.float32))]), FakeNano())
        try:
            h.tick()
            h.tick()
            t.init(make_frame(48, 64), BoundingBox(0.5, 0.5, 0.1, 0.1))
            assert t._state == State.LOCKED
            t.init(make_frame(48, 64), BoundingBox(0, 0, 0, 0))  # zero → reset
            assert t._state == State.ACQUIRE
        finally:
            t.close()

    def test_locked_to_reacq_after_drop_frames(self):
        t, State = _build(drop_frames=3)
        det = FakeYoloDetector()  # empty: won't re-lock during this test
        nano = FakeNano(scores=[0.9, 0.2, 0.2, 0.2, 0.2])  # locks then drops
        h = _Harness(t, det, nano)
        try:
            h.tick()
            h.tick()
            t.init(make_frame(48, 64), BoundingBox(0.5, 0.5, 0.1, 0.1))
            # one locked update
            h.tick()
            assert t._state == State.LOCKED
            # three sub-threshold updates → drop
            states = []
            for _ in range(4):
                h.tick()
                states.append(t._state)
            assert State.REACQ in states
            assert t._state == State.REACQ
        finally:
            t.close()


class TestReacquire:
    def _locked_then_dropped(self):
        t, State = _build(drop_frames=2,
                          lost_timeout_frames=5, search_timeout_frames=4)
        det = FakeYoloDetector(script=[
            (np.array([[0.5, 0.5, 0.1, 0.1]], np.float32), np.array([0.0], np.float32))])
        nano = FakeNano(scores=[0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        h = _Harness(t, det, nano)
        h.tick()
        h.tick()
        t.init(make_frame(48, 64), BoundingBox(0.5, 0.5, 0.1, 0.1))
        h.tick()
        for _ in range(3):       # drop into REACQ
            h.tick()
        assert t._state == State.REACQ
        return t, State, det, nano, h

    def test_reacq_holds_last_bbox_and_coasts(self):
        t, State, det, nano, h = self._locked_then_dropped()
        try:
            r = h.tick()
            assert r.status == TrackStatus.COASTING
            assert r.bbox.x == pytest.approx(0.5)   # held last-known
        finally:
            t.close()

    def test_reacq_searches_full_frame(self):
        t, State, det, nano, h = self._locked_then_dropped()
        try:
            # REACQ runs YOLO on the full frame immediately (no crop escalation).
            snap = t._control.read_latest()
            assert snap.mode == Mode.YOLO
            assert snap.crop.w == pytest.approx(1.0)  # full frame
            assert snap.crop.h == pytest.approx(1.0)
        finally:
            t.close()

    def test_relock_on_confident_detection(self):
        t, State = _build(drop_frames=2)
        # detector always finds a confident target → re-locks during re-acq
        det = FakeYoloDetector(script=[
            (np.array([[0.5, 0.5, 0.1, 0.1]], np.float32), np.array([0.9], np.float32))])
        nano = FakeNano(scores=[0.9, 0.1, 0.1, 0.1])  # lock then drop
        h = _Harness(t, det, nano)
        try:
            h.tick()
            h.tick()
            t.init(make_frame(48, 64), BoundingBox(0.5, 0.5, 0.1, 0.1))
            seen = set()
            final_gen = None
            for _ in range(10):
                h.tick()
                seen.add(t._state)
                snap = t._control.read_latest()
                if t._state == State.LOCKED and snap.lock_gen >= 2:
                    final_gen = snap.lock_gen
                    break
            assert State.REACQ in seen               # passed through re-acq
            assert final_gen is not None and final_gen >= 2  # re-locked
        finally:
            t.close()

    def test_reports_lost_then_resets_after_search_timeout(self):
        t, State, det, nano, h = self._locked_then_dropped()
        try:
            saw_lost = False
            for _ in range(30):
                r = h.tick()
                if r.status == TrackStatus.LOST:
                    saw_lost = True
                if t._state == State.ACQUIRE:
                    break
            assert saw_lost
            assert t._state == State.ACQUIRE  # search timeout → reset
        finally:
            t.close()


class TestMutualExclusion:
    def test_exactly_one_worker_active_each_state(self):
        t, State = _build(drop_frames=2)
        det = FakeYoloDetector()  # empty: dwell in REACQ (no re-lock) to observe mode
        nano = FakeNano(scores=[0.9, 0.1, 0.1, 0.1, 0.1])
        h = _Harness(t, det, nano)
        try:
            for _ in range(2):
                h.tick()
                assert t._control.read_latest().mode == Mode.YOLO  # ACQUIRE
            t.init(make_frame(48, 64), BoundingBox(0.5, 0.5, 0.1, 0.1))
            assert t._control.read_latest().mode == Mode.NANO       # LOCKED
            for _ in range(4):
                h.tick()
            assert t._state == State.REACQ
            assert t._control.read_latest().mode == Mode.YOLO       # REACQ
        finally:
            t.close()

"""Worker-body tests for AcquireTrack (Tasks 4-5).

Exercises the single-iteration step logic of the YOLO and NanoTrack workers
against in-process SHM channels and light stubs — no processes, no NPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.runtime.shm.control_channel import Mode
from tests.edgecv._acquire_stubs import (
    FakeNano,
    FakeYoloDetector,
    close_channels,
    make_channels,
    make_frame,
)

# ── Task 4: YOLO worker ─────────────────────────────────────────────────────

class TestYoloWorker:
    def test_detects_on_crop_and_publishes_when_mode_yolo(self):
        from edgecv.trackers.hybrid.acquire_workers import YoloWorker

        fr, ctrl, yolo, nano = make_channels()
        try:
            boxes = np.array([[0.4, 0.4, 0.1, 0.2]], np.float32)
            scores = np.array([0.8], np.float32)
            det = FakeYoloDetector(script=[(boxes, scores)])
            worker = YoloWorker(det, fr, ctrl, yolo)

            crop = BoundingBox(0.25, 0.25, 0.5, 0.5)
            ctrl.publish(mode=Mode.YOLO, crop=crop, lock_gen=0,
                         lock_bbox=BoundingBox(0, 0, 0, 0))
            fr.publish(make_frame(), seq=1, timestamp=11.0)

            assert worker.step() is True
            assert det.rois[-1].x == pytest.approx(0.25)  # cropped to control roi

            got = yolo.try_read()
            assert got is not None
            seq, arrays = got
            assert seq == 1
            assert arrays["boxes"].shape == (1, 4)
            assert arrays["scores"][0] == pytest.approx(0.8)
            assert arrays["src_ts"][0] == pytest.approx(11.0)
        finally:
            close_channels(fr, ctrl, yolo, nano)

    def test_idle_when_mode_not_yolo(self):
        from edgecv.trackers.hybrid.acquire_workers import YoloWorker

        fr, ctrl, yolo, nano = make_channels()
        try:
            det = FakeYoloDetector()
            worker = YoloWorker(det, fr, ctrl, yolo)
            ctrl.publish(mode=Mode.NANO, crop=BoundingBox(0, 0, 1, 1), lock_gen=1,
                         lock_bbox=BoundingBox(0.1, 0.1, 0.1, 0.1))
            fr.publish(make_frame(), seq=1, timestamp=1.0)

            assert worker.step() is False
            assert det.calls == 0
            assert yolo.try_read() is None  # nothing published
        finally:
            close_channels(fr, ctrl, yolo, nano)

    def test_skips_already_seen_frame(self):
        from edgecv.trackers.hybrid.acquire_workers import YoloWorker

        fr, ctrl, yolo, nano = make_channels()
        try:
            det = FakeYoloDetector(script=[(np.empty((0, 4), np.float32),
                                            np.empty((0,), np.float32))])
            worker = YoloWorker(det, fr, ctrl, yolo)
            ctrl.publish(mode=Mode.YOLO, crop=BoundingBox(0, 0, 1, 1), lock_gen=0,
                         lock_bbox=BoundingBox(0, 0, 0, 0))
            fr.publish(make_frame(), seq=5, timestamp=1.0)

            assert worker.step() is True
            assert worker.step() is False  # same seq → skip
            assert det.calls == 1
        finally:
            close_channels(fr, ctrl, yolo, nano)

    def test_publishes_empty_detection(self):
        """Zero detections still publishes a fresh seq (re-acq miss counting)."""
        from edgecv.trackers.hybrid.acquire_workers import YoloWorker

        fr, ctrl, yolo, nano = make_channels()
        try:
            det = FakeYoloDetector()  # default: empty
            worker = YoloWorker(det, fr, ctrl, yolo)
            ctrl.publish(mode=Mode.YOLO, crop=BoundingBox(0, 0, 1, 1), lock_gen=0,
                         lock_bbox=BoundingBox(0, 0, 0, 0))
            fr.publish(make_frame(), seq=2, timestamp=1.0)
            assert worker.step() is True
            seq, arrays = yolo.try_read()
            assert seq == 2
            assert arrays["boxes"].shape == (0, 4)
        finally:
            close_channels(fr, ctrl, yolo, nano)


# ── Task 5: NanoTrack worker ────────────────────────────────────────────────

class TestNanoWorker:
    def test_inits_on_new_lock_gen_then_updates(self):
        from edgecv.trackers.hybrid.acquire_workers import NanoWorker

        fr, ctrl, yolo, nano = make_channels()
        try:
            stub = FakeNano(scores=[0.9, 0.85])
            worker = NanoWorker(stub, fr, ctrl, nano)
            lockb = BoundingBox(0.4, 0.4, 0.1, 0.1)

            # New lock_gen → init + one update published
            ctrl.publish(mode=Mode.NANO, crop=BoundingBox(0, 0, 1, 1),
                         lock_gen=1, lock_bbox=lockb)
            fr.publish(make_frame(), seq=10, timestamp=5.0)
            assert worker.step() == "init"
            assert len(stub.inits) == 1
            assert stub.inits[0].x == pytest.approx(0.4)

            sample = nano.read_latest()
            assert sample is not None
            assert sample.status == TrackStatus.LOCKED
            assert sample.src_seq == 10
            assert sample.src_ts == pytest.approx(5.0)

            # Next frame → plain update, no re-init
            fr.publish(make_frame(), seq=11, timestamp=6.0)
            assert worker.step() == "update"
            assert len(stub.inits) == 1
            assert nano.read_latest().src_seq == 11
        finally:
            close_channels(fr, ctrl, yolo, nano)

    def test_reinits_on_lock_gen_bump(self):
        from edgecv.trackers.hybrid.acquire_workers import NanoWorker

        fr, ctrl, yolo, nano = make_channels()
        try:
            stub = FakeNano(scores=[0.9])
            worker = NanoWorker(stub, fr, ctrl, nano)
            ctrl.publish(mode=Mode.NANO, crop=BoundingBox(0, 0, 1, 1),
                         lock_gen=1, lock_bbox=BoundingBox(0.4, 0.4, 0.1, 0.1))
            fr.publish(make_frame(), seq=1, timestamp=1.0)
            worker.step()
            ctrl.publish(mode=Mode.NANO, crop=BoundingBox(0, 0, 1, 1),
                         lock_gen=2, lock_bbox=BoundingBox(0.6, 0.6, 0.1, 0.1))
            fr.publish(make_frame(), seq=2, timestamp=2.0)
            assert worker.step() == "init"
            assert len(stub.inits) == 2
            assert stub.inits[1].x == pytest.approx(0.6)
        finally:
            close_channels(fr, ctrl, yolo, nano)

    def test_idle_when_mode_not_nano(self):
        from edgecv.trackers.hybrid.acquire_workers import NanoWorker

        fr, ctrl, yolo, nano = make_channels()
        try:
            stub = FakeNano()
            worker = NanoWorker(stub, fr, ctrl, nano)
            ctrl.publish(mode=Mode.YOLO, crop=BoundingBox(0, 0, 1, 1), lock_gen=0,
                         lock_bbox=BoundingBox(0, 0, 0, 0))
            fr.publish(make_frame(), seq=1, timestamp=1.0)
            assert worker.step() == "idle"
            assert stub.inits == []
            assert nano.read_latest() is None
        finally:
            close_channels(fr, ctrl, yolo, nano)

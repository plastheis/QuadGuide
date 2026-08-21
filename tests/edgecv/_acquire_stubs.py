"""Stubs + helpers for AcquireTrack worker / state-machine tests (no NPU).

Light fakes that satisfy only the surfaces the AcquireTrack workers call:
- ``FakeYoloDetector.detect(frame, roi) -> DetectorOutput``
- ``FakeNano.init(frame, bbox)`` / ``FakeNano.update(frame) -> TrackResult``

The real YOLO crop/decode and NanoTrack decode are covered by test_yolo.py /
test_nanotrack.py; these isolate the worker-loop and state-machine logic.
"""

from __future__ import annotations

import time

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.fusion.policy import DetectorOutput
from edgecv.runtime.shm.control_channel import AcquireControlChannel
from edgecv.runtime.shm.frame_ring import FrameRing
from edgecv.runtime.shm.nano_result import NanoResultChannel
from edgecv.runtime.shm.payload import PayloadChannel


class FakeYoloDetector:
    """Returns scripted detections. `script` is a list of (boxes, scores); each
    detect() pops the next (cycling on the last entry). Records the rois seen."""

    def __init__(self, script=None):
        self._script = list(script) if script else [
            (np.empty((0, 4), np.float32), np.empty((0,), np.float32))
        ]
        self.calls = 0
        self.rois: list[BoundingBox] = []
        self.closed = False

    def detect(self, frame: np.ndarray, roi: BoundingBox) -> DetectorOutput:
        self.rois.append(roi)
        boxes, scores = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return DetectorOutput(boxes=np.asarray(boxes, np.float32),
                              scores=np.asarray(scores, np.float32),
                              meta={"search_roi": roi})

    def close(self) -> None:
        self.closed = True


class FakeNano:
    """Scripted NanoTrack. update() returns results from `scores` (cycling on the
    last), centred at the init bbox. Records init calls."""

    def __init__(self, scores=None):
        self._scores = list(scores) if scores else [0.9]
        self.inits: list[BoundingBox] = []
        self.calls = 0
        self._box = BoundingBox(0, 0, 0, 0)
        self.closed = False

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        self.inits.append(bbox)
        self._box = bbox
        self.calls = 0

    def update(self, frame: np.ndarray) -> TrackResult:
        conf = self._scores[min(self.calls, len(self._scores) - 1)]
        self.calls += 1
        status = (TrackStatus.LOCKED if conf >= 0.6
                  else TrackStatus.COASTING if conf >= 0.35
                  else TrackStatus.LOST)
        return TrackResult(bbox=self._box, confidence=conf, status=status,
                           timestamp=time.monotonic(), seq=self.calls)

    def close(self) -> None:
        self.closed = True


def make_channels(slots=4, h=48, w=64, c=3):
    """Create an in-process frame ring + control + yolo-result + nano-result set.

    Returns (frame_ring, control, yolo_result, nano_result) — all owner handles.
    Caller is responsible for close(unlink=True) on each.
    """
    fr = FrameRing.create(slots=slots, max_h=h, max_w=w, max_c=c, dtype="uint8")
    ctrl = AcquireControlChannel.create()
    yolo = PayloadChannel.create(capacity_bytes=64 * 1024, max_arrays=8)
    nano = NanoResultChannel.create()
    return fr, ctrl, yolo, nano


def close_channels(*chans) -> None:
    for ch in chans:
        ch.close(unlink=True)


def make_frame(h=48, w=64, c=3, fill=0):
    return np.full((h, w, c), fill, np.uint8)

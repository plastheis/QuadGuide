from __future__ import annotations
import time

import numpy as np

from quadguide.core.messages import BoundingBox, TrackerEstimate, TrackerHealth

__all__ = ["MOSSETracker"]


class MOSSETracker:
    """MOSSE tracker wrapping cv2.legacy.TrackerMOSSE.

    Requires opencv-contrib-python. Exposes no tunable parameters —
    OpenCV's MOSSE implementation has no public Params class.
    Confidence is always binary: 1.0 on success, 0.0 on failure.
    Health is therefore always NOMINAL or LOST, never UNCERTAIN.
    """

    def __init__(self) -> None:
        self._tracker     = None
        self._initialized = False

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        import cv2
        h, w = frame.shape[:2]
        self._tracker = cv2.legacy.TrackerMOSSE.create()
        bbox_px = (
            int(bbox.x * w),
            int(bbox.y * h),
            max(1, int(bbox.w * w)),
            max(1, int(bbox.h * h)),
        )
        self._tracker.init(frame, bbox_px)
        self._initialized = True

    def update(self, frame: np.ndarray) -> TrackerEstimate:
        if not self._initialized:
            return TrackerEstimate(
                timestamp_ns=time.monotonic_ns(),
                bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
                confidence=0.0,
                tracker_health=TrackerHealth.NO_LOCK,
            )
        h, w = frame.shape[:2]
        success, bbox_px = self._tracker.update(frame)
        if success:
            x, y, bw, bh = bbox_px
            return TrackerEstimate(
                timestamp_ns=time.monotonic_ns(),
                bbox=BoundingBox(
                    x=float(x) / w, y=float(y) / h,
                    w=float(bw) / w, h=float(bh) / h,
                ),
                confidence=1.0,
                tracker_health=TrackerHealth.NOMINAL,
            )
        return TrackerEstimate(
            timestamp_ns=time.monotonic_ns(),
            bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
            confidence=0.0,
            tracker_health=TrackerHealth.LOST,
        )

    def close(self) -> None:
        pass

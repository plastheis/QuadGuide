from __future__ import annotations
import time

import numpy as np

from quadguide.core.config import KCFConfig
from quadguide.core.messages import BoundingBox, TrackerEstimate, TrackerHealth

__all__ = ["KCFTracker"]


class KCFTracker:
    """KCF tracker wrapping cv2.TrackerKCF. No bus/IPC imports."""

    def __init__(self, config: KCFConfig) -> None:
        self._config      = config
        self._tracker     = None
        self._initialized = False
        self._frame_shape: tuple[int, int] | None = None  # (height, width)

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        """Initialise (or re-initialise) KCF on the given frame and bbox."""
        import cv2
        h, w = frame.shape[:2]
        self._frame_shape = (h, w)
        self._tracker     = self._build_cv_tracker()
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
            bbox_norm = BoundingBox(
                x=float(x) / w, y=float(y) / h,
                w=float(bw) / w, h=float(bh) / h,
            )
            return TrackerEstimate(
                timestamp_ns=time.monotonic_ns(),
                bbox=bbox_norm,
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

    def _build_cv_tracker(self):
        import cv2
        try:
            p = cv2.TrackerKCF.Params()
            p.sigma         = self._config.sigma
            p.lambda_       = self._config.lambda_
            p.detect_thresh = self._config.detect_thresh
            return cv2.TrackerKCF.create(p)
        except AttributeError:
            return cv2.TrackerKCF.create()

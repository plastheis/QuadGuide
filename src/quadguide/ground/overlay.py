from __future__ import annotations

import cv2
import numpy as np

from quadguide.core.messages import TargetEstimate, TrackerHealth

_JPEG_PARAMS     = [cv2.IMWRITE_JPEG_QUALITY, 80]
_COLOR_NOMINAL   = (0, 165, 255)   # orange BGR
_COLOR_UNCERTAIN = (0, 255, 255)   # yellow BGR


def draw_overlay(frame: np.ndarray, estimate: TargetEstimate | None) -> bytes:
    """Return frame encoded as JPEG, with tracking bbox drawn if tracker is active.

    Does not mutate the input frame. Returns a plain encode when estimate is
    None, NO_LOCK, or LOST — nothing is drawn in those states.
    """
    if estimate is None or estimate.tracker_health in (
        TrackerHealth.NO_LOCK, TrackerHealth.LOST
    ):
        return _encode(frame)

    h, w = frame.shape[:2]
    b = estimate.bbox
    x1 = int(b.x * w)
    y1 = int(b.y * h)
    x2 = int((b.x + b.w) * w)
    y2 = int((b.y + b.h) * h)
    color = (
        _COLOR_NOMINAL
        if estimate.tracker_health == TrackerHealth.NOMINAL
        else _COLOR_UNCERTAIN
    )
    out = frame.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return _encode(out)


def _encode(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, _JPEG_PARAMS)
    return buf.tobytes()

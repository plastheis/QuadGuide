from __future__ import annotations

import cv2
import numpy as np

from quadguide.core.messages import TrackerEstimate, TrackerHealth

_JPEG_PARAMS     = [cv2.IMWRITE_JPEG_QUALITY, 80]
_COLOR_NOMINAL   = (0, 165, 255)   # orange BGR
_COLOR_UNCERTAIN = (0, 255, 255)   # yellow BGR
_COLOR_ACQUIRING = (255, 255, 0)   # cyan BGR — pre-lock detection candidate

# Acquire-crop guideline: faint cyan square showing the central region YOLO scans
# before lock (AcquireTrack family only; see acquire_crop_from_config).
_COLOR_ACQUIRE_GUIDE = (255, 255, 0)   # cyan BGR, matches the candidate box
_ACQUIRE_GUIDE_ALPHA = 0.30            # blend weight — a low-opacity guideline
_ACQUIRE_TRACKERS    = ("acquire_track", "verified_acquire_track")
_DEFAULT_ACQUIRE_CROP = 0.5            # AcquireTrack's own default crop fraction

# Health states where nothing is drawn (no target / not tracking).
_NO_DRAW = (TrackerHealth.NO_LOCK, TrackerHealth.LOST)
_COLOR_BY_HEALTH = {
    TrackerHealth.NOMINAL:   _COLOR_NOMINAL,
    TrackerHealth.UNCERTAIN: _COLOR_UNCERTAIN,
    TrackerHealth.ACQUIRING: _COLOR_ACQUIRING,
}


def _percentile_stretch(frame: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    lo = float(np.percentile(frame, p_lo))
    hi = float(np.percentile(frame, p_hi))
    if hi <= lo:
        hi = lo + 1.0
    out = (frame.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def tonemap(
    frame: np.ndarray,
    mode: str = "percentile",
    p_lo: float = 1.0,
    p_hi: float = 99.5,
    gamma: float = 2.2,
) -> np.ndarray:
    """Reduce a mono uint16 (H,W) frame to a mono uint8 (H,W) for display.

    Display-only — never applied to the detector's raw feed.
      linear     : v >> 2 (naive, matches the old 10→8 truncation)
      percentile : stretch [p_lo, p_hi] then clip (robust to a flat bright sky)
      gamma      : percentile stretch, then a gamma curve for a photographic look
    """
    if mode == "linear":
        return (frame >> 2).astype(np.uint8)
    stretched = _percentile_stretch(frame, p_lo, p_hi)
    if mode == "gamma":
        lut = (((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255.0).astype(np.uint8)
        return lut[stretched]
    return stretched


def acquire_crop_from_config(config: dict | None) -> float | None:
    """Central acquire-crop side fraction to draw as a HUD guideline, or None.

    Only the EdgeCV AcquireTrack family scans a fixed central crop before lock, so
    the guideline is drawn for those trackers only. Mirrors AcquireTrack's
    ``_central_crop`` geometry: a centred square of side ``acquire_crop·min(w,h)``.
    """
    params = (config or {}).get("tracker", {}).get("params") or {}
    if params.get("tracker") not in _ACQUIRE_TRACKERS:
        return None
    return float(params.get("acquire_crop", _DEFAULT_ACQUIRE_CROP))


def draw_overlay(
    frame: np.ndarray,
    estimate: TrackerEstimate | None,
    acquire_crop: float | None = None,
    *,
    show_bbox: bool = True,
) -> bytes:
    """Return frame encoded as JPEG, with tracking bbox drawn if tracker is active.

    Does not mutate the input frame. The tracking box is drawn only when the
    tracker is active (not None/NO_LOCK/LOST); ACQUIRING draws the pre-lock
    candidate box in cyan (guidance ignores it; see guidance.worker). When
    ``acquire_crop`` is given (the AcquireTrack central-crop fraction), a faint
    cyan square marking that scan region is drawn underneath in every state.

    ``show_bbox=False`` (the operator's shared HUD toggle) suppresses both boxes,
    leaving a clean picture. Tracking itself is unaffected — this is draw-only.
    """
    out = None  # copy lazily — only when something is actually drawn

    if acquire_crop and show_bbox:
        out = frame.copy()
        _draw_acquire_guide(out, float(acquire_crop))

    if show_bbox and estimate is not None and estimate.tracker_health not in _NO_DRAW:
        if out is None:
            out = frame.copy()
        h, w = frame.shape[:2]
        b = estimate.bbox
        x1 = int(b.x * w)
        y1 = int(b.y * h)
        x2 = int((b.x + b.w) * w)
        y2 = int((b.y + b.h) * h)
        color = _COLOR_BY_HEALTH.get(estimate.tracker_health, _COLOR_UNCERTAIN)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    return _encode(out if out is not None else frame)


def _draw_acquire_guide(frame: np.ndarray, frac: float) -> None:
    """Blend a faint centred square (side = frac·min(w,h)) into ``frame`` in place."""
    h, w = frame.shape[:2]
    side = frac * min(h, w)
    x1 = int((w - side) / 2)
    y1 = int((h - side) / 2)
    x2 = int((w + side) / 2)
    y2 = int((h + side) / 2)
    layer = frame.copy()
    cv2.rectangle(layer, (x1, y1), (x2, y2), _COLOR_ACQUIRE_GUIDE, 1)
    cv2.addWeighted(layer, _ACQUIRE_GUIDE_ALPHA, frame, 1.0 - _ACQUIRE_GUIDE_ALPHA,
                    0.0, dst=frame)


def _encode(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", np.ascontiguousarray(frame), _JPEG_PARAMS)
    return buf.tobytes()

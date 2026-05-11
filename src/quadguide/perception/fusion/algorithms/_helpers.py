from __future__ import annotations

from quadguide.core.messages import ActiveTracker, BoundingBox, TargetEstimate, TrackerEstimate


def centroid(b: BoundingBox) -> tuple[float, float]:
    return (b.x + b.w / 2 - 0.5) * 2, (b.y + b.h / 2 - 0.5) * 2


def iou(a: BoundingBox, b: BoundingBox) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def passthrough_result(est: TrackerEstimate, label: ActiveTracker) -> TargetEstimate:
    return TargetEstimate(
        timestamp_ns=est.timestamp_ns,
        bbox=est.bbox,
        centroid_norm=centroid(est.bbox),
        confidence=est.confidence,
        tracker_health=est.tracker_health,
        active_tracker=label,
        latency_ns=est.latency_ns,
    )

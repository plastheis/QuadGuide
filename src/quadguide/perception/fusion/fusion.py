from __future__ import annotations
import time

from quadguide.core.messages import (
    ActiveTracker, BoundingBox, TargetEstimate, TrackerEstimate, TrackerHealth,
)

__all__ = ["fuse"]


def _centroid(b: BoundingBox) -> tuple[float, float]:
    return (b.x + b.w / 2 - 0.5) * 2, (b.y + b.h / 2 - 0.5) * 2


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _passthrough(est: TrackerEstimate, label: ActiveTracker) -> TargetEstimate:
    return TargetEstimate(
        timestamp_ns=est.timestamp_ns,
        bbox=est.bbox,
        centroid_norm=_centroid(est.bbox),
        confidence=est.confidence,
        tracker_health=est.tracker_health,
        active_tracker=label,
        latency_ns=est.latency_ns,
    )


def fuse(
    ccv: TrackerEstimate | None,
    ncv: TrackerEstimate | None,
    cfg,
) -> TargetEstimate | None:
    """Fuse CCV and NCV tracker estimates into a single TargetEstimate.

    Returns None before any estimate has arrived (startup transient).

    Passthrough: if only one tracker is publishing, its estimate is forwarded
    directly — no fusion arithmetic.  This means running with only a CCV tracker
    (dev bench, no NPU) or only an NCV tracker produces correct output without
    requiring the other slot to be present.

    Full fusion: when both trackers are publishing:
      1. Staleness-check NCV (drop if older than cfg.ncv_staleness_ms).
      2. If ncv confidence > cfg.confidence_gate → use ncv bbox.
      3. Otherwise → weighted-average bbox by confidence.
      4. IoU divergence check → UNCERTAIN health + confidence penalty.
    """
    now_ns = time.monotonic_ns()

    # drop NCV estimate if it has gone stale
    if ncv is not None:
        age_ms = (now_ns - ncv.timestamp_ns) / 1_000_000
        if age_ms > cfg.ncv_staleness_ms:
            ncv = None

    # passthrough: only one tracker is present
    if ccv is None and ncv is None:
        return None
    if ncv is None:
        return _passthrough(ccv, ActiveTracker.CCV)  # type: ignore[arg-type]
    if ccv is None:
        return _passthrough(ncv, ActiveTracker.NCV)

    # both present — propagate NO_LOCK if neither has initialised
    if (ccv.tracker_health == TrackerHealth.NO_LOCK
            and ncv.tracker_health == TrackerHealth.NO_LOCK):
        return _passthrough(ccv, ActiveTracker.CCV)

    # confidence gate: prefer nano when it is confident enough
    if ncv.confidence > cfg.confidence_gate:
        fused_bbox = ncv.bbox
        fused_conf = ncv.confidence
        active = ActiveTracker.NCV
        latency_ns = ncv.latency_ns
    else:
        total = ccv.confidence + ncv.confidence
        if total == 0.0:
            w_ccv, w_ncv = 0.5, 0.5
        else:
            w_ccv = ccv.confidence / total
            w_ncv = ncv.confidence / total
        fused_bbox = BoundingBox(
            x=ccv.bbox.x * w_ccv + ncv.bbox.x * w_ncv,
            y=ccv.bbox.y * w_ccv + ncv.bbox.y * w_ncv,
            w=ccv.bbox.w * w_ccv + ncv.bbox.w * w_ncv,
            h=ccv.bbox.h * w_ccv + ncv.bbox.h * w_ncv,
        )
        fused_conf = max(ccv.confidence, ncv.confidence)
        active = ActiveTracker.FUSED
        latency_ns = ccv.latency_ns  # CCV is the canonical latency reference in blended mode

    # IoU divergence: penalise confidence and flag health
    health = TrackerHealth.NOMINAL
    if _iou(ccv.bbox, ncv.bbox) < cfg.iou_divergence_thresh:
        health = TrackerHealth.UNCERTAIN
        fused_conf *= 0.5

    return TargetEstimate(
        timestamp_ns=now_ns,
        bbox=fused_bbox,
        centroid_norm=_centroid(fused_bbox),
        confidence=fused_conf,
        tracker_health=health,
        active_tracker=active,
        latency_ns=latency_ns,
    )

from __future__ import annotations
import time

from quadguide.core.messages import (
    ActiveTracker, BoundingBox, TargetEstimate, TrackerEstimate, TrackerHealth,
)

from ._helpers import centroid, iou, passthrough_result
from .base import BaseFusion


class ConfidenceWeightedFusion(BaseFusion):
    """Fuse CCV and NCV by confidence-weighted bbox blending.

    1. Drop NCV if older than cfg.ncv_staleness_ms.
    2. Passthrough whichever tracker is the only one present.
    3. If NCV confidence > cfg.confidence_gate → use NCV bbox exclusively.
    4. Otherwise blend bbox by confidence weights.
    5. IoU divergence < cfg.iou_divergence_thresh → UNCERTAIN + 50% confidence penalty.
    """

    def fuse(
        self,
        ccv: TrackerEstimate | None,
        ncv: TrackerEstimate | None,
        cfg,
    ) -> TargetEstimate | None:
        now_ns = time.monotonic_ns()

        if ncv is not None:
            age_ms = (now_ns - ncv.timestamp_ns) / 1_000_000
            if age_ms > cfg.ncv_staleness_ms:
                ncv = None

        if ccv is None and ncv is None:
            return None
        if ncv is None:
            return passthrough_result(ccv, ActiveTracker.CCV)  # type: ignore[arg-type]
        if ccv is None:
            return passthrough_result(ncv, ActiveTracker.NCV)

        if (ccv.tracker_health == TrackerHealth.NO_LOCK
                and ncv.tracker_health == TrackerHealth.NO_LOCK):
            return passthrough_result(ccv, ActiveTracker.CCV)

        if ncv.confidence > cfg.confidence_gate:
            fused_bbox = ncv.bbox
            fused_conf = ncv.confidence
            active = ActiveTracker.NCV
            latency_ns = ncv.latency_ns
        else:
            total = ccv.confidence + ncv.confidence
            w_ccv, w_ncv = (0.5, 0.5) if total == 0.0 else (ccv.confidence / total, ncv.confidence / total)
            fused_bbox = BoundingBox(
                x=ccv.bbox.x * w_ccv + ncv.bbox.x * w_ncv,
                y=ccv.bbox.y * w_ccv + ncv.bbox.y * w_ncv,
                w=ccv.bbox.w * w_ccv + ncv.bbox.w * w_ncv,
                h=ccv.bbox.h * w_ccv + ncv.bbox.h * w_ncv,
            )
            fused_conf = max(ccv.confidence, ncv.confidence)
            active = ActiveTracker.FUSED
            latency_ns = ccv.latency_ns

        health = TrackerHealth.NOMINAL
        if iou(ccv.bbox, ncv.bbox) < cfg.iou_divergence_thresh:
            health = TrackerHealth.UNCERTAIN
            fused_conf *= 0.5

        return TargetEstimate(
            timestamp_ns=now_ns,
            bbox=fused_bbox,
            centroid_norm=centroid(fused_bbox),
            confidence=fused_conf,
            tracker_health=health,
            active_tracker=active,
            latency_ns=latency_ns,
        )

from __future__ import annotations
import time

from quadguide.core.messages import (
    ActiveTracker, BoundingBox, TargetEstimate, TrackerEstimate, TrackerHealth,
)

from ._helpers import centroid, iou, passthrough_result
from .base import BaseFusion


class IoUGatedFusion(BaseFusion):
    """IoU-gated fusion with dead-reckoning correction for the slow tracker.

    Before IoU gating the slow tracker's bbox is age-corrected: the fast
    tracker's velocity (EMA over time) projects the slow bbox forward to
    the current timestamp so both are temporally aligned.

    Blend weights are determined by IoU between fast and age-corrected slow:
      iou > iou_thresh_high  → light blend toward slow (weight = 1 - thresh_high)
      iou > iou_thresh_low   → heavy blend toward slow (weight = 1 - thresh_low)
      iou == 0               → use slow bbox directly; confidence falls to slow's own score

    Which tracker is "fast" (sync, high-rate) vs "slow" (async, lower-rate) is
    controlled by cfg.fast_tracker ("ccv" | "ncv").
    """

    def __init__(self) -> None:
        self._v_cx: float = 0.0   # fast-tracker centroid velocity, normalized coords/s
        self._v_cy: float = 0.0
        self._prev_cx: float | None = None
        self._prev_cy: float | None = None
        self._prev_ts: int | None = None  # timestamp_ns of last fast estimate

    def _age_correct(self, bbox: BoundingBox, age_s: float) -> BoundingBox:
        return BoundingBox(
            x=bbox.x + age_s * self._v_cx,
            y=bbox.y + age_s * self._v_cy,
            w=bbox.w,
            h=bbox.h,
        )

    def _update_velocity(self, fast: TrackerEstimate, alpha: float) -> None:
        cx = fast.bbox.x + fast.bbox.w / 2
        cy = fast.bbox.y + fast.bbox.h / 2
        if self._prev_cx is not None and self._prev_ts is not None:
            dt = (fast.timestamp_ns - self._prev_ts) / 1e9
            if dt > 0:
                v_cx = (cx - self._prev_cx) / dt
                v_cy = (cy - self._prev_cy) / dt
                self._v_cx = alpha * v_cx + (1.0 - alpha) * self._v_cx
                self._v_cy = alpha * v_cy + (1.0 - alpha) * self._v_cy
        self._prev_cx = cx
        self._prev_cy = cy
        self._prev_ts = fast.timestamp_ns

    def fuse(
        self,
        ccv: TrackerEstimate | None,
        ncv: TrackerEstimate | None,
        cfg,
    ) -> TargetEstimate | None:
        fast_is_ccv = cfg.fast_tracker == "ccv"
        fast, slow = (ccv, ncv) if fast_is_ccv else (ncv, ccv)
        fast_label = ActiveTracker.CCV if fast_is_ccv else ActiveTracker.NCV
        slow_label = ActiveTracker.NCV if fast_is_ccv else ActiveTracker.CCV

        if fast is None and slow is None:
            return None
        if slow is None:
            if fast is not None:
                self._update_velocity(fast, cfg.iou_velocity_ema_alpha)
            return passthrough_result(fast, fast_label)  # type: ignore[arg-type]
        if fast is None:
            return passthrough_result(slow, slow_label)

        self._update_velocity(fast, cfg.iou_velocity_ema_alpha)

        now_ns = time.monotonic_ns()
        age_s = (now_ns - slow.timestamp_ns) / 1e9
        slow_corrected = self._age_correct(slow.bbox, age_s)
        overlap = iou(fast.bbox, slow_corrected)

        thresh_high = cfg.iou_thresh_high
        thresh_low  = cfg.iou_thresh_low

        def _blend(b1: BoundingBox, b2: BoundingBox, w: float) -> BoundingBox:
            """Return b1 + w*(b2 - b1)."""
            return BoundingBox(
                x=b1.x + w * (b2.x - b1.x),
                y=b1.y + w * (b2.y - b1.y),
                w=b1.w + w * (b2.w - b1.w),
                h=b1.h + w * (b2.h - b1.h),
            )

        if overlap > thresh_high:
            fused_bbox = _blend(fast.bbox, slow_corrected, 1.0 - thresh_high)
            active = ActiveTracker.FUSED
            conf = overlap
            latency_ns = fast.latency_ns
        elif overlap > thresh_low:
            fused_bbox = _blend(fast.bbox, slow_corrected, 1.0 - thresh_low)
            active = ActiveTracker.FUSED
            conf = overlap
            latency_ns = fast.latency_ns
        else:
            fused_bbox = slow_corrected
            active = slow_label
            conf = slow.confidence
            latency_ns = slow.latency_ns

        return TargetEstimate(
            timestamp_ns=now_ns,
            bbox=fused_bbox,
            centroid_norm=centroid(fused_bbox),
            confidence=conf,
            tracker_health=TrackerHealth.NOMINAL,
            active_tracker=active,
            latency_ns=latency_ns,
        )

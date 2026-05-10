import types
import time
import pytest

from quadguide.core.messages import (
    ActiveTracker, BoundingBox, TrackerEstimate, TrackerHealth,
)
from quadguide.perception.fusion.fusion import fuse

_CFG = types.SimpleNamespace(
    ncv_staleness_ms=100,
    confidence_gate=0.7,
    iou_divergence_thresh=0.3,
)

_BBOX = BoundingBox(0.2, 0.2, 0.3, 0.3)


def _ccv(health=TrackerHealth.NOMINAL, conf=0.5, latency_ns=1_000_000):
    return TrackerEstimate(
        timestamp_ns=time.monotonic_ns(),
        bbox=_BBOX,
        confidence=conf,
        tracker_health=health,
        latency_ns=latency_ns,
    )


def _ncv(health=TrackerHealth.NOMINAL, conf=0.8, latency_ns=3_000_000):
    return TrackerEstimate(
        timestamp_ns=time.monotonic_ns(),
        bbox=_BBOX,
        confidence=conf,
        tracker_health=health,
        latency_ns=latency_ns,
    )


class TestFuseLatencyPassthrough:
    def test_ccv_only_preserves_latency(self):
        result = fuse(_ccv(latency_ns=5_000_000), None, _CFG)
        assert result is not None
        assert result.latency_ns == 5_000_000

    def test_ncv_only_preserves_latency(self):
        result = fuse(None, _ncv(latency_ns=8_000_000), _CFG)
        assert result is not None
        assert result.latency_ns == 8_000_000

    def test_both_no_lock_uses_ccv_latency(self):
        ccv = _ccv(health=TrackerHealth.NO_LOCK, latency_ns=500_000)
        ncv = _ncv(health=TrackerHealth.NO_LOCK, latency_ns=1_000_000)
        result = fuse(ccv, ncv, _CFG)
        assert result is not None
        assert result.latency_ns == 500_000


class TestFuseLatencyFull:
    def test_ncv_active_uses_ncv_latency(self):
        # ncv.confidence (0.9) > gate (0.7) → NCV active
        ccv = _ccv(conf=0.5, latency_ns=1_000_000)
        ncv = _ncv(conf=0.9, latency_ns=3_000_000)
        result = fuse(ccv, ncv, _CFG)
        assert result is not None
        assert result.active_tracker == ActiveTracker.NCV
        assert result.latency_ns == 3_000_000

    def test_fused_active_uses_ccv_latency(self):
        # both confidences below gate (0.7) → FUSED
        ccv = _ccv(conf=0.5, latency_ns=1_000_000)
        ncv = _ncv(conf=0.5, latency_ns=3_000_000)
        result = fuse(ccv, ncv, _CFG)
        assert result is not None
        assert result.active_tracker == ActiveTracker.FUSED
        assert result.latency_ns == 1_000_000

    def test_stale_ncv_falls_back_to_ccv_latency(self):
        # ncv.timestamp_ns = 0, so age_ms will be huge → ncv dropped → passthrough ccv
        stale_ncv = TrackerEstimate(
            timestamp_ns=0,
            bbox=_BBOX,
            confidence=0.9,
            tracker_health=TrackerHealth.NOMINAL,
            latency_ns=9_000_000,
        )
        ccv = _ccv(latency_ns=1_000_000)
        result = fuse(ccv, stale_ncv, _CFG)
        assert result is not None
        assert result.active_tracker == ActiveTracker.CCV
        assert result.latency_ns == 1_000_000

from __future__ import annotations
import time
import types

import pytest

from quadguide.guidance.closing_vel import ClosingVelEstimator
from quadguide.core.messages import BoundingBox

_CFG = types.SimpleNamespace(
    closing_vel_fallback=2.0,
    closing_vel_ema_alpha=1.0,    # alpha=1 → no smoothing, predictable tests
    closing_vel_min_area_rate=0.001,
    closing_vel_area_scale=5.0,
)


def _bbox(w: float, h: float) -> BoundingBox:
    return BoundingBox(0.0, 0.0, w, h)


class TestClosingVelFallback:
    def test_first_call_returns_fallback(self):
        est = ClosingVelEstimator()
        result = est.update(_bbox(0.3, 0.3), time.monotonic_ns(), _CFG)
        assert result == pytest.approx(2.0)

    def test_stationary_bbox_returns_fallback(self):
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.3, 0.3), t0, _CFG)
        result = est.update(_bbox(0.3, 0.3), t0 + 20_000_000, _CFG)
        assert result == pytest.approx(2.0)

    def test_tiny_area_change_returns_fallback(self):
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.3, 0.3), t0, _CFG)
        # area change: 0.09 → 0.090001 in 1s → rate ≈ 0.000001 < 0.001 threshold
        result = est.update(_bbox(0.300003, 0.300003), t0 + 1_000_000_000, _CFG)
        assert result == pytest.approx(2.0)


class TestClosingVelNormal:
    def test_growing_bbox_returns_positive(self):
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.2, 0.2), t0, _CFG)
        # area: 0.04 → 0.09 in 0.1s → area_rate = 0.5 → v_c = 0.5 * 5 = 2.5
        result = est.update(_bbox(0.3, 0.3), t0 + 100_000_000, _CFG)
        assert result == pytest.approx(2.5, abs=0.01)

    def test_shrinking_bbox_clamped_to_fallback(self):
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.3, 0.3), t0, _CFG)
        # area shrinks → negative raw V_c clamped to fallback to avoid inverting PN law
        result = est.update(_bbox(0.2, 0.2), t0 + 100_000_000, _CFG)
        assert result == pytest.approx(_CFG.closing_vel_fallback)

    def test_scales_with_area_scale(self):
        cfg = types.SimpleNamespace(
            closing_vel_fallback=2.0,
            closing_vel_ema_alpha=1.0,
            closing_vel_min_area_rate=0.001,
            closing_vel_area_scale=10.0,   # double the default
        )
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.2, 0.2), t0, cfg)
        result = est.update(_bbox(0.3, 0.3), t0 + 100_000_000, cfg)
        assert result == pytest.approx(5.0, abs=0.01)   # 0.5 area_rate * 10


class TestClosingVelEMA:
    def test_alpha_one_is_raw(self):
        cfg = types.SimpleNamespace(
            closing_vel_fallback=2.0,
            closing_vel_ema_alpha=1.0,
            closing_vel_min_area_rate=0.001,
            closing_vel_area_scale=1.0,
        )
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.2, 0.2), t0, cfg)
        # area 0.04 → 0.09 in 1s → rate = 0.05
        result = est.update(_bbox(0.3, 0.3), t0 + 1_000_000_000, cfg)
        assert result == pytest.approx(0.05, rel=0.02)

    def test_ema_smooths_toward_zero(self):
        cfg = types.SimpleNamespace(
            closing_vel_fallback=0.0,    # disable fallback interference
            closing_vel_ema_alpha=0.5,
            closing_vel_min_area_rate=0.0,   # never fall back
            closing_vel_area_scale=1.0,
        )
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.2, 0.2), t0, cfg)
        # raw_rate ≈ 0.05; ema = 0.5*0.05 + 0.5*0.0 = 0.025
        result = est.update(_bbox(0.3, 0.3), t0 + 1_000_000_000, cfg)
        assert result == pytest.approx(0.025, rel=0.05)

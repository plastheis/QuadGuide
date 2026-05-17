from __future__ import annotations
import math
import time

import pytest

from quadguide.guidance.los import LOSRateEstimator
from quadguide.core.messages import AttitudeState, BoundingBox, LockOnCmd

FOV_H = 1.047   # ~60 degrees horizontal
ASPECT = 640 / 480   # image width / height


def _att(roll=0.0, pitch=0.0, yaw=0.0, rr=0.0, pr=0.0, yr=0.0) -> AttitudeState:
    return AttitudeState(time.monotonic_ns(), roll, pitch, yaw, rr, pr, yr)


def _lockon(seq: int) -> LockOnCmd:
    return LockOnCmd(time.monotonic_ns(), seq, BoundingBox(0.4, 0.4, 0.1, 0.1))


class TestLOSReset:
    def test_first_call_returns_zero(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        result = est.update((0.1, 0.1), _att(), None, time.monotonic_ns())
        assert result == (0.0, 0.0)

    def test_second_call_without_lock_returns_nonzero(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), None, t0)
        result = est.update((0.2, 0.0), _att(), None, t0 + 20_000_000)
        assert result != (0.0, 0.0)
        assert result[0] > 0.0

    def test_new_lockon_seq_resets_and_returns_zero(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), None, t0)
        est.update((0.2, 0.2), _att(), None, t0 + 20_000_000)
        # New lockon should reset
        result = est.update((0.5, 0.5), _att(), _lockon(seq=1), t0 + 40_000_000)
        assert result == (0.0, 0.0)

    def test_same_lockon_seq_does_not_reset(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        lockon = _lockon(seq=1)
        est.update((0.0, 0.0), _att(), lockon, t0)
        result = est.update((0.2, 0.0), _att(), lockon, t0 + 20_000_000)
        assert result != (0.0, 0.0)

    def test_subsequent_lockon_with_different_seq_resets(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), _lockon(seq=1), t0)
        est.update((0.1, 0.0), _att(), _lockon(seq=1), t0 + 20_000_000)
        result = est.update((0.5, 0.0), _att(), _lockon(seq=2), t0 + 40_000_000)
        assert result == (0.0, 0.0)


class TestLOSRateComputation:
    def test_centroid_moving_right_gives_positive_x(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), None, t0)
        # Move 0.1 in x over 100 ms → raw_rate_x = 1.0 centroid/s
        result = est.update((0.1, 0.0), _att(), None, t0 + 100_000_000)
        assert result[0] > 0.0

    def test_centroid_stationary_zero_body_rate_gives_zero(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.1, 0.1), _att(), None, t0)
        result = est.update((0.1, 0.1), _att(), None, t0 + 20_000_000)
        assert result == pytest.approx((0.0, 0.0), abs=1e-9)


class TestLOSBodyRateCorrection:
    def test_level_pitch_rate_subtracts_from_x(self):
        """Drone pitching at 1 rad/s with stationary centroid → negative LOS x rate."""
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), None, t0)
        att = _att(pr=1.0)   # pure pitch rate, level drone
        result = est.update((0.0, 0.0), att, None, t0 + 20_000_000)
        # correction_x = pitch_rate * 2/fov_h ≈ 1.91; raw_rate_x = 0
        # los_x = 0 - 1.91 ≈ -1.91
        expected_x = -(2.0 / FOV_H)
        assert result[0] == pytest.approx(expected_x, abs=0.02)

    def test_level_roll_rate_subtracts_from_y(self):
        """Drone rolling at 1 rad/s with stationary centroid → nonzero LOS y rate."""
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), None, t0)
        att = _att(rr=1.0)   # pure roll rate, level drone
        result = est.update((0.0, 0.0), att, None, t0 + 20_000_000)
        fov_v = FOV_H / ASPECT
        expected_y = 2.0 / fov_v   # roll moves centroid in y
        assert result[1] == pytest.approx(expected_y, abs=0.02)

    def test_yaw_rate_has_no_image_effect_at_level(self):
        """Yaw rotation around boresight does not move a centred target."""
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), None, t0)
        att = _att(yr=2.0)   # pure yaw rate
        result = est.update((0.0, 0.0), att, None, t0 + 20_000_000)
        assert result == pytest.approx((0.0, 0.0), abs=1e-6)

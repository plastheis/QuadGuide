from __future__ import annotations
import time

import pytest

from quadguide.guidance.los import LOSRateEstimator
from quadguide.core.messages import BoundingBox, IMUFrame, LockOnCmd

FOV_H = 1.047   # ~60° horizontal
ASPECT = 640 / 480


def _imu(gx=0.0, gy=0.0, gz=0.0, ax=0.0, ay=0.0, az=0.0) -> IMUFrame:
    return IMUFrame(time.monotonic_ns(), ax, ay, az, gx, gy, gz)


def _lockon(seq: int) -> LockOnCmd:
    return LockOnCmd(time.monotonic_ns(), seq, BoundingBox(0.4, 0.4, 0.1, 0.1))


class TestLOSReset:
    def test_first_call_returns_zero(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        result = est.update((0.1, 0.1), _imu(), None, time.monotonic_ns())
        assert result == (0.0, 0.0)

    def test_second_call_without_lock_returns_nonzero(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _imu(), None, t0)
        result = est.update((0.2, 0.0), _imu(), None, t0 + 20_000_000)
        assert result != (0.0, 0.0)
        assert result[0] > 0.0

    def test_new_lockon_seq_resets_and_returns_zero(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _imu(), None, t0)
        est.update((0.2, 0.2), _imu(), None, t0 + 20_000_000)
        result = est.update((0.5, 0.5), _imu(), _lockon(seq=1), t0 + 40_000_000)
        assert result == (0.0, 0.0)

    def test_same_lockon_seq_does_not_reset(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        lockon = _lockon(seq=1)
        est.update((0.0, 0.0), _imu(), lockon, t0)
        result = est.update((0.2, 0.0), _imu(), lockon, t0 + 20_000_000)
        assert result != (0.0, 0.0)

    def test_subsequent_lockon_with_different_seq_resets(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _imu(), _lockon(seq=1), t0)
        est.update((0.1, 0.0), _imu(), _lockon(seq=1), t0 + 20_000_000)
        result = est.update((0.5, 0.0), _imu(), _lockon(seq=2), t0 + 40_000_000)
        assert result == (0.0, 0.0)


class TestLOSRateComputation:
    def test_centroid_moving_right_gives_positive_x(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _imu(), None, t0)
        result = est.update((0.1, 0.0), _imu(), None, t0 + 100_000_000)
        assert result[0] > 0.0

    def test_centroid_stationary_zero_body_rate_gives_zero(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.1, 0.1), _imu(), None, t0)
        result = est.update((0.1, 0.1), _imu(), None, t0 + 20_000_000)
        assert result == pytest.approx((0.0, 0.0), abs=1e-9)


class TestLOSBodyRateDerotation:
    def test_pitch_rate_subtracts_from_x(self):
        """Body pitching (gy>0) with stationary centroid → negative LOS x rate."""
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _imu(), None, t0)
        result = est.update((0.0, 0.0), _imu(gy=1.0), None, t0 + 20_000_000)
        expected_x = -(2.0 / FOV_H)
        assert result[0] == pytest.approx(expected_x, abs=0.02)

    def test_roll_rate_subtracts_from_y(self):
        """Body rolling (gx>0) with stationary centroid → positive LOS y rate (sign flip)."""
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _imu(), None, t0)
        result = est.update((0.0, 0.0), _imu(gx=1.0), None, t0 + 20_000_000)
        fov_v = FOV_H / ASPECT
        expected_y = 2.0 / fov_v
        assert result[1] == pytest.approx(expected_y, abs=0.02)

    def test_yaw_rate_has_no_image_effect(self):
        """Yaw rotation about +Z bore-sight does not move a centred target."""
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _imu(), None, t0)
        result = est.update((0.0, 0.0), _imu(gz=2.0), None, t0 + 20_000_000)
        assert result == pytest.approx((0.0, 0.0), abs=1e-9)

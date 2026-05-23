import math
import time

import pytest

from quadguide.core.config import PurePursuitConfig
from quadguide.core.messages import (
    ActiveTracker, BoundingBox, IMUFrame, TargetEstimate, TrackerHealth,
)
from quadguide.guidance.pure_pursuit import PurePursuitGuidance


FOV_H = 1.047  # ~60° horizontal
ASPECT = 640 / 480


def _est(cx: float, cy: float) -> TargetEstimate:
    return TargetEstimate(
        timestamp_ns=time.monotonic_ns(),
        bbox=BoundingBox(0.4, 0.4, 0.1, 0.1),
        centroid_norm=(cx, cy),
        confidence=1.0,
        tracker_health=TrackerHealth.NOMINAL,
        active_tracker=ActiveTracker.FUSED,
    )


def _imu() -> IMUFrame:
    return IMUFrame(time.monotonic_ns(), 0, 0, 0, 0, 0, 0)


def test_zero_centroid_gives_zero_accel():
    pp = PurePursuitGuidance(PurePursuitConfig(K=6.0), FOV_H, ASPECT)
    ax, ay = pp.compute(_est(0.0, 0.0), _imu(), None, time.monotonic_ns())
    assert ax == 0.0
    assert ay == 0.0


def test_positive_cx_drives_positive_ax():
    pp = PurePursuitGuidance(PurePursuitConfig(K=6.0), FOV_H, ASPECT)
    ax, ay = pp.compute(_est(1.0, 0.0), _imu(), None, time.monotonic_ns())
    # K * 1.0 * (FOV_H / 2)
    assert ax == pytest.approx(6.0 * FOV_H * 0.5)
    assert ay == 0.0


def test_scales_with_K():
    pp1 = PurePursuitGuidance(PurePursuitConfig(K=1.0), FOV_H, ASPECT)
    pp2 = PurePursuitGuidance(PurePursuitConfig(K=10.0), FOV_H, ASPECT)
    a1 = pp1.compute(_est(0.5, 0.0), _imu(), None, time.monotonic_ns())
    a2 = pp2.compute(_est(0.5, 0.0), _imu(), None, time.monotonic_ns())
    assert math.isclose(a2[0], a1[0] * 10.0)


def test_vertical_scale_uses_vertical_fov():
    pp = PurePursuitGuidance(PurePursuitConfig(K=6.0), FOV_H, ASPECT)
    _, ay = pp.compute(_est(0.0, 1.0), _imu(), None, time.monotonic_ns())
    fov_v = FOV_H / ASPECT
    assert ay == pytest.approx(6.0 * fov_v * 0.5)


def test_name():
    pp = PurePursuitGuidance(PurePursuitConfig(K=6.0), FOV_H, ASPECT)
    assert pp.name() == "pure_pursuit"

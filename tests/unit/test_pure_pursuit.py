import math
import time

import pytest

from quadguide.core.config import PurePursuitConfig
from quadguide.core.messages import (
    BoundingBox, IMUFrame, TrackerEstimate, TrackerHealth,
)
from quadguide.guidance.pure_pursuit import PurePursuitGuidance


FOV_H = 1.047  # ~60° horizontal
ASPECT = 640 / 480


def _est_for_centroid(cx: float, cy: float) -> TrackerEstimate:
    """Build an estimate whose computed centroid equals (cx, cy)."""
    w = h = 0.1
    bbox_x = (cx / 2.0 + 0.5) - w * 0.5
    bbox_y = (cy / 2.0 + 0.5) - h * 0.5
    return TrackerEstimate(
        timestamp_ns=time.monotonic_ns(),
        bbox=BoundingBox(bbox_x, bbox_y, w, h),
        confidence=1.0,
        tracker_health=TrackerHealth.NOMINAL,
    )


def _imu() -> IMUFrame:
    return IMUFrame(time.monotonic_ns(), 0, 0, 0, 0, 0, 0)


def test_zero_centroid_gives_zero_accel():
    pp = PurePursuitGuidance(PurePursuitConfig(K=6.0), FOV_H, ASPECT)
    ax, ay = pp.compute(_est_for_centroid(0.0, 0.0), _imu(), None, time.monotonic_ns())
    assert ax == pytest.approx(0.0)
    assert ay == pytest.approx(0.0)


def test_positive_cx_drives_positive_ay():
    # Bore-up mount: horizontal image error (cx) is a lateral/roll offset → ay.
    pp = PurePursuitGuidance(PurePursuitConfig(K=6.0, deadband=0.0), FOV_H, ASPECT)
    ax, ay = pp.compute(_est_for_centroid(1.0, 0.0), _imu(), None, time.monotonic_ns())
    assert ay == pytest.approx(6.0 * FOV_H * 0.5)
    assert ax == pytest.approx(0.0)


def test_scales_with_K():
    pp1 = PurePursuitGuidance(PurePursuitConfig(K=1.0), FOV_H, ASPECT)
    pp2 = PurePursuitGuidance(PurePursuitConfig(K=10.0), FOV_H, ASPECT)
    a1 = pp1.compute(_est_for_centroid(0.5, 0.0), _imu(), None, time.monotonic_ns())
    a2 = pp2.compute(_est_for_centroid(0.5, 0.0), _imu(), None, time.monotonic_ns())
    assert math.isclose(a2[0], a1[0] * 10.0)


def test_vertical_scale_uses_vertical_fov():
    # Bore-up mount: vertical image error (cy) is a fore/aft pitch offset → ax.
    pp = PurePursuitGuidance(PurePursuitConfig(K=6.0, deadband=0.0), FOV_H, ASPECT)
    ax, _ = pp.compute(_est_for_centroid(0.0, 1.0), _imu(), None, time.monotonic_ns())
    fov_v = FOV_H / ASPECT
    assert ax == pytest.approx(6.0 * fov_v * 0.5)


def test_deadband_zeroes_small_centroid():
    pp = PurePursuitGuidance(PurePursuitConfig(K=6.0, deadband=0.05), FOV_H, ASPECT)
    # centroid inside the band on both axes → no command.
    ax, ay = pp.compute(_est_for_centroid(0.04, -0.03), _imu(), None, time.monotonic_ns())
    assert ax == pytest.approx(0.0)
    assert ay == pytest.approx(0.0)


def test_deadband_is_shifted_not_clamped():
    # Past the edge the response is continuous: output uses (|c| - db), so a soft
    # deadband produces strictly less than the no-deadband command at the same c.
    db = 0.1
    c = 0.5
    pp_db = PurePursuitGuidance(PurePursuitConfig(K=6.0, deadband=db), FOV_H, ASPECT)
    pp_no = PurePursuitGuidance(PurePursuitConfig(K=6.0, deadband=0.0), FOV_H, ASPECT)
    ay_db = pp_db.compute(_est_for_centroid(c, 0.0), _imu(), None, time.monotonic_ns())[1]
    ay_no = pp_no.compute(_est_for_centroid(c, 0.0), _imu(), None, time.monotonic_ns())[1]
    assert ay_db == pytest.approx(6.0 * (c - db) * FOV_H * 0.5)
    assert ay_db < ay_no


def test_name():
    pp = PurePursuitGuidance(PurePursuitConfig(K=6.0), FOV_H, ASPECT)
    assert pp.name() == "pure_pursuit"

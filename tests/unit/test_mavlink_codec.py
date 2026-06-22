import math
import pytest

from quadguide.link.mavlink_codec import (
    ATT_TARGET_IGNORE_RATES, MSG_ID_ATTITUDE, MSG_ID_RAW_IMU,
    euler_to_quaternion, make_mav,
)


def test_make_mav_sets_source_ids_and_robust_parsing():
    mav = make_mav(1, 191)
    assert mav.srcSystem == 1
    assert mav.srcComponent == 191
    assert mav.robust_parsing is True


def test_constants():
    assert ATT_TARGET_IGNORE_RATES == 0x07
    assert MSG_ID_ATTITUDE == 30
    assert MSG_ID_RAW_IMU == 27


def test_quaternion_identity():
    assert euler_to_quaternion(0.0, 0.0, 0.0) == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-9)


def test_quaternion_roll_90():
    q = euler_to_quaternion(math.pi / 2, 0.0, 0.0)
    assert q == pytest.approx((math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0), abs=1e-9)


def test_quaternion_pitch_90():
    q = euler_to_quaternion(0.0, math.pi / 2, 0.0)
    assert q == pytest.approx((math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0), abs=1e-9)


def test_quaternion_yaw_90():
    q = euler_to_quaternion(0.0, 0.0, math.pi / 2)
    assert q == pytest.approx((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), abs=1e-9)

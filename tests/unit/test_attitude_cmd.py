import math
import time

import pytest

from quadguide.control.attitude_cmd import compute
from quadguide.core.messages import AccelCmd

_G = 9.81


def _accel(ax: float, ay: float) -> AccelCmd:
    return AccelCmd(time.monotonic_ns(), ax, ay)


def test_zero_accel_gives_zero_angles():
    roll, pitch = compute(_accel(0.0, 0.0))
    assert roll == 0.0
    assert pitch == 0.0


def test_full_g_lateral_gives_45_deg_roll():
    roll, pitch = compute(_accel(0.0, _G))
    assert math.isclose(roll, 45.0, abs_tol=0.01)


def test_negative_lateral_gives_negative_roll():
    roll, pitch = compute(_accel(0.0, -_G))
    assert math.isclose(roll, -45.0, abs_tol=0.01)


def test_full_g_forward_gives_minus_45_deg_pitch():
    # Positive ax (forward accel) → nose up → negative pitch setpoint
    roll, pitch = compute(_accel(_G, 0.0))
    assert math.isclose(pitch, -45.0, abs_tol=0.01)


def test_negative_forward_gives_positive_pitch():
    roll, pitch = compute(_accel(-_G, 0.0))
    assert math.isclose(pitch, 45.0, abs_tol=0.01)


def test_roll_and_pitch_independent():
    roll, pitch = compute(_accel(_G, _G))
    assert math.isclose(roll, 45.0, abs_tol=0.01)
    assert math.isclose(pitch, -45.0, abs_tol=0.01)


def test_small_accel_proportional():
    roll1, _ = compute(_accel(0.0, 1.0))
    roll2, _ = compute(_accel(0.0, 2.0))
    assert roll2 == pytest.approx(roll1 * 2, rel=0.01)

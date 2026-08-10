import math
import time

import pytest

from quadguide.control.attitude_cmd import compute
from quadguide.core.messages import AccelCmd

_G = 9.81


def _accel(ax: float, ay: float) -> AccelCmd:
    return AccelCmd(time.monotonic_ns(), ax, ay)


# Sign convention hard-coded in attitude_cmd:
#   roll_deg  =  degrees(ay / g)
#   pitch_deg = -degrees(ax / g)
#
# MAVLink/ArduPilot attitude signs: +roll = right wing down (accelerates +Y,
# right), +pitch = nose up (accelerates -X, backward). So a body-frame accel
# demand of +ay needs +roll and +ax needs -pitch (nose down → forward).


def test_zero_accel_gives_zero_angles():
    roll, pitch = compute(_accel(0.0, 0.0))
    assert roll == 0.0
    assert pitch == 0.0


def test_positive_lateral_accel_gives_positive_roll():
    # ay > 0 (accelerate right) → roll right
    roll, _ = compute(_accel(0.0, 1.0))
    assert roll == pytest.approx(math.degrees(1.0 / _G), rel=1e-6)
    assert roll > 0.0


def test_negative_lateral_accel_gives_negative_roll():
    roll, _ = compute(_accel(0.0, -1.0))
    assert roll == pytest.approx(-math.degrees(1.0 / _G), rel=1e-6)
    assert roll < 0.0


def test_positive_forward_accel_gives_negative_pitch():
    # ax > 0 (accelerate forward) → nose down
    _, pitch = compute(_accel(1.0, 0.0))
    assert pitch == pytest.approx(-math.degrees(1.0 / _G), rel=1e-6)
    assert pitch < 0.0


def test_negative_forward_accel_gives_positive_pitch():
    _, pitch = compute(_accel(-1.0, 0.0))
    assert pitch == pytest.approx(math.degrees(1.0 / _G), rel=1e-6)
    assert pitch > 0.0


def test_roll_and_pitch_independent():
    roll, pitch = compute(_accel(1.0, 1.0))
    assert roll == pytest.approx(math.degrees(1.0 / _G), rel=1e-6)
    assert pitch == pytest.approx(-math.degrees(1.0 / _G), rel=1e-6)


def test_small_accel_proportional():
    roll1, _ = compute(_accel(0.0, 0.1))
    roll2, _ = compute(_accel(0.0, 0.2))
    assert roll2 == pytest.approx(roll1 * 2, rel=1e-6)

import math
import time

import pytest

from quadguide.control.attitude_cmd import compute
from quadguide.core.messages import AccelCmd

_G = 9.81


def _accel(ax: float, ay: float) -> AccelCmd:
    return AccelCmd(time.monotonic_ns(), ax, ay)


# Sign convention for the +Z (up-facing) bore-sight, hard-coded in attitude_cmd:
#   roll_deg  = -degrees(ay / g)
#   pitch_deg =  degrees(ax / g)
#
# (Conventional nadir camera would use +ay/g and -ax/g. Both axes are inverted
# here because the upward-facing image plane is mirrored about the horizon.)


def test_zero_accel_gives_zero_angles():
    roll, pitch = compute(_accel(0.0, 0.0))
    assert roll == 0.0
    assert pitch == 0.0


def test_positive_lateral_accel_gives_negative_roll():
    # ay > 0 → roll < 0 (sign flipped vs nadir camera)
    roll, _ = compute(_accel(0.0, 1.0))
    assert roll == pytest.approx(-math.degrees(1.0 / _G), rel=1e-6)
    assert roll < 0.0


def test_negative_lateral_accel_gives_positive_roll():
    roll, _ = compute(_accel(0.0, -1.0))
    assert roll == pytest.approx(math.degrees(1.0 / _G), rel=1e-6)
    assert roll > 0.0


def test_positive_forward_accel_gives_positive_pitch():
    # ax > 0 → pitch > 0 (sign flipped vs nadir camera)
    _, pitch = compute(_accel(1.0, 0.0))
    assert pitch == pytest.approx(math.degrees(1.0 / _G), rel=1e-6)
    assert pitch > 0.0


def test_negative_forward_accel_gives_negative_pitch():
    _, pitch = compute(_accel(-1.0, 0.0))
    assert pitch == pytest.approx(-math.degrees(1.0 / _G), rel=1e-6)
    assert pitch < 0.0


def test_roll_and_pitch_independent():
    roll, pitch = compute(_accel(1.0, 1.0))
    assert roll == pytest.approx(-math.degrees(1.0 / _G), rel=1e-6)
    assert pitch == pytest.approx(math.degrees(1.0 / _G), rel=1e-6)


def test_small_accel_proportional():
    roll1, _ = compute(_accel(0.0, 0.1))
    roll2, _ = compute(_accel(0.0, 0.2))
    assert roll2 == pytest.approx(roll1 * 2, rel=1e-6)

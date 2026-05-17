import math
import pytest
from quadguide.guidance.pronav import pronav


def test_zero_los_rate_gives_zero_accel():
    ax, ay = pronav((0.0, 0.0), 2.0, 4.0)
    assert ax == 0.0
    assert ay == 0.0


def test_positive_los_x_gives_positive_ax():
    ax, ay = pronav((1.0, 0.0), 2.0, 4.0)
    assert math.isclose(ax, 8.0)   # N * v_c * los_x = 4 * 2 * 1
    assert ay == 0.0


def test_positive_los_y_gives_positive_ay():
    ax, ay = pronav((0.0, 1.0), 2.0, 4.0)
    assert ax == 0.0
    assert math.isclose(ay, 8.0)


def test_negative_closing_vel_flips_sign():
    ax, _ = pronav((1.0, 0.0), -2.0, 4.0)
    assert math.isclose(ax, -8.0)


def test_scales_with_N():
    ax, ay = pronav((1.0, 1.0), 1.0, 3.0)
    assert math.isclose(ax, 3.0)
    assert math.isclose(ay, 3.0)


def test_zero_closing_vel_gives_zero_regardless_of_los():
    ax, ay = pronav((10.0, 10.0), 0.0, 4.0)
    assert ax == 0.0
    assert ay == 0.0

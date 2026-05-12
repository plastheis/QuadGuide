import math
import pytest
from quadguide.link.differentiator import AttitudeDifferentiator


def test_first_call_returns_zero_rates():
    diff = AttitudeDifferentiator(alpha=1.0)
    rates = diff.update(0.1, 0.2, 0.3, now_ns=0)
    assert rates == (0.0, 0.0, 0.0)


def test_rate_calculation_roll():
    diff = AttitudeDifferentiator(alpha=1.0)
    diff.update(0.0, 0.0, 0.0, now_ns=0)
    rates = diff.update(1.0, 0.0, 0.0, now_ns=int(1e9))  # 1 rad in 1 second
    assert rates[0] == pytest.approx(1.0, rel=1e-5)
    assert rates[1] == pytest.approx(0.0, abs=1e-9)
    assert rates[2] == pytest.approx(0.0, abs=1e-9)


def test_rate_calculation_pitch():
    diff = AttitudeDifferentiator(alpha=1.0)
    diff.update(0.0, 0.0, 0.0, now_ns=0)
    rates = diff.update(0.0, 2.0, 0.0, now_ns=int(2e9))  # 2 rad in 2 seconds
    assert rates[1] == pytest.approx(1.0, rel=1e-5)


def test_rate_calculation_half_second():
    diff = AttitudeDifferentiator(alpha=1.0)
    diff.update(0.0, 0.0, 0.0, now_ns=0)
    rates = diff.update(1.0, 0.0, 0.0, now_ns=int(500e6))  # 1 rad in 0.5 seconds
    assert rates[0] == pytest.approx(2.0, rel=1e-5)


def test_yaw_wraparound_positive_to_negative():
    diff = AttitudeDifferentiator(alpha=1.0)
    # Yaw crosses +π → -π boundary; shortest-path change is -0.2 rad
    diff.update(0.0, 0.0, math.pi - 0.1, now_ns=0)
    rates = diff.update(0.0, 0.0, -(math.pi - 0.1), now_ns=int(1e9))
    assert rates[2] == pytest.approx(-0.2, abs=1e-5)


def test_yaw_wraparound_negative_to_positive():
    diff = AttitudeDifferentiator(alpha=1.0)
    diff.update(0.0, 0.0, -(math.pi - 0.1), now_ns=0)
    rates = diff.update(0.0, 0.0, math.pi - 0.1, now_ns=int(1e9))
    assert rates[2] == pytest.approx(0.2, abs=1e-5)


def test_lowpass_filter_alpha_1_is_unfiltered():
    # alpha=1.0 means no filtering: output equals raw rate
    diff = AttitudeDifferentiator(alpha=1.0)
    diff.update(0.0, 0.0, 0.0, now_ns=0)
    diff.update(2.0, 0.0, 0.0, now_ns=int(1e9))  # raw = 2.0
    rates = diff.update(4.0, 0.0, 0.0, now_ns=int(2e9))  # raw = 2.0
    assert rates[0] == pytest.approx(2.0, rel=1e-5)


def test_lowpass_filter_smoothing():
    # alpha=0.5: filtered = 0.5*raw + 0.5*prev_filtered
    diff = AttitudeDifferentiator(alpha=0.5)
    diff.update(0.0, 0.0, 0.0, now_ns=0)
    # Step 1: raw=2.0, filtered = 0.5*2.0 + 0.5*0.0 = 1.0
    diff.update(2.0, 0.0, 0.0, now_ns=int(1e9))
    # Step 2: raw=2.0, filtered = 0.5*2.0 + 0.5*1.0 = 1.5
    rates = diff.update(4.0, 0.0, 0.0, now_ns=int(2e9))
    assert rates[0] == pytest.approx(1.5, abs=1e-6)


def test_zero_dt_returns_previous_rates():
    diff = AttitudeDifferentiator(alpha=1.0)
    diff.update(0.0, 0.0, 0.0, now_ns=0)
    diff.update(1.0, 0.0, 0.0, now_ns=int(1e9))  # sets rate to 1.0
    # Same timestamp — dt=0, should return previous filtered rates unchanged
    rates = diff.update(2.0, 0.0, 0.0, now_ns=int(1e9))
    assert rates[0] == pytest.approx(1.0, rel=1e-5)

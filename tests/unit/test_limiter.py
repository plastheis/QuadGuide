from __future__ import annotations
import time

import pytest

from quadguide.control.limiter import failsafe_cmd, saturate, slew_rate
from quadguide.core.config import ControlLimitsConfig
from quadguide.core.messages import ControlCmd

_LIMITS = ControlLimitsConfig(
    max_roll_deg=35.0,
    max_pitch_deg=35.0,
    max_roll_rate_dps=200.0,
    max_pitch_rate_dps=200.0,
)


def _cmd(roll: float, pitch: float, throttle: float = 0.55) -> ControlCmd:
    return ControlCmd(time.monotonic_ns(), roll, pitch, 0.0, throttle)


class TestSaturate:
    def test_within_limits_unchanged(self):
        roll, pitch = saturate(10.0, -10.0, _LIMITS)
        assert roll == 10.0
        assert pitch == -10.0

    def test_roll_clamped_positive(self):
        roll, _ = saturate(50.0, 0.0, _LIMITS)
        assert roll == 35.0

    def test_roll_clamped_negative(self):
        roll, _ = saturate(-50.0, 0.0, _LIMITS)
        assert roll == -35.0

    def test_pitch_clamped_positive(self):
        _, pitch = saturate(0.0, 50.0, _LIMITS)
        assert pitch == 35.0

    def test_pitch_clamped_negative(self):
        _, pitch = saturate(0.0, -50.0, _LIMITS)
        assert pitch == -35.0

    def test_at_limit_boundary_unchanged(self):
        roll, pitch = saturate(35.0, -35.0, _LIMITS)
        assert roll == 35.0
        assert pitch == -35.0


class TestSlewRate:
    def test_no_prev_passes_through_unchanged(self):
        roll, pitch = slew_rate(20.0, -20.0, None, _LIMITS, 0.01)
        assert roll == 20.0
        assert pitch == -20.0

    def test_large_positive_step_clamped(self):
        prev = _cmd(0.0, 0.0)
        # max_delta = 200 dps * 0.01 s = 2 deg
        roll, _ = slew_rate(10.0, 0.0, prev, _LIMITS, 0.01)
        assert roll == pytest.approx(2.0)

    def test_large_negative_step_clamped(self):
        prev = _cmd(0.0, 0.0)
        roll, _ = slew_rate(-10.0, 0.0, prev, _LIMITS, 0.01)
        assert roll == pytest.approx(-2.0)

    def test_small_step_passes_through(self):
        prev = _cmd(0.0, 0.0)
        roll, _ = slew_rate(1.0, 0.0, prev, _LIMITS, 0.01)
        assert roll == pytest.approx(1.0)

    def test_pitch_clamped_independently(self):
        prev = _cmd(0.0, 0.0)
        _, pitch = slew_rate(0.0, 10.0, prev, _LIMITS, 0.01)
        assert pitch == pytest.approx(2.0)

    def test_step_from_nonzero_prev(self):
        prev = _cmd(30.0, 0.0)
        # from 30 deg, requesting 35 deg: delta = 5 deg > 2 deg limit → clamp to 32 deg
        roll, _ = slew_rate(35.0, 0.0, prev, _LIMITS, 0.01)
        assert roll == pytest.approx(32.0)

    def test_exact_max_delta_allowed(self):
        prev = _cmd(0.0, 0.0)
        roll, _ = slew_rate(2.0, 0.0, prev, _LIMITS, 0.01)
        assert roll == pytest.approx(2.0)


class TestFailsafeCmd:
    def test_roll_pitch_yaw_are_zero(self):
        cmd = failsafe_cmd(0.55)
        assert cmd.roll_deg == 0.0
        assert cmd.pitch_deg == 0.0
        assert cmd.yaw_rate_dps == 0.0

    def test_throttle_matches_argument(self):
        cmd = failsafe_cmd(0.55)
        assert cmd.throttle_norm == pytest.approx(0.55)

    def test_different_throttle_value(self):
        cmd = failsafe_cmd(0.4)
        assert cmd.throttle_norm == pytest.approx(0.4)

    def test_returns_control_cmd_type(self):
        cmd = failsafe_cmd(0.55)
        assert isinstance(cmd, ControlCmd)

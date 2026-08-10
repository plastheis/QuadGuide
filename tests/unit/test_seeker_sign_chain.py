"""End-to-end seeker sign chain: image half → commanded roll/pitch.

Guards the one property the whole loop hangs on for the strapdown bore-up
mount, across *both* guidance methods:

    target in the BOTTOM half of the image  → pitch DOWN (pitch_deg < 0)
    target in the RIGHT  half of the image  → roll  RIGHT (roll_deg  > 0)

Chain under test: bbox (top-left origin, y down) → centroid_norm → guidance
(ax, ay) → attitude_cmd (roll_deg, pitch_deg), with MAVLink attitude signs
(+roll = right wing down, +pitch = nose up).
"""
from __future__ import annotations
import time

from quadguide.control.attitude_cmd import compute as attitude_cmd
from quadguide.core.config import PronavConfig, PurePursuitConfig
from quadguide.core.messages import (
    AccelCmd, BoundingBox, IMUFrame, TrackerEstimate, TrackerHealth,
)
from quadguide.guidance.pronav import ProNavGuidance
from quadguide.guidance.pure_pursuit import PurePursuitGuidance

FOV_H = 1.047  # ~60° horizontal
ASPECT = 640 / 480


def _est(cx: float, cy: float, ts_ns: int) -> TrackerEstimate:
    """Estimate whose centroid_norm is (cx, cy); cy > 0 = lower half."""
    w = h = 0.1
    return TrackerEstimate(
        timestamp_ns=ts_ns,
        bbox=BoundingBox((cx / 2.0 + 0.5) - w * 0.5, (cy / 2.0 + 0.5) - h * 0.5, w, h),
        confidence=1.0,
        tracker_health=TrackerHealth.NOMINAL,
    )


def _imu() -> IMUFrame:
    return IMUFrame(time.monotonic_ns(), 0, 0, 0, 0, 0, 0)


def _angles(ax: float, ay: float) -> tuple[float, float]:
    return attitude_cmd(AccelCmd(time.monotonic_ns(), ax, ay))


def _pp() -> PurePursuitGuidance:
    return PurePursuitGuidance(PurePursuitConfig(K=15.0, deadband=0.03), FOV_H, ASPECT)


class TestPurePursuitSignChain:
    def test_bottom_half_pitches_down(self):
        roll, pitch = _angles(*_pp().compute(_est(0.0, 0.5, time.monotonic_ns()),
                                             _imu(), None, time.monotonic_ns()))
        assert pitch < 0.0
        assert roll == 0.0

    def test_top_half_pitches_up(self):
        _, pitch = _angles(*_pp().compute(_est(0.0, -0.5, time.monotonic_ns()),
                                          _imu(), None, time.monotonic_ns()))
        assert pitch > 0.0

    def test_right_half_rolls_right(self):
        roll, pitch = _angles(*_pp().compute(_est(0.5, 0.0, time.monotonic_ns()),
                                             _imu(), None, time.monotonic_ns()))
        assert roll > 0.0
        assert pitch == 0.0

    def test_left_half_rolls_left(self):
        roll, _ = _angles(*_pp().compute(_est(-0.5, 0.0, time.monotonic_ns()),
                                         _imu(), None, time.monotonic_ns()))
        assert roll < 0.0

    def test_bottom_right_quadrant_pitches_down_and_rolls_right(self):
        roll, pitch = _angles(*_pp().compute(_est(0.5, 0.5, time.monotonic_ns()),
                                             _imu(), None, time.monotonic_ns()))
        assert roll > 0.0
        assert pitch < 0.0


class TestProNavSignChain:
    """ProNav commands off LOS *rate*, so the guard is on drift direction."""

    def _drift(self, dcx: float, dcy: float) -> tuple[float, float]:
        pn = ProNavGuidance(PronavConfig(N=3.0, closing_vel_fallback=1.0), FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        pn.compute(_est(0.0, 0.0, t0), _imu(), None, t0)
        return _angles(*pn.compute(_est(dcx, dcy, t0 + 50_000_000), _imu(), None,
                                   t0 + 50_000_000))

    def test_target_drifting_down_pitches_down(self):
        roll, pitch = self._drift(0.0, 0.2)
        assert pitch < 0.0
        assert roll == 0.0

    def test_target_drifting_right_rolls_right(self):
        roll, pitch = self._drift(0.2, 0.0)
        assert roll > 0.0
        assert pitch == 0.0

    def test_target_drifting_up_left_pitches_up_and_rolls_left(self):
        roll, pitch = self._drift(-0.2, -0.2)
        assert roll < 0.0
        assert pitch > 0.0

from __future__ import annotations

from quadguide.core.clock import monotonic_ns
from quadguide.core.config import ControlLimitsConfig
from quadguide.core.messages import ControlCmd


def saturate(
    roll: float,
    pitch: float,
    limits: ControlLimitsConfig,
) -> tuple[float, float]:
    roll  = max(-limits.max_roll_deg,  min(limits.max_roll_deg,  roll))
    pitch = max(-limits.max_pitch_deg, min(limits.max_pitch_deg, pitch))
    return roll, pitch


def slew_rate(
    roll: float,
    pitch: float,
    prev_cmd: ControlCmd | None,
    limits: ControlLimitsConfig,
    dt: float,
) -> tuple[float, float]:
    if prev_cmd is None:
        return roll, pitch
    max_dr = limits.max_roll_rate_dps  * dt
    max_dp = limits.max_pitch_rate_dps * dt
    roll  = prev_cmd.roll_deg  + max(-max_dr, min(max_dr,  roll  - prev_cmd.roll_deg))
    pitch = prev_cmd.pitch_deg + max(-max_dp, min(max_dp,  pitch - prev_cmd.pitch_deg))
    return roll, pitch


def failsafe_cmd(throttle_hold: float) -> ControlCmd:
    return ControlCmd(monotonic_ns(), 0.0, 0.0, 0.0, throttle_hold)

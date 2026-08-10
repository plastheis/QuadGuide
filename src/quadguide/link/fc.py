"""Semantic map between ArduPilot MAVLink messages and quadguide bus dataclasses.

Decoders take a parsed pymavlink message plus the monotonic receive timestamp
(stamped by the RX loop) and return the bus dataclass. Encoders (Task 4) take a
codec `mav` object and return packed MAVLink2 bytes.
"""
from __future__ import annotations
import math

from pymavlink import mavutil

from quadguide.core.messages import AttitudeState, ControlCmd, IMUFrame
from quadguide.link.mavlink_codec import (
    ATT_TARGET_IGNORE_RATES, MAV_AUTOPILOT_NONE, MAV_TYPE_COMPANION,
    euler_to_quaternion,
)

_G_MPS2 = 9.80665


def decode_attitude(msg, recv_ns: int) -> AttitudeState:
    """ATTITUDE (#30) → AttitudeState. Angles rad, rates rad/s, body frame — native."""
    return AttitudeState(
        timestamp_ns=recv_ns,
        roll_rad=msg.roll,
        pitch_rad=msg.pitch,
        yaw_rad=msg.yaw,
        roll_rate_rps=msg.rollspeed,
        pitch_rate_rps=msg.pitchspeed,
        yaw_rate_rps=msg.yawspeed,
    )


def decode_imu(msg, recv_ns: int) -> IMUFrame:
    """RAW_IMU (#27) / SCALED_IMU2 (#116) → IMUFrame.

    ArduPilot units: accel in milli-g (1000 = 1 g), gyro in milli-rad/s, body
    NED/FRD. Returned units: m/s² and rad/s.
    """
    return IMUFrame(
        timestamp_ns=recv_ns,
        ax=(msg.xacc / 1000.0) * _G_MPS2,
        ay=(msg.yacc / 1000.0) * _G_MPS2,
        az=(msg.zacc / 1000.0) * _G_MPS2,
        gx=msg.xgyro / 1000.0,
        gy=msg.ygyro / 1000.0,
        gz=msg.zgyro / 1000.0,
    )


def decode_vfr_hud(msg) -> tuple[int, float, float]:
    """VFR_HUD (#74) → (throttle_pct, climb_mps, alt_m). Diagnostic only.

    ``throttle`` is the FC's own commanded throttle in percent — compare it
    against the thrust we sent in SET_ATTITUDE_TARGET. A steady mismatch, or a
    non-zero ``climb`` under constant thrust, means ArduPilot is treating the
    thrust field as a CLIMB RATE (GUID_OPTIONS bit 3 clear) rather than thrust.
    """
    return int(msg.throttle), float(msg.climb), float(msg.alt)


def decode_heartbeat(msg) -> tuple[bool, int]:
    """HEARTBEAT → (armed, custom_mode)."""
    armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    return armed, msg.custom_mode


def attitude_setpoint(
    cmd: ControlCmd | None, max_roll_deg: float, max_pitch_deg: float,
) -> tuple[float, float, float]:
    """`cmd` → the (roll_deg, pitch_deg, thrust) that will go on the wire.

    Split out of ``encode_attitude_target`` so the TX loop can record exactly
    what was commanded — post-clamp — without re-deriving the limits. Comparing
    that against the FC's own ATTITUDE reply is the only way to tell "the FC
    ignored SET_ATTITUDE_TARGET" from "the FC flew the attitude but gated the
    throttle": the two look identical in VFR_HUD alone.
    """
    if cmd is None:
        return 0.0, 0.0, 0.0
    return (
        _clamp(cmd.roll_deg, -max_roll_deg, max_roll_deg),
        _clamp(cmd.pitch_deg, -max_pitch_deg, max_pitch_deg),
        _clamp(cmd.throttle_norm, 0.0, 1.0),
    )


def encode_attitude_target(
    mav, cmd: ControlCmd | None, yaw_hold: float,
    target_sys: int, target_comp: int,
    max_roll_deg: float, max_pitch_deg: float, now_ms: int,
) -> bytes:
    """Build a SET_ATTITUDE_TARGET (#82) frame.

    roll/pitch come from `cmd` (clamped to limits) and yaw from the latched
    `yaw_hold`, all baked into the quaternion (type_mask ignores body rates).
    thrust = clamped throttle_norm. `cmd is None` → level attitude, zero thrust.
    """
    roll_deg, pitch_deg, thrust = attitude_setpoint(cmd, max_roll_deg, max_pitch_deg)
    q = euler_to_quaternion(
        math.radians(roll_deg), math.radians(pitch_deg), yaw_hold)
    msg = mav.set_attitude_target_encode(
        now_ms, target_sys, target_comp, ATT_TARGET_IGNORE_RATES,
        list(q), 0.0, 0.0, 0.0, thrust,
    )
    return msg.pack(mav)


def encode_arm(mav, arm: bool, target_sys: int, target_comp: int) -> bytes:
    """COMMAND_LONG / MAV_CMD_COMPONENT_ARM_DISARM. param1: 1=arm, 0=disarm."""
    msg = mav.command_long_encode(
        target_sys, target_comp,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1.0 if arm else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    return msg.pack(mav)


def encode_set_mode(mav, custom_mode: int, target_sys: int, target_comp: int) -> bytes:
    """COMMAND_LONG / MAV_CMD_DO_SET_MODE.

    param1 = MAV_MODE_FLAG_CUSTOM_MODE_ENABLED (interpret param2 as a custom
    mode), param2 = the ArduCopter custom_mode number (e.g. 9 = LAND).
    """
    msg = mav.command_long_encode(
        target_sys, target_comp,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
        float(custom_mode), 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    return msg.pack(mav)


def encode_set_message_interval(
    mav, msg_id: int, rate_hz: float, target_sys: int, target_comp: int
) -> bytes:
    """COMMAND_LONG / MAV_CMD_SET_MESSAGE_INTERVAL. param2 is the interval in µs."""
    interval_us = 0.0 if rate_hz <= 0 else 1_000_000.0 / rate_hz
    msg = mav.command_long_encode(
        target_sys, target_comp,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        float(msg_id), interval_us, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    return msg.pack(mav)


def encode_heartbeat(mav) -> bytes:
    """Companion HEARTBEAT so the FC sees quadguide as a live onboard controller."""
    msg = mav.heartbeat_encode(
        MAV_TYPE_COMPANION, MAV_AUTOPILOT_NONE, 0, 0,
        mavutil.mavlink.MAV_STATE_ACTIVE,
    )
    return msg.pack(mav)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

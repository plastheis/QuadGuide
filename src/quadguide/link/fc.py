"""Semantic map between ArduPilot MAVLink messages and quadguide bus dataclasses.

Decoders take a parsed pymavlink message plus the monotonic receive timestamp
(stamped by the RX loop) and return the bus dataclass. Encoders (Task 4) take a
codec `mav` object and return packed MAVLink2 bytes.
"""
from __future__ import annotations

from pymavlink import mavutil

from quadguide.core.messages import AttitudeState, IMUFrame

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


def decode_heartbeat(msg) -> tuple[bool, int]:
    """HEARTBEAT → (armed, custom_mode)."""
    armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    return armed, msg.custom_mode

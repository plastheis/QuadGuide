"""MAVLink2 codec helpers for the ArduPilot link.

pymavlink is used as a *codec only* — the MAVLink object is built with no
connection (`file=None`) and only parses incoming bytes / serializes outgoing
messages. The transport (UART or TCP) stays in serial_port.py / tcp_serial.py,
so the RX/TX loops and reconnect machinery are transport-agnostic.
"""
from __future__ import annotations
import math

from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mavlink2

# Telemetry message ids we request via SET_MESSAGE_INTERVAL.
MSG_ID_ATTITUDE = mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE  # 30
MSG_ID_RAW_IMU = mavutil.mavlink.MAVLINK_MSG_ID_RAW_IMU    # 27
# VFR_HUD is diagnostic only (never a guidance input, never watchdogged): it is
# the ONLY way to see what the FC actually did with the thrust we sent — its
# throttle % and climb rate. Measured 2026-08-10: with GUID_OPTIONS bit 3 set
# (thrust-as-thrust) and a commanded thrust of 0.20 held for 8.3 s while armed
# in GUIDED_NOGPS, the FC reported 0% throttle throughout — so the field is
# reaching ArduPilot correctly and being gated somewhere past the mode handler.
MSG_ID_VFR_HUD = mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD    # 74

# SET_ATTITUDE_TARGET type_mask: ignore the three body-rate fields (bits 0,1,2).
# ArduPilot GUIDED_NOGPS ignores them regardless, so attitude — yaw included — is
# carried entirely by the quaternion + thrust.
ATT_TARGET_IGNORE_RATES = 0x07

# Companion identity (quadguide is an onboard controller, not an autopilot).
MAV_TYPE_COMPANION = mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER  # 18
MAV_AUTOPILOT_NONE = mavutil.mavlink.MAV_AUTOPILOT_INVALID        # 8


def make_mav(system_id: int, component_id: int) -> mavlink2.MAVLink:
    """Build a codec-mode MAVLink2 object (no connection — parse/serialize only)."""
    mav = mavlink2.MAVLink(None, srcSystem=system_id, srcComponent=component_id)
    # Garbage bytes on a real UART must not raise — swallow framing/CRC errors.
    mav.robust_parsing = True
    return mav


def euler_to_quaternion(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """Aircraft ZYX euler (radians) → quaternion (w, x, y, z) in MAVLink order."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)

from __future__ import annotations
import struct

from quadguide.core.messages import AttitudeState, ControlCmd, IMUFrame
from quadguide.link.crsf import CRSF_RC_CHANNELS, CRSFFrame, build_frame, pack_channels
from quadguide.link.differentiator import AttitudeDifferentiator

_ROLL_PITCH_SCALE = 90.0   # ±90° maps to ±500 µs full deflection
_YAW_RATE_SCALE   = 200.0  # ±200 dps maps to ±500 µs full deflection

def us_to_ticks(us: float) -> int:
    return int(_clamp((us - 1500.0) * 8.0 / 5.0 + 992.0, 172.0, 1811.0))


def ticks_to_us(ticks: int) -> float:
    return (ticks - 992) * 5.0 / 8.0 + 1500.0

_STICK_HALF_RANGE_US = 400.0
_THROTTLE_MIN_US     = 1100.0
_THROTTLE_MAX_US     = 1900.0
_NEUTRAL_US = 1500.0
_FLIGHTMODE_US = [1165, 1295, 1425, 1555, 1685, 1815]
_MODE_INDEX = 0
_NEUTRAL      = 992   # 1500 µs — center
_THROTTLE_MIN = us_to_ticks(_THROTTLE_MIN_US)   # 1100 µs — minimum throttle
_ARM_HIGH     = 1811  # 2000 µs — armed
_ARM_LOW      = 172   # 1000 µs — disarmed


def decode_attitude(frame: CRSFFrame, diff: AttitudeDifferentiator
                    ) -> tuple[AttitudeState, IMUFrame]:
    pitch_raw, roll_raw, yaw_raw = struct.unpack(">hhh", frame.payload[:6])
    roll_rad  = roll_raw  * 1e-4
    pitch_rad = pitch_raw * 1e-4
    yaw_rad   = yaw_raw   * 1e-4
    rr, pr, yr = diff.update(roll_rad, pitch_rad, yaw_rad, frame.timestamp_ns)
    att = AttitudeState(
        timestamp_ns=frame.timestamp_ns,
        roll_rad=roll_rad, pitch_rad=pitch_rad, yaw_rad=yaw_rad,
        roll_rate_rps=rr, pitch_rate_rps=pr, yaw_rate_rps=yr,
    )
    imu = IMUFrame(
        timestamp_ns=frame.timestamp_ns,
        ax=0.0, ay=0.0, az=0.0,
        gx=rr, gy=pr, gz=yr,
    )
    return att, imu


def encode_rc(cmd: ControlCmd | None, armed: bool) -> bytes:
    if cmd is None:
        ch_roll, ch_pitch, ch_throttle, ch_yaw = (
            _NEUTRAL, _NEUTRAL, _THROTTLE_MIN, _NEUTRAL
        )
    else:
        ch_roll     = us_to_ticks(1500.0 + _clamp(cmd.roll_deg     /        _ROLL_PITCH_SCALE, -1, 1) * _STICK_HALF_RANGE_US)
        ch_pitch    = us_to_ticks(1500.0 + _clamp(cmd.pitch_deg    /        _ROLL_PITCH_SCALE, -1, 1) * _STICK_HALF_RANGE_US)
        ch_throttle = us_to_ticks(_THROTTLE_MIN_US + _clamp(cmd.throttle_norm, 0, 1) * (_THROTTLE_MAX_US - _THROTTLE_MIN_US))
        ch_yaw      = us_to_ticks(1500.0 + _clamp(cmd.yaw_rate_dps / _YAW_RATE_SCALE,  -1, 1) * _STICK_HALF_RANGE_US)
    channels = [
        ch_roll, ch_pitch, ch_throttle, ch_yaw,
        _ARM_HIGH if armed else _ARM_LOW,  # ch5 arm
        us_to_ticks(_FLIGHTMODE_US[_MODE_INDEX]),
        *([_NEUTRAL] * 10),                # ch7–16
    ]
    return build_frame(CRSF_RC_CHANNELS, pack_channels(channels))





def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

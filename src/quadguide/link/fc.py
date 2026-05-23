from __future__ import annotations
import struct
from dataclasses import dataclass

import math

from quadguide.core.messages import AttitudeState, ControlCmd, IMUFrame
from quadguide.link.crsf import CRSF_RC_CHANNELS, CRSFFrame, build_frame, pack_channels

_G_MPS2     = 9.80665
_DEG_TO_RAD = math.pi / 180.0

_MID_US = 1500.0  # neutral µs for unused channels


@dataclass(frozen=True)
class ChannelConfig:
    roll_min_us:              float
    roll_mid_us:              float
    roll_max_us:              float
    pitch_min_us:             float
    pitch_mid_us:             float
    pitch_max_us:             float
    throttle_min_us:          float
    throttle_max_us:          float
    yaw_min_us:               float
    yaw_mid_us:               float
    yaw_max_us:               float
    arm_disarmed_us:          float
    arm_armed_us:             float
    flight_mode_positions_us: tuple[float, ...]
    flight_mode_index:        int
    roll_pitch_scale:         float  # full-stick roll/pitch limit (deg in angle mode, dps in rate mode)
    yaw_rate_scale:           float  # full-stick yaw rate limit (dps)


def channel_config_from_cfg(cfg: dict) -> ChannelConfig:
    ch = cfg["link"]["channels"]
    return ChannelConfig(
        roll_min_us=float(ch["roll"]["min_us"]),
        roll_mid_us=float(ch["roll"]["mid_us"]),
        roll_max_us=float(ch["roll"]["max_us"]),
        pitch_min_us=float(ch["pitch"]["min_us"]),
        pitch_mid_us=float(ch["pitch"]["mid_us"]),
        pitch_max_us=float(ch["pitch"]["max_us"]),
        throttle_min_us=float(ch["throttle"]["min_us"]),
        throttle_max_us=float(ch["throttle"]["max_us"]),
        yaw_min_us=float(ch["yaw"]["min_us"]),
        yaw_mid_us=float(ch["yaw"]["mid_us"]),
        yaw_max_us=float(ch["yaw"]["max_us"]),
        arm_disarmed_us=float(ch["arm"]["disarmed_us"]),
        arm_armed_us=float(ch["arm"]["armed_us"]),
        flight_mode_positions_us=tuple(float(v) for v in ch["flight_mode"]["positions_us"]),
        flight_mode_index=int(ch["flight_mode"]["mode_index"]),
        roll_pitch_scale=float(ch["roll_pitch_scale"]),
        yaw_rate_scale=float(ch["yaw_rate_scale"]),
    )


def decode_attitude(
    frame: CRSFFrame,
    have_imu_frame: bool,
    last_gyro: tuple[float, float, float] | None = None,
) -> AttitudeState:
    """Decode a CRSF 0x1E attitude frame.

    Field order on the wire is pitch, roll, yaw (radians × 10000).
    Body-rate fields come from `last_gyro` (populated from 0x80 IMU) when
    `have_imu_frame` is True; otherwise zeroed until the first 0x80 arrives.
    """
    pitch_raw, roll_raw, yaw_raw = struct.unpack(">hhh", frame.payload[:6])
    roll_rad  = roll_raw  * 1e-4
    pitch_rad = pitch_raw * 1e-4
    yaw_rad   = yaw_raw   * 1e-4
    if have_imu_frame and last_gyro is not None:
        rr, pr, yr = last_gyro
    else:
        rr, pr, yr = 0.0, 0.0, 0.0
    return AttitudeState(
        timestamp_ns=frame.timestamp_ns,
        roll_rad=roll_rad, pitch_rad=pitch_rad, yaw_rad=yaw_rad,
        roll_rate_rps=rr, pitch_rate_rps=pr, yaw_rate_rps=yr,
    )


def decode_imu(frame: CRSFFrame) -> IMUFrame:
    """Decode a CRSF 0x80 IMU RAW frame (12 bytes, big-endian int16, NED body).

    Wire order: ax, ay, az, gx, gy, gz.
    Units on the wire: milli-G for accel, deci-deg/s for gyro.
    Returned units: m/s² and rad/s.
    """
    ax_r, ay_r, az_r, gx_r, gy_r, gz_r = struct.unpack(">hhhhhh", frame.payload[:12])
    return IMUFrame(
        timestamp_ns=frame.timestamp_ns,
        ax=(ax_r / 1000.0) * _G_MPS2,
        ay=(ay_r / 1000.0) * _G_MPS2,
        az=(az_r / 1000.0) * _G_MPS2,
        gx=(gx_r / 10.0) * _DEG_TO_RAD,
        gy=(gy_r / 10.0) * _DEG_TO_RAD,
        gz=(gz_r / 10.0) * _DEG_TO_RAD,
    )


def decode_flight_mode(frame: CRSFFrame) -> str:
    """Decode a CRSF 0x21 flight mode frame to a printable ASCII string.

    Wire format: null-terminated ASCII (≤16 bytes). Leading '*' when armed,
    optional satellite-count prefix when > 0.
    """
    raw = bytes(frame.payload)
    nul = raw.find(b"\x00")
    if nul >= 0:
        raw = raw[:nul]
    return raw.decode("ascii", errors="replace")


def encode_rc(cmd: ControlCmd | None, armed: bool, ch_cfg: ChannelConfig) -> bytes:
    """Build a CRSF RC_CHANNELS_PACKED frame.

    When disarmed: throttle=min, arm_ch=disarmed regardless of cmd.
    When armed:    throttle and attitude come from cmd (ground or guidance).
    """
    fm_us = ch_cfg.flight_mode_positions_us[ch_cfg.flight_mode_index]

    if not armed:
        channels_us = [
            ch_cfg.roll_mid_us, ch_cfg.pitch_mid_us,
            ch_cfg.throttle_min_us, ch_cfg.yaw_mid_us,
            ch_cfg.arm_disarmed_us, fm_us,
            *([_MID_US] * 10),
        ]
        return build_frame(CRSF_RC_CHANNELS, pack_channels(channels_us))

    if cmd is None:
        roll_us     = ch_cfg.roll_mid_us
        pitch_us    = ch_cfg.pitch_mid_us
        throttle_us = ch_cfg.throttle_min_us
        yaw_us      = ch_cfg.yaw_mid_us
    else:
        half_roll  = (ch_cfg.roll_max_us  - ch_cfg.roll_min_us)  / 2
        half_pitch = (ch_cfg.pitch_max_us - ch_cfg.pitch_min_us) / 2
        half_yaw   = (ch_cfg.yaw_max_us   - ch_cfg.yaw_min_us)   / 2
        roll_us     = ch_cfg.roll_mid_us  + _clamp(cmd.roll_deg     / ch_cfg.roll_pitch_scale, -1, 1) * half_roll
        pitch_us    = ch_cfg.pitch_mid_us + _clamp(cmd.pitch_deg    / ch_cfg.roll_pitch_scale, -1, 1) * half_pitch
        throttle_us = ch_cfg.throttle_min_us + _clamp(cmd.throttle_norm, 0, 1) * (ch_cfg.throttle_max_us - ch_cfg.throttle_min_us)
        yaw_us      = ch_cfg.yaw_mid_us   + _clamp(cmd.yaw_rate_dps / ch_cfg.yaw_rate_scale,  -1, 1) * half_yaw

    channels_us = [
        roll_us, pitch_us, throttle_us, yaw_us,
        ch_cfg.arm_armed_us, fm_us,
        *([_MID_US] * 10),
    ]
    return build_frame(CRSF_RC_CHANNELS, pack_channels(channels_us))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

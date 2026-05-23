import math
import struct
import pytest
from quadguide.link.crsf import (
    build_frame, CRSF_ATTITUDE, CRSF_FLIGHT_MODE, CRSF_IMU_RAW, CRSF_RC_CHANNELS,
    CRSFFrame, us_to_ticks, ticks_to_us,
)
from quadguide.link.differentiator import AttitudeDifferentiator
from quadguide.link.fc import (
    ChannelConfig, channel_config_from_cfg,
    decode_attitude, decode_flight_mode, decode_imu, encode_rc,
)
from quadguide.core.messages import AttitudeState, ControlCmd, IMUFrame

_G = 9.80665


@pytest.fixture
def ch_cfg() -> ChannelConfig:
    """madflight channel calibration — rate mode values."""
    return ChannelConfig(
        roll_min_us=1100.0, roll_mid_us=1500.0, roll_max_us=1900.0,
        pitch_min_us=1100.0, pitch_mid_us=1500.0, pitch_max_us=1900.0,
        throttle_min_us=1100.0, throttle_max_us=1900.0,
        yaw_min_us=1100.0, yaw_mid_us=1500.0, yaw_max_us=1900.0,
        arm_disarmed_us=988.0, arm_armed_us=2012.0,
        flight_mode_positions_us=(1165.0, 1295.0, 1425.0, 1555.0, 1685.0, 1815.0),
        flight_mode_index=0,
        roll_pitch_scale=60.0,
        yaw_rate_scale=160.0,
    )


# ── us_to_ticks / ticks_to_us ────────────────────────────────────────────────

def test_us_to_ticks_center():
    assert us_to_ticks(1500.0) == 992

def test_us_to_ticks_arm_disarmed():
    assert us_to_ticks(988.0) == 172

def test_us_to_ticks_arm_armed():
    assert us_to_ticks(2012.0) == 1811

def test_us_to_ticks_clamps_high():
    assert us_to_ticks(3000.0) == 1811

def test_us_to_ticks_clamps_low():
    assert us_to_ticks(0.0) == 172

def test_ticks_to_us_center():
    assert ticks_to_us(992) == pytest.approx(1500.0)

def test_ticks_to_us_arm_disarmed():
    assert ticks_to_us(172) == pytest.approx(988.0, abs=1.0)

def test_ticks_to_us_arm_armed():
    assert ticks_to_us(1811) == pytest.approx(2012.0, abs=1.0)


# ── decode_attitude ──────────────────────────────────────────────────────────

def _make_attitude_frame(pitch_raw: int, roll_raw: int, yaw_raw: int) -> CRSFFrame:
    payload = struct.pack(">hhh", pitch_raw, roll_raw, yaw_raw)
    return CRSFFrame(type=CRSF_ATTITUDE, payload=payload, timestamp_ns=int(1e9))


def test_decode_attitude_angles():
    frame = _make_attitude_frame(pitch_raw=1000, roll_raw=500, yaw_raw=-200)
    diff = AttitudeDifferentiator(alpha=1.0)
    att = decode_attitude(frame, diff, have_imu_frame=False)
    assert isinstance(att, AttitudeState)
    assert att.pitch_rad == pytest.approx(0.1,   rel=1e-5)
    assert att.roll_rad  == pytest.approx(0.05,  rel=1e-5)
    assert att.yaw_rad   == pytest.approx(-0.02, rel=1e-5)


def test_decode_attitude_fallback_first_call_rates_zero():
    frame = _make_attitude_frame(1000, 500, 200)
    diff = AttitudeDifferentiator(alpha=1.0)
    att = decode_attitude(frame, diff, have_imu_frame=False)
    assert att.roll_rate_rps  == pytest.approx(0.0, abs=1e-9)
    assert att.pitch_rate_rps == pytest.approx(0.0, abs=1e-9)
    assert att.yaw_rate_rps   == pytest.approx(0.0, abs=1e-9)


def test_decode_attitude_uses_last_gyro_when_have_imu():
    frame = _make_attitude_frame(0, 0, 0)
    diff = AttitudeDifferentiator(alpha=1.0)
    att = decode_attitude(frame, diff, have_imu_frame=True, last_gyro=(0.5, -0.25, 1.0))
    assert att.roll_rate_rps  == pytest.approx(0.5,   rel=1e-9)
    assert att.pitch_rate_rps == pytest.approx(-0.25, rel=1e-9)
    assert att.yaw_rate_rps   == pytest.approx(1.0,   rel=1e-9)


def test_decode_attitude_falls_back_when_have_imu_but_no_gyro_yet():
    frame = _make_attitude_frame(0, 0, 0)
    diff = AttitudeDifferentiator(alpha=1.0)
    att = decode_attitude(frame, diff, have_imu_frame=True, last_gyro=None)
    assert att.roll_rate_rps == pytest.approx(0.0, abs=1e-9)


# ── decode_imu ───────────────────────────────────────────────────────────────

def _make_imu_frame(ax_r, ay_r, az_r, gx_r, gy_r, gz_r) -> CRSFFrame:
    payload = struct.pack(">hhhhhh", ax_r, ay_r, az_r, gx_r, gy_r, gz_r)
    return CRSFFrame(type=CRSF_IMU_RAW, payload=payload, timestamp_ns=int(2e9))


def test_decode_imu_returns_imuframe():
    frame = _make_imu_frame(0, 0, 1000, 0, 0, 0)
    imu = decode_imu(frame)
    assert isinstance(imu, IMUFrame)
    assert imu.timestamp_ns == int(2e9)


def test_decode_imu_accel_milli_g_to_mps2():
    # 1000 milli-G = 1 G = 9.80665 m/s²
    frame = _make_imu_frame(1000, -500, 250, 0, 0, 0)
    imu = decode_imu(frame)
    assert imu.ax == pytest.approx(_G,          rel=1e-6)
    assert imu.ay == pytest.approx(-0.5 * _G,   rel=1e-6)
    assert imu.az == pytest.approx(0.25 * _G,   rel=1e-6)


def test_decode_imu_gyro_deci_dps_to_rad_per_s():
    # gx = 1800 (deci-deg/s) = 180 deg/s = π rad/s
    frame = _make_imu_frame(0, 0, 0, 1800, -900, 100)
    imu = decode_imu(frame)
    assert imu.gx == pytest.approx(math.pi,         rel=1e-5)
    assert imu.gy == pytest.approx(-math.pi / 2,    rel=1e-5)
    assert imu.gz == pytest.approx(10 * math.pi / 180, rel=1e-5)


# ── decode_flight_mode ───────────────────────────────────────────────────────

def test_decode_flight_mode_strips_null():
    frame = CRSFFrame(type=CRSF_FLIGHT_MODE, payload=b"ANGLE\x00", timestamp_ns=0)
    assert decode_flight_mode(frame) == "ANGLE"


def test_decode_flight_mode_armed_prefix():
    frame = CRSFFrame(type=CRSF_FLIGHT_MODE, payload=b"*ACRO\x00", timestamp_ns=0)
    assert decode_flight_mode(frame) == "*ACRO"


# ── encode_rc ────────────────────────────────────────────────────────────────

def _decode_channels(frame_bytes: bytes) -> list[int]:
    payload = frame_bytes[3:25]
    bits = int.from_bytes(payload, "little")
    return [(bits >> (i * 11)) & 0x7FF for i in range(16)]


def test_encode_rc_none_center_roll_pitch_yaw(ch_cfg):
    channels = _decode_channels(encode_rc(None, armed=False, ch_cfg=ch_cfg))
    assert channels[0] == 992
    assert channels[1] == 992
    assert channels[3] == 992


def test_encode_rc_none_min_throttle(ch_cfg):
    channels = _decode_channels(encode_rc(None, armed=False, ch_cfg=ch_cfg))
    assert channels[2] == us_to_ticks(1100.0)


def test_encode_rc_none_disarmed(ch_cfg):
    channels = _decode_channels(encode_rc(None, armed=False, ch_cfg=ch_cfg))
    assert channels[4] == 172


def test_encode_rc_armed_sets_ch5(ch_cfg):
    channels = _decode_channels(encode_rc(None, armed=True, ch_cfg=ch_cfg))
    assert channels[4] == 1811


def test_encode_rc_ch6_flight_mode_pos0(ch_cfg):
    channels = _decode_channels(encode_rc(None, armed=False, ch_cfg=ch_cfg))
    assert channels[5] == us_to_ticks(1165.0)


def test_encode_rc_ch7_to_ch16_neutral(ch_cfg):
    channels = _decode_channels(encode_rc(None, armed=False, ch_cfg=ch_cfg))
    for i in range(6, 16):
        assert channels[i] == 992


def test_encode_rc_roll_full_right(ch_cfg):
    cmd = ControlCmd(timestamp_ns=0, roll_deg=ch_cfg.roll_pitch_scale, pitch_deg=0.0,
                     yaw_rate_dps=0.0, throttle_norm=0.0)
    channels = _decode_channels(encode_rc(cmd, armed=False, ch_cfg=ch_cfg))
    assert channels[0] == us_to_ticks(1900.0)


def test_encode_rc_roll_full_left(ch_cfg):
    cmd = ControlCmd(timestamp_ns=0, roll_deg=-ch_cfg.roll_pitch_scale, pitch_deg=0.0,
                     yaw_rate_dps=0.0, throttle_norm=0.0)
    channels = _decode_channels(encode_rc(cmd, armed=False, ch_cfg=ch_cfg))
    assert channels[0] == us_to_ticks(1100.0)


def test_encode_rc_throttle_half(ch_cfg):
    cmd = ControlCmd(timestamp_ns=0, roll_deg=0.0, pitch_deg=0.0,
                     yaw_rate_dps=0.0, throttle_norm=0.5)
    channels = _decode_channels(encode_rc(cmd, armed=False, ch_cfg=ch_cfg))
    assert channels[2] == 992


def test_encode_rc_throttle_full(ch_cfg):
    cmd = ControlCmd(timestamp_ns=0, roll_deg=0.0, pitch_deg=0.0,
                     yaw_rate_dps=0.0, throttle_norm=1.0)
    channels = _decode_channels(encode_rc(cmd, armed=False, ch_cfg=ch_cfg))
    assert channels[2] == us_to_ticks(1900.0)


def test_encode_rc_valid_crsf_frame(ch_cfg):
    data = encode_rc(None, armed=False, ch_cfg=ch_cfg)
    assert data[0] == 0xC8
    assert data[2] == CRSF_RC_CHANNELS
    assert len(data) == 26


# ── channel_config_from_cfg ──────────────────────────────────────────────────

def test_channel_config_from_cfg():
    cfg = {
        "link": {
            "channels": {
                "roll":        {"min_us": 1100, "mid_us": 1500, "max_us": 1900},
                "pitch":       {"min_us": 1100, "mid_us": 1500, "max_us": 1900},
                "throttle":    {"min_us": 1100, "max_us": 1900},
                "yaw":         {"min_us": 1100, "mid_us": 1500, "max_us": 1900},
                "arm":         {"disarmed_us": 988, "armed_us": 2012},
                "flight_mode": {"mode_index": 0,
                                "positions_us": [1165, 1295, 1425, 1555, 1685, 1815]},
                "roll_pitch_scale": 60.0,
                "yaw_rate_scale":  160.0,
            }
        }
    }
    c = channel_config_from_cfg(cfg)
    assert c.roll_min_us == 1100.0
    assert c.throttle_min_us == 1100.0
    assert c.arm_disarmed_us == 988.0
    assert c.arm_armed_us == 2012.0
    assert c.flight_mode_positions_us[0] == 1165.0
    assert c.flight_mode_index == 0
    assert c.roll_pitch_scale == 60.0
    assert c.yaw_rate_scale == 160.0

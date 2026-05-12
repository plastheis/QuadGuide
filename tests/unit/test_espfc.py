import math
import struct
import pytest
from quadguide.link.crsf import build_frame, pack_channels, CRSF_ATTITUDE, CRSF_RC_CHANNELS, CRSFFrame
from quadguide.link.differentiator import AttitudeDifferentiator
from quadguide.link.espfc import decode_attitude, encode_rc, us_to_ticks, ticks_to_us
from quadguide.core.messages import AttitudeState, ControlCmd, IMUFrame


# ── us_to_ticks / ticks_to_us ─────────────────────────────────────────────

def test_us_to_ticks_center():
    assert us_to_ticks(1500.0) == 992

def test_us_to_ticks_full_high():
    assert us_to_ticks(2000.0) == 1792

def test_us_to_ticks_full_low():
    assert us_to_ticks(1000.0) == 192

def test_us_to_ticks_clamps_high():
    assert us_to_ticks(3000.0) == 1811

def test_us_to_ticks_clamps_low():
    assert us_to_ticks(0.0) == 172

def test_ticks_to_us_center():
    assert ticks_to_us(992) == pytest.approx(1500.0)

def test_ticks_to_us_full_high():
    assert ticks_to_us(1792) == pytest.approx(2000.0)

def test_ticks_to_us_full_low():
    assert ticks_to_us(192) == pytest.approx(1000.0)


# ── decode_attitude ───────────────────────────────────────────────────────

def _make_attitude_frame(pitch_raw: int, roll_raw: int, yaw_raw: int) -> CRSFFrame:
    payload = struct.pack(">hhh", pitch_raw, roll_raw, yaw_raw)
    frame = build_frame(CRSF_ATTITUDE, payload)
    # Build CRSFFrame directly (timestamp doesn't matter for unit test)
    return CRSFFrame(type=CRSF_ATTITUDE, payload=payload, timestamp_ns=int(1e9))


def test_decode_attitude_angles():
    frame = _make_attitude_frame(pitch_raw=1000, roll_raw=500, yaw_raw=-200)
    diff = AttitudeDifferentiator(alpha=1.0)
    att, imu = decode_attitude(frame, diff)
    assert isinstance(att, AttitudeState)
    assert att.pitch_rad == pytest.approx(0.1,   rel=1e-5)   # 1000 * 1e-4
    assert att.roll_rad  == pytest.approx(0.05,  rel=1e-5)   # 500  * 1e-4
    assert att.yaw_rad   == pytest.approx(-0.02, rel=1e-5)   # -200 * 1e-4


def test_decode_attitude_first_call_body_rates_zero():
    frame = _make_attitude_frame(1000, 500, 200)
    diff = AttitudeDifferentiator(alpha=1.0)
    att, _ = decode_attitude(frame, diff)
    assert att.roll_rate_rps  == pytest.approx(0.0, abs=1e-9)
    assert att.pitch_rate_rps == pytest.approx(0.0, abs=1e-9)
    assert att.yaw_rate_rps   == pytest.approx(0.0, abs=1e-9)


def test_decode_attitude_imu_gyro_matches_att_rates():
    diff = AttitudeDifferentiator(alpha=1.0)
    decode_attitude(_make_attitude_frame(0, 0, 0), diff)
    # Second frame: 1 second later, 1 rad change in roll
    frame2 = CRSFFrame(type=CRSF_ATTITUDE,
                       payload=struct.pack(">hhh", 0, 10000, 0),
                       timestamp_ns=int(2e9))
    att, imu = decode_attitude(frame2, diff)
    assert imu.gx == pytest.approx(att.roll_rate_rps,  rel=1e-5)
    assert imu.gy == pytest.approx(att.pitch_rate_rps, rel=1e-5)
    assert imu.gz == pytest.approx(att.yaw_rate_rps,   rel=1e-5)


def test_decode_attitude_imu_accel_zero():
    frame = _make_attitude_frame(0, 0, 0)
    diff = AttitudeDifferentiator(alpha=1.0)
    _, imu = decode_attitude(frame, diff)
    assert isinstance(imu, IMUFrame)
    assert imu.ax == 0.0
    assert imu.ay == 0.0
    assert imu.az == 0.0


# ── encode_rc ─────────────────────────────────────────────────────────────

def _decode_channels(frame_bytes: bytes) -> list[int]:
    """Helper: extract 16 CRSF channel values from a built RC_CHANNELS frame."""
    payload = frame_bytes[3:25]   # skip sync(1), len(1), type(1); payload is 22 bytes
    bits = int.from_bytes(payload, "little")
    return [(bits >> (i * 11)) & 0x7FF for i in range(16)]


def test_encode_rc_none_cmd_center_roll_pitch_yaw():
    channels = _decode_channels(encode_rc(None, armed=False))
    assert channels[0] == 992   # ch1 roll  — neutral
    assert channels[1] == 992   # ch2 pitch — neutral
    assert channels[3] == 992   # ch4 yaw   — neutral


def test_encode_rc_none_cmd_min_throttle():
    channels = _decode_channels(encode_rc(None, armed=False))
    assert channels[2] == 172   # ch3 throttle — minimum


def test_encode_rc_none_cmd_disarmed():
    channels = _decode_channels(encode_rc(None, armed=False))
    assert channels[4] == 172   # ch5 — disarmed


def test_encode_rc_armed_sets_ch5_max():
    channels = _decode_channels(encode_rc(None, armed=True))
    assert channels[4] == 1811  # ch5 — armed


def test_encode_rc_ch6_to_ch16_neutral():
    channels = _decode_channels(encode_rc(None, armed=False))
    for i in range(5, 16):
        assert channels[i] == 992


def test_encode_rc_roll_right():
    # roll_deg = +90 → 2000 µs → 1792 ticks
    cmd = ControlCmd(timestamp_ns=0, roll_deg=90.0, pitch_deg=0.0,
                     yaw_rate_dps=0.0, throttle_norm=0.0)
    channels = _decode_channels(encode_rc(cmd, armed=False))
    assert channels[0] == 1792


def test_encode_rc_roll_left():
    cmd = ControlCmd(timestamp_ns=0, roll_deg=-90.0, pitch_deg=0.0,
                     yaw_rate_dps=0.0, throttle_norm=0.0)
    channels = _decode_channels(encode_rc(cmd, armed=False))
    assert channels[0] == 192


def test_encode_rc_throttle_half():
    # throttle_norm=0.5 → 1500 µs → 992 ticks
    cmd = ControlCmd(timestamp_ns=0, roll_deg=0.0, pitch_deg=0.0,
                     yaw_rate_dps=0.0, throttle_norm=0.5)
    channels = _decode_channels(encode_rc(cmd, armed=False))
    assert channels[2] == 992


def test_encode_rc_throttle_full():
    cmd = ControlCmd(timestamp_ns=0, roll_deg=0.0, pitch_deg=0.0,
                     yaw_rate_dps=0.0, throttle_norm=1.0)
    channels = _decode_channels(encode_rc(cmd, armed=False))
    assert channels[2] == 1792


def test_encode_rc_returns_valid_crsf_frame():
    data = encode_rc(None, armed=False)
    assert data[0] == 0xC8              # sync
    assert data[2] == CRSF_RC_CHANNELS  # type
    assert len(data) == 26              # sync+len+type+22+crc

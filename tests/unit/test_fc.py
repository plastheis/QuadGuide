import math
import pytest
from pymavlink import mavutil

from quadguide.link.mavlink_codec import (
    ATT_TARGET_IGNORE_RATES, MSG_ID_ATTITUDE, euler_to_quaternion, make_mav,
)
from quadguide.link.fc import (
    decode_attitude, decode_heartbeat, decode_imu,
    encode_arm, encode_attitude_target, encode_heartbeat,
    encode_set_message_interval, encode_set_mode,
)
from quadguide.core.messages import AttitudeState, ControlCmd, IMUFrame

_G = 9.80665


@pytest.fixture
def mav():
    return make_mav(1, 191)


# ── decode_attitude ──────────────────────────────────────────────────────────

def test_decode_attitude_maps_angles_and_rates(mav):
    msg = mav.attitude_encode(0, 0.05, 0.1, -0.02, 0.5, -0.25, 1.0)
    att = decode_attitude(msg, recv_ns=123)
    assert isinstance(att, AttitudeState)
    assert att.timestamp_ns == 123
    assert att.roll_rad == pytest.approx(0.05)
    assert att.pitch_rad == pytest.approx(0.1)
    assert att.yaw_rad == pytest.approx(-0.02)
    assert att.roll_rate_rps == pytest.approx(0.5)
    assert att.pitch_rate_rps == pytest.approx(-0.25)
    assert att.yaw_rate_rps == pytest.approx(1.0)


# ── decode_imu ───────────────────────────────────────────────────────────────

def test_decode_imu_scales_accel_mg_to_mps2(mav):
    # accel in milli-g: 1000 mG = 1 g = 9.80665 m/s²
    msg = mav.raw_imu_encode(0, 1000, -500, 250, 0, 0, 0, 0, 0, 0)
    imu = decode_imu(msg, recv_ns=7)
    assert isinstance(imu, IMUFrame)
    assert imu.timestamp_ns == 7
    assert imu.ax == pytest.approx(_G, rel=1e-4)
    assert imu.ay == pytest.approx(-0.5 * _G, rel=1e-4)
    assert imu.az == pytest.approx(0.25 * _G, rel=1e-4)


def test_decode_imu_scales_gyro_mrad_to_rad(mav):
    # gyro in milli-rad/s: 1571 mrad/s ≈ π/2 rad/s
    msg = mav.raw_imu_encode(0, 0, 0, 0, 1571, -785, 100, 0, 0, 0)
    imu = decode_imu(msg, recv_ns=0)
    assert imu.gx == pytest.approx(1.571, rel=1e-4)
    assert imu.gy == pytest.approx(-0.785, rel=1e-4)
    assert imu.gz == pytest.approx(0.1, rel=1e-4)


def test_decode_imu_accepts_scaled_imu2(mav):
    msg = mav.scaled_imu2_encode(0, 0, 0, 1000, 0, 0, 0, 0, 0, 0)
    imu = decode_imu(msg, recv_ns=0)
    assert imu.az == pytest.approx(_G, rel=1e-4)


# ── decode_heartbeat ─────────────────────────────────────────────────────────

def test_decode_heartbeat_armed(mav):
    msg = mav.heartbeat_encode(
        mavutil.mavlink.MAV_TYPE_QUADROTOR,
        mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
        mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED, 20, 0,
    )
    armed, mode = decode_heartbeat(msg)
    assert armed is True
    assert mode == 20


def test_decode_heartbeat_disarmed(mav):
    msg = mav.heartbeat_encode(
        mavutil.mavlink.MAV_TYPE_QUADROTOR,
        mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA, 0, 0, 0,
    )
    armed, mode = decode_heartbeat(msg)
    assert armed is False
    assert mode == 0


def _roundtrip(data: bytes):
    """Parse packed MAVLink2 bytes back into a message via a fresh codec."""
    rx = make_mav(1, 1)
    out = None
    for b in data:
        m = rx.parse_char(bytes([b]))
        if m is not None:
            out = m
    return out


# ── encode_attitude_target ───────────────────────────────────────────────────

def test_encode_attitude_target_mask_thrust_and_quaternion(mav):
    cmd = ControlCmd(0, roll_deg=0.0, pitch_deg=0.0, yaw_rate_dps=0.0, throttle_norm=0.4)
    msg = _roundtrip(encode_attitude_target(
        mav, cmd, yaw_hold=0.0, target_sys=1, target_comp=1,
        max_roll_deg=35.0, max_pitch_deg=35.0, now_ms=0))
    assert msg.get_type() == "SET_ATTITUDE_TARGET"
    assert msg.type_mask == ATT_TARGET_IGNORE_RATES
    assert msg.thrust == pytest.approx(0.4)
    assert list(msg.q) == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)


def test_encode_attitude_target_clamps_roll_to_limit(mav):
    cmd = ControlCmd(0, roll_deg=90.0, pitch_deg=0.0, yaw_rate_dps=0.0, throttle_norm=0.0)
    msg = _roundtrip(encode_attitude_target(
        mav, cmd, 0.0, 1, 1, max_roll_deg=35.0, max_pitch_deg=35.0, now_ms=0))
    expected = euler_to_quaternion(math.radians(35.0), 0.0, 0.0)
    assert list(msg.q) == pytest.approx(list(expected), abs=1e-5)


def test_encode_attitude_target_clamps_thrust(mav):
    cmd = ControlCmd(0, 0.0, 0.0, 0.0, throttle_norm=2.0)
    msg = _roundtrip(encode_attitude_target(mav, cmd, 0.0, 1, 1, 35.0, 35.0, 0))
    assert msg.thrust == pytest.approx(1.0)


def test_encode_attitude_target_none_is_level_zero_thrust(mav):
    msg = _roundtrip(encode_attitude_target(mav, None, 0.0, 1, 1, 35.0, 35.0, 0))
    assert msg.thrust == pytest.approx(0.0)
    assert list(msg.q) == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)


def test_encode_attitude_target_bakes_yaw_hold(mav):
    cmd = ControlCmd(0, 0.0, 0.0, 0.0, 0.0)
    msg = _roundtrip(encode_attitude_target(
        mav, cmd, yaw_hold=math.pi / 2, target_sys=1, target_comp=1,
        max_roll_deg=35.0, max_pitch_deg=35.0, now_ms=0))
    assert list(msg.q) == pytest.approx(
        [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], abs=1e-6)


# ── encode_arm ───────────────────────────────────────────────────────────────

def test_encode_arm_arms(mav):
    msg = _roundtrip(encode_arm(mav, True, 1, 1))
    assert msg.get_type() == "COMMAND_LONG"
    assert msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    assert msg.param1 == pytest.approx(1.0)


def test_encode_arm_disarms(mav):
    msg = _roundtrip(encode_arm(mav, False, 1, 1))
    assert msg.param1 == pytest.approx(0.0)


# ── encode_set_mode ──────────────────────────────────────────────────────────

def test_encode_set_mode_sets_custom_mode(mav):
    msg = _roundtrip(encode_set_mode(mav, 9, 1, 1))  # 9 = ArduCopter LAND
    assert msg.get_type() == "COMMAND_LONG"
    assert msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE
    assert msg.param1 == pytest.approx(
        float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED))
    assert msg.param2 == pytest.approx(9.0)


# ── encode_set_message_interval ──────────────────────────────────────────────

def test_encode_set_message_interval_converts_hz_to_us(mav):
    msg = _roundtrip(encode_set_message_interval(mav, MSG_ID_ATTITUDE, 50.0, 1, 1))
    assert msg.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL
    assert msg.param1 == pytest.approx(MSG_ID_ATTITUDE)
    assert msg.param2 == pytest.approx(20000.0)  # 1e6 / 50


# ── encode_heartbeat ─────────────────────────────────────────────────────────

def test_encode_heartbeat_is_onboard_controller(mav):
    msg = _roundtrip(encode_heartbeat(mav))
    assert msg.get_type() == "HEARTBEAT"
    assert msg.type == mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER

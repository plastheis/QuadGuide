import math
import pytest
from pymavlink import mavutil

from quadguide.link.mavlink_codec import make_mav
from quadguide.link.fc import decode_attitude, decode_heartbeat, decode_imu
from quadguide.core.messages import AttitudeState, IMUFrame

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

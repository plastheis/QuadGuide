import asyncio
import logging

import pytest
from pymavlink import mavutil

from quadguide.core.messages import AttitudeState, IMUFrame
from quadguide.link.mavlink_codec import make_mav
from quadguide.link.worker import _ArmController, _LinkState, _rx_loop, latch_yaw


# ── _ArmController ───────────────────────────────────────────────────────────

def test_arm_controller_silent_when_steady_disarmed():
    arm = _ArmController(retry_count=3, resend_every_ticks=2)
    assert arm.on_arm_state(False) is None
    assert arm.on_arm_state(False) is None


def test_arm_controller_emits_arm_on_rising_edge():
    arm = _ArmController(retry_count=3, resend_every_ticks=2)
    assert arm.on_arm_state(True) is True


def test_arm_controller_emits_disarm_on_falling_edge():
    arm = _ArmController(retry_count=3, resend_every_ticks=2)
    arm.on_arm_state(True)
    arm.on_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
               mavutil.mavlink.MAV_RESULT_ACCEPTED)
    assert arm.on_arm_state(False) is False


def test_arm_controller_resends_until_retries_exhausted():
    arm = _ArmController(retry_count=2, resend_every_ticks=2)
    assert arm.on_arm_state(True) is True   # edge
    assert arm.on_arm_state(True) is None   # tick 1
    assert arm.on_arm_state(True) is True   # tick 2 → resend (retries 2→1)
    assert arm.on_arm_state(True) is None   # tick 1
    assert arm.on_arm_state(True) is True   # tick 2 → resend (retries 1→0)
    assert arm.on_arm_state(True) is None   # exhausted
    assert arm.on_arm_state(True) is None


def test_arm_controller_stops_after_ack():
    arm = _ArmController(retry_count=5, resend_every_ticks=2)
    assert arm.on_arm_state(True) is True
    arm.on_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
               mavutil.mavlink.MAV_RESULT_ACCEPTED)
    assert arm.on_arm_state(True) is None
    assert arm.on_arm_state(True) is None


def test_arm_controller_ignores_unrelated_ack():
    arm = _ArmController(retry_count=5, resend_every_ticks=1)
    arm.on_arm_state(True)
    arm.on_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
               mavutil.mavlink.MAV_RESULT_ACCEPTED)
    assert arm.on_arm_state(True) is True   # still pending → resends


# ── latch_yaw ────────────────────────────────────────────────────────────────

def test_latch_yaw_latches_on_arm_edge():
    assert latch_yaw(armed=True, prev_armed=False, last_yaw=0.7, held=0.0) == pytest.approx(0.7)


def test_latch_yaw_holds_between_ticks():
    assert latch_yaw(True, True, 1.2, 0.7) == pytest.approx(0.7)


def test_latch_yaw_zero_when_no_attitude_yet():
    assert latch_yaw(True, False, None, 0.0) == 0.0


def test_latch_yaw_keeps_held_while_disarmed():
    assert latch_yaw(False, True, 0.9, 0.7) == pytest.approx(0.7)


# ── _LinkState ───────────────────────────────────────────────────────────────

def test_link_state_defaults():
    s = _LinkState()
    assert s.have_heartbeat is False
    assert s.have_raw_imu is False
    assert s.last_yaw is None
    assert s.target_system == 0
    assert s.target_component == 0


# ── _rx_loop ──────────────────────────────────────────────────────────────────

class _FakeSerial:
    """Async byte-stream stub that yields a fixed sequence once."""
    def __init__(self, data: bytes):
        self._data = data

    async def read_stream(self):
        for b in self._data:
            yield b


class _FakeBus:
    def __init__(self):
        self.published: list[tuple[str, object]] = []

    def publish(self, topic, msg):
        self.published.append((topic, msg))

    def latest(self, topic):
        return None


def _enc(fn) -> bytes:
    """Pack a message on a fresh FC-side codec (sys=1, comp=1)."""
    m = make_mav(1, 1)
    return fn(m)


def _run_rx(data: bytes):
    serial = _FakeSerial(data)
    bus = _FakeBus()
    state = _LinkState()
    arm = _ArmController(retry_count=5, resend_every_ticks=25)
    log = logging.getLogger("test")
    mav = make_mav(1, 191)
    asyncio.run(_rx_loop(serial, mav, state, bus, arm, log))
    return bus, state, arm


def test_rx_publishes_attitude_and_tracks_yaw():
    data = _enc(lambda m: m.attitude_encode(0, 0.05, 0.1, -0.02, 0.0, 0.0, 0.0).pack(m))
    bus, state, _ = _run_rx(data)
    atts = [msg for t, msg in bus.published if t == "fc/attitude"]
    assert len(atts) == 1 and isinstance(atts[0], AttitudeState)
    assert atts[0].roll_rad == pytest.approx(0.05)
    assert state.last_yaw == pytest.approx(-0.02)


def test_rx_publishes_imu_from_raw_imu():
    data = _enc(lambda m: m.raw_imu_encode(0, 0, 0, 1000, 0, 0, 0, 0, 0, 0).pack(m))
    bus, state, _ = _run_rx(data)
    imus = [msg for t, msg in bus.published if t == "fc/imu"]
    assert len(imus) == 1 and isinstance(imus[0], IMUFrame)
    assert state.have_raw_imu is True


def test_rx_scaled_imu2_is_fallback_only_until_raw_imu():
    m1 = _enc(lambda m: m.scaled_imu2_encode(0, 0, 0, 1000, 0, 0, 0, 0, 0, 0).pack(m))
    m2 = _enc(lambda m: m.raw_imu_encode(0, 0, 0, 1000, 0, 0, 0, 0, 0, 0).pack(m))
    m3 = _enc(lambda m: m.scaled_imu2_encode(0, 0, 0, 1000, 0, 0, 0, 0, 0, 0).pack(m))
    bus, state, _ = _run_rx(m1 + m2 + m3)
    imus = [msg for t, msg in bus.published if t == "fc/imu"]
    assert len(imus) == 2  # scaled (fallback) + raw; the post-raw scaled is ignored


def test_rx_heartbeat_learns_fc_target_ids():
    data = _enc(lambda m: m.heartbeat_encode(
        mavutil.mavlink.MAV_TYPE_QUADROTOR,
        mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA, 0, 0, 0).pack(m))
    _, state, _ = _run_rx(data)
    assert state.have_heartbeat is True
    assert state.target_system == 1
    assert state.target_component == 1


def test_rx_ignores_gcs_heartbeat():
    data = _enc(lambda m: m.heartbeat_encode(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0).pack(m))
    _, state, _ = _run_rx(data)
    assert state.have_heartbeat is False


def test_rx_command_ack_acks_pending_arm():
    arm = _ArmController(retry_count=5, resend_every_ticks=25)
    arm.on_arm_state(True)  # rising edge → pending, not acked
    data = _enc(lambda m: m.command_ack_encode(
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        mavutil.mavlink.MAV_RESULT_ACCEPTED).pack(m))
    serial = _FakeSerial(data)
    bus = _FakeBus()
    state = _LinkState()
    log = logging.getLogger("test")
    mav = make_mav(1, 191)
    asyncio.run(_rx_loop(serial, mav, state, bus, arm, log))
    assert arm.on_arm_state(True) is None  # acked → nothing more to send

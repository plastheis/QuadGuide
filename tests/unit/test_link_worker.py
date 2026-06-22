import pytest
from pymavlink import mavutil

from quadguide.link.worker import _ArmController, _LinkState, latch_yaw


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

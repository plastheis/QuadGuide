from __future__ import annotations
import asyncio
import logging

from pymavlink import mavutil

from quadguide.core.clock import monotonic_ns
from quadguide.link.fc import decode_attitude, decode_heartbeat, decode_imu


class _LinkState:
    """Per-connection mutable state shared between the RX/TX loops."""

    def __init__(self) -> None:
        self.target_system: int = 0
        self.target_component: int = 0
        self.have_heartbeat: bool = False
        self.have_raw_imu: bool = False
        self.last_yaw: float | None = None
        self.fc_armed: bool = False
        self.fc_mode: int = -1


class _ArmController:
    """Edge-triggered MAVLink arm/disarm with bounded retransmits until ACK.

    Call `on_arm_state(desired)` once per TX tick with the latest arm/cmd state.
    It returns the arm value (True/False) to transmit this tick, or None to send
    nothing. On a new edge it emits immediately, then re-emits every
    `resend_every_ticks` ticks up to `retry_count` times until `on_ack` confirms.
    """

    def __init__(self, retry_count: int, resend_every_ticks: int) -> None:
        self._desired: bool = False          # assume disarmed at startup; no spurious cmd
        self._acked: bool = True
        self._retries_left: int = 0
        self._ticks: int = 0
        self._retry_count = retry_count
        self._resend_every = resend_every_ticks

    def on_arm_state(self, desired: bool) -> bool | None:
        if desired != self._desired:
            self._desired = desired
            self._acked = False
            self._retries_left = self._retry_count
            self._ticks = 0
            return desired
        if self._acked or self._retries_left <= 0:
            return None
        self._ticks += 1
        if self._ticks >= self._resend_every:
            self._ticks = 0
            self._retries_left -= 1
            return self._desired
        return None

    def on_ack(self, command: int, result: int) -> None:
        if (command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
                and result == mavutil.mavlink.MAV_RESULT_ACCEPTED):
            self._acked = True


def latch_yaw(
    armed: bool, prev_armed: bool, last_yaw: float | None, held: float
) -> float:
    """Hold-heading: latch the current yaw on the disarmed->armed edge; else keep."""
    if armed and not prev_armed:
        return last_yaw if last_yaw is not None else 0.0
    return held


def _on_heartbeat(msg, state: _LinkState, log: logging.Logger) -> None:
    """Learn FC ids on the first heartbeat; log arm/mode transitions."""
    if not state.have_heartbeat:
        state.target_system = msg.get_srcSystem()
        state.target_component = msg.get_srcComponent()
        state.have_heartbeat = True
        log.info("FC HEARTBEAT: sys=%d comp=%d", state.target_system, state.target_component)
    armed, mode = decode_heartbeat(msg)
    if armed != state.fc_armed:
        log.info("FC arm state → %s", "ARMED" if armed else "DISARMED")
        state.fc_armed = armed
    if mode != state.fc_mode:
        log.info("FC custom_mode → %d", mode)
        state.fc_mode = mode


async def _rx_loop(serial, mav, state: _LinkState, bus,
                   arm_ctrl: _ArmController, log: logging.Logger) -> None:
    async for byte in serial.read_stream():
        msg = mav.parse_char(bytes([byte]))
        if msg is None:
            continue
        t = msg.get_type()
        if t == "ATTITUDE":
            state.last_yaw = msg.yaw
            bus.publish("fc/attitude", decode_attitude(msg, monotonic_ns()))
        elif t == "RAW_IMU":
            state.have_raw_imu = True
            bus.publish("fc/imu", decode_imu(msg, monotonic_ns()))
        elif t == "SCALED_IMU2":
            if not state.have_raw_imu:          # fallback until RAW_IMU arrives
                bus.publish("fc/imu", decode_imu(msg, monotonic_ns()))
        elif t == "HEARTBEAT":
            if msg.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID:  # ignore GCS
                _on_heartbeat(msg, state, log)
        elif t == "COMMAND_ACK":
            arm_ctrl.on_ack(msg.command, msg.result)

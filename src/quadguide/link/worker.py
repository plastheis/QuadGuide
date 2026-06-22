from __future__ import annotations
import asyncio
import logging
import signal

from pymavlink import mavutil

from quadguide.core.clock import monotonic_ns
from quadguide.core.config import cfg_airframe, cfg_diag
from quadguide.core.diagtrace import DiagTrace
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, ProcessState
from quadguide.link.fc import (
    decode_attitude, decode_heartbeat, decode_imu,
    encode_arm, encode_attitude_target, encode_heartbeat,
    encode_set_message_interval,
)
from quadguide.link.mavlink_codec import (
    MSG_ID_ATTITUDE, MSG_ID_RAW_IMU, make_mav,
)
from quadguide.link.serial_port import SerialPort
from quadguide.link.tcp_serial import TCPSerialPort


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


_NS_PER_MS = 1_000_000


def _link_cfg(config: dict) -> dict:
    """Flatten the link + airframe-limit fields the loops need into one dict."""
    link = config["link"]
    acfg = cfg_airframe(config)
    return {
        "tx_rate_hz": link["tx_rate_hz"],
        "stream_rate_hz": link.get("stream_rate_hz", 50),
        "system_id": link.get("system_id", 1),
        "component_id": link.get("component_id", 191),
        "target_system": link.get("target_system", 1),
        "target_component": link.get("target_component", 1),
        "arm_retry_count": link.get("arm_retry_count", 5),
        "heartbeat_wait_s": link.get("heartbeat_wait_s", 5.0),
        "max_roll_deg": acfg.control_limits.max_roll_deg,
        "max_pitch_deg": acfg.control_limits.max_pitch_deg,
    }


async def _tx_loop(serial, mav, bus, state: _LinkState, arm_ctrl: _ArmController,
                   lc: dict, log: logging.Logger, trace) -> None:
    interval = 1.0 / lc["tx_rate_hz"]
    prev_armed = False
    yaw_hold = 0.0
    while True:
        cmd = bus.latest("control/cmd")
        arm_cmd = bus.latest("arm/cmd")
        armed = bool(arm_cmd and arm_cmd.armed)

        to_send = arm_ctrl.on_arm_state(armed)
        if to_send is not None and state.have_heartbeat:
            await serial.write(encode_arm(mav, to_send,
                                          state.target_system, state.target_component))
            log.info("arm command → %s", "ARM" if to_send else "DISARM")

        yaw_hold = latch_yaw(armed, prev_armed, state.last_yaw, yaw_hold)
        prev_armed = armed

        now = monotonic_ns()
        tsys = state.target_system or lc["target_system"]
        tcomp = state.target_component or lc["target_component"]
        await serial.write(encode_attitude_target(
            mav, cmd, yaw_hold, tsys, tcomp,
            lc["max_roll_deg"], lc["max_pitch_deg"], now // _NS_PER_MS))
        # Actuation point: glass→TX latency for the command just sent to the FC.
        if cmd is not None and cmd.origin_ns > 0:
            trace.latency(now, cmd.timestamp_ns, cmd.origin_ns)
        await asyncio.sleep(interval)


async def _heartbeat_loop(serial, mav) -> None:
    while True:
        await serial.write(encode_heartbeat(mav))
        await asyncio.sleep(1.0)


async def _stream_setup_loop(serial, mav, state: _LinkState, lc: dict,
                             log: logging.Logger) -> None:
    """Wait (bounded) for the first FC heartbeat, then request the telemetry streams.

    Runs once per connection and returns. The RX loop is the sole reader, so this
    polls `state.have_heartbeat` rather than reading bytes itself.
    """
    waited = 0.0
    while not state.have_heartbeat and waited < lc["heartbeat_wait_s"]:
        await asyncio.sleep(0.1)
        waited += 0.1
    tsys = state.target_system or lc["target_system"]
    tcomp = state.target_component or lc["target_component"]
    for mid in (MSG_ID_ATTITUDE, MSG_ID_RAW_IMU):
        await serial.write(encode_set_message_interval(
            mav, mid, lc["stream_rate_hz"], tsys, tcomp))
    log.info("requested ATTITUDE+RAW_IMU @ %d Hz from sys=%d comp=%d",
             lc["stream_rate_hz"], tsys, tcomp)


async def _health_loop(bus, state: _LinkState, trace) -> None:
    while True:
        mode = str(state.fc_mode) if state.have_heartbeat else ""
        bus.publish("system/health",
                    HealthReport(monotonic_ns(), "link", ProcessState.OK, mode))
        trace.health(monotonic_ns(), "ok", mode)
        await asyncio.sleep(0.2)


def _serial_factory(config: dict, log: logging.Logger):
    """Pick the link transport from platform.serial.mode.

    "uart" → MAVLink2 over the real UART; "tcp" → MAVLink2 over a TCP socket to
    ArduPilot SITL. Both satisfy the same async port interface, so the loops are
    transport-agnostic.
    """
    scfg = config["platform"]["serial"]
    mode = scfg.get("mode", "uart")
    if mode == "tcp":
        host = scfg["tcp_host"]
        port = scfg["tcp_port"]
        log.info(f"HIL: MAVLink2 over TCP → {host}:{port} (SITL)")
        return (lambda: TCPSerialPort(host, port), f"tcp {host}:{port}")
    if mode != "uart":
        raise ValueError(f"Unknown serial mode {mode!r}. Valid values: 'uart', 'tcp'")
    dev = scfg["port"]
    baud = scfg["baud"]
    return (lambda: SerialPort(dev, baud), f"uart {dev} @ {baud}")


async def _run_async(config: dict, bus) -> None:
    log = setup_logging("link", config)
    lc = _link_cfg(config)
    make_serial, transport = _serial_factory(config, log)

    dcfg = cfg_diag(config)
    trace = DiagTrace("link", enabled=dcfg.trace,
                      dir=dcfg.trace_dir, max_rows=dcfg.trace_max_rows)

    loop = asyncio.get_running_loop()

    def _on_sigterm(*_):
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGTERM, _on_sigterm)

    while True:
        serial = make_serial()
        mav = make_mav(lc["system_id"], lc["component_id"])
        state = _LinkState()
        arm_ctrl = _ArmController(
            lc["arm_retry_count"], max(1, int(lc["tx_rate_hz"] // 2)))
        tasks: list[asyncio.Task] = []
        try:
            await serial.open()
            log.info(f"Link opened ({transport})")
            tasks = [
                asyncio.create_task(_rx_loop(serial, mav, state, bus, arm_ctrl, log)),
                asyncio.create_task(_tx_loop(serial, mav, bus, state, arm_ctrl, lc, log, trace)),
                asyncio.create_task(_heartbeat_loop(serial, mav)),
                asyncio.create_task(_stream_setup_loop(serial, mav, state, lc, log)),
                asyncio.create_task(_health_loop(bus, state, trace)),
            ]
            await asyncio.gather(*tasks)

        except ConnectionError as exc:
            log.error(f"Serial error: {exc}")
            bus.publish("system/health",
                        HealthReport(monotonic_ns(), "link", ProcessState.DEGRADED, ""))

        except asyncio.CancelledError:
            break

        finally:
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            serial.close()

        log.info("Reconnecting in 500 ms...")
        await asyncio.sleep(0.5)

    trace.flush()
    bus.detach()
    log.info("Link worker stopped.")


def run(config: dict, bus) -> None:
    asyncio.run(_run_async(config, bus))

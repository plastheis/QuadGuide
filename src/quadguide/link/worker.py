from __future__ import annotations
import asyncio
import logging
import signal

from quadguide.core.clock import monotonic_ns
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, ProcessState
from quadguide.link.crsf import (
    CRSF_ATTITUDE, CRSF_FLIGHT_MODE, CRSF_IMU_RAW, CRSFParser,
)
from quadguide.link.fc import (
    ChannelConfig, channel_config_from_cfg,
    decode_attitude, decode_flight_mode, decode_imu, encode_rc,
)
from quadguide.link.serial_port import SerialPort


class _LinkState:
    """Per-connection mutable state shared between RX handlers."""

    def __init__(self) -> None:
        self.have_imu_frame: bool = False
        self.last_gyro: tuple[float, float, float] | None = None
        self.flight_mode: str = ""


async def _rx_loop(serial, parser: CRSFParser, state: _LinkState,
                   bus, log: logging.Logger) -> None:
    async for byte in serial.read_stream():
        frame = parser.feed(byte)
        if frame is None:
            continue

        if frame.type == CRSF_IMU_RAW:
            imu = decode_imu(frame)
            state.last_gyro = (imu.gx, imu.gy, imu.gz)
            if not state.have_imu_frame:
                state.have_imu_frame = True
                log.info("0x80 IMU RAW frame detected — using gyro for body rates")
            bus.publish("fc/imu", imu)

        elif frame.type == CRSF_ATTITUDE:
            att = decode_attitude(frame, state.have_imu_frame, state.last_gyro)
            bus.publish("fc/attitude", att)

        elif frame.type == CRSF_FLIGHT_MODE:
            mode = decode_flight_mode(frame)
            if mode != state.flight_mode:
                log.info("FC flight mode → %s", mode)
                state.flight_mode = mode


async def _tx_loop(serial, bus, tx_rate_hz: float,
                   ch_cfg: ChannelConfig, log: logging.Logger) -> None:
    interval = 1.0 / tx_rate_hz
    prev_armed = None
    while True:
        cmd     = bus.latest("control/cmd")
        arm_cmd = bus.latest("arm/cmd")
        armed   = arm_cmd.armed if arm_cmd else False
        if armed != prev_armed:
            log.info("arm state → %s", "ARMED" if armed else "DISARMED")
            prev_armed = armed
        await serial.write(encode_rc(cmd, armed, ch_cfg))
        await asyncio.sleep(interval)


async def _health_loop(bus, state: _LinkState, log: logging.Logger) -> None:
    while True:
        bus.publish(
            "system/health",
            HealthReport(monotonic_ns(), "link", ProcessState.OK, state.flight_mode),
        )
        await asyncio.sleep(0.2)


async def _run_async(config: dict, bus) -> None:
    log        = setup_logging("link", config)
    tx_rate_hz = config["link"]["tx_rate_hz"]
    ch_cfg     = channel_config_from_cfg(config)
    port       = config["platform"]["serial"]["port"]
    baud       = config["platform"]["serial"]["baud"]

    loop = asyncio.get_running_loop()

    def _on_sigterm(*_):
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGTERM, _on_sigterm)

    while True:
        serial = SerialPort(port, baud)
        tasks: list[asyncio.Task] = []
        state = _LinkState()
        try:
            await serial.open()
            log.info(f"Serial opened {port} @ {baud}")

            tasks = [
                asyncio.create_task(_rx_loop(serial, CRSFParser(), state, bus, log)),
                asyncio.create_task(_tx_loop(serial, bus, tx_rate_hz, ch_cfg, log)),
                asyncio.create_task(_health_loop(bus, state, log)),
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

    bus.detach()
    log.info("Link worker stopped.")


def run(config: dict, bus) -> None:
    asyncio.run(_run_async(config, bus))

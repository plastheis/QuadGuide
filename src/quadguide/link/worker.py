from __future__ import annotations
import asyncio
import logging
import signal

from quadguide.core.clock import monotonic_ns
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, ProcessState
from quadguide.link.crsf import CRSF_ATTITUDE, CRSFParser
from quadguide.link.differentiator import AttitudeDifferentiator
from quadguide.link.espfc import decode_attitude, encode_rc
from quadguide.link.serial_port import SerialPort


async def _rx_loop(serial, parser: CRSFParser,
                   diff: AttitudeDifferentiator, bus, log: logging.Logger) -> None:
    async for byte in serial.read_stream():
        frame = parser.feed(byte)
        if frame is None:
            continue
        if frame.type == CRSF_ATTITUDE:
            att, imu = decode_attitude(frame, diff)
            bus.publish("fc/attitude", att)
            bus.publish("fc/imu", imu)


async def _tx_loop(serial, bus, tx_rate_hz: float, log: logging.Logger) -> None:
    interval = 1.0 / tx_rate_hz
    while True:
        cmd     = bus.latest("control/cmd")
        arm_cmd = bus.latest("arm/cmd")
        armed   = arm_cmd.armed if arm_cmd else False
        await serial.write(encode_rc(cmd, armed))
        await asyncio.sleep(interval)


async def _health_loop(bus, log: logging.Logger) -> None:
    while True:
        bus.publish("system/health",
                    HealthReport(monotonic_ns(), "link", ProcessState.OK, ""))
        await asyncio.sleep(0.2)


async def _run_async(config: dict, bus) -> None:
    log        = setup_logging("link", config)
    diff       = AttitudeDifferentiator(config["link"]["diff_lowpass_alpha"])
    tx_rate_hz = config["link"]["tx_rate_hz"]
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
        try:
            await serial.open()
            log.info(f"Serial opened {port} @ {baud}")

            tasks = [
                asyncio.create_task(_rx_loop(serial, CRSFParser(), diff, bus, log)),
                asyncio.create_task(_tx_loop(serial, bus, tx_rate_hz, log)),
                asyncio.create_task(_health_loop(bus, log)),
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

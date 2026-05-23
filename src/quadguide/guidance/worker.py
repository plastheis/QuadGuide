from __future__ import annotations
import signal

from quadguide.core.bus import Bus
from quadguide.core.clock import RateLimiter, monotonic_ns
from quadguide.core.config import cfg_guidance, cfg_platform, cfg_watchdog
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import (
    AccelCmd, HealthReport, ProcessState, TrackerHealth,
)
from quadguide.guidance.factory import get_guidance

__all__ = ["run"]

_HEALTH_EVERY = 10   # iterations; 50 Hz / 10 = 5 Hz health rate


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    log = setup_logging("guidance", config)
    gcfg = cfg_guidance(config)
    pcfg = cfg_platform(config)
    wcfg = cfg_watchdog(config)
    fc_imu_timeout_ns = wcfg.fc_imu_ms * 1_000_000

    aspect = pcfg.camera.width / pcfg.camera.height
    method = get_guidance(gcfg, aspect)
    rate = RateLimiter(hz=50)

    stop = False

    def _on_sigterm(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    i = 0
    log.info("guidance: started (method=%s, throttle_hold=%.2f)", method.name(), gcfg.throttle_hold)

    while not stop:
        rate.sleep()

        est        = bus.latest("target/estimate")
        att        = bus.latest("fc/attitude")
        imu        = bus.latest("fc/imu")
        lockon_cmd = bus.latest("lockon/cmd")

        if est is None or att is None or imu is None:
            continue
        if est.tracker_health in (TrackerHealth.LOST, TrackerHealth.NO_LOCK):
            continue

        now_ns = monotonic_ns()
        if now_ns - imu.timestamp_ns > fc_imu_timeout_ns:
            continue

        ax, ay = method.compute(est, imu, lockon_cmd, now_ns)

        bus.publish("guidance/accel", AccelCmd(now_ns, ax, ay))

        i += 1
        if i % _HEALTH_EVERY == 0:
            bus.publish(
                "system/health",
                HealthReport(monotonic_ns(), "guidance", ProcessState.OK, ""),
            )

    bus.detach()
    log.info("guidance: stopped")

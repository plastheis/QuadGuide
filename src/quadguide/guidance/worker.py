from __future__ import annotations
import signal

from quadguide.core.bus import Bus
from quadguide.core.clock import RateLimiter, monotonic_ns
from quadguide.core.config import cfg_guidance, cfg_platform
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import (
    AccelCmd, HealthReport, ProcessState, TrackerHealth,
)
from quadguide.guidance.closing_vel import ClosingVelEstimator
from quadguide.guidance.los import LOSRateEstimator
from quadguide.guidance.pronav import pronav

__all__ = ["run"]

_HEALTH_EVERY = 10   # iterations; 50 Hz / 10 = 5 Hz health rate


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    log = setup_logging("guidance", config)
    gcfg = cfg_guidance(config)
    pcfg = cfg_platform(config)

    aspect = pcfg.camera.width / pcfg.camera.height
    los = LOSRateEstimator(gcfg.fov_horizontal_rad, aspect)
    cv = ClosingVelEstimator()
    rate = RateLimiter(hz=50)

    stop = False

    def _on_sigterm(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    i = 0
    log.info("guidance: started (N=%.1f, throttle_hold=%.2f)", gcfg.N, gcfg.throttle_hold)

    while not stop:
        rate.sleep()

        est        = bus.latest("target/estimate")
        att        = bus.latest("fc/attitude")
        lockon_cmd = bus.latest("lockon/cmd")

        if est is None or att is None:
            continue
        if est.tracker_health in (TrackerHealth.LOST, TrackerHealth.NO_LOCK):
            continue

        now_ns = monotonic_ns()
        los_r  = los.update(est.centroid_norm, att, lockon_cmd, now_ns)
        v_c    = cv.update(est.bbox, now_ns, gcfg)
        ax, ay = pronav(los_r, v_c, gcfg.N)

        bus.publish("guidance/accel", AccelCmd(now_ns, ax, ay))

        i += 1
        if i % _HEALTH_EVERY == 0:
            bus.publish(
                "system/health",
                HealthReport(monotonic_ns(), "guidance", ProcessState.OK, ""),
            )

    bus.detach()
    log.info("guidance: stopped")

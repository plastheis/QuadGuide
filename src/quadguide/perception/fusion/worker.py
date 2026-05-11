from __future__ import annotations
import signal

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.config import cfg_tracker
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, ProcessState, TrackerEstimate
from quadguide.perception.fusion.algorithms import build_fusion_algorithm

__all__ = ["run"]

_HEALTH_EVERY = 100


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    log = setup_logging("fusion", config)
    tcfg = cfg_tracker(config)
    fcfg = tcfg.fusion

    algorithm = build_fusion_algorithm(fcfg.algorithm)

    subscribe_topics = []
    if tcfg.ccv is not None:
        subscribe_topics.append("ccv_tracker/estimate")
    if tcfg.ncv is not None:
        subscribe_topics.append("ncv_tracker/estimate")

    stop = False

    def _on_sigterm(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    latest_ccv: TrackerEstimate | None = None
    latest_ncv: TrackerEstimate | None = None
    i = 0

    log.info("fusion: started (algorithm=%s, fast=%s)", fcfg.algorithm, fcfg.fast_tracker)
    while not stop:
        try:
            topic, msg = bus.subscribe_any(subscribe_topics)
        except (InterruptedError, OSError):
            break

        if topic == "ccv_tracker/estimate":
            latest_ccv = msg
        else:
            latest_ncv = msg

        estimate = algorithm.fuse(latest_ccv, latest_ncv, fcfg)
        if estimate is not None:
            bus.publish("target/estimate", estimate)

        i += 1
        if i % _HEALTH_EVERY == 0:
            bus.publish(
                "system/health",
                HealthReport(monotonic_ns(), "fusion", ProcessState.OK, ""),
            )

    bus.detach()
    log.info("fusion: stopped")

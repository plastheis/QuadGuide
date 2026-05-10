from __future__ import annotations
import signal

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.config import cfg_tracker
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, ProcessState, TrackerEstimate
from quadguide.perception.fusion.fusion import fuse

__all__ = ["run"]

_HEALTH_EVERY = 100
_SUBSCRIBE_TOPICS = ["ccv_tracker/estimate", "ncv_tracker/estimate"]


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    log = setup_logging("fusion", config)
    fcfg = cfg_tracker(config).fusion
    stop = False

    def _on_sigterm(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    latest_ccv: TrackerEstimate | None = None
    latest_ncv: TrackerEstimate | None = None
    i = 0

    log.info("fusion: started")
    while not stop:
        try:
            topic, msg = bus.subscribe_any(_SUBSCRIBE_TOPICS)
        except (InterruptedError, OSError):
            break

        if topic == "ccv_tracker/estimate":
            latest_ccv = msg
        else:
            latest_ncv = msg

        estimate = fuse(latest_ccv, latest_ncv, fcfg)
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

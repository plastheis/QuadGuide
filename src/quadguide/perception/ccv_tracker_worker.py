from __future__ import annotations
import os
import signal
import time

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, LockOnCmd, ProcessState

__all__ = ["CCVTrackerWorker"]

_HEALTH_EVERY = 50   # publish HealthReport every N iterations


class CCVTrackerWorker:
    """IPC loop for the classical CV tracker slot.

    Publishes to ccv_tracker/estimate. Sets CPU affinity if requested.
    Tracker object must implement init(frame, bbox) and update(frame).
    """

    def __init__(self, tracker, bus: Bus, frame_buffer: FrameBuffer,
                 cpu_core: int = 1, config: dict | None = None) -> None:
        self._tracker      = tracker
        self._bus          = bus
        self._fb           = frame_buffer
        self._cpu_core     = cpu_core
        self._config       = config or {}
        self._last_seq: int | None = None
        self._stop         = False

    def run(self) -> None:
        log = setup_logging("ccv_tracker", self._config)

        signal.signal(signal.SIGTERM, self._handle_sigterm)

        try:
            os.sched_setaffinity(0, {self._cpu_core})
        except (AttributeError, OSError):
            pass  # dev machine or permission denied — continue without affinity

        log.info("ccv_tracker: started")
        i = 0
        while not self._stop:
            self._check_lockon()
            frame, _ = self._fb.read_latest()
            if frame is not None:
                est = self._tracker.update(frame)
                self._bus.publish("ccv_tracker/estimate", est)
            i += 1
            if i % _HEALTH_EVERY == 0:
                self._bus.publish(
                    "system/health",
                    HealthReport(monotonic_ns(), "ccv_tracker", ProcessState.OK, ""),
                )

        self._tracker.close()
        self._bus.detach()
        log.info("ccv_tracker: stopped")

    def _check_lockon(self) -> None:
        cmd: LockOnCmd | None = self._bus.latest("lockon/cmd")
        if cmd is None:
            return
        if cmd.seq != self._last_seq:
            self._last_seq = cmd.seq
            frame, _ = self._fb.read_latest()
            if frame is not None:
                self._tracker.init(frame, cmd.bbox)

    def _handle_sigterm(self, sig, frame) -> None:
        self._stop = True

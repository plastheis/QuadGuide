from __future__ import annotations
import dataclasses
import os
import signal
import time

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, LockOnCmd, ProcessState

__all__ = ["CCVTrackerWorker", "run_from_config"]

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
        self._proc_name    = f"ccv_{tracker.name()}"

    def run(self) -> None:
        log = setup_logging(self._proc_name, self._config)

        signal.signal(signal.SIGTERM, self._handle_sigterm)

        try:
            os.sched_setaffinity(0, {self._cpu_core})
        except (AttributeError, OSError):
            pass  # dev machine or permission denied — continue without affinity

        log.info(f"{self._proc_name}: started")
        i = 0
        while not self._stop:
            self._check_lockon()
            frame, frame_ts = self._fb.read_latest()
            if frame is not None:
                est = self._tracker.update(frame)
                latency_ns = min(monotonic_ns() - frame_ts, 0xFFFF_FFFF) if frame_ts > 0 else 0
                est = dataclasses.replace(est, latency_ns=latency_ns)
                self._bus.publish("ccv_tracker/estimate", est)
            i += 1
            if i % _HEALTH_EVERY == 0:
                self._bus.publish(
                    "system/health",
                    HealthReport(monotonic_ns(), self._proc_name, ProcessState.OK, ""),
                )

        self._tracker.close()
        self._bus.detach()
        log.info(f"{self._proc_name}: stopped")

    def _check_lockon(self) -> None:
        cmd: LockOnCmd | None = self._bus.latest("lockon/cmd")
        if cmd is None:
            return
        if cmd.seq != self._last_seq:
            self._last_seq = cmd.seq
            if cmd.bbox.w == 0.0 and cmd.bbox.h == 0.0:
                self._tracker.reset()
            else:
                frame, _ = self._fb.read_latest()
                if frame is not None:
                    self._tracker.init(frame, cmd.bbox)

    def _handle_sigterm(self, sig, frame) -> None:
        self._stop = True


def run_from_config(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    """Construct the CCV tracker selected by config.tracker.ccv and run it."""
    from quadguide.core.config import cfg_platform
    from quadguide.perception.tracker_factories import get_ccv_tracker
    pcfg    = cfg_platform(config)
    tracker = get_ccv_tracker(config)
    CCVTrackerWorker(
        tracker, bus, frame_buffer,
        cpu_core=pcfg.realtime.kcf_cpu_core,
        config=config,
    ).run()

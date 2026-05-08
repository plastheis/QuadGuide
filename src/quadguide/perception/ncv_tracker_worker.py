from __future__ import annotations
import signal

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, LockOnCmd, ProcessState

__all__ = ["NCVTrackerWorker"]

_HEALTH_EVERY = 10   # NCV runs slower (~30 Hz); publish health more frequently


class NCVTrackerWorker:
    """IPC loop for the neural CV tracker slot.

    Publishes to ncv_tracker/estimate. Calls tracker.close() on SIGTERM
    to release the NPU handle before exit — critical for RKNN.
    """

    def __init__(self, tracker, bus: Bus, frame_buffer: FrameBuffer,
                 config: dict | None = None) -> None:
        self._tracker      = tracker
        self._bus          = bus
        self._fb           = frame_buffer
        self._config       = config or {}
        self._last_seq: int | None = None
        self._stop         = False

    def run(self) -> None:
        log = setup_logging("ncv_tracker", self._config)
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        log.info("ncv_tracker: started")
        i = 0
        while not self._stop:
            self._check_lockon()
            frame, _ = self._fb.read_latest()
            if frame is not None:
                est = self._tracker.update(frame)
                self._bus.publish("ncv_tracker/estimate", est)
            i += 1
            if i % _HEALTH_EVERY == 0:
                self._bus.publish(
                    "system/health",
                    HealthReport(monotonic_ns(), "ncv_tracker", ProcessState.OK, ""),
                )

        self._tracker.close()  # release NPU handle before exit
        self._bus.detach()
        log.info("ncv_tracker: stopped")

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

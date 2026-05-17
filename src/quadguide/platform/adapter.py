from __future__ import annotations

import os
import logging

log = logging.getLogger(__name__)

try:
    import ctypes
    import ctypes.util
    _HAS_SCHED = hasattr(os, "sched_setscheduler")
except Exception:
    _HAS_SCHED = False


class PlatformAdapter:
    """Thin OS-level adapter for CPU affinity and real-time scheduling."""

    def __init__(self, config: dict) -> None:
        self._cfg = config

    def set_realtime(self, cpu_core: int, fifo_prio: int) -> None:
        """Pin the current process to *cpu_core* and optionally enable SCHED_FIFO."""
        # CPU affinity
        try:
            os.sched_setaffinity(0, {cpu_core})
            log.debug("set CPU affinity to core %d", cpu_core)
        except (AttributeError, OSError) as e:
            log.debug("set_affinity skipped: %s", e)

        # SCHED_FIFO — check config flag before attempting
        try:
            sched_fifo = self._cfg["platform"]["realtime"]["control_sched_fifo"]
        except (KeyError, TypeError):
            sched_fifo = False

        if sched_fifo:
            try:
                SCHED_FIFO = 1
                param = os.sched_param(fifo_prio)
                os.sched_setscheduler(0, SCHED_FIFO, param)
                log.debug("set SCHED_FIFO priority %d", fifo_prio)
            except (AttributeError, PermissionError, OSError) as e:
                log.debug("set SCHED_FIFO skipped (need CAP_SYS_NICE): %s", e)

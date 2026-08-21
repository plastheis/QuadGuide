"""Worker child entrypoint helpers (ARCHITECTURE.md §7.4).

Children attach to shared memory (never unlink), detach the resource_tracker for
attached segments to avoid multiprocessing double-unlink warnings, request death
with the parent via PR_SET_PDEATHSIG (Linux), and initialise their backend
in-process (NPU/RKNN contexts do not survive fork and must be created here).
"""

from __future__ import annotations

import ctypes
import logging
import platform

log = logging.getLogger("edgecv.worker")

_PR_SET_PDEATHSIG = 1  # from <sys/prctl.h>


def request_death_with_parent() -> None:
    """Ask the kernel to send SIGTERM to this process when the parent dies."""
    if platform.system() != "Linux":
        return
    try:
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception as e:  # pragma: no cover - platform dependent
        log.warning("PR_SET_PDEATHSIG failed: %s", e)


def detach_resource_tracker(shm_name: str) -> None:
    """Stop this process's resource_tracker from trying to unlink an attached segment.

    Only the orchestrator (owner) unlinks. Children attach only (§7.4 / §14.8)."""
    try:
        from multiprocessing import resource_tracker

        resource_tracker.unregister(f"/{shm_name}", "shared_memory")
    except Exception as e:  # pragma: no cover - internal API drift
        log.debug("resource_tracker.unregister(%s) failed: %s", shm_name, e)


def child_main(target, args: tuple) -> None:
    """Generic child bootstrap: install death signal, then run the worker target."""
    request_death_with_parent()
    # detach_resource_tracker is wired in here once SHM-attaching workers land
    # (the foundation has no worker that attaches a segment yet).
    target(*args)

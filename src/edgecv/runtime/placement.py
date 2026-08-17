"""Declarative process->hardware placement (ARCHITECTURE.md §7.6).

A board profile maps each process to CPU affinity, optional SCHED_FIFO, an NPU
core, and a backend. Shipped defaults for rk3588; fully user-overridable. No
placement is ever hardcoded in a tracker. Applying affinity/sched is best-effort:
it needs privileges (CAP_SYS_NICE) that may be absent in CI, so failures are
swallowed with a warning rather than raised.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import yaml

log = logging.getLogger("edgecv.placement")


@dataclass
class ProcessPlacement:
    cpu_affinity: list[int] | None = None
    sched: dict | None = None
    npu_core: int | None = None
    backend: str | None = None

    def apply(self) -> None:
        """Best-effort: pin affinity and (optionally) set SCHED_FIFO for this process."""
        if self.cpu_affinity:
            try:
                os.sched_setaffinity(0, set(self.cpu_affinity))
            except Exception as e:  # missing on non-Linux, or EPERM in CI
                log.warning("could not set CPU affinity %s: %s", self.cpu_affinity, e)
        if self.sched and self.sched.get("policy") == "FIFO":
            try:
                param = os.sched_param(int(self.sched.get("priority", 1)))
                os.sched_setscheduler(0, os.SCHED_FIFO, param)
            except Exception as e:
                log.warning("could not set SCHED_FIFO: %s", e)


@dataclass
class BoardProfile:
    board: str
    processes: dict[str, ProcessPlacement] = field(default_factory=dict)


def _from_dict(data: dict) -> BoardProfile:
    procs = {
        name: ProcessPlacement(
            cpu_affinity=spec.get("cpu_affinity"),
            sched=spec.get("sched"),
            npu_core=spec.get("npu_core"),
            backend=spec.get("backend"),
        )
        for name, spec in (data.get("processes") or {}).items()
    }
    return BoardProfile(board=data.get("board", "unknown"), processes=procs)


def load_profile(path: str | os.PathLike) -> BoardProfile:
    data = yaml.safe_load(Path(path).read_text())
    return _from_dict(data)


def default_profile() -> BoardProfile:
    """Shipped default (rk3588). Single source of truth is the packaged YAML in
    models/profiles, so the default cannot drift from the file users override."""
    data = yaml.safe_load(
        (files("edgecv.models") / "profiles" / "rk3588.yaml").read_text()
    )
    return _from_dict(data)

"""Process-group orchestrator (ARCHITECTURE.md §7.4).

Spawns workers with the 'spawn' (or 'forkserver') start method — never 'fork',
because NPU runtime contexts do not survive fork. Owns shared-memory lifecycle:
the orchestrator creates and unlinks all segments; children only attach. Provides
a heartbeat reaper and deterministic teardown via close()/context manager.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass, field

from edgecv.runtime.worker import child_main

log = logging.getLogger("edgecv.orchestrator")


@dataclass
class WorkerSpec:
    name: str
    target: Callable
    args: tuple = field(default_factory=tuple)


class Orchestrator:
    def __init__(self, mp_context: str = "spawn"):
        if mp_context == "fork":
            raise ValueError("fork is forbidden: NPU contexts do not survive fork (§7.4)")
        self._ctx = mp.get_context(mp_context)
        self._specs: dict[str, WorkerSpec] = {}
        self._procs: dict[str, mp.process.BaseProcess] = {}
        self._closed = False

    def add_worker(self, spec: WorkerSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate worker name: {spec.name}")
        self._specs[spec.name] = spec

    def start(self) -> None:
        for name, spec in self._specs.items():
            if name in self._procs and self._procs[name].is_alive():
                continue
            proc = self._ctx.Process(  # type: ignore[attr-defined]
                target=child_main, args=(spec.target, spec.args), name=name, daemon=True
            )
            proc.start()
            self._procs[name] = proc
            log.info("started worker %s (pid=%s)", name, proc.pid)

    def is_alive(self, name: str) -> bool:
        proc = self._procs.get(name)
        return bool(proc and proc.is_alive())

    def reap(self, restart: bool = False) -> None:
        """Join finished workers; optionally restart any that died."""
        for name, proc in list(self._procs.items()):
            if not proc.is_alive():
                proc.join(timeout=0)
                log.info("worker %s exited (code=%s)", name, proc.exitcode)
                if restart:
                    self._procs.pop(name)
        if restart:
            self.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _name, proc in self._procs.items():
            if proc.is_alive():
                proc.terminate()
        for name, proc in self._procs.items():
            proc.join(timeout=5.0)
            if proc.is_alive():  # pragma: no cover - escalation path
                proc.kill()
                proc.join(timeout=5.0)
            log.info("worker %s reaped", name)
        self._procs.clear()

    def __enter__(self) -> Orchestrator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

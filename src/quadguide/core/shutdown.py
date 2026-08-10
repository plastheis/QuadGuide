"""Uniform worker shutdown signalling.

Workers are forked by scripts/run.py, which stops them with SIGTERM. But when
run.py is launched from a terminal, Ctrl-C makes the tty deliver SIGINT to
*every* process in the foreground group at once — the workers get it directly,
before the orchestrator has run a line of its own shutdown path. Python's
default SIGINT handler raises KeyboardInterrupt out of whatever the worker was
doing, so the post-loop `trace.flush()` never runs and that worker's
`{process}.jsonl` is silently missing from the run.

That is exactly the failure that lost control.jsonl and guidance.jsonl from the
20260810-120828 trace. Handling both signals identically makes Ctrl-C and
`kill` produce the same clean exit, so a bench run started in a shell captures
the same trace set as one stopped by the orchestrator.
"""
from __future__ import annotations
import signal
from typing import Callable

__all__ = ["install_shutdown_handler"]


def install_shutdown_handler(handler: Callable[[int, object], None]) -> None:
    """Register `handler` for both SIGTERM and SIGINT."""
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

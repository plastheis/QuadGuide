"""Seqlock for wait-free cross-process reads (ARCHITECTURE.md §7.3).

The writer bumps the seq word odd, writes the payload, then bumps it even. A
reader retries while the seq is odd or changed across the read. Reads never block
the writer.

Honest caveat: pure Python has no explicit memory barriers, so this is "correct
in practice for aligned word-size stores on ARM64/x86" rather than provably
correct. If stronger guarantees are needed, back the control word with a tiny C
extension or a microsecond-held lock on ONLY the control word.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# os.sched_yield exists on POSIX; fall back to a no-op elsewhere.
_yield = getattr(os, "sched_yield", lambda: None)


class SeqLock:
    """Wraps a uint64 seq word living at `offset` inside a shared buffer."""

    def __init__(self, buf, offset: int = 0):
        self._word = ctypes.c_uint64.from_buffer(buf, offset)

    def release(self) -> None:
        """Drop the ctypes view into the shared buffer.

        ``ctypes.from_buffer`` keeps the buffer exported; on Python 3.12+ the
        owning ``SharedMemory.close()`` raises ``BufferError`` while any export
        is alive. Owners must call this before closing their segment.
        """
        self._word = None  # type: ignore[assignment]

    def write_begin(self) -> None:
        self._word.value += 1          # -> odd: a write is in progress

    def write_end(self) -> None:
        self._word.value += 1          # -> even: write complete

    def read(self, fn: Callable[[], T], max_retries: int = 10_000) -> T:
        """Run `fn` (which reads the guarded payload) under seqlock retry.

        Uncontended reads return immediately (the word is even and unchanged).
        On contention we yield the CPU/GIL before retrying: a busy `continue`
        would otherwise let a same-core or same-interpreter writer starve — under
        the GIL a pure-Python spin holds the interpreter so a *writer thread*
        can never finish its odd→even bump. Yielding keeps reads lock-free while
        letting the writer make progress; it never blocks the writer.
        """
        for _ in range(max_retries):
            before = self._word.value
            if before & 1:             # writer mid-update
                _yield()
                continue
            value = fn()
            after = self._word.value
            if before == after:
                return value
            _yield()                   # writer published between our reads; retry
        raise RuntimeError("seqlock read exceeded max_retries (writer starving reader?)")

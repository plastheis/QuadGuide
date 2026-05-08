from __future__ import annotations
import ctypes
import time

__all__ = ["monotonic_ns", "sleep_until", "RateLimiter"]

_CLOCK_MONOTONIC = 1
_TIMER_ABSTIME   = 1


class _timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


try:
    _librt = ctypes.CDLL("librt.so.1", use_errno=True)
    _clock_nanosleep_fn = _librt.clock_nanosleep
    _clock_nanosleep_fn.restype = ctypes.c_int
    _clock_nanosleep_fn.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_timespec),
        ctypes.POINTER(_timespec),
    ]
    _HAS_LIBRT = True
except OSError:
    _HAS_LIBRT = False


def monotonic_ns() -> int:
    return time.monotonic_ns()


def sleep_until(target_ns: int) -> None:
    """Sleep until target_ns (monotonic clock). Uses clock_nanosleep when available."""
    if _HAS_LIBRT:
        ts = _timespec(target_ns // 1_000_000_000, target_ns % 1_000_000_000)
        _clock_nanosleep_fn(_CLOCK_MONOTONIC, _TIMER_ABSTIME, ctypes.byref(ts), None)
    else:
        delay_s = (target_ns - time.monotonic_ns()) / 1e9
        if delay_s > 0:
            time.sleep(delay_s)


class RateLimiter:
    """Fixed-rate loop pacer that accounts for iteration execution time."""

    def __init__(self, hz: float) -> None:
        self._interval_ns = int(1_000_000_000 / hz)
        self._next_ns: int | None = None

    def sleep(self) -> None:
        now = time.monotonic_ns()
        if self._next_ns is None:
            self._next_ns = now + self._interval_ns
            return
        sleep_until(self._next_ns)
        self._next_ns += self._interval_ns
        # Fallen behind (e.g. after blocking I/O): reset rather than spinning to catch up
        if self._next_ns < time.monotonic_ns():
            self._next_ns = time.monotonic_ns() + self._interval_ns

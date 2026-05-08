from __future__ import annotations
import time
from enum import Enum

__all__ = ["HealthFault", "FailsafeState", "Watchdog"]


class HealthFault(Exception):
    """Raised by Watchdog.check_all() when any watched topic goes stale."""


class FailsafeState(Enum):
    NOMINAL  = "nominal"
    LEVEL    = "level"
    DISARMED = "disarmed"


class Watchdog:
    """Staleness checker for a fixed set of bus topics.

    Constructed once before the control loop starts. check_all() is called on
    every loop iteration; it raises HealthFault on the first stale topic found.
    """

    def __init__(self, topics: list[tuple[str, int]], bus) -> None:
        # topics: [(topic_name, timeout_ms), ...]
        self._topics = topics
        self._bus = bus

    def check_all(self) -> None:
        """Raise HealthFault if any topic has no message or its timestamp is stale."""
        now_ns = time.monotonic_ns()
        for topic, timeout_ms in self._topics:
            msg = self._bus.latest(topic)
            if msg is None:
                raise HealthFault(f"{topic}: no message received")
            age_ms = (now_ns - msg.timestamp_ns) / 1_000_000
            if age_ms > timeout_ms:
                raise HealthFault(
                    f"{topic}: stale {age_ms:.1f} ms (limit {timeout_ms} ms)"
                )

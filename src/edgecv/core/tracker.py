"""Tracker abstract base class (ARCHITECTURE.md §5.3)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackResult, TrackStatus


class Tracker(ABC):
    @abstractmethod
    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None: ...

    @abstractmethod
    def update(self, frame: np.ndarray) -> TrackResult:
        """Non-blocking. For hybrids this publishes the frame and returns the
        latest fused estimate; early calls may return status=INITIALIZING."""

    @property
    @abstractmethod
    def status(self) -> TrackStatus: ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable tracker name, e.g. "MOSSE", "SiamFC", "MAFiD"."""

    def close(self) -> None:  # noqa: B027 - intentional concrete no-op, not abstract
        """Tear down any owned process group / shared memory. No-op for inline trackers."""

    def __enter__(self) -> Tracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

"""Tracker output types (ARCHITECTURE.md §5.2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from edgecv.core.bbox import BoundingBox


class TrackStatus(IntEnum):
    INITIALIZING = 0  # workers warming up / no lock yet
    LOCKED = 1        # confident track
    COASTING = 2      # low confidence, extrapolating / awaiting correction
    LOST = 3          # track lost


@dataclass
class TrackResult:
    bbox: BoundingBox | None       # None when no estimate is available
    confidence: float | None       # None when the tracker has no meaningful score
    status: TrackStatus
    timestamp: float               # monotonic seconds, source-frame time
    seq: int                       # frame sequence number this result corresponds to

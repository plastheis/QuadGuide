"""Search ROI channel (MAFiD spec §3.2).

Latest-only, wait-free, single-writer (caller -> detector worker). Carries a
normalised BoundingBox defining the search/crop area for local detection.
Uses the seqlock pattern from payload.py, with a fixed-size struct.
"""

from __future__ import annotations

import ctypes
import gc
from multiprocessing import shared_memory

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.runtime.shm.seqlock import SeqLock
from edgecv.runtime.shm.structs import ABI_VERSION, MAGIC, SearchROIControl, validate_header

_SEQLOCK_OFFSET = SearchROIControl.seqlock.offset


class SearchROIChannel:
    """Shared-memory channel carrying the latest search ROI (caller -> worker).

    Writer:  Caller process (publishes at full frame rate).
    Reader:  Detector worker (reads latest ROI for local detection).
    """

    def __init__(self, shm, owner: bool) -> None:
        self._shm = shm
        self._owner = owner
        self._header = SearchROIControl.from_buffer(shm.buf, 0)
        self._seqlock = SeqLock(shm.buf, _SEQLOCK_OFFSET)
        if owner:
            self._header.magic = MAGIC
            self._header.abi_version = ABI_VERSION
            self._header.seqlock = 0
            self._header.seq = 0
            self._header.x = 0.0
            self._header.y = 0.0
            self._header.w = 0.0
            self._header.h = 0.0
            self._header.timestamp = 0.0
        else:
            validate_header(self._header.magic, self._header.abi_version)

    @property
    def name(self) -> str:
        return self._shm.name

    @classmethod
    def create(cls, name: str | None = None) -> SearchROIChannel:
        """Create a new shared-memory segment for the search ROI channel."""
        size = ctypes.sizeof(SearchROIControl)
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        return cls(shm, owner=True)

    @classmethod
    def attach(cls, name: str) -> SearchROIChannel:
        """Attach to an existing search ROI shared-memory segment."""
        shm = shared_memory.SharedMemory(name=name)
        return cls(shm, owner=False)

    def publish(self, bbox: BoundingBox, *, seq: int | None = None,
                timestamp: float | None = None) -> None:
        """Publish a search ROI (writer side).

        Args:
            bbox: Normalised bounding box (the crop region for local detection).
            seq: Optional frame sequence number. If None, increments internal seq.
            timestamp: Optional timestamp. If None, uses monotonic time.
        """
        import time as _time

        self._seqlock.write_begin()
        if seq is not None:
            self._header.seq = seq
        else:
            self._header.seq += 1
        self._header.x = bbox.x
        self._header.y = bbox.y
        self._header.w = bbox.w
        self._header.h = bbox.h
        self._header.timestamp = timestamp if timestamp is not None else _time.monotonic()
        self._seqlock.write_end()

    def read_latest(self) -> BoundingBox | None:
        """Read the latest search ROI (reader side).

        Returns a BoundingBox if data has been published at least once,
        or None before the first publish.
        """
        def snapshot() -> BoundingBox | None:
            seq = int(self._header.seq)
            if seq == 0:
                return None
            return BoundingBox(
                x=float(self._header.x),
                y=float(self._header.y),
                w=float(self._header.w),
                h=float(self._header.h),
            )

        return self._seqlock.read(snapshot)

    def close(self, unlink: bool = False) -> None:
        """Close the shared-memory segment.

        Args:
            unlink: Whether to unlink (remove) the segment. Only the owner
                    should unlink; readers pass unlink=False.
        """
        self._header = None  # type: ignore[assignment]
        self._seqlock.release()
        gc.collect()
        self._shm.close()
        if unlink and self._owner:
            self._shm.unlink()

"""NanoTrack result channel (spec 2026-06-14-acquire-track-design §3.2).

Latest-only, wait-free, single-writer (NanoTrack worker → parent). Carries one
bounding box plus its confidence, status, and the source frame's seq/timestamp
(for latency lineage, §4). Same seqlock + fixed-struct pattern as SearchROIChannel.
"""

from __future__ import annotations

import ctypes
import gc
import time
from dataclasses import dataclass
from multiprocessing import shared_memory

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.runtime.shm.seqlock import SeqLock
from edgecv.runtime.shm.structs import ABI_VERSION, MAGIC, NanoResult, validate_header

_SEQLOCK_OFFSET = NanoResult.seqlock.offset


@dataclass(frozen=True)
class NanoSample:
    bbox: BoundingBox
    confidence: float
    status: TrackStatus
    src_seq: int
    src_ts: float


class NanoResultChannel:
    """Shared-memory result word (NanoTrack worker → parent).

    Writer:  the NanoTrack worker.
    Reader:  the AcquireTrack parent.
    """

    def __init__(self, shm, owner: bool) -> None:
        self._shm = shm
        self._owner = owner
        self._header = NanoResult.from_buffer(shm.buf, 0)
        self._seqlock = SeqLock(shm.buf, _SEQLOCK_OFFSET)
        if owner:
            self._header.magic = MAGIC
            self._header.abi_version = ABI_VERSION
            self._header.seqlock = 0
            self._header.seq = 0
            self._header.src_seq = 0
            self._header.status = int(TrackStatus.INITIALIZING)
            self._header.x = 0.0
            self._header.y = 0.0
            self._header.w = 0.0
            self._header.h = 0.0
            self._header.confidence = 0.0
            self._header.src_ts = 0.0
        else:
            validate_header(self._header.magic, self._header.abi_version)

    @property
    def name(self) -> str:
        return self._shm.name

    @classmethod
    def create(cls, name: str | None = None) -> NanoResultChannel:
        size = ctypes.sizeof(NanoResult)
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        return cls(shm, owner=True)

    @classmethod
    def attach(cls, name: str) -> NanoResultChannel:
        shm = shared_memory.SharedMemory(name=name)
        return cls(shm, owner=False)

    def publish(self, bbox: BoundingBox, *, confidence: float,
                status: TrackStatus, src_seq: int, src_ts: float | None = None,
                seq: int | None = None) -> None:
        self._seqlock.write_begin()
        if seq is not None:
            self._header.seq = seq
        else:
            self._header.seq += 1
        self._header.x = bbox.x
        self._header.y = bbox.y
        self._header.w = bbox.w
        self._header.h = bbox.h
        self._header.confidence = float(confidence)
        self._header.status = int(status)
        self._header.src_seq = int(src_seq)
        self._header.src_ts = src_ts if src_ts is not None else time.monotonic()
        self._seqlock.write_end()

    def read_latest(self) -> NanoSample | None:
        """Latest result, or None before the first publish."""

        def snapshot() -> NanoSample | None:
            if int(self._header.seq) == 0:
                return None
            return NanoSample(
                bbox=BoundingBox(
                    x=float(self._header.x), y=float(self._header.y),
                    w=float(self._header.w), h=float(self._header.h),
                ),
                confidence=float(self._header.confidence),
                status=TrackStatus(int(self._header.status)),
                src_seq=int(self._header.src_seq),
                src_ts=float(self._header.src_ts),
            )

        return self._seqlock.read(snapshot)

    def close(self, unlink: bool = False) -> None:
        self._header = None  # type: ignore[assignment]
        self._seqlock.release()
        gc.collect()
        self._shm.close()
        if unlink and self._owner:
            self._shm.unlink()

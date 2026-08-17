"""AcquireTrack control channel (spec 2026-06-14-acquire-track-design §3.2).

Latest-only, wait-free, single-writer (parent → both workers). Carries the active
worker selector (`mode`), the normalised crop region fed to YOLO, and a monotone
`lock_gen` + `lock_bbox` the NanoTrack worker watches to (re-)initialise its
template. Generalises ``SearchROIChannel``: same seqlock + fixed-struct pattern.
"""

from __future__ import annotations

import ctypes
import gc
import time
from dataclasses import dataclass
from enum import IntEnum
from multiprocessing import shared_memory

from edgecv.core.bbox import BoundingBox
from edgecv.runtime.shm.seqlock import SeqLock
from edgecv.runtime.shm.structs import (
    ABI_VERSION,
    MAGIC,
    AcquireControl,
    validate_header,
)

_SEQLOCK_OFFSET = AcquireControl.seqlock.offset


class Mode(IntEnum):
    """Which worker(s) are active.

    Base AcquireTrack runs exactly one non-IDLE worker at a time (YOLO or NANO).
    BOTH runs NanoTrack and YOLO concurrently — used by VerifiedAcquireTrack to
    check the locked NanoTrack box against live YOLO detections during LOCKED
    (spec 2026-06-16-acquire-track-lock-verification-design). Workers test mode by
    membership, so BOTH is backward compatible: a tracker that never publishes it
    behaves exactly as before.
    """

    IDLE = 0
    YOLO = 1
    NANO = 2
    BOTH = 3


@dataclass(frozen=True)
class ControlSnapshot:
    mode: Mode
    crop: BoundingBox
    lock_gen: int
    lock_bbox: BoundingBox


class AcquireControlChannel:
    """Shared-memory control word (parent → workers).

    Writer:  the AcquireTrack parent (publishes at full frame rate).
    Readers: the YOLO worker and the NanoTrack worker.
    """

    def __init__(self, shm, owner: bool) -> None:
        self._shm = shm
        self._owner = owner
        self._header = AcquireControl.from_buffer(shm.buf, 0)
        self._seqlock = SeqLock(shm.buf, _SEQLOCK_OFFSET)
        if owner:
            self._header.magic = MAGIC
            self._header.abi_version = ABI_VERSION
            self._header.seqlock = 0
            self._header.seq = 0
            self._header.lock_gen = 0
            self._header.mode = int(Mode.IDLE)
            self._header.cx = 0.0
            self._header.cy = 0.0
            self._header.cw = 0.0
            self._header.ch = 0.0
            self._header.lx = 0.0
            self._header.ly = 0.0
            self._header.lw = 0.0
            self._header.lh = 0.0
            self._header.timestamp = 0.0
        else:
            validate_header(self._header.magic, self._header.abi_version)

    @property
    def name(self) -> str:
        return self._shm.name

    @classmethod
    def create(cls, name: str | None = None) -> AcquireControlChannel:
        size = ctypes.sizeof(AcquireControl)
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        return cls(shm, owner=True)

    @classmethod
    def attach(cls, name: str) -> AcquireControlChannel:
        shm = shared_memory.SharedMemory(name=name)
        return cls(shm, owner=False)

    def publish(self, *, mode: Mode, crop: BoundingBox, lock_gen: int,
                lock_bbox: BoundingBox, seq: int | None = None,
                timestamp: float | None = None) -> None:
        self._seqlock.write_begin()
        if seq is not None:
            self._header.seq = seq
        else:
            self._header.seq += 1
        self._header.mode = int(mode)
        self._header.lock_gen = int(lock_gen)
        self._header.cx = crop.x
        self._header.cy = crop.y
        self._header.cw = crop.w
        self._header.ch = crop.h
        self._header.lx = lock_bbox.x
        self._header.ly = lock_bbox.y
        self._header.lw = lock_bbox.w
        self._header.lh = lock_bbox.h
        self._header.timestamp = timestamp if timestamp is not None else time.monotonic()
        self._seqlock.write_end()

    def read_latest(self) -> ControlSnapshot:
        """Latest control snapshot. Returns the IDLE default before first publish."""

        def snapshot() -> ControlSnapshot:
            return ControlSnapshot(
                mode=Mode(int(self._header.mode)),
                crop=BoundingBox(
                    x=float(self._header.cx), y=float(self._header.cy),
                    w=float(self._header.cw), h=float(self._header.ch),
                ),
                lock_gen=int(self._header.lock_gen),
                lock_bbox=BoundingBox(
                    x=float(self._header.lx), y=float(self._header.ly),
                    w=float(self._header.lw), h=float(self._header.lh),
                ),
            )

        return self._seqlock.read(snapshot)

    def close(self, unlink: bool = False) -> None:
        self._header = None  # type: ignore[assignment]
        self._seqlock.release()
        gc.collect()
        self._shm.close()
        if unlink and self._owner:
            self._shm.unlink()

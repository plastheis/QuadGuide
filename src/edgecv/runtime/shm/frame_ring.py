"""Zero-copy, latest-only frame ring (ARCHITECTURE.md §7.1).

N fixed-size slots sized for the max supported resolution. The single producer
writes the next slot then publishes the control word under a seqlock. Consumers
read a zero-copy numpy view of the newest slot; a consumer that fell behind jumps
to the newest seq rather than draining (latest-only). Triple-or-more buffering
plus latest-only reads handle slot recycling without refcounts.
"""

from __future__ import annotations

import ctypes
import gc
from multiprocessing import shared_memory

import numpy as np

from edgecv.runtime.shm.seqlock import SeqLock
from edgecv.runtime.shm.structs import (
    ABI_VERSION,
    MAGIC,
    FrameControl,
    dtype_to_code,
    validate_header,
)

_CONTROL_SIZE = ctypes.sizeof(FrameControl)
# seqlock word for the control struct is the FrameControl.seqlock field; locate it.
_SEQLOCK_OFFSET = FrameControl.seqlock.offset


class FrameRing:
    def __init__(self, shm: shared_memory.SharedMemory, slots: int, max_h: int,
                 max_w: int, max_c: int, dtype: str, owner: bool):
        self._shm = shm
        self._slots = slots
        self._max_h = max_h
        self._max_w = max_w
        self._max_c = max_c
        self._dtype = np.dtype(dtype)
        self._owner = owner
        self._slot_bytes = max_h * max_w * max_c * self._dtype.itemsize
        self._data_offset = _CONTROL_SIZE
        self._control = FrameControl.from_buffer(shm.buf, 0)  # type: ignore[arg-type]
        self._seqlock = SeqLock(shm.buf, _SEQLOCK_OFFSET)
        self._write_count = 0
        if owner:
            self._control.magic = MAGIC
            self._control.abi_version = ABI_VERSION
            self._control.seq = 0
            self._control.seqlock = 0
            self._control.slot = 0
        else:
            validate_header(self._control.magic, self._control.abi_version)

    @property
    def name(self) -> str:
        return self._shm.name

    @staticmethod
    def _size(slots: int, max_h: int, max_w: int, max_c: int, dtype: str) -> int:
        slot_bytes = max_h * max_w * max_c * np.dtype(dtype).itemsize
        return _CONTROL_SIZE + slots * slot_bytes

    @classmethod
    def create(cls, slots: int, max_h: int, max_w: int, max_c: int,
               dtype: str = "uint8", name: str | None = None) -> FrameRing:
        size = cls._size(slots, max_h, max_w, max_c, dtype)
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        return cls(shm, slots, max_h, max_w, max_c, dtype, owner=True)

    @classmethod
    def attach(cls, name: str, slots: int, max_h: int, max_w: int, max_c: int,
               dtype: str = "uint8") -> FrameRing:
        shm = shared_memory.SharedMemory(name=name)
        return cls(shm, slots, max_h, max_w, max_c, dtype, owner=False)

    def _slot_view(self, slot: int, h: int, w: int, c: int) -> np.ndarray:
        offset = self._data_offset + slot * self._slot_bytes
        return np.ndarray((h, w, c), dtype=self._dtype,
                          buffer=self._shm.buf, offset=offset)

    def publish(self, frame: np.ndarray, seq: int, timestamp: float) -> None:
        if frame.dtype != self._dtype:
            raise ValueError(f"frame dtype {frame.dtype} != ring dtype {self._dtype}")
        h, w = frame.shape[0], frame.shape[1]
        c = frame.shape[2] if frame.ndim == 3 else 1
        if h > self._max_h or w > self._max_w or c > self._max_c:
            raise ValueError(f"frame {frame.shape} exceeds ring capacity "
                             f"({self._max_h},{self._max_w},{self._max_c})")
        slot = self._write_count % self._slots
        dst = self._slot_view(slot, h, w, c)
        dst[...] = frame.reshape(h, w, c)
        del dst  # drop the view so it never lingers as a buffer export
        self._seqlock.write_begin()
        self._control.slot = slot
        self._control.seq = seq
        self._control.timestamp = timestamp
        self._control.h = h
        self._control.w = w
        self._control.c = c
        self._control.dtype_code = dtype_to_code(self._dtype)
        self._seqlock.write_end()
        self._write_count += 1

    def read_latest(self) -> tuple[np.ndarray, int, float] | None:
        def snapshot():
            return (int(self._control.seq), int(self._control.slot),
                    float(self._control.timestamp), int(self._control.h),
                    int(self._control.w), int(self._control.c))
        seq, slot, ts, h, w, c = self._seqlock.read(snapshot)
        if seq == 0:
            return None
        view = self._slot_view(slot, h, w, c).copy()  # decouple from later writers
        return view, seq, ts

    def close(self, unlink: bool) -> None:
        # Drop every export into the buffer (the control struct and the seqlock
        # word) before closing it; py3.12+ SharedMemory.close raises otherwise.
        self._control = None  # type: ignore[assignment]
        self._seqlock.release()
        gc.collect()
        self._shm.close()
        if unlink and self._owner:
            self._shm.unlink()

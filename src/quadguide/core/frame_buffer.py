from __future__ import annotations
import multiprocessing
import struct
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

__all__ = ["FrameBuffer"]

_TS_FMT  = struct.Struct("!Q")   # 8-byte big-endian uint64 for timestamp_ns
_TS_SIZE = _TS_FMT.size          # 8


class FrameBuffer:
    """Shared-memory ring buffer for zero-copy camera frame delivery.

    Created once in the parent process before forking. Workers inherit the
    SharedMemory handle and the atomic head Value across fork.

    Layout per slot:
        [0:8]       timestamp_ns  (big-endian uint64)
        [8:8+N]     frame bytes   (width × height × channels, row-major uint8)

    The camera worker is the sole writer. Readers call read_latest() which
    returns a numpy view into the live shared memory — callers must not hold
    the view past the next write cycle (~100 ms at 60 fps with 6 slots).
    """

    def __init__(
        self,
        width: int,
        height: int,
        channels: int = 3,
        n_slots: int = 6,
    ) -> None:
        self._width       = width
        self._height      = height
        self._channels    = channels
        self._n_slots     = n_slots
        self._frame_bytes = width * height * channels
        self._slot_bytes  = _TS_SIZE + self._frame_bytes

        self._shm  = SharedMemory(create=True, size=n_slots * self._slot_bytes)
        self._head = multiprocessing.Value("i", -1)  # -1 = nothing written yet

    def write_frame(self, arr: np.ndarray, timestamp_ns: int | None = None) -> None:
        """Write arr into the next ring slot and advance the head.

        arr must have shape (height, width, channels) and dtype uint8.
        timestamp_ns defaults to monotonic_ns() if not provided.
        """
        if timestamp_ns is None:
            timestamp_ns = time.monotonic_ns()

        slot   = (self._head.value + 1) % self._n_slots
        offset = slot * self._slot_bytes

        _TS_FMT.pack_into(self._shm.buf, offset, timestamp_ns)
        frame_start = offset + _TS_SIZE
        # tobytes() is an O(N) copy — unavoidable when writing numpy → shm buf
        data = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
        self._shm.buf[frame_start : frame_start + self._frame_bytes] = data

        self._head.value = slot  # atomic store; readers see consistent slot

    def read_latest(self) -> tuple[np.ndarray, int]:
        """Return (frame_view, timestamp_ns) for the most recently written slot.

        frame_view is a zero-copy numpy view into shared memory.  Reshape is
        performed here so callers always receive (H, W, C) uint8 arrays.
        Returns (None, 0) if no frame has been written yet.
        """
        h = self._head.value
        if h == -1:
            return None, 0

        offset = h * self._slot_bytes
        ts     = _TS_FMT.unpack_from(self._shm.buf, offset)[0]
        frame  = np.frombuffer(
            self._shm.buf,
            dtype=np.uint8,
            count=self._frame_bytes,
            offset=offset + _TS_SIZE,
        ).reshape(self._height, self._width, self._channels)
        return frame, ts

    def close(self) -> None:
        """Release this process's reference to the shared memory segment."""
        self._shm.close()

    def unlink(self) -> None:
        """Destroy the shared memory segment. Only the parent process should call this."""
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass

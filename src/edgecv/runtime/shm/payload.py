"""Variable-shape numpy payload channel (ARCHITECTURE.md §7.2).

Carries detector boxes+scores AND the candidate FilterState (complex arrays whose
shape depends on ROI size). Layout: a fixed header (magic, abi, seqlock, seq,
n_arrays) + a fixed-size array-descriptor table + a max-size data region. All
reads go through the seqlock; a fixed ctypes struct alone is insufficient because
the filter is variable-shape.
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
    code_to_dtype,
    dtype_to_code,
    validate_header,
)

_MAX_NAME = 24
_MAX_NDIM = 6


class _Header(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("seqlock", ctypes.c_uint64),
        ("seq", ctypes.c_uint64),
        ("n_arrays", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
    ]


class _ArrayDesc(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * _MAX_NAME),
        ("dtype_code", ctypes.c_uint32),
        ("ndim", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * _MAX_NDIM),
        ("offset", ctypes.c_uint64),   # bytes from start of data region
        ("nbytes", ctypes.c_uint64),
    ]


_HEADER_SIZE = ctypes.sizeof(_Header)
_DESC_SIZE = ctypes.sizeof(_ArrayDesc)
_SEQLOCK_OFFSET = _Header.seqlock.offset


class PayloadChannel:
    def __init__(self, shm, capacity_bytes: int, max_arrays: int, owner: bool):
        self._shm = shm
        self._capacity = capacity_bytes
        self._max_arrays = max_arrays
        self._owner = owner
        self._header = _Header.from_buffer(shm.buf, 0)
        self._desc_offset = _HEADER_SIZE
        self._data_offset = _HEADER_SIZE + max_arrays * _DESC_SIZE
        self._seqlock = SeqLock(shm.buf, _SEQLOCK_OFFSET)
        if owner:
            self._header.magic = MAGIC
            self._header.abi_version = ABI_VERSION
            self._header.seqlock = 0
            self._header.seq = 0
            self._header.n_arrays = 0
        else:
            validate_header(self._header.magic, self._header.abi_version)

    @property
    def name(self) -> str:
        return self._shm.name

    @staticmethod
    def _size(capacity_bytes: int, max_arrays: int) -> int:
        return _HEADER_SIZE + max_arrays * _DESC_SIZE + capacity_bytes

    @classmethod
    def create(cls, capacity_bytes: int, max_arrays: int = 8,
               name: str | None = None) -> PayloadChannel:
        shm = shared_memory.SharedMemory(
            create=True, size=cls._size(capacity_bytes, max_arrays), name=name)
        return cls(shm, capacity_bytes, max_arrays, owner=True)

    @classmethod
    def attach(cls, name: str, capacity_bytes: int, max_arrays: int = 8) -> PayloadChannel:
        shm = shared_memory.SharedMemory(name=name)
        return cls(shm, capacity_bytes, max_arrays, owner=False)

    def _desc(self, i: int) -> _ArrayDesc:
        return _ArrayDesc.from_buffer(self._shm.buf, self._desc_offset + i * _DESC_SIZE)

    def publish(self, arrays: dict[str, np.ndarray], seq: int) -> None:
        if len(arrays) > self._max_arrays:
            raise ValueError(f"{len(arrays)} arrays exceeds max_arrays={self._max_arrays}")
        # Pre-plan layout and check capacity before touching the seqlock.
        plan = []
        cursor = 0
        for name, arr in arrays.items():
            if len(name.encode()) >= _MAX_NAME:
                raise ValueError(f"array name too long: {name!r}")
            if arr.ndim > _MAX_NDIM:
                raise ValueError(f"array {name!r} ndim {arr.ndim} > {_MAX_NDIM}")
            arr = np.ascontiguousarray(arr)
            nbytes = arr.nbytes
            if cursor + nbytes > self._capacity:
                raise ValueError("payload exceeds channel capacity")
            plan.append((name, arr, cursor, nbytes))
            cursor += nbytes

        self._seqlock.write_begin()
        self._header.n_arrays = len(plan)
        for i, (name, arr, off, nbytes) in enumerate(plan):
            desc = self._desc(i)
            desc.name = name.encode()
            desc.dtype_code = dtype_to_code(arr.dtype)
            desc.ndim = arr.ndim
            for d in range(_MAX_NDIM):
                desc.shape[d] = arr.shape[d] if d < arr.ndim else 0
            desc.offset = off
            desc.nbytes = nbytes
            dst = np.ndarray((nbytes,), dtype=np.uint8, buffer=self._shm.buf,
                             offset=self._data_offset + off)
            dst[...] = arr.view(np.uint8).reshape(-1)
            del dst, desc  # drop buffer exports promptly
        self._header.seq = seq
        self._seqlock.write_end()

    def try_read(self) -> tuple[int, dict[str, np.ndarray]] | None:
        def snapshot():
            seq = int(self._header.seq)
            n = int(self._header.n_arrays)
            out: dict[str, np.ndarray] = {}
            for i in range(n):
                desc = self._desc(i)
                name = bytes(desc.name).rstrip(b"\x00").decode()
                dtype = code_to_dtype(desc.dtype_code)
                shape = tuple(int(desc.shape[d]) for d in range(int(desc.ndim)))
                raw = np.ndarray((int(desc.nbytes),), dtype=np.uint8,
                                 buffer=self._shm.buf,
                                 offset=self._data_offset + int(desc.offset))
                out[name] = raw.view(dtype).reshape(shape).copy()
            return seq, out
        seq, out = self._seqlock.read(snapshot)
        if seq == 0:
            return None
        return seq, out

    def close(self, unlink: bool) -> None:
        # Drop exports into the buffer (header struct + seqlock word) before close.
        self._header = None  # type: ignore[assignment]
        self._seqlock.release()
        gc.collect()
        self._shm.close()
        if unlink and self._owner:
            self._shm.unlink()

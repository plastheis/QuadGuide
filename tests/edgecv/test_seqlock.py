import gc
import threading
from multiprocessing import shared_memory

import numpy as np

from edgecv.runtime.shm.seqlock import SeqLock
from edgecv.runtime.shm.structs import ABI_VERSION, MAGIC, code_to_dtype, dtype_to_code


def test_magic_and_abi_present():
    assert isinstance(MAGIC, int)
    assert ABI_VERSION >= 1


def test_dtype_code_roundtrip():
    for name in ("float32", "uint8", "int8", "complex64", "float64"):
        assert code_to_dtype(dtype_to_code(np.dtype(name))) == np.dtype(name)


def test_seqlock_torn_read_is_retried():
    shm = shared_memory.SharedMemory(create=True, size=64)
    try:
        lock = SeqLock(shm.buf, offset=0)
        payload = np.ndarray((4,), dtype=np.int64, buffer=shm.buf, offset=8)
        payload[:] = [0, 0, 0, 0]

        stop = threading.Event()
        torn = {"seen": False}

        def writer():
            v = 0
            while not stop.is_set():
                v += 1
                lock.write_begin()
                for i in range(4):
                    payload[i] = v       # multi-field write, not atomic
                lock.write_end()

        def reader():
            for _ in range(20000):
                def read():
                    return [int(payload[i]) for i in range(4)]
                vals = lock.read(read)
                if len(set(vals)) != 1:
                    torn["seen"] = True
                    break

        w = threading.Thread(target=writer)
        w.start()
        reader()
        stop.set()
        w.join()

        assert torn["seen"] is False  # seqlock retried away every torn read
    finally:
        # Release every export into shm.buf before closing (py3.12+ BufferError):
        # the ndarray view plus the closures (writer/reader) that capture it.
        # Rebind (not `del`) so the names stay defined for the closure bodies.
        lock.release()
        lock = payload = writer = reader = None  # noqa: F841
        gc.collect()
        shm.close()
        shm.unlink()

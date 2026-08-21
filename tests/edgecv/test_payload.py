import numpy as np
import pytest

from edgecv.runtime.shm.payload import PayloadChannel


def test_try_read_before_publish_is_none():
    ch = PayloadChannel.create(capacity_bytes=64 * 1024, max_arrays=8)
    try:
        assert ch.try_read() is None
    finally:
        ch.close(unlink=True)


def test_variable_shape_roundtrip():
    ch = PayloadChannel.create(capacity_bytes=256 * 1024, max_arrays=8)
    try:
        arrays = {
            "boxes": np.array([[0.1, 0.2, 0.3, 0.4]], np.float32),
            "scores": np.array([0.9], np.float32),
            "H": (np.random.rand(13, 21) + 1j * np.random.rand(13, 21)).astype(np.complex64),
        }
        ch.publish(arrays, seq=5)
        out = ch.try_read()
        assert out is not None
        seq, got = out
        assert seq == 5
        assert set(got) == set(arrays)
        for k in arrays:
            np.testing.assert_array_equal(got[k], arrays[k])
    finally:
        ch.close(unlink=True)


def test_latest_publish_wins():
    ch = PayloadChannel.create(capacity_bytes=64 * 1024, max_arrays=4)
    try:
        ch.publish({"a": np.array([1], np.int32)}, seq=1)
        ch.publish({"a": np.array([2], np.int32)}, seq=2)
        seq, got = ch.try_read()
        assert seq == 2 and int(got["a"][0]) == 2
    finally:
        ch.close(unlink=True)


def test_capacity_overflow_raises():
    ch = PayloadChannel.create(capacity_bytes=128, max_arrays=2)
    try:
        with pytest.raises(ValueError):
            ch.publish({"big": np.zeros(10_000, np.float64)}, seq=1)
    finally:
        ch.close(unlink=True)


def test_attach_reads_other_handle():
    a = PayloadChannel.create(capacity_bytes=64 * 1024, max_arrays=4)
    try:
        a.publish({"x": np.array([3.0], np.float32)}, seq=7)
        b = PayloadChannel.attach(a.name, capacity_bytes=64 * 1024, max_arrays=4)
        try:
            seq, got = b.try_read()
            assert seq == 7 and got["x"][0] == 3.0
        finally:
            b.close(unlink=False)
    finally:
        a.close(unlink=True)

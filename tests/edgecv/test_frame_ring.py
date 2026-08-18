import numpy as np

from edgecv.runtime.shm.frame_ring import FrameRing


def test_publish_then_read_latest_roundtrips():
    ring = FrameRing.create(slots=4, max_h=8, max_w=8, max_c=3, dtype="uint8")
    try:
        frame = (np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
        ring.publish(frame, seq=1, timestamp=10.0)
        got = ring.read_latest()
        assert got is not None
        view, seq, ts = got
        assert seq == 1 and ts == 10.0
        np.testing.assert_array_equal(view, frame)
    finally:
        ring.close(unlink=True)


def test_latest_only_skips_to_newest():
    ring = FrameRing.create(slots=4, max_h=4, max_w=4, max_c=1, dtype="uint8")
    try:
        for s in range(1, 6):  # more than slots, forces wraparound
            ring.publish(np.full((4, 4, 1), s, np.uint8), seq=s, timestamp=float(s))
        view, seq, ts = ring.read_latest()
        assert seq == 5
        assert int(view[0, 0, 0]) == 5
    finally:
        ring.close(unlink=True)


def test_read_before_any_publish_returns_none():
    ring = FrameRing.create(slots=2, max_h=4, max_w=4, max_c=1, dtype="uint8")
    try:
        assert ring.read_latest() is None
    finally:
        ring.close(unlink=True)


def test_attach_reads_producer_frames():
    producer = FrameRing.create(slots=3, max_h=4, max_w=4, max_c=1, dtype="uint8")
    try:
        producer.publish(np.full((4, 4, 1), 7, np.uint8), seq=99, timestamp=1.0)
        consumer = FrameRing.attach(producer.name, slots=3, max_h=4, max_w=4,
                                    max_c=1, dtype="uint8")
        try:
            view, seq, ts = consumer.read_latest()
            assert seq == 99 and int(view[0, 0, 0]) == 7
        finally:
            consumer.close(unlink=False)
    finally:
        producer.close(unlink=True)

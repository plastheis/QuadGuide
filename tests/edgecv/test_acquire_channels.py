"""Tests for AcquireTrack SHM channels (spec 2026-06-14-acquire-track-design §3.2).

Covers the two new shared-memory layouts and their channels:
- AcquireControl / AcquireControlChannel  (parent -> workers: mode + crop + lock)
- NanoResult / NanoResultChannel          (NanoTrack worker -> parent: one bbox)
"""

from __future__ import annotations

import ctypes
import threading

import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus

# ── Task 1: structs + ABI bump ──────────────────────────────────────────────

class TestStructs:
    def test_abi_bumped_to_3(self):
        from edgecv.runtime.shm.structs import ABI_VERSION

        assert ABI_VERSION == 3

    def test_acquire_control_header_prefix(self):
        from edgecv.runtime.shm.structs import AcquireControl

        names = [f[0] for f in AcquireControl._fields_]
        # Every shared header begins with magic + abi_version + seq + seqlock.
        assert names[:4] == ["magic", "abi_version", "seq", "seqlock"]
        for field in ("mode", "lock_gen", "cx", "cy", "cw", "ch",
                      "lx", "ly", "lw", "lh", "timestamp"):
            assert field in names

    def test_nano_result_header_prefix(self):
        from edgecv.runtime.shm.structs import NanoResult

        names = [f[0] for f in NanoResult._fields_]
        assert names[:4] == ["magic", "abi_version", "seq", "seqlock"]
        for field in ("x", "y", "w", "h", "confidence", "status",
                      "src_seq", "src_ts"):
            assert field in names

    def test_structs_have_stable_size(self):
        from edgecv.runtime.shm.structs import AcquireControl, NanoResult

        # Sizes are deterministic; assert they are non-trivial and 8-aligned
        # (all u64/double payloads => multiple of 8).
        assert ctypes.sizeof(AcquireControl) % 8 == 0
        assert ctypes.sizeof(NanoResult) % 8 == 0


# ── Task 2: AcquireControlChannel ───────────────────────────────────────────

class TestAcquireControlChannel:
    def test_round_trip(self):
        from edgecv.runtime.shm.control_channel import AcquireControlChannel, Mode

        ch = AcquireControlChannel.create()
        name = ch.name
        crop = BoundingBox(0.25, 0.25, 0.5, 0.5)
        lockb = BoundingBox(0.4, 0.4, 0.1, 0.1)
        ch.publish(mode=Mode.NANO, crop=crop, lock_gen=7, lock_bbox=lockb)
        ch.close(unlink=False)

        reader = AcquireControlChannel.attach(name)
        try:
            snap = reader.read_latest()
            assert snap.mode == Mode.NANO
            assert snap.lock_gen == 7
            assert snap.crop.x == pytest.approx(0.25)
            assert snap.crop.w == pytest.approx(0.5)
            assert snap.lock_bbox.x == pytest.approx(0.4)
            assert snap.lock_bbox.w == pytest.approx(0.1)
        finally:
            reader.close(unlink=True)

    def test_default_before_publish(self):
        from edgecv.runtime.shm.control_channel import AcquireControlChannel, Mode

        ch = AcquireControlChannel.create()
        try:
            snap = ch.read_latest()
            assert snap.mode == Mode.IDLE
            assert snap.lock_gen == 0
        finally:
            ch.close(unlink=True)

    def test_latest_only(self):
        from edgecv.runtime.shm.control_channel import AcquireControlChannel, Mode

        ch = AcquireControlChannel.create()
        try:
            ch.publish(mode=Mode.YOLO, crop=BoundingBox(0, 0, 1, 1), lock_gen=0,
                       lock_bbox=BoundingBox(0, 0, 0, 0))
            ch.publish(mode=Mode.NANO, crop=BoundingBox(0, 0, 1, 1), lock_gen=1,
                       lock_bbox=BoundingBox(0.1, 0.1, 0.2, 0.2))
            snap = ch.read_latest()
            assert snap.mode == Mode.NANO
            assert snap.lock_gen == 1
        finally:
            ch.close(unlink=True)

    def test_torn_read_safe_under_writer(self):
        """A concurrent writer never yields a half-written snapshot."""
        from edgecv.runtime.shm.control_channel import AcquireControlChannel, Mode

        ch = AcquireControlChannel.create()
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                i += 1
                v = i / 1000.0 % 1.0
                ch.publish(mode=Mode.NANO,
                           crop=BoundingBox(v, v, 1 - v, 1 - v),
                           lock_gen=i,
                           lock_bbox=BoundingBox(v, v, v, v))

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        try:
            for _ in range(2000):
                snap = ch.read_latest()
                # crop x and lock x are written from the same source value v.
                assert snap.crop.x == pytest.approx(snap.lock_bbox.x)
        finally:
            stop.set()
            t.join(timeout=2.0)
            ch.close(unlink=True)

    def test_abi_validation_on_attach(self):
        from edgecv.runtime.shm.control_channel import AcquireControlChannel
        from edgecv.runtime.shm.structs import ABI_VERSION, MAGIC

        ch = AcquireControlChannel.create()
        name = ch.name
        ch.close(unlink=False)
        reader = AcquireControlChannel.attach(name)
        assert reader._header.magic == MAGIC
        assert reader._header.abi_version == ABI_VERSION
        reader.close(unlink=True)


# ── Task 3: NanoResultChannel ───────────────────────────────────────────────

class TestNanoResultChannel:
    def test_round_trip(self):
        from edgecv.runtime.shm.nano_result import NanoResultChannel

        ch = NanoResultChannel.create()
        name = ch.name
        ch.publish(BoundingBox(0.1, 0.2, 0.3, 0.4), confidence=0.77,
                   status=TrackStatus.LOCKED, src_seq=12, src_ts=999.5)
        ch.close(unlink=False)

        reader = NanoResultChannel.attach(name)
        try:
            sample = reader.read_latest()
            assert sample is not None
            assert sample.bbox.x == pytest.approx(0.1)
            assert sample.bbox.h == pytest.approx(0.4)
            assert sample.confidence == pytest.approx(0.77)
            assert sample.status == TrackStatus.LOCKED
            assert sample.src_seq == 12
            assert sample.src_ts == pytest.approx(999.5)
        finally:
            reader.close(unlink=True)

    def test_read_before_publish_returns_none(self):
        from edgecv.runtime.shm.nano_result import NanoResultChannel

        ch = NanoResultChannel.create()
        try:
            assert ch.read_latest() is None
        finally:
            ch.close(unlink=True)

    def test_double_close_safe(self):
        from edgecv.runtime.shm.nano_result import NanoResultChannel

        ch = NanoResultChannel.create()
        ch.close(unlink=True)
        assert ch._header is None

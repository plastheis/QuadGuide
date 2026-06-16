"""Tests for the raw-TCP HIL camera source (raw_frame_source.RawFrameCamera).

Exercises the wire protocol against an in-process loopback server on an
ephemeral 127.0.0.1 port: header round-trip, latest-wins delivery, recv-boundary
reassembly, and the CameraSource contract / worker registration.
"""
from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np
import pytest

from quadguide.perception.camera.sources import CameraSource
from quadguide.perception.camera.raw_frame_source import (
    RawFrameCamera,
    _HEADER_FMT,
    _MAGIC,
)


def _pack(frame: np.ndarray, seq: int, stamp_ns: int, pixfmt: int = 0) -> bytes:
    h, w = frame.shape[0], frame.shape[1]
    body = frame.tobytes()
    hdr = struct.pack(
        _HEADER_FMT, _MAGIC, 1, pixfmt, seq, w, h, len(body), stamp_ns
    )
    return hdr + body


class _LoopbackServer:
    """Minimal RawFrameServer stand-in: accepts one client and writes frames."""

    def __init__(self, frames: list[bytes], chunk: int | None = None):
        self._frames = frames
        self._chunk = chunk
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        conn, _ = self._srv.accept()
        with conn:
            payload = b"".join(self._frames)
            if self._chunk:
                for i in range(0, len(payload), self._chunk):
                    conn.sendall(payload[i : i + self._chunk])
                    time.sleep(0.005)
            else:
                conn.sendall(payload)
            # Hold the connection open so the reader doesn't see EOF mid-test.
            time.sleep(1.0)

    def close(self) -> None:
        try:
            self._srv.close()
        except OSError:
            pass


class _Cfg:
    """getattr-compatible stand-in for CameraConfig."""

    def __init__(self, port: int, width: int = 4, height: int = 3):
        self.raw_tcp_host = "127.0.0.1"
        self.raw_tcp_port = port
        self.width = width
        self.height = height


class TestContract:
    def test_is_camera_source(self):
        assert issubclass(RawFrameCamera, CameraSource)

    def test_open_times_out_without_server(self):
        cam = RawFrameCamera(_Cfg(port=1))  # nothing listening
        with pytest.raises(RuntimeError):
            cam.open()
        cam.close()


class TestProtocol:
    def test_roundtrip_single_frame(self):
        frame = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(3, 4, 3)
        srv = _LoopbackServer([_pack(frame, seq=1, stamp_ns=12345)])
        cam = RawFrameCamera(_Cfg(srv.port))
        try:
            before = time.monotonic_ns()
            cam.open()
            out, ts = cam.read()
            # ts is stamped on the SBC clock at receive, NOT the sender's header
            # stamp_ns (12345) — see RawFrameCamera module docstring.
            assert ts >= before
            np.testing.assert_array_equal(out, frame)
        finally:
            cam.close()
            srv.close()

    def test_recv_boundary_reassembly(self):
        frame = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(3, 4, 3)
        # 7-byte chunks split header (32B) and body across recv() boundaries.
        srv = _LoopbackServer([_pack(frame, seq=1, stamp_ns=99)], chunk=7)
        cam = RawFrameCamera(_Cfg(srv.port))
        try:
            cam.open()
            out, _ts = cam.read()
            np.testing.assert_array_equal(out, frame)
        finally:
            cam.close()
            srv.close()

    def test_latest_wins(self):
        frames = [
            _pack(np.full((3, 4, 3), i, dtype=np.uint8), seq=i, stamp_ns=i)
            for i in range(1, 6)
        ]
        srv = _LoopbackServer(frames)
        cam = RawFrameCamera(_Cfg(srv.port))
        try:
            cam.open()
            time.sleep(0.1)  # let all frames arrive; reader keeps only the newest
            out, _ts = cam.read()
            # frame i is filled with value i; newest (seq 5) must win.
            assert int(out.flat[0]) == 5
        finally:
            cam.close()
            srv.close()

    def test_resize_on_dim_mismatch(self):
        frame = np.full((3, 4, 3), 7, dtype=np.uint8)
        srv = _LoopbackServer([_pack(frame, seq=1, stamp_ns=1)])
        cam = RawFrameCamera(_Cfg(srv.port, width=8, height=6))
        try:
            cam.open()
            out, _ = cam.read()
            assert out.shape[:2] == (6, 8)
        finally:
            cam.close()
            srv.close()


def test_registered_as_backend():
    from quadguide.perception.camera.worker import _SOURCES
    assert _SOURCES["raw_tcp"] is RawFrameCamera

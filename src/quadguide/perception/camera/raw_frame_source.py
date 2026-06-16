"""Raw-frame network camera source — reads uncompressed BGR/GRAY frames over TCP (HIL).

This is the SBC-side receiver for the hil-test ``RawFrameServer`` raw-TCP frame
transport (design: 2026-06-14-tcp-raw-camera-transport-design.md). It replaces
the HTTP/MJPEG path (network_source.py:NetworkCamera) on a direct-connect 1GbE
HIL link: no JPEG decode, no cv2.VideoCapture buffering, latest-wins delivery for
minimal glass→track latency. Enabled via config: platform.camera.backend =
"raw_tcp".

A background reader thread keeps the newest frame in a single slot; ``read()``
blocks for the *next* frame (new seq) so the camera worker gets one fresh frame
per call (same contract as USBCamera/NetworkCamera) instead of busy-spinning on
duplicates. Depends only on numpy + stdlib (+ cv2 only for the rare resize).

The frame timestamp returned by ``read()`` is stamped on the SBC when the frame
finishes arriving — NOT the sender's ``stamp_ns`` from the header. quadguide's
latency telemetry subtracts this value (as ``origin_ns``) from SBC-side
CLOCK_MONOTONIC reads (tracker_worker, ground/server, diagtrace); the dev-machine
render clock is a different monotonic epoch, so feeding it through would corrupt
every latency number. The header ``stamp_ns`` is for sender-side glass-to-glass
measurement, which needs cross-machine clock sync we don't have. Same rationale
as NetworkCamera stamping at SBC-receive.

Wire protocol — 32-byte big-endian header then ``frame_bytes`` of raw pixels:
    >2sBBIHHIQ8x : magic 'HF', version, pixfmt, seq, width, height,
                   frame_bytes, stamp_ns, 8x reserved
"""
from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np

from .sources import CameraSource

_MAGIC = b"HF"
_HEADER_FMT = ">2sBBIHHIQ8x"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 32
_CHANNELS = {0: 3, 1: 1}


class RawFrameCamera(CameraSource):
    """HIL virtual camera over the raw-TCP frame transport."""

    def __init__(self, config) -> None:
        self._host   = getattr(config, "raw_tcp_host", "") or "127.0.0.1"
        self._port   = int(getattr(config, "raw_tcp_port", 0) or 8091)
        self._width  = getattr(config, "width",  640)
        self._height = getattr(config, "height", 480)

        self._sock: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._latest = None    # (frame, recv_ts_ns, seq)
        self._last_seq = -1     # newest seq handed to read()

    def open(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="raw-frame-camera", daemon=True
        )
        self._thread.start()
        # Block until the first frame arrives, mirroring NetworkCamera's connect
        # settle: the worker must not start reading before the stream is live.
        with self._cond:
            if not self._cond.wait_for(lambda: self._latest is not None, timeout=5.0):
                raise RuntimeError(
                    f"RawFrameCamera: no frames from {self._host}:{self._port}"
                )

    def read(self) -> tuple[np.ndarray, int]:
        # Wait for a frame newer than the last one we returned, so each read()
        # yields one fresh frame (no duplicate timestamps into the frame buffer).
        # On teardown _running drops and we surface the same fault contract the
        # camera worker expects from USBCamera.read().
        with self._cond:
            self._cond.wait_for(
                lambda: not self._running
                or (self._latest is not None and self._latest[2] != self._last_seq)
            )
            if not self._running or self._latest is None:
                raise RuntimeError("RawFrameCamera: frame capture failed (stream ended?)")
            frame, recv_ts, seq = self._latest
            self._last_seq = seq
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            import cv2
            frame = cv2.resize(frame, (self._width, self._height))
        return frame, recv_ts

    def close(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()  # unblock a blocking recv in the reader thread
            except OSError:
                pass
            self._sock = None
        with self._cond:
            self._cond.notify_all()  # release a read()/open() blocked on the condition
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    # internals -------------------------------------------------------------
    def _run(self) -> None:
        while self._running:
            try:
                self._sock = socket.create_connection(
                    (self._host, self._port), timeout=5.0
                )
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._read_stream()
            except OSError:
                pass
            finally:
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                    self._sock = None
            if self._running:
                time.sleep(0.5)  # backoff before reconnect

    def _read_stream(self) -> None:
        while self._running:
            hdr = self._recv_exact(HEADER_SIZE)
            if hdr is None:
                return
            magic, _ver, pixfmt, seq, w, h, frame_bytes, _stamp_ns = struct.unpack(
                _HEADER_FMT, hdr
            )
            if magic != _MAGIC:
                return  # desync → drop + reconnect
            body = self._recv_exact(frame_bytes)
            if body is None:
                return
            # Stamp arrival on the SBC clock (see module docstring) — the header's
            # _stamp_ns is the dev-machine render clock and is intentionally dropped.
            recv_ts = time.monotonic_ns()
            ch = _CHANNELS.get(pixfmt, 3)
            frame = np.frombuffer(body, dtype=np.uint8).reshape(h, w, ch).copy()
            with self._cond:
                self._latest = (frame, recv_ts, seq)
                self._cond.notify_all()

    def _recv_exact(self, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n and self._running:
            try:
                chunk = self._sock.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf) if len(buf) == n else None

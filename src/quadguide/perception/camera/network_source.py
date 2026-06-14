"""Network camera source — reads frames from an HTTP MJPEG stream (HIL).

In HIL mode the dev machine renders synthetic camera frames and serves the
latest one as a multipart MJPEG stream. The SBC reads it through
cv2.VideoCapture, which handles the HTTP + JPEG decode transparently. This is
the camera-side half of the HIL link (the serial-side half is
link/tcp_serial.py:TCPSerialPort). Enabled via config: platform.camera.backend
= "network".
"""
from __future__ import annotations

import time

import numpy as np

from .sources import CameraSource


class NetworkCamera(CameraSource):
    """Read frames from an HTTP MJPEG stream.

    Receives a ``CameraConfig`` dataclass (same as the other sources); the
    ``url`` field selects the stream.
    """

    def __init__(self, config) -> None:
        self._url    = getattr(config, "url", "") or "http://localhost:8090/camera"
        self._width  = getattr(config, "width",  640)
        self._height = getattr(config, "height", 480)
        self._cap    = None

    def open(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self._url)
        # Keep only the newest frame. cv2's default capture queue hands back
        # stale buffered frames, which would inflate the glass→track latency the
        # tracker's new-frame gate is built to minimise (ARCHITECTURE §13).
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        time.sleep(0.5)  # let the HTTP stream connect before the first read
        if not self._cap.isOpened():
            raise RuntimeError(f"NetworkCamera: failed to open {self._url}")

    def read(self) -> tuple[np.ndarray, int]:
        ret, frame = self._cap.read()
        ts = time.monotonic_ns()
        if not ret:
            # Treated as teardown if SIGTERM interrupted the read, else a fault —
            # same contract the camera worker expects from USBCamera.read().
            raise RuntimeError("NetworkCamera: frame capture failed (stream ended?)")
        if frame.shape[0] != self._height or frame.shape[1] != self._width:
            import cv2
            frame = cv2.resize(frame, (self._width, self._height))
        return frame, ts

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

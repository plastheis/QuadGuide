from __future__ import annotations
import abc
import time
import numpy as np

__all__ = ["CameraSource", "USBCamera", "CSICamera", "VirtualCamera"]


class CameraSource(abc.ABC):
    """Abstract base for all camera input sources."""

    @abc.abstractmethod
    def open(self) -> None:
        """Open the camera device or pipeline."""

    @abc.abstractmethod
    def read(self) -> tuple[np.ndarray, int]:
        """Return (frame_bgr, timestamp_ns). Blocks until a frame is available."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the camera device or pipeline."""

    def __iter__(self):
        while True:
            yield self.read()


class USBCamera(CameraSource):
    """V4L2 USB camera via cv2.VideoCapture."""

    _DEVICE = "/dev/video0"  # matches cv2.VideoCapture(0)

    def __init__(self, config) -> None:
        # config is a CameraConfig dataclass or dict-like with width/height/fps
        self._width  = getattr(config, "width",  640)
        self._height = getattr(config, "height", 480)
        self._fps    = getattr(config, "fps",     30)
        self._cap    = None

    def open(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        if not self._cap.isOpened():
            raise RuntimeError("USBCamera: failed to open /dev/video0")
        self._force_constant_framerate()

    def _force_constant_framerate(self) -> None:
        # UVC webcams (e.g. Logitech C920) ship with exposure_dynamic_framerate=1,
        # which lets auto-exposure lengthen exposure in low light and silently drop
        # the frame rate (measured 30 → 24 fps on this rig). Force a constant rate so
        # the tracker/guidance loop sees frames at the configured cadence. Best-effort:
        # needs v4l2-ctl and is a harmless no-op if the control is absent.
        import subprocess
        for ctrl in ("exposure_dynamic_framerate", "exposure_auto_priority"):
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", self._DEVICE, "-c", f"{ctrl}=0"],
                    capture_output=True, timeout=2,
                )
            except (OSError, subprocess.SubprocessError):
                pass

    def read(self) -> tuple[np.ndarray, int]:
        ret, frame = self._cap.read()
        ts = time.monotonic_ns()
        if not ret:
            raise RuntimeError("USBCamera: frame capture failed")
        return frame, ts

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CSICamera(CameraSource):
    """CSI camera via a GStreamer pipeline string, read through cv2.VideoCapture."""

    def __init__(self, config) -> None:
        self._pipeline = getattr(config, "pipeline", "")
        self._cap      = None

    def open(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(f"CSICamera: failed to open pipeline: {self._pipeline!r}")

    def read(self) -> tuple[np.ndarray, int]:
        ret, frame = self._cap.read()
        ts = time.monotonic_ns()
        if not ret:
            raise RuntimeError("CSICamera: frame capture failed")
        return frame, ts

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VirtualCamera(CameraSource):
    """Stub camera for HIL mode.

    # STUB: HIL not yet implemented — see hil/virtual_source.py for the full
    # implementation that renders synthetic frames from the dynamics simulation.
    # Replace this class body when building the hil/ module.
    """

    def open(self) -> None:
        raise NotImplementedError("VirtualCamera: HIL not yet implemented")

    def read(self) -> tuple[np.ndarray, int]:
        raise NotImplementedError("VirtualCamera: HIL not yet implemented")

    def close(self) -> None:
        pass

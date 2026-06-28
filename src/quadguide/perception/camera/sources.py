from __future__ import annotations
import abc
import re
import subprocess
import threading
import time
import numpy as np

__all__ = [
    "CameraSource", "USBCamera", "CSICamera", "CSIY10Camera", "VirtualCamera",
    "unpack_raw10_to_gray8",
]


def unpack_raw10_to_gray8(buf, width: int, height: int, stride: int) -> np.ndarray:
    """Unpack one MIPI RAW10-packed mono frame to an 8-bit greyscale image.

    The ROCK 5C rkcif node delivers the OV9281's ``Y10`` as MIPI RAW10: four
    pixels per five bytes (four high-8-bit bytes, then one byte packing the four
    2-bit LSBs as ``p3 p2 p1 p0`` from the high bits down), with each row padded
    to ``stride`` bytes. GStreamer has no Y10 mapping, so we read the raw V4L2
    buffer and unpack here. Returns an ``(height, width)`` uint8 array (the 10-bit
    value scaled down by ``>> 2``).
    """
    packed_per_row = width * 5 // 4
    a = np.frombuffer(buf, dtype=np.uint8)[: stride * height]
    a = a.reshape(height, stride)[:, :packed_per_row]
    g = a.reshape(height, -1, 5).astype(np.uint16)
    lo = g[..., 4]
    p0 = (g[..., 0] << 2) | (lo & 0x3)
    p1 = (g[..., 1] << 2) | ((lo >> 2) & 0x3)
    p2 = (g[..., 2] << 2) | ((lo >> 4) & 0x3)
    p3 = (g[..., 3] << 2) | ((lo >> 6) & 0x3)
    px10 = np.stack((p0, p1, p2, p3), axis=-1).reshape(height, width)
    return (px10 >> 2).astype(np.uint8)


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


class CSIY10Camera(CameraSource):
    """OV9281 global-shutter mono CSI camera (ROCK 5C / rkcif), via direct V4L2.

    The OV9281 advertises only ``Y10`` (10-bit mono) and the rkcif node delivers it
    as MIPI RAW10-packed multiplanar buffers. GStreamer's ``v4l2src`` has no Y10
    mapping (``cv2.CAP_GSTREAMER`` returns not-negotiated), so the GStreamer
    ``CSICamera`` path can't be used. Instead we stream raw frames from ``v4l2-ctl``
    and unpack to BGR here. A background thread keeps only the newest raw frame
    (latest-wins → minimal glass→track latency); ``read()`` unpacks on demand so
    only consumed frames cost CPU. Selected via config: platform.camera.backend = "csi".
    """

    _SENSOR_W = 1280   # OV9281 native resolution
    _SENSOR_H = 800
    _SENSOR_NAME = "ov9281"   # substring used to find the sensor media entity
    _FOURCC = "Y10 "          # V4L2 'Y10 ' pixelformat — the trailing space IS significant

    def __init__(self, config) -> None:
        self._device = getattr(config, "device", "") or "/dev/video0"
        self._media  = getattr(config, "media", "")  or "/dev/media0"
        self._out_w  = getattr(config, "width",  640)
        self._out_h  = getattr(config, "height", 400)

        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._latest = None        # (raw_bytes, ts_ns, seq)
        self._last_seq = -1
        self._seq = 0
        self._running = False
        self._sizeimage = 0
        self._stride = 0

    def open(self) -> None:
        # Sensor is Y10-only; pin the pad format so the rkcif link validates, set the
        # capture-node format, then learn the exact packed buffer geometry (stride is
        # padded — 1792 B/row for 1280px, not 1280*1.25). Pad setup is best-effort: the
        # pad is already Y10 after bind, but we set it so a fresh format negotiates.
        self._set_sensor_pad()
        self._v4l2([self._fmt_arg()])
        self._sizeimage, self._stride = self._query_geometry()
        if self._sizeimage <= 0 or self._stride <= 0:
            raise RuntimeError(
                f"CSIY10Camera: could not determine frame geometry for {self._device}"
            )

        self._running = True
        # stderr -> DEVNULL: v4l2-ctl prints a '<' per frame to stderr, which would
        # fill and block a pipe; on failure we re-probe via _diagnose() instead.
        self._proc = subprocess.Popen(
            ["v4l2-ctl", "-d", self._device, self._fmt_arg(),
             "--stream-mmap", "--stream-to=-", "--stream-count=0"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
        )
        self._thread = threading.Thread(
            target=self._run, name="csi-y10-camera", daemon=True
        )
        self._thread.start()
        with self._cond:
            if not self._cond.wait_for(lambda: self._latest is not None, timeout=5.0):
                err = self._diagnose()
                raise RuntimeError(
                    f"CSIY10Camera: no frames from {self._device}. "
                    f"v4l2-ctl: {err or '(no diagnostic output)'}"
                )

    def read(self) -> tuple[np.ndarray, int]:
        # Block for a frame newer than the last returned, then unpack on demand:
        # only frames perception actually consumes pay the RAW10→BGR cost.
        with self._cond:
            self._cond.wait_for(
                lambda: not self._running
                or (self._latest is not None and self._latest[2] != self._last_seq)
            )
            if not self._running or self._latest is None:
                raise RuntimeError("CSIY10Camera: frame capture failed (stream ended?)")
            raw, ts, seq = self._latest
            self._last_seq = seq
        gray = unpack_raw10_to_gray8(raw, self._SENSOR_W, self._SENSOR_H, self._stride)
        import cv2
        if (self._out_w, self._out_h) != (self._SENSOR_W, self._SENSOR_H):
            gray = cv2.resize(gray, (self._out_w, self._out_h),
                              interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # mono → 3-ch BGR for perception
        return frame, ts

    def close(self) -> None:
        self._running = False
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    # internals -------------------------------------------------------------
    def _run(self) -> None:
        n = self._sizeimage
        stream = self._proc.stdout
        while self._running:
            raw = self._read_exact(stream, n)
            if raw is None:
                break  # v4l2-ctl exited / stream ended
            ts = time.monotonic_ns()
            with self._cond:
                self._seq += 1
                self._latest = (raw, ts, self._seq)
                self._cond.notify_all()
        with self._cond:           # wake any read() blocked on a dead stream
            self._running = False
            self._cond.notify_all()

    @staticmethod
    def _read_exact(stream, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _fmt_arg(self) -> str:
        # Note the trailing space in the fourcc — 'Y10' (3 chars) is rejected by
        # v4l2-ctl ("pixelformat 'Y10' is invalid"); the V4L2 fourcc is 'Y10 '.
        return ("--set-fmt-video=width=%d,height=%d,pixelformat=%s"
                % (self._SENSOR_W, self._SENSOR_H, self._FOURCC))

    def _diagnose(self) -> str:
        """One-shot capture to surface why the stream produced no frames."""
        try:
            r = subprocess.run(
                ["v4l2-ctl", "-d", self._device, self._fmt_arg(),
                 "--stream-mmap", "--stream-count=1", "--stream-to=/dev/null"],
                capture_output=True, text=True, timeout=5,
            )
            return (r.stderr or r.stdout or "").strip().replace("\n", " ")[:300]
        except (OSError, subprocess.SubprocessError) as exc:
            return str(exc)

    def _v4l2(self, args: list[str]) -> str:
        try:
            out = subprocess.run(
                ["v4l2-ctl", "-d", self._device, *args],
                capture_output=True, text=True, timeout=5,
            )
            return out.stdout
        except (OSError, subprocess.SubprocessError):
            return ""

    def _query_geometry(self) -> tuple[int, int]:
        """Return (sizeimage, bytesperline) from the node's current format."""
        text = self._v4l2(["--get-fmt-video"])
        size = re.search(r"Size Image\s*:\s*(\d+)", text)
        line = re.search(r"Bytes per Line\s*:\s*(\d+)", text)
        sizeimage = int(size.group(1)) if size else 0
        stride = int(line.group(1)) if line else 0
        if sizeimage and not stride:        # fall back if stride wasn't reported
            stride = sizeimage // self._SENSOR_H
        return sizeimage, stride

    def _set_sensor_pad(self) -> None:
        """Best-effort: set the OV9281 sensor pad to Y10 so the rkcif link validates."""
        graph = self._media_ctl(["-p"])
        m = re.search(r"entity \d+: (\S[^\n(]*%s[^\n(]*) \(" % self._SENSOR_NAME, graph)
        if not m:
            return
        name = m.group(1).strip()
        self._media_ctl([
            "--set-v4l2",
            '"%s":0[fmt:Y10_1X10/%dx%d]' % (name, self._SENSOR_W, self._SENSOR_H),
        ])

    def _media_ctl(self, args: list[str]) -> str:
        try:
            out = subprocess.run(
                ["media-ctl", "-d", self._media, *args],
                capture_output=True, text=True, timeout=5,
            )
            return out.stdout
        except (OSError, subprocess.SubprocessError):
            return ""


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

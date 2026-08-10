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
    """Unpack one MIPI RAW10 bit-packed mono frame to an 8-bit greyscale image.

    The ROCK 5C rkcif node delivers the OV9281's ``Y10`` as genuine MIPI RAW10:
    four 10-bit pixels bit-packed little-endian across five bytes (40 bits), with
    each row padded to ``stride`` bytes. For bytes ``b0..b4`` of a group the pixel
    values are::

        p0 =  b0        | (b1 & 0x03) << 8
        p1 = (b1 >> 2)  | (b2 & 0x0f) << 6
        p2 = (b2 >> 4)  | (b3 & 0x3f) << 4
        p3 = (b3 >> 6)  | (b4 << 2)

    (This is NOT the CSI-2 "four high-8-bit bytes + one LSB-pack byte" layout —
    reading it that way leaves only every 4th column correct, producing vertical
    striping and a concentric-ring moiré once downscaled.) GStreamer has no Y10
    mapping, so we read the raw V4L2 buffer and unpack here. Returns an
    ``(height, width)`` uint8 array (the 10-bit value scaled down by ``>> 2``).
    """
    packed_per_row = width * 5 // 4
    a = np.frombuffer(buf, dtype=np.uint8)[: stride * height]
    a = a.reshape(height, stride)[:, :packed_per_row]
    g = a.reshape(height, -1, 5).astype(np.uint16)
    b0, b1, b2, b3, b4 = (g[..., k] for k in range(5))
    p0 = b0 | ((b1 & 0x03) << 8)
    p1 = (b1 >> 2) | ((b2 & 0x0f) << 6)
    p2 = (b2 >> 4) | ((b3 & 0x3f) << 4)
    p3 = (b3 >> 6) | (b4 << 2)
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
    """CSI camera via a GStreamer pipeline string, read through cv2.VideoCapture.

    On a ``libcamerasrc`` pipeline (the RPi OV9281 path) the exposure/gain controls
    from config are injected as element properties — see ``_libcamera_props``.
    """

    # libcamera builds this name as LIBCAMERA_<ipa>_TUNING_FILE; for the RPi IPA it is
    # RPI (NOT RPI_VC4 — verified on the board: the VC4 spelling is ignored and the
    # stock file loads silently).
    _TUNING_ENV = "LIBCAMERA_RPI_TUNING_FILE"

    def __init__(self, config) -> None:
        self._pipeline = self._with_libcamera_controls(
            getattr(config, "pipeline", ""), config
        )
        self._tuning_file = self._resolve_tuning_file(getattr(config, "tuning_file", ""))
        self._cap      = None

    @staticmethod
    def _resolve_tuning_file(tuning_file: str) -> str:
        """Absolute path of the configured IPA tuning file ('' if unset)."""
        if not tuning_file:
            return ""
        from pathlib import Path
        path = Path(tuning_file)
        if not path.is_absolute():
            # parents[4] == repo root (src/quadguide/perception/camera/sources.py)
            path = Path(__file__).resolve().parents[4] / path
        if not path.is_file():
            raise RuntimeError(f"CSICamera: tuning file not found: {path}")
        return str(path)

    @staticmethod
    def _libcamera_props(config) -> list[str]:
        """libcamerasrc property assignments for the configured exposure/gain policy.

        libcamera exposes the sensor's AE/AGC as camera controls, which libcamerasrc
        surfaces as GObject properties. There is no ISP auto-exposure to fall back on
        for a raw mono sensor other than this, so it is the only exposure control on
        the RPi path.

        ``auto_exposure`` picks the branch. With AE on we only bias it (EV, metering,
        constraint, exposure mode) and leave the loop to libcamera's AGC — which is
        tuned per-sensor by ``/usr/share/libcamera/ipa/rpi/vc4/ov9281_mono.json``.
        With AE off, ``ae-enable=false`` puts BOTH exposure time and gain into manual
        (libcamera ties the two modes to that one control), so the sensor holds
        exactly the configured values — the deterministic choice for tracking, where a
        hunting AE changes target contrast frame to frame.
        """
        props: list[str] = []
        auto = bool(getattr(config, "auto_exposure", True))
        props.append("ae-enable=%s" % ("true" if auto else "false"))
        if auto:
            # Bias knobs; these are ignored by libcamera once a mode is manual.
            for prop, attr in (("ae-exposure-mode",   "ae_exposure_mode"),
                               ("ae-metering-mode",   "ae_metering_mode"),
                               ("ae-constraint-mode", "ae_constraint_mode")):
                value = getattr(config, attr, "")
                if value:
                    props.append("%s=%s" % (prop, value))
            ev = float(getattr(config, "exposure_value", 0.0) or 0.0)
            if ev:
                props.append("exposure-value=%s" % ev)
        gain = float(getattr(config, "analogue_gain", 0.0) or 0.0)
        exposure_us = int(getattr(config, "exposure_time_us", 0) or 0)
        # Set even when auto: on the manual branch these ARE the operating point, and
        # a value supplied in the same request that switches to manual is applied
        # immediately (no frame at a stale AE value).
        if gain:
            props.append("analogue-gain=%s" % gain)
        if exposure_us:
            props.append("exposure-time=%d" % exposure_us)
        return props

    @classmethod
    def _with_libcamera_controls(cls, pipeline: str, config) -> str:
        """Splice the exposure/gain properties into the pipeline's libcamerasrc.

        A property already written into the configured pipeline string wins: the
        pipeline is the more specific statement of intent, and silently overriding a
        hand-tuned one would be a confusing debugging trap.
        """
        if "libcamerasrc" not in pipeline:
            return pipeline
        head = pipeline.split("!", 1)[0]     # the libcamerasrc element and its properties
        add = [p for p in cls._libcamera_props(config)
               if not re.search(r"\b%s\s*=" % re.escape(p.split("=", 1)[0]), head)]
        if not add:
            return pipeline
        return pipeline.replace("libcamerasrc", "libcamerasrc " + " ".join(add), 1)

    def open(self) -> None:
        import cv2
        if self._tuning_file:
            # Must be exported BEFORE the pipeline goes to READY: libcamerasrc creates the
            # camera manager there, and the IPA reads this once at load. Setting it in this
            # (per-worker) process is enough — no service-wide Environment= needed.
            import os
            os.environ[self._TUNING_ENV] = self._tuning_file
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

    _SENSOR_NAME = "ov9281"   # substring used to find the sensor media entity
    _FOURCC = "Y10 "          # V4L2 'Y10 ' pixelformat — the trailing space IS significant
    # OV9281 native array. We always capture the FULL frame and downscale to the
    # configured output: on this driver, asking the raw rkcif node for a smaller
    # size returns a top-left CROP (verified by cross-correlation), not a binned or
    # subsampled full frame, so a sub-native capture would silently lose ~75% of the
    # field of view. Downscaling 1280x800 -> output preserves the full FoV.
    _SENSOR_W = 1280
    _SENSOR_H = 800

    # Sensor control ranges (from VIDIOC_QUERYCTRL on the OV9281 subdev).
    _GAIN_MIN, _GAIN_MAX = 16, 248
    _EXP_MIN,  _EXP_MAX  = 4, 3600
    # Software auto-exposure. There is no ISP/AEC on the raw path. The scene is a flat,
    # bright sky that fills most of the frame with a target object silhouetted against
    # it, so metering the *mean* would let the sky dominate and either blow out (hiding
    # a bright target) or hunt on the uniform field. Instead we meter a high percentile
    # (~the sky) and hold it just BELOW clipping: the sky stays bright but unsaturated,
    # leaving headroom so a target against it stays distinguishable, and a dark target
    # reads as a clean high-contrast silhouette.
    _AE_PCTL     = 95     # metered percentile of frame brightness (≈ the sky level)
    _AE_INTERVAL = 12     # frames between adjustments (let a change settle; no hunting)
    _AE_DEADBAND = 0.06   # skip if metered within ±6% of target (stability on flat sky)
    _AE_STEP_LO, _AE_STEP_HI = 0.7, 1.5   # per-step brightness factor clamp (no overshoot)
    _AE_EXP_CAP  = 1500   # prefer exposure up to here (low noise), then gain (limits blur)

    def __init__(self, config) -> None:
        self._device = getattr(config, "device", "") or "/dev/video0"
        self._media  = getattr(config, "media", "")  or "/dev/media0"
        self._cap_w  = self._SENSOR_W
        self._cap_h  = self._SENSOR_H
        self._out_w  = getattr(config, "width",  640)
        self._out_h  = getattr(config, "height", 400)
        self._fps    = getattr(config, "fps",    0)
        self._auto_exposure = getattr(config, "auto_exposure", True)
        self._ae_target = getattr(config, "ae_target", 210)
        # Seed gain/exposure: config values if set, else sane AE start points. These are
        # the live values the AE loop drives; written to the sensor in open() and on each
        # adjustment in read().
        self._gain     = getattr(config, "gain", 0)     or (self._GAIN_MIN if self._auto_exposure else 0)
        self._exposure = getattr(config, "exposure", 0) or (800 if self._auto_exposure else 0)
        self._ae_count = 0

        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._latest = None        # (raw_bytes, ts_ns, seq)
        self._last_seq = -1
        self._seq = 0
        self._running = False
        self._sizeimage = 0
        self._stride = 0
        self._act_w = self._cap_w   # actual delivered width/height (re-read after set-fmt)
        self._act_h = self._cap_h
        self._subdev = ""           # sensor subdev node, discovered from the media graph

    def open(self) -> None:
        # Sensor is Y10-only; pin the pad format so the rkcif link validates, set the
        # capture-node format, then learn the exact packed buffer geometry (stride is
        # padded — e.g. 1024 B/row for 640px, 1792 for 1280px). Pad setup is best-effort.
        self._subdev = self._find_sensor_subdev()
        self._set_sensor_pad()
        self._v4l2([self._fmt_arg()])
        self._act_w, self._act_h, self._sizeimage, self._stride = self._query_geometry()
        if self._sizeimage <= 0 or self._stride <= 0 or self._act_w <= 0:
            raise RuntimeError(
                f"CSIY10Camera: could not determine frame geometry for {self._device}"
            )
        # Exposure/gain (no ISP/AEC on the raw path — the sensor would otherwise sit at
        # minimum gain and produce a very dark frame) and frame rate, all best-effort.
        self._apply_sensor_controls()

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
        gray = unpack_raw10_to_gray8(raw, self._act_w, self._act_h, self._stride)
        import cv2
        if (self._out_w, self._out_h) != (self._act_w, self._act_h):
            gray = cv2.resize(gray, (self._out_w, self._out_h),
                              interpolation=cv2.INTER_AREA)
        if self._auto_exposure:
            self._ae_step(gray)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # mono → 3-ch BGR for perception
        return frame, ts

    def _ae_step(self, gray: np.ndarray) -> None:
        """Drive sensor gain/exposure toward holding the metered percentile at target.

        Brightness is ~linear in exposure*gain, so we treat their product as a single
        light budget and nudge it by the brightness error (clamped per step to avoid
        overshoot/hunting), then allocate the budget exposure-first up to a blur cap and
        only spill into gain beyond that — keeping noise low while bounding motion blur.
        """
        self._ae_count += 1
        if self._subdev == "" or self._ae_count % self._AE_INTERVAL:
            return
        measured = float(np.percentile(gray, self._AE_PCTL))
        if measured < 1.0:
            # Essentially black (also avoids div-by-zero): boost by the full step.
            ratio = self._AE_STEP_HI
        elif measured >= 250.0:
            # Saturated: the true brightness is clipped away, so the metered ratio
            # underestimates how over-exposed we are — cut by the full step instead.
            ratio = self._AE_STEP_LO
        else:
            ratio = self._ae_target / measured
            if abs(1.0 - ratio) < self._AE_DEADBAND:
                return
            ratio = min(max(ratio, self._AE_STEP_LO), self._AE_STEP_HI)

        budget = self._exposure * self._gain * ratio
        budget = min(max(budget, self._EXP_MIN * self._GAIN_MIN),
                     self._AE_EXP_CAP * self._GAIN_MAX)
        exp = int(round(min(max(budget / self._GAIN_MIN, self._EXP_MIN), self._AE_EXP_CAP)))
        gain = int(round(min(max(budget / exp, self._GAIN_MIN), self._GAIN_MAX)))
        if (exp, gain) != (self._exposure, self._gain):
            self._exposure, self._gain = exp, gain
            self._v4l2_dev(self._subdev,
                           ["-c", "analogue_gain=%d,exposure=%d" % (gain, exp)])

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
                % (self._cap_w, self._cap_h, self._FOURCC))

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
        return self._v4l2_dev(self._device, args)

    def _v4l2_dev(self, dev: str, args: list[str]) -> str:
        try:
            out = subprocess.run(
                ["v4l2-ctl", "-d", dev, *args],
                capture_output=True, text=True, timeout=5,
            )
            return out.stdout
        except (OSError, subprocess.SubprocessError):
            return ""

    def _query_geometry(self) -> tuple[int, int, int, int]:
        """Return (width, height, sizeimage, bytesperline) from the node's format."""
        text = self._v4l2(["--get-fmt-video"])
        wh   = re.search(r"Width/Height\s*:\s*(\d+)/(\d+)", text)
        size = re.search(r"Size Image\s*:\s*(\d+)", text)
        line = re.search(r"Bytes per Line\s*:\s*(\d+)", text)
        width  = int(wh.group(1)) if wh else self._cap_w
        height = int(wh.group(2)) if wh else self._cap_h
        sizeimage = int(size.group(1)) if size else 0
        stride = int(line.group(1)) if line else 0
        if sizeimage and not stride and height:   # fall back if stride wasn't reported
            stride = sizeimage // height
        return width, height, sizeimage, stride

    def _sensor_entity(self) -> str:
        """Name of the OV9281 source entity in the media graph (or '')."""
        graph = self._media_ctl(["-p"])
        m = re.search(r"entity \d+: (\S[^\n(]*%s[^\n(]*) \(" % self._SENSOR_NAME, graph)
        return m.group(1).strip() if m else ""

    def _find_sensor_subdev(self) -> str:
        """The /dev/v4l-subdevN node backing the OV9281, parsed from the media graph."""
        graph = self._media_ctl(["-p"])
        m = re.search(
            r"entity \d+: \S[^\n(]*%s[^\n(]*\(.*?device node name (/dev/v4l-subdev\d+)"
            % self._SENSOR_NAME, graph, re.S)
        return m.group(1) if m else ""

    def _set_sensor_pad(self) -> None:
        """Best-effort: set the OV9281 sensor pad to Y10 so the rkcif link validates."""
        name = self._sensor_entity()
        if not name:
            return
        self._media_ctl([
            "--set-v4l2",
            '"%s":0[fmt:Y10_1X10/%dx%d]' % (name, self._cap_w, self._cap_h),
        ])

    def _apply_sensor_controls(self) -> None:
        """Best-effort exposure/gain and frame rate on the sensor subdev.

        The raw rkcif path has no ISP, so there is no auto-exposure: the OV9281
        otherwise sits at minimum analogue_gain and the frame is very dark. fps must
        go through the subdev frame interval (the capture node rejects S_PARM).
        """
        if not self._subdev:
            return
        ctrls = []
        if self._gain:
            ctrls.append("analogue_gain=%d" % self._gain)
        if self._exposure:
            ctrls.append("exposure=%d" % self._exposure)
        if ctrls:
            self._v4l2_dev(self._subdev, ["-c", ",".join(ctrls)])
        if self._fps:
            self._v4l2_dev(self._subdev, ["--set-subdev-fps", "pad=0,fps=%d" % self._fps])

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

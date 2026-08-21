# 10-bit Mono Capture + Frame Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver native 1280×800 linear uint16 mono OV9281 frames into shared memory on the RPi 4B (via picamera2 raw capture under libcamera's tuned AGC), with the HUD showing a tonemapped live view — the frame foundation the SP3 contrast detector will consume.

**Architecture:** A new `Picamera2RawCamera` `CameraSource` captures a raw-only stream and returns unpacked uint16 mono frames. `FrameBuffer` is generalized to carry an arbitrary dtype + channel count (defaulting to today's uint8/3ch), so it can hold one uint16 mono plane. The ground worker tonemaps mono16 → 8-bit BGR at the MJPEG read boundary; the overlay/JPEG code is unchanged. The 10-bit path is a new config-selected backend; every legacy BGR path is byte-for-byte untouched.

**Tech Stack:** Python 3.10–3.12, numpy, multiprocessing shared memory, picamera2 (board only, lazy-imported), OpenCV (ground display), pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-tenbit-capture-transport-design.md`

## Global Constraints

- **Platform:** RPi 4B (4× Cortex-A72). Capture target 1280×800 @ 60 fps; **achievable fps is measured on-board, not assumed.**
- **Back-compat is mandatory:** `FrameBuffer` defaults stay `channels=3, dtype=uint8`; every existing `FrameBuffer(...)` call site and legacy camera backend (`gstreamer`/`v4l2`/`csi`/`network`/`raw_tcp`) must behave exactly as before. No legacy test may change behavior.
- **`picamera2` MUST be lazy-imported inside `Picamera2RawCamera.open()`** — never at `sources.py` module top level, or the whole camera module fails to import on dev machines (Windows/non-Pi). This is a hard correctness rule: the module must import cleanly with no picamera2 installed.
- **Mono frames are `(H, W)`** from `FrameBuffer.read_latest()` when `channels == 1`; multi-channel stays `(H, W, C)`.
- **One writer per buffer** (camera). Zero-copy reads preserved.
- **Time:** always `quadguide.core.clock.monotonic_ns()` / `time.monotonic_ns()` for stamps — never `time.time()`.
- **CI gate:** code must pass `ruff` and `mypy` (the CI workflow gates on both). Keep type hints on new public functions.
- Tonemap and any 10-bit→8-bit reduction is **display-only** and must never touch the detector's raw feed.

---

### Task 1: dtype-aware FrameBuffer

**Files:**
- Modify: `src/quadguide/core/frame_buffer.py`
- Test: `tests/unit/test_frame_buffer.py`

**Interfaces:**
- Consumes: nothing (foundation task).
- Produces: `FrameBuffer(width, height, channels=3, n_slots=6, dtype="uint8")`. `write_frame(arr, timestamp_ns=None)` accepts arrays of the buffer's dtype. `read_latest() -> (frame, ts)` returns `(H,W)` for `channels==1`, else `(H,W,C)`, dtype = buffer dtype; `(None, 0)` when empty.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_frame_buffer.py`:

```python
import numpy as np
from quadguide.core.frame_buffer import FrameBuffer


def test_mono16_roundtrip_shape_dtype_and_values():
    fb = FrameBuffer(4, 3, channels=1, dtype="uint16")
    frame = (np.arange(12, dtype=np.uint16).reshape(3, 4) * 100)  # values > 255
    fb.write_frame(frame, timestamp_ns=42)
    out, ts = fb.read_latest()
    assert out.dtype == np.uint16
    assert out.shape == (3, 4)          # mono → (H, W), no channel axis
    assert ts == 42
    np.testing.assert_array_equal(out, frame)


def test_mono16_preserves_sub_8bit_low_bits():
    fb = FrameBuffer(2, 2, channels=1, dtype="uint16")
    frame = np.array([[1023, 512], [3, 300]], dtype=np.uint16)  # true 10-bit values
    fb.write_frame(frame)
    out, _ = fb.read_latest()
    np.testing.assert_array_equal(out, frame)   # >8-bit values survive intact


def test_uint8_bgr_backcompat_unchanged():
    fb = FrameBuffer(4, 3)  # defaults: channels=3, dtype=uint8
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    frame[1, 2] = (10, 20, 30)
    fb.write_frame(frame)
    out, _ = fb.read_latest()
    assert out.dtype == np.uint8 and out.shape == (3, 4, 3)
    np.testing.assert_array_equal(out, frame)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_frame_buffer.py -v -k "mono16 or backcompat"`
Expected: the two mono16 tests FAIL (`FrameBuffer() got an unexpected keyword argument 'dtype'`); backcompat may pass already.

- [ ] **Step 3: Generalize FrameBuffer**

In `src/quadguide/core/frame_buffer.py`, change `__init__` and the read/write to be dtype-aware. `dtype` is added **after** `n_slots` so all existing positional calls (`FrameBuffer(64, 64, 3, n_slots=2)`) keep working:

```python
def __init__(
    self,
    width: int,
    height: int,
    channels: int = 3,
    n_slots: int = 6,
    dtype: object = "uint8",
) -> None:
    self._width       = width
    self._height      = height
    self._channels    = channels
    self._n_slots     = n_slots
    self._dtype       = np.dtype(dtype)          # accepts "uint16", np.uint16, etc.
    self._count       = width * height * channels
    self._frame_bytes = self._count * self._dtype.itemsize
    self._slot_bytes  = _TS_SIZE + self._frame_bytes

    self._shm  = SharedMemory(create=True, size=n_slots * self._slot_bytes)
    self._head = multiprocessing.Value("i", -1)  # -1 = nothing written yet
```

In `write_frame`, use the buffer dtype instead of forcing uint8:

```python
    data = np.ascontiguousarray(arr, dtype=self._dtype).tobytes()
    self._shm.buf[frame_start : frame_start + self._frame_bytes] = data
```

In `read_latest`, read with the buffer dtype and reshape by channel count:

```python
    flat = np.frombuffer(
        self._shm.buf,
        dtype=self._dtype,
        count=self._count,
        offset=offset + _TS_SIZE,
    )
    shape = ((self._height, self._width) if self._channels == 1
             else (self._height, self._width, self._channels))
    frame = flat.reshape(shape)
    return frame, ts
```

Update the class docstring's "row-major uint8" / "(H, W, C) uint8 arrays" lines to note the buffer now carries its configured dtype and returns `(H, W)` for mono.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_frame_buffer.py -v`
Expected: all PASS (including the pre-existing uint8 tests — back-compat intact).

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/core/frame_buffer.py tests/unit/test_frame_buffer.py
git commit -m "feat: dtype-aware FrameBuffer (uint16 mono support)"
```

---

### Task 2: CameraConfig.bit_depth + frame_spec() + orchestrator wiring

**Files:**
- Modify: `src/quadguide/core/config.py` (`CameraConfig` ~L145, `cfg_platform` ~L226)
- Modify: `scripts/run.py:143`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `FrameBuffer(..., dtype=...)` from Task 1.
- Produces: `CameraConfig.bit_depth: int = 8`; `frame_spec(camera: CameraConfig) -> tuple[int, str]` returning `(channels, dtype_str)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_config.py`:

```python
from quadguide.core.config import CameraConfig, frame_spec


def _cam(**kw) -> CameraConfig:
    base = dict(backend="gstreamer", pipeline="", width=1280, height=800)
    base.update(kw)
    return CameraConfig(**base)


def test_frame_spec_defaults_to_bgr_uint8():
    assert frame_spec(_cam()) == (3, "uint8")


def test_frame_spec_picamera2_backend_is_mono16():
    assert frame_spec(_cam(backend="picamera2", bit_depth=10)) == (1, "uint16")


def test_frame_spec_high_bit_depth_forces_mono16():
    assert frame_spec(_cam(bit_depth=10)) == (1, "uint16")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config.py -v -k frame_spec`
Expected: FAIL (`cannot import name 'frame_spec'` / `CameraConfig() got unexpected keyword 'bit_depth'`).

- [ ] **Step 3: Add the field, the loader mapping, and frame_spec**

In `CameraConfig` (frozen dataclass), add after the existing scalar fields:

```python
    bit_depth: int = 8          # 8 = BGR uint8 path; >8 (e.g. 10) = raw mono uint16
```

In `cfg_platform`, where `CameraConfig(...)` is built, pass it through:

```python
            bit_depth=cam.get("bit_depth", 8),
```

Add the helper near `cfg_platform` (kept numpy-free — returns a dtype **string** so config has no numpy import; `FrameBuffer`/`np.frombuffer` accept the string):

```python
def frame_spec(camera: CameraConfig) -> tuple[int, str]:
    """(channels, dtype) for the FrameBuffer backing this camera source.

    The raw picamera2 path (or any bit_depth > 8) carries one uint16 mono
    plane; every other backend keeps the legacy 3-channel BGR uint8 frame.
    """
    if camera.backend == "picamera2" or camera.bit_depth > 8:
        return 1, "uint16"
    return 3, "uint8"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Wire the orchestrator**

In `scripts/run.py` at the `FrameBuffer` construction (currently
`frame_buffer = FrameBuffer(pcfg.camera.width, pcfg.camera.height)`), derive
channels + dtype from the camera config:

```python
    from quadguide.core.config import frame_spec
    _ch, _dt = frame_spec(pcfg.camera)
    frame_buffer = FrameBuffer(
        pcfg.camera.width, pcfg.camera.height, channels=_ch, dtype=_dt
    )
```

(Legacy configs → `frame_spec` returns `(3, "uint8")`, reproducing the current call exactly.)

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/core/config.py scripts/run.py tests/unit/test_config.py
git commit -m "feat: frame_spec + camera bit_depth wiring for mono16 buffer"
```

---

### Task 3: Tonemap functions (10-bit → 8-bit, display only)

**Files:**
- Modify: `src/quadguide/ground/overlay.py`
- Test: `tests/unit/test_tonemap.py` (new)

**Interfaces:**
- Consumes: nothing.
- Produces: `tonemap(frame: np.ndarray, mode: str = "percentile", p_lo: float = 1.0, p_hi: float = 99.5, gamma: float = 2.2) -> np.ndarray` — takes a mono uint16 `(H,W)`, returns mono uint8 `(H,W)`. Pure numpy, no OpenCV.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tonemap.py`:

```python
import numpy as np
from quadguide.ground.overlay import tonemap


def _sky_with_target(h=32, w=32):
    """Bright flat sky (~900/1023) with a small dark target patch (~150)."""
    frame = np.full((h, w), 900, dtype=np.uint16)
    frame[14:18, 14:18] = 150
    return frame


def test_tonemap_outputs_uint8_full_range():
    out = tonemap(_sky_with_target(), mode="percentile")
    assert out.dtype == np.uint8
    assert out.shape == (32, 32)
    assert out.min() < 40 and out.max() > 215     # stretch actually uses the range


def test_linear_map_is_shift_right_2():
    frame = np.array([[0, 4, 1020, 1023]], dtype=np.uint16)
    out = tonemap(frame, mode="linear")
    np.testing.assert_array_equal(out, (frame >> 2).astype(np.uint8))


def test_percentile_separates_target_from_sky_better_than_linear():
    frame = _sky_with_target()
    lin = tonemap(frame, mode="linear")
    pct = tonemap(frame, mode="percentile")
    sky = (slice(0, 4), slice(0, 4))
    tgt = (slice(14, 18), slice(14, 18))
    # Percentile stretch widens the sky/target 8-bit separation vs the naive >>2.
    assert (int(pct[sky].mean()) - int(pct[tgt].mean())) > \
           (int(lin[sky].mean()) - int(lin[tgt].mean()))


def test_gamma_mode_is_monotonic_uint8():
    frame = (np.arange(1024, dtype=np.uint16)).reshape(32, 32)
    out = tonemap(frame, mode="gamma")
    assert out.dtype == np.uint8
    assert out.flatten()[0] <= out.flatten()[-1]   # monotonic overall
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_tonemap.py -v`
Expected: FAIL (`cannot import name 'tonemap'`).

- [ ] **Step 3: Implement the tonemap**

Add to `src/quadguide/ground/overlay.py` (pure numpy — do **not** use cv2 here, so the tests stay OpenCV-free):

```python
def _percentile_stretch(frame: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    lo = float(np.percentile(frame, p_lo))
    hi = float(np.percentile(frame, p_hi))
    if hi <= lo:
        hi = lo + 1.0
    out = (frame.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def tonemap(
    frame: np.ndarray,
    mode: str = "percentile",
    p_lo: float = 1.0,
    p_hi: float = 99.5,
    gamma: float = 2.2,
) -> np.ndarray:
    """Reduce a mono uint16 (H,W) frame to a mono uint8 (H,W) for display.

    Display-only — never applied to the detector's raw feed.
      linear     : v >> 2 (naive, matches the old 10→8 truncation)
      percentile : stretch [p_lo, p_hi] then clip (robust to a flat bright sky)
      gamma      : percentile stretch, then a gamma curve for a photographic look
    """
    if mode == "linear":
        return (frame >> 2).astype(np.uint8)
    stretched = _percentile_stretch(frame, p_lo, p_hi)
    if mode == "gamma":
        lut = (((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255.0).astype(np.uint8)
        return lut[stretched]
    return stretched
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_tonemap.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/ground/overlay.py tests/unit/test_tonemap.py
git commit -m "feat: 10-bit->8-bit display tonemap (linear/percentile/gamma)"
```

---

### Task 4: HUD read-boundary adapter + ground wiring

**Files:**
- Modify: `src/quadguide/ground/overlay.py` (add `to_display_bgr`)
- Modify: `src/quadguide/ground/server.py` (`create_app` lifespan ~L59, `_mjpeg` ~L194)
- Test: `tests/unit/test_ground_overlay.py`

**Interfaces:**
- Consumes: `tonemap(...)` from Task 3.
- Produces: `to_display_bgr(frame: np.ndarray, tonemap_mode: str = "percentile") -> np.ndarray` — mono16 `(H,W)` → BGR uint8 `(H,W,3)`; BGR uint8 input returned unchanged (legacy passthrough).

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_ground_overlay.py`:

```python
import numpy as np
from quadguide.ground.overlay import to_display_bgr


def test_to_display_bgr_converts_mono16():
    frame = np.full((8, 10), 800, dtype=np.uint16)
    out = to_display_bgr(frame, tonemap_mode="percentile")
    assert out.dtype == np.uint8
    assert out.shape == (8, 10, 3)


def test_to_display_bgr_passthrough_for_bgr8():
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    out = to_display_bgr(frame)
    assert out is frame          # legacy path: no copy, no conversion
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_ground_overlay.py -v -k to_display_bgr`
Expected: FAIL (`cannot import name 'to_display_bgr'`).

- [ ] **Step 3: Implement the adapter**

Add to `src/quadguide/ground/overlay.py` (this one uses cv2 — overlay.py is a ground/display module where cv2 is already a dependency):

```python
def to_display_bgr(frame: np.ndarray, tonemap_mode: str = "percentile") -> np.ndarray:
    """Normalise a frame-buffer frame to BGR uint8 for the MJPEG overlay/JPEG path.

    Legacy BGR uint8 frames pass straight through; a mono uint16 (H,W) frame is
    tonemapped to 8-bit and expanded to 3-channel so the coloured overlay works.
    """
    import cv2
    if frame.ndim == 3:
        return frame                                   # already BGR uint8
    gray8 = tonemap(frame, mode=tonemap_mode)
    return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ground_overlay.py -v`
Expected: all PASS.

- [ ] **Step 5: Wire the ground server**

In `src/quadguide/ground/server.py`:

Read the tonemap mode in `create_app` (near `ui_mode`) and stash it on app state in the lifespan:

```python
    tonemap_mode = (config or {}).get("ground", {}).get("tonemap", "percentile")
```
```python
        app.state.tonemap_mode = tonemap_mode
```

In `_mjpeg`, normalise the frame before overlay (the `else` branch where `frame is not None`):

```python
        else:
            frame = overlay.to_display_bgr(frame, app.state.tonemap_mode)
            estimate = app.state.bus.latest("target/estimate")
            jpeg = overlay.draw_overlay(
                frame, estimate, app.state.acquire_crop,
                show_bbox=app.state.ui["show_bbox"],
            )
```

- [ ] **Step 6: Run the ground test suite for regressions**

Run: `pytest tests/unit/test_ground_overlay.py tests/unit/test_ground_server.py -v`
Expected: all PASS (legacy BGR frames still flow unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/quadguide/ground/overlay.py src/quadguide/ground/server.py tests/unit/test_ground_overlay.py
git commit -m "feat: HUD tonemap adapter for mono16 frames"
```

---

### Task 5: Picamera2RawCamera source (board-only)

**Files:**
- Modify: `src/quadguide/perception/camera/sources.py` (add class)
- Modify: `src/quadguide/perception/camera/worker.py` (import + `_SOURCES` ~L18)

**Interfaces:**
- Consumes: the `CameraSource` ABC; the mono16 `FrameBuffer` (Task 1); `frame_spec`'s `picamera2` branch (Task 2).
- Produces: `Picamera2RawCamera(config)` with `open()/read()/close()`; `read() -> (np.ndarray[(H,W) uint16], ts_ns)`. Registered as `_SOURCES["picamera2"]`.

> **No pytest here.** picamera2 + the sensor exist only on the Pi. The verification gate is `scripts/diag_raw10.py` (Task 7) run on-board. The one thing that MUST hold on every machine: `sources.py` still imports with no picamera2 installed (picamera2 is imported inside `open()`), so run the import check in Step 3.

- [ ] **Step 1: Add the source class**

In `src/quadguide/perception/camera/sources.py`, add to `__all__` and define the class. **picamera2 is imported inside `open()` only.** The exact configuration call is verified on-board; this is the intended shape:

```python
class Picamera2RawCamera(CameraSource):
    """OV9281 mono global-shutter CSI camera on the RPi 4B, raw 10-bit via picamera2.

    Captures a RAW-ONLY stream (no ISP/main stream) so libcamera's tuned AGC still
    drives sensor exposure but we consume the LINEAR uint16 frame before gamma/
    tonemap. picamera2 returns an already-unpacked (H, W) uint16 array, so there is
    no custom RAW10 bit-unpacking. Selected via config: platform.camera.backend =
    "picamera2". Mono uint16 flows through the dtype-aware FrameBuffer; the HUD
    tonemaps it for display.
    """

    _TUNING_ENV = "LIBCAMERA_RPI_TUNING_FILE"

    def __init__(self, config) -> None:
        self._width  = getattr(config, "width",  1280)
        self._height = getattr(config, "height", 800)
        self._fps    = getattr(config, "fps",    0)
        self._auto_exposure   = getattr(config, "auto_exposure", True)
        self._gain            = float(getattr(config, "analogue_gain", 0.0) or 0.0)
        self._exposure_us     = int(getattr(config, "exposure_time_us", 0) or 0)
        self._tuning_file     = CSICamera._resolve_tuning_file(
            getattr(config, "tuning_file", ""))
        self._picam = None

    def open(self) -> None:
        import os
        from picamera2 import Picamera2   # board-only; MUST stay inside open()
        if self._tuning_file:
            os.environ[self._TUNING_ENV] = self._tuning_file
        self._picam = Picamera2()
        cfg = self._picam.create_video_configuration(
            raw={"size": (self._width, self._height)},
            buffer_count=6,
        )
        self._picam.configure(cfg)
        self._apply_exposure()
        self._picam.start()

    def _apply_exposure(self) -> None:
        controls = {}
        if not self._auto_exposure:
            controls["AeEnable"] = False
            if self._exposure_us:
                controls["ExposureTime"] = self._exposure_us
            if self._gain:
                controls["AnalogueGain"] = self._gain
        else:
            controls["AeEnable"] = True
        if self._fps:
            frame_us = int(1_000_000 / self._fps)
            controls["FrameDurationLimits"] = (frame_us, frame_us)
        if controls:
            self._picam.set_controls(controls)

    def set_exposure(self, *, auto: bool, exposure_us: int = 0,
                     gain: float = 0.0) -> None:
        """Runtime auto<->lock switch. SP2's sky-calibration button drives this."""
        self._auto_exposure = auto
        if exposure_us:
            self._exposure_us = exposure_us
        if gain:
            self._gain = gain
        if self._picam is not None:
            self._apply_exposure()

    def read(self) -> tuple[np.ndarray, int]:
        arr = self._picam.capture_array("raw")   # (H, W) uint16, libcamera-unpacked
        ts = time.monotonic_ns()
        return arr, ts

    def close(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None
```

- [ ] **Step 2: Register the backend**

In `src/quadguide/perception/camera/worker.py`, add `Picamera2RawCamera` to the `from ...sources import (...)` line and to `_SOURCES`:

```python
    "picamera2": Picamera2RawCamera,   # OV9281 raw 10-bit mono (RPi 4B, picamera2)
```

- [ ] **Step 3: Verify the module still imports with no picamera2**

Run (on the dev machine, where picamera2 is absent):
`python -c "import quadguide.perception.camera.worker; import quadguide.perception.camera.sources; print('import ok')"`
Expected: prints `import ok` (proves picamera2 is not imported at module load).

Also run the existing camera-source tests for regressions:
`pytest tests/unit/test_camera_sources.py -v`
Expected: PASS (existing sources unaffected).

- [ ] **Step 4: Commit**

```bash
git add src/quadguide/perception/camera/sources.py src/quadguide/perception/camera/worker.py
git commit -m "feat: Picamera2RawCamera raw 10-bit mono source (RPi 4B)"
```

---

### Task 6: rpi4b_raw10 config preset + dependency wiring

**Files:**
- Create: `configs/rpi4b_raw10.yaml`
- Modify: `scripts/firstboot_install_rpi.sh`, `requirements.txt`
- Test: `tests/unit/test_rpi4b_config.py`

**Interfaces:**
- Consumes: `frame_spec` (Task 2), the `picamera2` backend (Task 5).
- Produces: a loadable preset that resolves to a `(1, "uint16")` frame spec.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_rpi4b_config.py`:

```python
from pathlib import Path
from quadguide.core.config import load_config, cfg_platform, frame_spec

_REPO = Path(__file__).resolve().parents[2]


def test_rpi4b_raw10_preset_is_mono16():
    cfg = load_config(str(_REPO / "configs" / "rpi4b_raw10.yaml"))
    pcfg = cfg_platform(cfg)
    assert pcfg.camera.backend == "picamera2"
    assert pcfg.camera.width == 1280 and pcfg.camera.height == 800
    assert pcfg.camera.bit_depth == 10
    assert frame_spec(pcfg.camera) == (1, "uint16")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_rpi4b_config.py -v -k raw10`
Expected: FAIL (file not found).

- [ ] **Step 3: Create the preset**

Create `configs/rpi4b_raw10.yaml` by copying `configs/rpi4b.yaml` and changing **only** the camera block + adding `ground.tonemap` (leave serial/guidance/failsafe/link/etc. as in `rpi4b.yaml`). The camera block:

```yaml
platform:
  name: rpi4b
  camera:
    # RPi 4B OV9281 raw 10-bit bring-up (SP1). picamera2 captures a RAW-ONLY stream;
    # libcamera's tuned AGC drives exposure, we consume the LINEAR uint16 frame.
    # NOTE: no tracker consumes mono16 until the SP3 seeker exists — verify this
    # preset with scripts/diag_raw10.py and the HUD (camera + ground only).
    backend: picamera2
    pipeline: ""              # ignored by the picamera2 backend
    width: 1280
    height: 800
    fps: 60                   # TARGET; achievable rate is MEASURED on-board (diag_raw10)
    bit_depth: 10             # drives the uint16 FrameBuffer + picamera2 raw config
    auto_exposure: true       # libcamera AGC; false → manual gain/exposure (SP2 lock)
    tuning_file: configs/libcamera/ov9281_mono_quadguide.json
    analogue_gain: 0.0        # manual operating point, applied only when auto_exposure=false
    exposure_time_us: 0
  # serial / realtime: keep identical to configs/rpi4b.yaml
```

And in the `ground:` block of the copied file, add:

```yaml
ground:
  port: 8080
  ui_mode: minimal
  tonemap: percentile        # HUD display: linear | percentile | gamma
```

(Copy the remaining sections — `airframe`, `guidance`, `watchdog`, `failsafe`, `link`, `mission`, `logging`, `bus`, `diag`, and `serial`/`realtime` under `platform` — verbatim from `configs/rpi4b.yaml`. There is **no** `tracker:` requirement for SP1 bring-up; keep the section but it is not exercised until SP3.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_rpi4b_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Wire the dependency**

In `scripts/firstboot_install_rpi.sh`, add `python3-picamera2` to the apt install list (system libcamera + numpy come with it; the venv uses `--system-site-packages` so the worker sees it).

In `requirements.txt`, add a comment noting the system dependency (it is **not** pip-installable in the venv — it is an apt/system package):

```
# picamera2 (RPi 4B 10-bit raw capture) is a SYSTEM package: `apt install python3-picamera2`
# (installed by scripts/firstboot_install_rpi.sh; not pip-installable). Not needed on dev machines.
```

- [ ] **Step 6: Commit**

```bash
git add configs/rpi4b_raw10.yaml scripts/firstboot_install_rpi.sh requirements.txt tests/unit/test_rpi4b_config.py
git commit -m "feat: rpi4b_raw10 preset + picamera2 dependency wiring"
```

---

### Task 7: diag_raw10.py on-board verification (board-only)

**Files:**
- Create: `scripts/diag_raw10.py`

**Interfaces:**
- Consumes: `Picamera2RawCamera` (Task 5), `tonemap` (Task 3), `load_config`/`cfg_platform` (Task 2).
- Produces: an on-board script proving genuine 10-bit capture; no pytest (hardware-only).

> **No pytest here** — this runs on the Pi. It is the Definition-of-Done gate for the capture path.

- [ ] **Step 1: Write the diagnostic script**

Create `scripts/diag_raw10.py`:

```python
#!/usr/bin/env python3
"""Verify genuine 10-bit raw capture from the OV9281 via Picamera2RawCamera.

    sudo systemctl stop quadguide
    sudo .venv/bin/python scripts/diag_raw10.py --config configs/rpi4b_raw10.yaml
    sudo systemctl start quadguide

Prints per-frame stats and PASS/FAIL on the "genuinely >8-bit" check, and writes
a tonemapped PNG + a raw .npy to /tmp/raw10 for inspection.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from quadguide.core.config import cfg_platform, load_config
from quadguide.ground.overlay import tonemap
from quadguide.perception.camera.sources import Picamera2RawCamera

OUT = "/tmp/raw10"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rpi4b_raw10.yaml")
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pcfg = cfg_platform(load_config(args.config))
    cam = Picamera2RawCamera(pcfg.camera)
    cam.open()
    try:
        last = None
        levels = 0
        for i in range(args.frames):
            frame, _ts = cam.read()
            last = frame
            levels = int(np.unique(frame).size)
            low2 = int(np.count_nonzero(frame & 0x3))   # populated low 2 bits ⇒ true 10-bit
            print(f"frame {i:02d} {frame.shape} dtype={frame.dtype} "
                  f"min={frame.min():4d} mean={frame.mean():7.1f} "
                  f"p95={np.percentile(frame, 95):6.1f} max={frame.max():4d} "
                  f"levels={levels} low2bits={low2}")
    finally:
        cam.close()

    if last is None:
        print("FAIL: no frames captured")
        return 1
    np.save(f"{OUT}/frame.npy", last)
    try:
        import cv2
        cv2.imwrite(f"{OUT}/frame.png", tonemap(last, mode="percentile"))
    except Exception:
        pass

    p95 = float(np.percentile(last, 95))
    genuine_10bit = levels > 256 and last.dtype == np.uint16
    sky_headroom = p95 < 1000                     # below the 10-bit clip (1023)
    print(f"\nlevels={levels} (>256 ⇒ true >8-bit), dtype={last.dtype}, "
          f"sky p95={p95:.0f} (<1023 clip)")
    print("PASS" if (genuine_10bit and sky_headroom) else
          "FAIL: not genuine 10-bit or sky clipped")
    return 0 if (genuine_10bit and sky_headroom) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify it imports on the dev machine**

Run: `python -c "import ast; ast.parse(open('scripts/diag_raw10.py').read()); print('parse ok')"`
Expected: `parse ok` (the actual run happens on the Pi).

- [ ] **Step 3: Commit**

```bash
git add scripts/diag_raw10.py
git commit -m "feat: diag_raw10 on-board 10-bit capture verification"
```

- [ ] **Step 4 (on-board, not part of the dev commit cycle): run the gate**

On the Pi 4B: `sudo .venv/bin/python scripts/diag_raw10.py --config configs/rpi4b_raw10.yaml`
Expected: `PASS`, `levels > 256`, populated low-2-bits, sky p95 < 1023. Record the achieved fps (frame cadence) here — this is where the 60 fps target is confirmed or the binned-640×400 fallback is decided.

---

## Self-Review

**1. Spec coverage:**
- Capture (`Picamera2RawCamera`, raw stream, AGC, unpacked uint16) → Task 5. ✓
- dtype-aware `FrameBuffer` → Task 1. ✓
- HUD tonemap adapter + `ground.tonemap` → Tasks 3, 4. ✓
- Config surface (`bit_depth`, `backend: picamera2`, preset) + guidance-unaffected note → Tasks 2, 6. ✓
- Exposure auto + lock hook (`set_exposure`) → Task 5. ✓
- Windows-testable pieces (buffer, config, tonemap, adapter) → Tasks 1–4, 6. ✓
- Board-only verification (`diag_raw10.py`, genuine-10-bit, fps) → Task 7. ✓
- Dependency (`python3-picamera2`) + installer/requirements → Task 6. ✓
- Back-compat (defaults uint8/3ch; legacy paths untouched) → Tasks 1, 2 + Task 4 regression run. ✓
- No gap found.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N". The one intentional forward-reference (SP2 wires `set_exposure`) is a scope note, not a missing step. ✓

**3. Type consistency:** `frame_spec -> (int, str)` used identically in Tasks 2 and 6; `FrameBuffer(..., dtype=str|np.dtype)` consistent across Tasks 1, 2, 5; `tonemap(frame, mode=...)` consistent Tasks 3, 4, 7; `to_display_bgr(frame, tonemap_mode=...)` consistent Task 4 ↔ server wiring; `Picamera2RawCamera.read() -> ((H,W) uint16, ts)` matches the buffer's mono16 expectation. ✓

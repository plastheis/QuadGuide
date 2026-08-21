# 10-bit Mono Capture + Frame Transport — Design Spec

Date: 2026-08-21

---

## Overview

QuadGuide's RPi 4B camera path today is `libcamerasrc → videoconvert →
video/x-raw,format=BGR → appsink`, read through `cv2.VideoCapture`
(`perception/camera/sources.py:CSICamera`, `backend: gstreamer`). The OV9281 is a
10-bit mono global-shutter sensor, but **all of that dynamic range is consumed
inside libcamera's ISP**: its AGC (tuned by `configs/libcamera/ov9281_mono_quadguide.json`
to hold the sky just under clipping) plus a gamma-2.2 curve collapse the raw R10
to 8-bit BGR before any consumer sees it. A contrast-measure detector (SP3) needs
the *linear* 10-bit data, at full sensor resolution, to separate a small target
from the sky.

This spec (SP1 of a four-part effort) delivers **linear uint16 mono frames at the
OV9281's native 1280×800** into shared memory on the Pi 4B, while keeping the HUD
live. It does **not** build the detector (SP3), the sky-calibration button (SP2),
or any tracker/guidance change (SP4). SP1's deliverable is *pristine 10-bit frames
in the buffer, provably 10-bit, HUD intact.*

The capture mechanism is **picamera2 configured for a raw stream only**, with
libcamera's AGC still driving the sensor (so exposure behavior matches today) but
the **raw** frame consumed before gamma/tonemap. picamera2 returns an
already-unpacked uint16 array, so there is **no custom RAW10 bit-unpacking** and
none of the packing-layout risk that striped the ROCK 5C rkcif path
(`CSIY10Camera`). The frame travels through a **dtype-aware `FrameBuffer`** carrying
one uint16 mono plane; the HUD tonemaps that to 8-bit on read for JPEG display.

Net change: one new `CameraSource`, a small generalization of `FrameBuffer`
(dtype + channels), a read-boundary tonemap in the ground worker, additive config,
and a verification diag script. Legacy BGR configs are byte-for-byte unchanged.

---

## Goals

- Deliver native **1280×800 linear uint16 mono** frames from the OV9281 into
  `FrameBuffer` on the RPi 4B.
- Keep libcamera's tuned AGC (`ov9281_mono_quadguide.json`) driving sensor
  exposure — the raw frame reflects the same sky-under-clip operating point the
  8-bit path uses today, just before the gamma/tonemap.
- Keep the HUD live: a tonemapped 8-bit view with the lock-on box, unchanged
  operator workflow.
- Generalize `FrameBuffer` to carry dtype + channels, defaulting to today's
  `uint8`/3-channel so every existing config and the legacy BGR path are unchanged.
- Expose the exposure plumbing (auto AE now; a fixed-operating-point + auto↔lock
  switch) that **SP2** will wire to the launcher calibration button.
- Keep every new pure-logic piece (dtype-aware buffer, tonemap, config parsing)
  unit-testable on Windows without the Linux stack.
- Prove the frames are genuinely 10-bit (not an 8-bit upcast) with an on-board
  diagnostic.

## Non-goals

- **The detector.** SP1 puts 10-bit frames in the buffer; the MPCM/Otsu detector
  that consumes them is SP3. Until then the mono16 buffer is verified by the diag
  script and the HUD, not by a tracker (existing trackers expect BGR — see below).
- **Sky calibration / background model.** The launcher button, the `POST`, the FPN
  / hot-pixel / vignetting / sky-statistics model are SP2. SP1 only exposes the
  exposure auto↔lock hook SP2 needs.
- **Wire-format / guidance changes.** `target/estimate`, `pronav`, and the tracker
  protocol are untouched here (SP4).
- **The ROCK 5C rkcif path.** `CSIY10Camera` (`backend: csi`) and its
  `unpack_raw10_to_gray8` stay exactly as they are. SP1 is the RPi 4B (unicam /
  libcamera) path only.
- **Replacing the legacy BGR path.** `backend: gstreamer`/`v4l2`/`network`/`raw_tcp`
  and their uint8 BGR `FrameBuffer` remain the default. The 10-bit path is a new,
  config-selected backend, not a global switch.

---

## Architecture

### Data flow

```
                         ┌───────────────────────────────────────────────┐
                         │  SHARED MEMORY                                │
                         │  frame_buffer (dtype-aware ring)              │
                         │    mono16 path: (800,1280) uint16, 1 channel  │
                         └───────────────────────────────────────────────┘
                              ↑ write (2.05 MB/slot)     ↓ read (zero-copy)

[camera worker]                                    [ground worker]  (SP1 consumer)
  Picamera2RawCamera.open()                          _mjpeg @ 15 Hz:
    Picamera2(), raw stream only,                      frame_u16 = fb.read_latest()
    tuning_file, AGC on                                bgr8 = tonemap(frame_u16)   ← new
  loop:                                                jpeg = overlay(bgr8, estimate)
    frame_u16, ts = cam.read()                         yield jpeg
    frame_buffer.write_frame(frame_u16, ts)
                                                    [tracker worker]  (SP3+; not SP1)
                                                       reads mono16 natively
```

The camera worker loop (`perception/camera/worker.py:run`) is **unchanged** — it
already calls `source.read() → frame_buffer.write_frame(frame, ts)` for any
`CameraSource`. SP1 adds a new source and lets the frame carry a different
dtype/shape end to end.

### Component 1 — `Picamera2RawCamera` (new `CameraSource`)

New class in `perception/camera/sources.py`, registered in
`perception/camera/worker.py:_SOURCES` as `"picamera2": Picamera2RawCamera`.
Implements the existing ABC (`open`/`read`/`close`), so the worker needs no branch.

- **`open()`** — constructs `Picamera2()` **in the child process** (imported lazily
  inside `open()`, exactly as `CSICamera.open()` lazy-imports `cv2` — libcamera
  objects must not cross the `fork()` in `run.py`). Configures a **raw stream
  only**:
  - `create_still_configuration` / `create_video_configuration` with a `raw`
    stream at `{"size": (1280, 800)}` and **no `main`/ISP stream** (or a minimal
    unused one if the API requires `main` — the raw stream is the only one we
    `capture_array` from). Requesting no ISP main stream is what frees the CPU the
    old `videoconvert` spent.
  - `bit_depth: 10`, sensor mode pinned to the full 1280×800 array (full FoV, no
    crop).
  - Export `LIBCAMERA_RPI_TUNING_FILE` before start (same env var and timing as
    `CSICamera`), so the tuned AGC/tonemap file loads. Only the AGC half matters
    to us (it sets sensor exposure/gain); the tonemap half is bypassed by reading
    raw.
  - AGC left **on** by default (auto exposure), or set to the manual operating
    point when `auto_exposure: false` (see Exposure control).
  - `picam2.start()`.
- **`read()`** — `arr = picam2.capture_array("raw")` returns an already-unpacked
  `(800, 1280)` uint16 array (libcamera unpacks R10_CSI2P → 16-bit; the low bits
  are real sensor data, the top 6 bits zero). Stamp `ts = monotonic_ns()` on the
  SBC clock (same rationale as every other source). Return `(arr, ts)`.
  - picamera2 delivers the newest completed frame, so the existing "one fresh
    frame per read" contract holds without extra bookkeeping.
- **`close()`** — `picam2.stop()`, `picam2.close()`. Best-effort, idempotent.

**Frame dtype/shape contract:** `read()` returns `(H, W)` uint16 (single mono
plane). This is what the SP3 detector will consume directly and what the HUD
adapter tonemaps.

### Component 2 — dtype-aware `FrameBuffer`

`core/frame_buffer.py` today hard-codes `uint8`, 3 channels, and forces
`np.ascontiguousarray(arr, dtype=np.uint8)` on write. Generalize:

- Constructor gains `dtype: np.dtype = np.uint8` (channels already a parameter).
  `self._itemsize = np.dtype(dtype).itemsize`; `frame_bytes = w · h · channels ·
  itemsize`. Slot layout is unchanged: `[ts: 8 bytes big-endian][frame: N bytes]`.
- `write_frame` uses the buffer's dtype (`np.ascontiguousarray(arr, dtype=self._dtype)`)
  instead of forcing uint8. The `.tobytes()` O(N) copy is unchanged.
- `read_latest` reads with the buffer's dtype/itemsize and reshapes:
  - `channels == 1` → `(H, W)`  (natural grayscale shape for cv2/numpy ops)
  - `channels > 1` → `(H, W, C)` (today's behavior)
  Zero-copy is preserved (`np.frombuffer(..., dtype=self._dtype)`).
- **Back-compat:** defaults `dtype=uint8, channels=3` reproduce today's behavior
  exactly. Every existing `FrameBuffer(w, h)` call site (run.py, dev scripts,
  tests) is unchanged.
- **Construction from config** (`scripts/run.py`, currently
  `FrameBuffer(pcfg.camera.width, pcfg.camera.height)`): derive channels + dtype
  from the camera format — mono16 for the `picamera2` backend / `bit_depth: 10`,
  BGR uint8 otherwise. A single helper (e.g. `frame_spec(camera_config) →
  (channels, dtype)`) keeps this in one place and is unit-testable.

Memory: 1280×800×1×2 = **2.05 MB/slot**, ~12.3 MB for the default 6 slots. Fine.

### Component 3 — HUD / ground read-boundary tonemap

`ground/server.py:_mjpeg` and `ground/overlay.py:draw_overlay` assume BGR uint8.
Add a thin adapter at the read boundary; the overlay/JPEG code stays BGR-based and
untouched.

- New `overlay.to_display_bgr(frame, tonemap="percentile")`:
  - `frame.ndim == 3` (BGR uint8) → return as-is (legacy path, zero cost).
  - `frame.dtype == uint16` mono → tonemap → 8-bit → `cv2.cvtColor(GRAY2BGR)`.
- **Tonemap** (10-bit → 8-bit, display only, never touches the detector feed):
  - `linear` — `(v >> 2)`, the cheap/naive map (matches the old `>> 2`).
  - `percentile` — stretch `[p_lo, p_hi]` (robust to the flat bright sky) then
    clip; **default**, because a fixed linear map wastes most of the 8-bit range on
    a sky that occupies a narrow high band.
  - `gamma` — percentile stretch + gamma, for a more photographic operator view.
  - Selected by `ground.tonemap`; runs only at the 15 Hz MJPEG rate in the ground
    worker, so cost is negligible.
- Lock-on math is normalized (0–1) on both the client canvas and the frame, so the
  1280×800 capture vs 640×400 overlay canvas mismatch needs no special handling
  (it is already how the 1280×800 CSICamera setups worked).

### Where the tracker fits (context, not SP1 work)

Existing trackers (NanoTrack via `EdgeCVTracker`, cv2 trackers) expect BGR uint8
and would choke on a mono16 frame. That is fine: they run on the legacy BGR
backend, and the seeker that consumes mono16 is SP3/SP4. SP1 therefore does **not**
wire a tracker to the mono16 buffer. On-board SP1 verification runs **camera +
ground only** (plus the diag script) — no tracker worker — so nothing consumes a
format it can't read.

---

## Config surface (additive)

New/reused fields under `platform.camera`, plus one `ground` key. Added to a new
preset so the working flight config (`rpi4b.yaml`, nanotrack on BGR) is untouched:

```yaml
# configs/rpi4b_raw10.yaml  (new preset — RPi 4B, 10-bit capture bring-up)
platform:
  camera:
    backend: picamera2       # NEW source: raw uint16 mono via picamera2 + libcamera AGC
    width: 1280
    height: 800
    fps: 60                  # target; ACHIEVED RATE IS MEASURED ON-BOARD (see Risks)
    bit_depth: 10            # drives FrameBuffer dtype (uint16) + picamera2 raw config
    auto_exposure: true      # libcamera AGC (as today); false → manual operating point
    tuning_file: configs/libcamera/ov9281_mono_quadguide.json   # reused verbatim
    # analogue_gain / exposure_time_us: manual operating point, applied only when
    # auto_exposure=false (the auto↔lock hook SP2 drives). Reuse existing fields.
ground:
  tonemap: percentile        # HUD display only: linear | percentile | gamma
```

- `CameraConfig` (`core/config.py`) gains `bit_depth: int = 8` and the loader maps
  it in `cfg_platform`. `tonemap` is read under the `ground` block by the server
  (mirrors the existing `ground.ui_mode` handling); no new dataclass required.
- Guidance is unaffected: aspect `1280/800 = 1.6` equals `640/400`, and it is the
  full FoV, so `guidance.fov_horizontal_rad` is unchanged from any 1280×800 setup.
- The `picamera2` backend ignores the `pipeline` string (like `v4l2`/`csi`); no
  caps-vs-config cross-check (that check is specific to the GStreamer path).

---

## Exposure control (auto now; lock hook for SP2)

- **Default (SP1):** `auto_exposure: true` → libcamera AGC runs with the tuned
  file, exactly the exposure behavior of today's 8-bit path. The raw frame we read
  reflects the AGC-chosen gain/exposure in **linear** space, before gamma — i.e.
  the sky sits just under clipping with maximum linear headroom for a target
  against it.
- **Manual / lock hook (plumbing only in SP1):** `Picamera2RawCamera` accepts
  `auto_exposure: false` + `analogue_gain`/`exposure_time_us`, which it applies as
  picamera2 controls (`AeEnable=False`, `ExposureTime`, `AnalogueGain`). It also
  exposes a method to switch **auto ↔ locked at runtime** (set controls live). SP1
  proves both modes deliver frames; **SP2 wires the launcher "calibrate against
  sky" button** to: read the sky under AGC → freeze the operating point → lock for
  the engagement (a hunting AE changes target contrast frame to frame, which the
  detector must not fight during terminal closure).

---

## Testing

### Windows / mock (TDD here)

- **`FrameBuffer` dtype/channels** — uint16 mono round-trip (write → `read_latest`
  returns `(H,W)` uint16, values preserved incl. the low bits); `(H,W,C)` for
  multi-channel; back-compat default `uint8`/3ch reproduces current bytes; itemsize
  arithmetic. Extend `tests/unit/test_frame_buffer.py`.
- **Tonemap** — `linear`/`percentile`/`gamma` on synthetic 10-bit inputs: output is
  uint8, monotonic, correct range; a synthetic "sky + dark target" separates
  better under `percentile` than `linear` (asserts the stretch actually uses the
  8-bit range). New `tests/unit/test_tonemap.py`.
- **`to_display_bgr` adapter** — mono16 → BGR uint8 shape/dtype; BGR passthrough
  unchanged. Extend `tests/unit/test_ground_overlay.py`.
- **Config** — `bit_depth`/`backend: picamera2` parse; `frame_spec()` maps camera
  config → `(channels, dtype)` correctly for both paths. Extend
  `tests/unit/test_config.py` / `test_rpi4b_config.py`.
- **Camera-worker loop with a fake source** — a `FakeRawCamera` yielding synthetic
  uint16 frames drives `perception/camera/worker.py:run` into a mono16
  `FrameBuffer` with no hardware. Extend the perception loop tests.

`picamera2` is import-guarded (lazy, inside `open()`), so none of the above imports
it; the Windows suite never needs the package.

### Board-only (diagnostic, not unit tests)

- Real `Picamera2RawCamera` capture, achievable fps, and the end-to-end HUD.
  Covered by `scripts/diag_raw10.py` + a camera+ground bring-up, run on the Pi.

---

## Verification & risks

### Prove it is genuinely 10-bit

`scripts/diag_raw10.py` (mirrors `scripts/diag_y10_verify.py`): capture N frames via
`Picamera2RawCamera`, and report — distinct grey-level count (**> 256** ⇒ real >8-bit;
== 256 or fewer nonzero low bits ⇒ an 8-bit upcast, a red flag), low-2-bit
population, per-frame min/mean/p95, and sky-headroom (p95 below the 10-bit clip
1023). Writes a PNG (tonemapped) and a raw `.npy` for inspection. This is part of
Definition of Done.

### Risks

- **Full-res fps on the A72.** The old path hit **34.9 fps at 1280×800** *because of
  `videoconvert`* (measured, in `rpi4b.yaml`). The raw path skips debayer/videoconvert,
  so 60 fps is plausible — but it is **measured on hardware, not assumed**.
  *Fallback if 60 is unreachable and the rate matters:* leave it to on-board
  measurement (the recommendation), with a 2×2-binned **640×400 10-bit** mode (still
  full FoV) as the escape hatch — captured here, decided on the board, not now.
- **New dependency `python3-picamera2`** (apt; pulls system libcamera). Add to
  `scripts/firstboot_install_rpi.sh` and note in `requirements.txt`
  (system/apt-installed, not pip). The venv already uses `--system-site-packages`,
  so the worker sees it.
- **picamera2 raw-stream API specifics** (exact config call, whether a `main`
  stream must be declared, control names for AE lock) are verified on the board
  during bring-up; the design does not depend on a particular call spelling.
- **Off the table:** the RAW10 packing-layout unknown that plagued the rkcif path —
  libcamera/picamera2 unpacks, so no custom unpacker and no striping/moiré risk.

---

## Definition of Done

1. `FrameBuffer` carries uint16 mono (and keeps uint8/3ch back-compat); Windows
   unit tests green.
2. `Picamera2RawCamera` streams native 1280×800 linear 10-bit under the tuned AGC
   on the Pi 4B.
3. `scripts/diag_raw10.py` confirms genuine 10-bit (>256 levels, populated low
   bits) with sky headroom under the clip.
4. The HUD shows a live tonemapped view with the lock-on box, via the
   `to_display_bgr` adapter; `ground.tonemap` switches the map.
5. Exposure auto **and** locked (manual operating point) both deliver frames — the
   hook SP2 will drive.
6. Legacy BGR configs (`gstreamer`/`v4l2`/HIL) unchanged; their tests still green.

---

## File-by-file changes

| File | Change |
| --- | --- |
| `perception/camera/sources.py` | **New** `Picamera2RawCamera(CameraSource)`; lazy picamera2 import; raw uint16 mono capture + auto/lock exposure. |
| `perception/camera/worker.py` | Register `"picamera2"` in `_SOURCES`. |
| `core/frame_buffer.py` | Add `dtype` param + itemsize; dtype-aware `write_frame`/`read_latest`; mono → `(H,W)`. Back-compat defaults. |
| `core/config.py` | `CameraConfig.bit_depth`; map in `cfg_platform`; `frame_spec(camera) → (channels, dtype)` helper. |
| `scripts/run.py` | Build `FrameBuffer` with channels+dtype from `frame_spec`. |
| `ground/overlay.py` | **New** `to_display_bgr(frame, tonemap)` + tonemap fns (`linear`/`percentile`/`gamma`). |
| `ground/server.py` | `_mjpeg` calls `to_display_bgr` before overlay; read `ground.tonemap`. |
| `configs/rpi4b_raw10.yaml` | **New** preset (picamera2 backend, 1280×800, bit_depth 10, tonemap). |
| `scripts/diag_raw10.py` | **New** on-board 10-bit verification. |
| `scripts/firstboot_install_rpi.sh`, `requirements.txt` | Add/note `python3-picamera2`. |
| `tests/unit/test_frame_buffer.py`, `test_tonemap.py` (new), `test_ground_overlay.py`, `test_config.py`/`test_rpi4b_config.py` | Cover the pure-logic pieces above. |

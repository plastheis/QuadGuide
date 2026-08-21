# Webcam tracker tool — design

> **Status:** approved design. Scope: a host-only, interactive tool for qualitatively
> testing edgecv trackers against a live webcam. Lives in repo-root `tools/`, not in the
> shipped wheel. Drives trackers exclusively through the public `Tracker` API.

## 1. Purpose and scope

A single-file command-line tool, `tools/track_webcam.py`, that lets a human point a webcam
at an object, lock a tracker onto it, and watch the result box in real time. It exists to
**qualitatively** judge tracker behaviour (lock stability, drift, PSR/status under motion and
occlusion), not to benchmark or to be part of any runtime.

**In scope**
- Capturing frames from a webcam and rendering an interactive window.
- Driving any `edgecv` tracker through the public API only: `init()`, `update()`,
  `status`, `name()` — exactly as a downstream user would.
- A keyboard-driven init/release workflow with a scalable selection box.

**Out of scope**
- Being a runtime dependency of the library. The tool is host-only (ARCHITECTURE.md §11,
  §13) and lives in repo-root `tools/`, parallel to the `edgecv/` package, excluded from
  the wheel.
- Recording, benchmarking, metrics export, multi-object, or scale adaptation in the tool
  itself. The tool only feeds frames; trackers own all tracking logic.
- Unit-testing the camera/render loop. Only the pure helper functions are tested.

## 2. Dependencies

- `numpy` (already a core dependency).
- `opencv-python` for `VideoCapture`, drawing, and the display window. This is host-only;
  it is already listed in the `[fast]` optional extra. It is **not** added as a runtime
  dependency of the library.
- Standard library `argparse`, `time`.

If `cv2` cannot be imported, the tool exits with a clear message:
`opencv-python is required for this host tool: pip install opencv-python`.

## 3. The contract the tool relies on

A user — and therefore this tool — interacts with a tracker like this:

```python
from edgecv.trackers.cf import Mosse
from edgecv import BoundingBox

tracker = Mosse()
tracker.init(rgb_frame, bbox)        # bbox is normalised 0–1
result = tracker.update(rgb_frame)   # TrackResult: bbox (normalised), confidence (PSR), status
```

- `bbox` passed to `init` is a normalised `BoundingBox` (0–1), built from a pixel-space
  selection via `BoundingBox.from_pixels(PixelBox(...), width, height)`.
- `update` returns a `TrackResult` whose `bbox` is normalised; the tool converts back to
  pixels with `bbox.to_pixels(width, height)` for drawing.
- `result.status` is a `TrackStatus` (`LOCKED` / `COASTING` / `LOST` / `INITIALIZING`);
  `result.confidence` is the PSR for CF trackers (may be `None` for trackers with no score).
- The tool treats the tracker as a black box behind this contract. It never imports tracker
  internals, ops, or filter state.

### 3.1 Color order (the one correctness detail)

`edgecv.trackers.cf.ops.features._to_gray` converts color to luminance with **RGB** luma
weights. OpenCV's `VideoCapture` yields **BGR**. The tool is the "caller that owns frames"
(ARCHITECTURE.md §1), so it is responsible for handing the tracker the color order its luma
weights expect. Therefore, per frame:

```
bgr   = capture.read()                     # native OpenCV order
rgb   = cv2.cvtColor(bgr, COLOR_BGR2RGB)   # fed to tracker.init / tracker.update
display = bgr.copy()                       # all overlays drawn here; never fed back
```

The RGB frame is what the tracker sees; the BGR copy is only for display.

## 4. Interaction model

A two-state machine, advanced by `cv2.waitKey(1)`:

### SETUP (no lock)
- A **white square** is drawn centered on the frame, side length `box_px` (default 96).
- `+` / `=` grows the square; `-` / `_` shrinks it. Step ~16 px. Clamped to a minimum
  tracker-safe size (`MIN_BOX_PX`, e.g. 24) and to the smaller frame dimension.
- `Space` snaps the current square to a normalised `BoundingBox`, constructs a **fresh**
  tracker instance from the registry, calls `tracker.init(rgb, bbox)`, and transitions to
  TRACKING. (Fresh instance each lock → repeatable re-testing in one session.)

### TRACKING (locked)
- Each frame: `result = tracker.update(rgb)`. The result `bbox` is drawn as a rectangle,
  **color-coded by status**:
  - `LOCKED` → orange
  - `COASTING` → yellow
  - `LOST` → red
  - (`INITIALIZING` → orange, treated as the nominal post-init case)
- `r` discards the tracker and returns to SETUP; the white square reappears at the last
  size. `+` / `-` are inert in TRACKING (MOSSE has no scale adaptation; the box size is
  fixed at init).

### Either state
- `q` or `ESC` quits.

## 5. Rendering / HUD

All overlays are drawn on the BGR `display` copy.

- **SETUP:** white square outline.
- **TRACKING:** status-colored result rectangle.
- **HUD (top-left text):** `"<NAME> | PSR <psr> | <STATUS> | <fps> FPS"`, e.g.
  `MOSSE | PSR 12.3 | LOCKED | 28 FPS`. PSR shows `--` when confidence is `None`. In SETUP
  the HUD shows the tracker name and `SETUP` in place of status.
- **Key-hint line:** `[space] lock  [r] release  [+/-] size  [q] quit`.
- **FPS:** smoothed (exponential moving average) frame-to-frame time measured by the tool.

## 6. Components (all in `tools/track_webcam.py`)

| Component | Responsibility |
|---|---|
| `TRACKERS: dict[str, Callable[[], Tracker]]` | Name→factory registry, `{"mosse": Mosse}`. `--tracker` selects (default `mosse`). Adding a tracker later is one line. |
| `centered_square(frame_h, frame_w, size_px) -> PixelBox` | Build a centered square `PixelBox` of the given side, clamped to the frame. Pure. |
| `clamp_box_size(size_px, frame_h, frame_w) -> int` | Clamp a requested square side to `[MIN_BOX_PX, min(frame_h, frame_w)]`. Pure. |
| `status_color(status: TrackStatus) -> tuple[int,int,int]` | Map status → BGR color. Pure. |
| `draw_square`, `draw_result`, `draw_hud` | OpenCV drawing onto the display frame. |
| `main()` | Argparse, capture open + failure check, the loop, guaranteed teardown. |

### CLI

```
tools/track_webcam.py [--camera N] [--tracker NAME] [--width W] [--height H] [--list]
```

- `--camera` (default 0): `VideoCapture` index.
- `--tracker` (default `mosse`): registry key.
- `--width` / `--height` (optional): requested capture resolution.
- `--list`: print available tracker names and exit.

## 7. Error handling

- **Missing `cv2`:** caught at import, prints the install hint, exits non-zero.
- **Camera open failure** (`cap.isOpened()` false, or first `read()` fails): clear message
  naming the camera index, exits non-zero.
- **Unknown `--tracker`:** argparse-level error listing valid names.
- **Degenerate init box:** `MIN_BOX_PX` clamp in SETUP guarantees `init` never receives a
  too-small box.
- **State safety:** `update` is only ever called in TRACKING, which is only entered after a
  successful `init`.
- **Teardown:** `cap.release()` and `cv2.destroyAllWindows()` run in a `finally` so the
  camera is freed on any exit path (including exceptions and `q`/`ESC`).

## 8. Testing strategy

The capture/render loop is interactive and camera-bound; it is validated manually
(qualitatively — which is the tool's whole purpose). To keep that surface small, all pure
logic lives in module-level functions that are unit-tested without a camera or `cv2` window:

- `clamp_box_size` — min/max clamping, including frames smaller than `MIN_BOX_PX`.
- `centered_square` — correct center and side for representative frame sizes.
- `status_color` — every `TrackStatus` maps to its expected BGR tuple.

These tests import only the helper functions and `TrackStatus`; they must not require a
camera or open a window. Tests run via `.venv/bin/pytest`.

## 9. Constants (defaults)

| Name | Value | Meaning |
|---|---|---|
| `DEFAULT_BOX_PX` | 96 | initial square side |
| `MIN_BOX_PX` | 24 | smallest allowed square side |
| `BOX_STEP_PX` | 16 | `+`/`-` increment |
| orange / yellow / red | BGR tuples | LOCKED / COASTING / LOST result-box colors |
| white | (255,255,255) | SETUP square color |

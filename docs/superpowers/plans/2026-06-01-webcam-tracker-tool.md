# Webcam Tracker Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tools/track_webcam.py`, a host-only interactive tool that feeds webcam frames to an edgecv tracker and renders the result, for qualitative testing.

**Architecture:** A single file owning only camera I/O and rendering. Pure helpers (box sizing, centered-square geometry, status→color) are module-level and unit-tested without a camera. The capture/render loop drives any tracker through the public `Tracker` API (`init`/`update`/`status`/`name`). `cv2` is imported under a `TYPE_CHECKING`/`try` guard so the module imports (and its helpers test) even where `opencv-python` is absent.

**Tech Stack:** Python 3.10+, numpy, opencv-python (host-only), argparse. Tests via `.venv/bin/pytest`.

Reference spec: `docs/superpowers/specs/2026-06-01-webcam-tracker-tool-design.md`.

---

### Task 1: Pure helpers + module scaffold (TDD)

**Files:**
- Create: `tools/track_webcam.py`
- Test: `tests/test_track_webcam.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_track_webcam.py`. It loads the tool via `importlib` from its file path (the
`tools/` dir is not a package and is not on `sys.path`), which works because the module's `cv2`
import is guarded — so importing it does not require `opencv-python`.

```python
import importlib.util
from pathlib import Path

import pytest

from edgecv.core.result import TrackStatus

_PATH = Path(__file__).resolve().parent.parent / "tools" / "track_webcam.py"
_spec = importlib.util.spec_from_file_location("track_webcam", _PATH)
tw = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(tw)


def test_clamp_box_size_within_bounds_unchanged():
    assert tw.clamp_box_size(96, 480, 640) == 96


def test_clamp_box_size_floors_to_min():
    assert tw.clamp_box_size(8, 480, 640) == tw.MIN_BOX_PX


def test_clamp_box_size_caps_to_smaller_frame_dim():
    assert tw.clamp_box_size(500, 480, 640) == 480


def test_clamp_box_size_tiny_frame_returns_frame_dim():
    # frame smaller than MIN_BOX_PX -> return the frame dimension
    assert tw.clamp_box_size(96, 10, 20) == 10


def test_centered_square_is_centered():
    pix = tw.centered_square(480, 640, 96)
    assert pix.w == 96 and pix.h == 96
    assert pix.x == (640 - 96) / 2.0
    assert pix.y == (480 - 96) / 2.0
    assert pix.center == (320.0, 240.0)


def test_centered_square_clamps_size():
    pix = tw.centered_square(480, 640, 5000)
    assert pix.w == 480 and pix.h == 480


@pytest.mark.parametrize(
    "status,expected",
    [
        (TrackStatus.LOCKED, tw.ORANGE),
        (TrackStatus.INITIALIZING, tw.ORANGE),
        (TrackStatus.COASTING, tw.YELLOW),
        (TrackStatus.LOST, tw.RED),
    ],
)
def test_status_color(status, expected):
    assert tw.status_color(status) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_track_webcam.py -q`
Expected: FAIL — `FileNotFoundError` / module load error because `tools/track_webcam.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `tools/track_webcam.py` with the guarded import, constants, registry, and the three
pure helpers (the loop/rendering come in Task 2):

```python
#!/usr/bin/env python3
"""Host-only interactive webcam harness for qualitatively testing edgecv trackers.

Owns ONLY camera I/O and rendering; drives trackers through the public Tracker API
(init/update/status/name), exactly as a downstream user would. Not a runtime dependency
of the library (ARCHITECTURE.md §11, §13). Lives in repo-root tools/, excluded from the wheel.

Controls:
  [space] lock tracker on the white square    [r] release back to setup
  [+/-]   grow/shrink the selection square     [q]/[ESC] quit
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackStatus
from edgecv.core.tracker import Tracker
from edgecv.trackers.cf import Mosse

if TYPE_CHECKING:
    import cv2
else:
    try:
        import cv2
    except ImportError:  # host tool only; keep the module importable for unit tests
        cv2 = None

# --- constants ---
DEFAULT_BOX_PX = 96
MIN_BOX_PX = 24
BOX_STEP_PX = 16

# BGR colors (OpenCV order)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

TRACKERS: dict[str, Callable[[], Tracker]] = {
    "mosse": Mosse,
}


# --- pure helpers (unit-tested; no cv2) ---
def clamp_box_size(size_px: int, frame_h: int, frame_w: int) -> int:
    """Clamp a square side to [MIN_BOX_PX, min(frame_h, frame_w)]."""
    upper = min(frame_h, frame_w)
    if upper < MIN_BOX_PX:
        return upper
    return max(MIN_BOX_PX, min(size_px, upper))


def centered_square(frame_h: int, frame_w: int, size_px: int) -> PixelBox:
    """A centered square PixelBox of side ``size_px`` (clamped to the frame)."""
    side = clamp_box_size(size_px, frame_h, frame_w)
    x = (frame_w - side) / 2.0
    y = (frame_h - side) / 2.0
    return PixelBox(x=x, y=y, w=float(side), h=float(side))


def status_color(status: TrackStatus) -> tuple[int, int, int]:
    """Map track status to a BGR result-box color."""
    if status == TrackStatus.COASTING:
        return YELLOW
    if status == TrackStatus.LOST:
        return RED
    return ORANGE  # LOCKED / INITIALIZING -> nominal
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_track_webcam.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/track_webcam.py tests/test_track_webcam.py
git commit -m "feat(tools): webcam tracker tool helpers + scaffold

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Capture/render loop + CLI

**Files:**
- Modify: `tools/track_webcam.py` (append rendering, CLI, and `main()`)

This task adds the interactive loop. It is camera-bound and validated manually; the only
automated checks are that the module still imports without `cv2`, that `--list` works without
`cv2`, and that ruff/mypy pass.

- [ ] **Step 1: Append rendering helpers, CLI, and main()**

Add to the end of `tools/track_webcam.py`:

```python
# --- rendering (cv2) ---
def _draw_box(display, pix: PixelBox, color: tuple[int, int, int], thickness: int = 2) -> None:
    x0, y0 = int(round(pix.x)), int(round(pix.y))
    x1, y1 = int(round(pix.x + pix.w)), int(round(pix.y + pix.h))
    cv2.rectangle(display, (x0, y0), (x1, y1), color, thickness)


def _draw_hud(
    display, name: str, status_text: str, psr: float | None, fps: float
) -> None:
    psr_text = f"{psr:.1f}" if psr is not None else "--"
    line1 = f"{name} | PSR {psr_text} | {status_text} | {fps:.0f} FPS"
    line2 = "[space] lock  [r] release  [+/-] size  [q] quit"
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, y, scale in ((line1, 24, 0.6), (line2, 48, 0.5)):
        cv2.putText(display, text, (10, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, text, (10, y), font, scale, WHITE, 1, cv2.LINE_AA)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualitatively test edgecv trackers on a live webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="VideoCapture index (default 0)")
    parser.add_argument(
        "--tracker", choices=sorted(TRACKERS), default="mosse", help="tracker to run"
    )
    parser.add_argument("--width", type=int, default=None, help="requested capture width")
    parser.add_argument("--height", type=int, default=None, help="requested capture height")
    parser.add_argument("--list", action="store_true", help="list trackers and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list:
        print("\n".join(sorted(TRACKERS)))
        return 0
    if cv2 is None:
        print(
            "opencv-python is required for this host tool: pip install opencv-python",
            file=sys.stderr,
        )
        return 1

    cap = cv2.VideoCapture(args.camera)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"could not open camera index {args.camera}", file=sys.stderr)
        return 1

    make_tracker = TRACKERS[args.tracker]
    tracker: Tracker | None = None
    box_px = DEFAULT_BOX_PX
    fps = 0.0
    last = time.monotonic()
    win = "edgecv tracker"
    cv2.namedWindow(win)
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                print("camera read failed", file=sys.stderr)
                return 1
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # tracker sees RGB (luma weights)
            display = bgr.copy()                        # overlays only; never fed back

            now = time.monotonic()
            dt = now - last
            last = now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst

            if tracker is None:
                box_px = clamp_box_size(box_px, h, w)
                _draw_box(display, centered_square(h, w, box_px), WHITE)
                _draw_hud(display, args.tracker.upper(), "SETUP", None, fps)
            else:
                result = tracker.update(rgb)
                if result.bbox is not None:
                    _draw_box(display, result.bbox.to_pixels(w, h), status_color(result.status))
                _draw_hud(display, tracker.name(), result.status.name, result.confidence, fps)

            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                tracker = None
            elif key == ord(" ") and tracker is None:
                box_px = clamp_box_size(box_px, h, w)
                bbox = BoundingBox.from_pixels(centered_square(h, w, box_px), w, h)
                tracker = make_tracker()
                tracker.init(rgb, bbox)
            elif key in (ord("+"), ord("=")) and tracker is None:
                box_px = clamp_box_size(box_px + BOX_STEP_PX, h, w)
            elif key in (ord("-"), ord("_")) and tracker is None:
                box_px = clamp_box_size(box_px - BOX_STEP_PX, h, w)
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify module still imports and unit tests still pass without cv2**

Run: `.venv/bin/pytest tests/test_track_webcam.py -q`
Expected: PASS (8 tests) — the guarded import keeps the module loadable without `opencv-python`.

- [ ] **Step 3: Verify `--list` works without cv2**

Run: `.venv/bin/python tools/track_webcam.py --list`
Expected: prints `mosse` and exits 0 (the `--list` branch returns before touching `cv2`).

- [ ] **Step 4: Verify the no-cv2 error path**

Run: `.venv/bin/python tools/track_webcam.py`
Expected: prints `opencv-python is required for this host tool: pip install opencv-python` to
stderr and exits 1 (because `cv2` is not installed in `.venv`).

- [ ] **Step 5: Lint and type-check**

Run: `.venv/bin/ruff check tools/track_webcam.py tests/test_track_webcam.py`
Expected: no errors.

Run: `.venv/bin/mypy tools/track_webcam.py`
Expected: no errors (`cv2` resolves via the `TYPE_CHECKING` import + `ignore_missing_imports`).

- [ ] **Step 6: Commit**

```bash
git add tools/track_webcam.py
git commit -m "feat(tools): webcam capture/render loop + CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Manual verification (post-implementation, requires a webcam)

On a host with `opencv-python` and a camera:

```bash
pip install opencv-python
python tools/track_webcam.py            # default camera 0, mosse
```

- White square appears centered; `+`/`-` resize it.
- `Space` locks: box turns orange and follows the object; HUD shows PSR/status/FPS.
- Move/occlude the target: box color tracks status (orange→yellow→red).
- `r` releases back to the white square; `q`/`ESC` quits and frees the camera.

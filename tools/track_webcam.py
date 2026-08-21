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
from pathlib import Path
from typing import TYPE_CHECKING

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackStatus
from edgecv.core.tracker import Tracker
from edgecv.models.manifest import load_manifest
from edgecv.models.paths import resolve_artifact_path
from edgecv.trackers.cf import Mosse
from edgecv.trackers.nn import NanoTrack, SiamFC, YoloTracker
from edgecv.trackers.nn.base import select_backend

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

_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = _ROOT / "src" / "edgecv" / "models" / "manifests"
MODELS_DIR = _ROOT / "models"

_NN_TRACKERS: dict[str, tuple[Callable[..., Tracker], str]] = {
    "siamfc": (SiamFC, "siamfc_generic.yaml"),
    "yolo": (YoloTracker, "yolo26n.yaml"),
}
# NanoTrack needs two separate ONNX models (backbone + head).
_NANOTRACK_MANIFEST = "nanotrack.yaml"
# All selectable tracker names.
TRACKERS: tuple[str, ...] = (
    "mosse", *_NN_TRACKERS, "nanotrack"
)


def build_tracker(name: str, model_path: str | None = None,
                  backend: str = "auto") -> Tracker:
    """Construct a tracker by name.

    ``mosse`` needs no model. NN trackers (``siamfc``/``yolo``) load a single
    model. ``nanotrack`` loads two models (backbone + head). ``backend`` selects
    the inference backend (``onnx`` on the host, ``rknn`` on the Rockchip NPU;
    ``auto`` prefers rknn then onnx).

    Raises FileNotFoundError if required model weights are absent.
    """
    if name == "mosse":
        return Mosse()

    # --- nanotrack: split backbone + head models ---
    if name == "nanotrack":
        return _build_nanotrack(model_path, backend)

    # --- single-model NN trackers ---
    if name in _NN_TRACKERS:
        cls, manifest_file = _NN_TRACKERS[name]
    else:
        raise ValueError(f"unknown tracker: {name}")

    manifest = load_manifest(MANIFESTS_DIR / manifest_file)

    # Standalone NN trackers: validate the ONNX model exists.
    if model_path:
        weights = Path(model_path)
    else:
        artifact = manifest.artifacts.get("onnx") or {}
        weights = MODELS_DIR / Path(artifact.get("path", f"{name}.onnx")).name
    if not weights.is_file():
        raise FileNotFoundError(
            f"{name} needs an ONNX model; expected {weights}. "
            f"Pass --model PATH, or drop the weights file in {MODELS_DIR}/."
        )
    manifest.artifacts["onnx"] = {"path": str(weights)}
    return cls(manifest, backend="onnx")


def _build_nanotrack(model_path: str | None = None,
                     backend: str = "auto") -> NanoTrack:
    """Build NanoTrack from its split backbone + head artifacts.

    ``backend`` picks the inference backend: ``onnx`` (host) or ``rknn`` (Rockchip
    NPU); ``auto`` prefers rknn then onnx. The same logical manifest names both the
    .onnx and .rknn artifacts, so only the backend changes between host and device.
    ``--model`` is ignored here: NanoTrack needs two distinct model files.
    """
    manifest = load_manifest(MANIFESTS_DIR / _NANOTRACK_MANIFEST)
    name = select_backend(backend)

    # Preflight: a friendly error if either artifact file for this backend is
    # missing (the backend's own load() would otherwise fail more opaquely).
    for key in ("backbone", "head"):
        art = (manifest.artifacts.get(key) or {}).get(name) or {}
        path = art.get("path")
        if path and not Path(resolve_artifact_path(path)).is_file():
            raise FileNotFoundError(
                f"nanotrack[{name}] needs a model for {key!r}; expected "
                f"{resolve_artifact_path(path)}. Convert/drop the weights in "
                f"{MODELS_DIR}/ (see tools/CONVERSION.md for ONNX->RKNN)."
            )

    return NanoTrack.from_manifest(manifest, backend=name)


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


# --- rendering (cv2) ---
def _draw_box(display, pix: PixelBox, color: tuple[int, int, int], thickness: int = 2) -> None:
    x0, y0 = int(round(pix.x)), int(round(pix.y))
    x1, y1 = int(round(pix.x + pix.w)), int(round(pix.y + pix.h))
    cv2.rectangle(display, (x0, y0), (x1, y1), color, thickness)


def _draw_hud(
    display, name: str, status_text: str, conf: float | None, fps: float
) -> None:
    conf_text = f"{conf:.1f}" if conf is not None else "--"
    line1 = f"{name} | conf {conf_text} | {status_text} | {fps:.0f} FPS"
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
        "--tracker", choices=sorted(TRACKERS), default=None,
        help="tracker to run (default: show available and exit)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="ONNX model path for NN trackers (siamfc/yolo); "
             "default models/<artifact>.onnx",
    )
    parser.add_argument(
        "--backend", choices=["auto", "onnx", "rknn"], default="auto",
        help="inference backend for NN/nanotrack trackers "
             "(onnx=host, rknn=Rockchip NPU; default auto)",
    )
    parser.add_argument("--width", type=int, default=None, help="requested capture width")
    parser.add_argument("--height", type=int, default=None, help="requested capture height")
    parser.add_argument("--list", action="store_true", help="list trackers and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list or args.tracker is None:
        print("Available trackers:")
        for t in sorted(TRACKERS):
            print(f"  {t}")
        if args.list:
            return 0
        print("\nPass --tracker NAME to select one.")
        return 0
    if cv2 is None:
        print(
            "opencv-python is required for this host tool: pip install opencv-python",
            file=sys.stderr,
        )
        return 1

    # Build once up front: fail fast on a missing/unloadable model, and reuse the
    # (possibly heavy) ONNX session across lock/release cycles via init().
    try:
        tracker = build_tracker(args.tracker, args.model, args.backend)
    except (FileNotFoundError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"could not open camera index {args.camera}", file=sys.stderr)
        tracker.close()
        return 1

    locked = False
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

            if not locked:
                box_px = clamp_box_size(box_px, h, w)
                _draw_box(display, centered_square(h, w, box_px), WHITE)
                _draw_hud(display, tracker.name(), "SETUP — press SPACE to lock", None, fps)
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
                locked = False
            elif key == ord(" ") and not locked:
                box_px = clamp_box_size(box_px, h, w)
                bbox = BoundingBox.from_pixels(centered_square(h, w, box_px), w, h)
                tracker.init(rgb, bbox)
                locked = True
            elif key in (ord("+"), ord("=")) and not locked:
                box_px = clamp_box_size(box_px + BOX_STEP_PX, h, w)
            elif key in (ord("-"), ord("_")) and not locked:
                box_px = clamp_box_size(box_px - BOX_STEP_PX, h, w)
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    raise SystemExit(main())

# NN Trackers (SiamFC + class-agnostic YOLO) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first dense-network trackers as **standalone, inline** trackers per the spec `docs/superpowers/specs/2026-06-04-nn-trackers-design.md`: a shared `NNTracker` base, `SiamFC` (multi-scale), and a class-agnostic `YoloDetector` + `YoloTracker` (local-crop association). Plus manifests and device-only RKNN model wiring.

**Architecture:** Trackers depend only on the HAL (`InferenceBackend`/`Model`/`ModelManifest`, ARCHITECTURE §10), never a vendor runtime. A `model=` dependency-injection seam lets tests drive the exact geometry/decode/association logic with **deterministic stub `Model`s** (real weights are deferred to host-only `tools/`, spec §1). Image math lives in module-level pure functions in `trackers/nn/preprocess.py` (importable in a future `spawn`ed worker); SiamFC/YOLO wire them into the `Tracker` API.

**Tech Stack:** Python 3.10+, numpy only for the trackers (an inference backend extra is needed for a *real* model, but every test here uses a stub or the `mock` backend). Reuses `ops.psr` (generic response-map statistic). Tests with pytest via `.venv/bin/python -m pytest`.

**Conventions pinned for this plan:**
- **Box format in `DetectorOutput.boxes`:** `(N, 4)` **normalised xywh top-left** (matches `BoundingBox` field order). Documented at the producer (`YoloDetector.detect`).
- **Crop↔frame mapping** is centre-aligned: output pixel `o` ↔ frame coord `(top_left) + (o+0.5)/out * size`. The single inversion path is `CropXform.to_frame` / `LetterboxXform.to_orig_xyxy` (ARCHITECTURE §5.1 — norm/pixel mixing is the #1 bug source).
- **YOLO search window is square** (side = `search_factor * max(w_px, h_px)`) in v1 to avoid aspect distortion through the crop→letterbox→detect→invert chain. (Spec §6.2 said "per axis"; square is the v1 simplification — note it in the spec if it sticks.)
- **No `clamp()` on tracker output** — scale-adaptive boxes report off-frame coords truthfully (`bbox.py` docstring; ARCHITECTURE §9).

---

## File Structure

```
edgecv/trackers/nn/
├── preprocess.py     # CREATE — resize_bilinear, crop_with_context+CropXform, letterbox+LetterboxXform, to_input, class_agnostic_nms
├── base.py           # CREATE — Template, resolve_model/_select_backend, NNTracker(Tracker)
├── siamfc.py         # CREATE — SiamFC(NNTracker), multi-scale
├── yolo.py           # CREATE — YoloDetector (-> DetectorOutput) + YoloTracker(NNTracker)
└── __init__.py       # MODIFY — export SiamFC, YoloDetector, YoloTracker

edgecv/models/manifests/
├── siamfc_generic.yaml   # CREATE
└── yolo_generic.yaml     # CREATE

edgecv/backends/rknn/__init__.py   # MODIFY — RknnBackend.load -> RknnModel (device-only)
pyproject.toml                     # MODIFY — package the manifests

tests/
├── test_nn_preprocess.py   # CREATE
├── test_nn_base.py         # CREATE
├── test_siamfc.py          # CREATE
├── test_yolo.py            # CREATE
├── test_manifests_nn.py    # CREATE (manifest load + packaging)
├── _nn_stubs.py            # CREATE (deterministic stub Model, Task 5)
├── _onnx_synth.py          # CREATE (synthetic ONNX model builders, Task 11)
└── test_nn_onnx.py         # CREATE (onnx-backend integration, Task 11)
```

Run all tests with: `.venv/bin/python -m pytest -q`

A reusable stub `Model` is defined once (Task 5 test file) and imported by the SiamFC/YOLO tests:
`tests/_nn_stubs.py` (see Task 5). Place it under `tests/` so pytest's rootdir import works.

---

### Task 1: `resize_bilinear` + `crop_with_context` + `CropXform`

**Files:**
- Create: `edgecv/trackers/nn/preprocess.py`
- Test: `tests/test_nn_preprocess.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_nn_preprocess.py`:

```python
import numpy as np
import pytest

from edgecv.trackers.nn.preprocess import crop_with_context, resize_bilinear


def test_resize_bilinear_identity():
    img = np.random.default_rng(0).standard_normal((8, 8, 1)).astype(np.float32)
    out = resize_bilinear(img, (8, 8))
    np.testing.assert_allclose(out, img, atol=1e-5)


def test_resize_bilinear_changes_shape_keeps_channels():
    img = np.zeros((4, 4, 3), np.float32)
    out = resize_bilinear(img, (8, 16))
    assert out.shape == (8, 16, 3)


def test_crop_with_context_centre_and_shape():
    frame = np.zeros((100, 120, 3), np.uint8)
    frame[48:52, 58:62] = 200  # bright square at centre (~ (60, 50))
    patch, xf = crop_with_context(frame, (60.0, 50.0), (20.0, 20.0), (40, 40))
    assert patch.shape == (40, 40, 3)
    # brightest output pixel maps back to ~the frame centre we cropped around
    py, px = np.unravel_index(int(patch[..., 0].argmax()), patch.shape[:2])
    fx, fy = xf.to_frame((px, py))
    assert fx == pytest.approx(60.0, abs=2.0)
    assert fy == pytest.approx(50.0, abs=2.0)


def test_crop_with_context_edge_replicates_off_frame():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    # centre at a corner: window crosses the border, must still return full size
    patch, _ = crop_with_context(frame, (0.0, 0.0), (6.0, 6.0), (12, 12))
    assert patch.shape == (12, 12)
    assert np.isfinite(patch).all()


def test_crop_xform_to_frame_roundtrip():
    _, xf = crop_with_context(np.zeros((50, 50)), (25.0, 30.0), (10.0, 20.0), (32, 64))
    # output centre maps to the crop centre
    fx, fy = xf.to_frame((32.0 - 0.5, 16.0 - 0.5))  # (ow/2-0.5, oh/2-0.5)
    assert fx == pytest.approx(25.0, abs=1e-6)
    assert fy == pytest.approx(30.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_nn_preprocess.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.trackers.nn.preprocess'`

- [ ] **Step 3: Write minimal implementation**

Create `edgecv/trackers/nn/preprocess.py`:

```python
"""Pure numpy preprocessing for NN trackers (ARCHITECTURE.md §6.2, §7.4).

Module-level functions only, so a future spawned worker can import them. Numpy
reference today; RK RGA / DMA crop-resize can swap in behind this boundary later
with no tracker change (ARCHITECTURE §16). All crop<->frame and letterbox<->image
coordinate inversion lives here, in one place (ARCHITECTURE §5.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sample_clamped(img: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Bilinear sample at float (gx, gy) frame coords; clamp = edge-replicate."""
    h, w = img.shape[0], img.shape[1]
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    wx = (gx - x0).astype(np.float32)
    wy = (gy - y0).astype(np.float32)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    if img.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    ia, ib = img[y0c, x0c], img[y0c, x1c]
    ic, idd = img[y1c, x0c], img[y1c, x1c]
    top = ia * (1.0 - wx) + ib * wx
    bot = ic * (1.0 - wx) + idd * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def resize_bilinear(img: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize (H,W[,C]) to out_hw with centre-aligned bilinear sampling."""
    oh, ow = out_hw
    h, w = img.shape[0], img.shape[1]
    xs = (np.arange(ow) + 0.5) * (w / ow) - 0.5
    ys = (np.arange(oh) + 0.5) * (h / oh) - 0.5
    gx, gy = np.meshgrid(xs, ys)
    return _sample_clamped(img.astype(np.float32), gx, gy)


@dataclass(frozen=True)
class CropXform:
    center: tuple[float, float]    # crop centre in frame px (cx, cy)
    size_px: tuple[float, float]   # crop side in frame px (sh, sw)
    out_size: tuple[int, int]      # resized output (oh, ow)

    def to_frame(self, out_xy: tuple[float, float]) -> tuple[float, float]:
        ox, oy = out_xy
        oh, ow = self.out_size
        sh, sw = self.size_px
        cx, cy = self.center
        fx = (cx - sw / 2.0) + (ox + 0.5) / ow * sw
        fy = (cy - sh / 2.0) + (oy + 0.5) / oh * sh
        return fx, fy


def crop_with_context(
    frame: np.ndarray,
    center: tuple[float, float],
    size_px: tuple[float, float],
    out_size: tuple[int, int],
) -> tuple[np.ndarray, CropXform]:
    """Crop a (sh, sw)-px window centred at `center`, edge-replicate at borders,
    resize to out_size in one gather. Returns the patch and the inversion transform."""
    cx, cy = center
    sh, sw = size_px
    oh, ow = out_size
    fx = (cx - sw / 2.0) + (np.arange(ow) + 0.5) / ow * sw
    fy = (cy - sh / 2.0) + (np.arange(oh) + 0.5) / oh * sh
    gx, gy = np.meshgrid(fx, fy)
    patch = _sample_clamped(frame, gx, gy)
    if frame.ndim == 2:
        patch = patch.reshape(oh, ow)
    return patch, CropXform(center, (sh, sw), (oh, ow))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_nn_preprocess.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/preprocess.py tests/test_nn_preprocess.py
git commit -m "feat(nn/preprocess): bilinear resize + border-safe crop_with_context"
```

---

### Task 2: `letterbox` + `LetterboxXform`

**Files:**
- Modify: `edgecv/trackers/nn/preprocess.py`
- Test: `tests/test_nn_preprocess.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_nn_preprocess.py` (extend imports to include `letterbox`):

```python
def test_letterbox_preserves_aspect_and_pads():
    img = np.zeros((50, 100, 3), np.uint8)  # 2:1 wide
    out, xf = letterbox(img, (64, 64), pad_value=114)
    assert out.shape == (64, 64, 3)
    # 100->64 sets scale 0.64; height 50*0.64=32, padded symmetrically in 64
    assert xf.scale == pytest.approx(0.64)
    assert xf.pad[1] == pytest.approx((64 - 32) / 2.0)  # vertical pad


def test_letterbox_inverts_box():
    img = np.zeros((50, 100, 3), np.uint8)
    _, xf = letterbox(img, (64, 64))
    # a box covering the whole original maps from the unpadded letterbox region
    px, py = xf.pad
    s = xf.scale
    x1, y1, x2, y2 = xf.to_orig_xyxy((px, py, px + 100 * s, py + 50 * s))
    assert (x1, y1, x2, y2) == pytest.approx((0.0, 0.0, 100.0, 50.0), abs=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_nn_preprocess.py -k letterbox -q`
Expected: FAIL — `ImportError: cannot import name 'letterbox'`

- [ ] **Step 3: Write minimal implementation**

Add to `edgecv/trackers/nn/preprocess.py`:

```python
@dataclass(frozen=True)
class LetterboxXform:
    scale: float                   # uniform resize factor applied to the original
    pad: tuple[float, float]       # (pad_x, pad_y) added in output px
    out_size: tuple[int, int]      # (oh, ow)
    orig_size: tuple[int, int]     # (h, w)

    def to_orig_xyxy(self, xyxy: tuple[float, float, float, float]):
        x1, y1, x2, y2 = xyxy
        px, py = self.pad
        s = self.scale
        return ((x1 - px) / s, (y1 - py) / s, (x2 - px) / s, (y2 - py) / s)


def letterbox(
    image: np.ndarray, out_size: tuple[int, int], *, pad_value: float = 114.0
) -> tuple[np.ndarray, LetterboxXform]:
    """Aspect-preserving resize + symmetric pad into out_size (YOLO convention)."""
    oh, ow = out_size
    h, w = image.shape[0], image.shape[1]
    s = min(oh / h, ow / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = resize_bilinear(image, (nh, nw))
    ch = image.shape[2] if image.ndim == 3 else 1
    canvas = np.full((oh, ow, ch), pad_value, np.float32)
    pad_y = (oh - nh) / 2.0
    pad_x = (ow - nw) / 2.0
    y0, x0 = int(round(pad_y)), int(round(pad_x))
    block = resized if resized.ndim == 3 else resized[..., None]
    canvas[y0:y0 + nh, x0:x0 + nw] = block
    out = canvas if image.ndim == 3 else canvas[..., 0]
    return out, LetterboxXform(s, (float(x0), float(y0)), (oh, ow), (h, w))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_nn_preprocess.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/preprocess.py tests/test_nn_preprocess.py
git commit -m "feat(nn/preprocess): aspect-preserving letterbox + box inversion"
```

---

### Task 3: `to_input` + `class_agnostic_nms`

**Files:**
- Modify: `edgecv/trackers/nn/preprocess.py`
- Test: `tests/test_nn_preprocess.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_nn_preprocess.py` (extend imports to include `class_agnostic_nms`, `to_input`; add `from edgecv.backends.base import TensorSpec`):

```python
def test_to_input_gray_layout_and_scale():
    patch = np.full((8, 8, 3), 255, np.uint8)
    spec = TensorSpec(name="exemplar", shape=(1, 1, 8, 8), dtype="float32")
    arr = to_input(patch, spec, color="gray", scale=1 / 255)
    assert arr.shape == (1, 1, 8, 8)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, 1.0, atol=1e-4)


def test_to_input_int8_quant():
    patch = np.zeros((4, 4, 3), np.uint8)
    spec = TensorSpec(name="x", shape=(1, 3, 4, 4), dtype="int8",
                      quant={"scale": 0.5, "zero_point": -3})
    arr = to_input(patch, spec, color="rgb", scale=1 / 255)
    assert arr.dtype == np.int8
    # value 0.0 -> round(0/0.5) + (-3) = -3
    assert int(arr.flat[0]) == -3


def test_class_agnostic_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = class_agnostic_nms(boxes, scores, iou_thresh=0.5)
    assert set(keep.tolist()) == {0, 2}  # box 1 overlaps the higher-scored box 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_nn_preprocess.py -k "to_input or nms" -q`
Expected: FAIL — `ImportError: cannot import name 'to_input'`

- [ ] **Step 3: Write minimal implementation**

Add to `edgecv/trackers/nn/preprocess.py` (add the `TensorSpec` import at top):

```python
from edgecv.backends.base import TensorSpec
```

```python
def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img[..., None]
    if img.shape[2] == 1:
        return img
    w = np.array([0.299, 0.587, 0.114], np.float32)
    return (img[..., :3] * w).sum(axis=2, keepdims=True)


def to_input(patch: np.ndarray, spec: TensorSpec, *, color: str = "rgb",
             scale: float = 1.0 / 255.0, mean=None, std=None) -> np.ndarray:
    """Colour-convert, normalise, pack to NCHW, cast to spec.dtype (quantise if INT8)."""
    img = patch.astype(np.float32)
    if color == "gray":
        img = _to_gray(img)
    elif img.ndim == 2:
        img = img[..., None]
    img = img * scale
    if mean is not None:
        img = (img - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    chw = np.transpose(img, (2, 0, 1))[None]          # 1,C,H,W
    if spec.quant:
        q = np.round(chw / spec.quant["scale"]) + spec.quant["zero_point"]
        info = np.iinfo(np.dtype(spec.dtype))
        return np.clip(q, info.min, info.max).astype(spec.dtype)
    return chw.astype(np.dtype(spec.dtype))


def class_agnostic_nms(boxes_xyxy: np.ndarray, scores: np.ndarray,
                       iou_thresh: float) -> np.ndarray:
    """Greedy NMS over a single pool (class labels ignored). Returns kept indices."""
    if len(scores) == 0:
        return np.empty((0,), np.int64)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return np.array(keep, np.int64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_nn_preprocess.py -q`
Expected: PASS (all preprocess tests)

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/preprocess.py tests/test_nn_preprocess.py
git commit -m "feat(nn/preprocess): to_input (color/scale/quant) + class-agnostic NMS"
```

---

### Task 4: Manifests + packaging

**Files:**
- Create: `edgecv/models/manifests/siamfc_generic.yaml`
- Create: `edgecv/models/manifests/yolo_generic.yaml`
- Modify: `pyproject.toml`
- Test: `tests/test_manifests_nn.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_manifests_nn.py`:

```python
from pathlib import Path

from edgecv.models.manifest import load_manifest

MANIFESTS = Path("edgecv/models/manifests")


def test_siamfc_manifest_loads():
    m = load_manifest(MANIFESTS / "siamfc_generic.yaml")
    assert m.name == "siamfc_generic"
    assert {i["name"] for i in m.inputs} == {"exemplar", "search"}
    assert m.outputs[0]["name"] == "score_map"
    assert m.preprocessing["color"] == "gray"


def test_yolo_manifest_loads():
    m = load_manifest(MANIFESTS / "yolo_generic.yaml")
    assert m.name == "yolo_generic"
    assert m.task == "detection"
    assert m.preprocessing["class_agnostic"] is True
    assert m.preprocessing["output_format"] == "yolov5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_manifests_nn.py -q`
Expected: FAIL — `FileNotFoundError` (manifests do not exist)

- [ ] **Step 3: Write minimal implementation**

Create `edgecv/models/manifests/siamfc_generic.yaml`:

```yaml
name: siamfc_generic
task: sot_template_matching
preprocessing:
  color: gray
  exemplar: 127
  search: 255
  context: 0.5
  total_stride: 8
  response_up: 16
  scale_num: 3
  scale_step: 1.0375
  scale_penalty: 0.9745
  scale_lr: 0.59
  window_influence: 0.176
io:
  inputs:
    - { name: exemplar, shape: [1, 1, 127, 127], dtype: float32 }
    - { name: search,   shape: [1, 1, 255, 255], dtype: float32 }
  outputs:
    - { name: score_map, shape: [1, 1, 17, 17], dtype: float32 }
artifacts:
  onnx: { path: siamfc_generic.onnx }
  rknn: { path: siamfc_generic.rk3588.rknn, quant: int8 }
```

Create `edgecv/models/manifests/yolo_generic.yaml`:

```yaml
name: yolo_generic
task: detection
preprocessing:
  color: rgb
  input: 640
  scale: 0.00392156862745098
  output_format: yolov5
  class_agnostic: true
  conf_thresh: 0.25
  iou_thresh: 0.45
io:
  inputs:
    - { name: images, shape: [1, 3, 640, 640], dtype: float32 }
  outputs:
    - { name: output0, shape: [1, -1, 85], dtype: float32 }
artifacts:
  onnx: { path: yolo_generic.onnx }
  rknn: { path: yolo_generic.rk3588.rknn, quant: int8 }
```

In `pyproject.toml`, add the manifests to the wheel force-include (next to the existing profiles entry):

```toml
[tool.hatch.build.targets.wheel.force-include]
"edgecv/models/profiles" = "edgecv/models/profiles"
"edgecv/models/manifests" = "edgecv/models/manifests"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_manifests_nn.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add edgecv/models/manifests pyproject.toml tests/test_manifests_nn.py
git commit -m "feat(models): siamfc + class-agnostic yolo manifests, packaged as data"
```

---

### Task 5: `NNTracker` base — `Template`, backend resolution, lifecycle + DI seam

**Files:**
- Create: `edgecv/trackers/nn/base.py`
- Create: `tests/_nn_stubs.py` (shared deterministic stub `Model`)
- Test: `tests/test_nn_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/_nn_stubs.py` (a deterministic `Model`, reused by Tasks 6–9):

```python
"""Deterministic stub Models for NN-tracker tests (no weights, no backend)."""
from __future__ import annotations

import numpy as np

from edgecv.backends.base import IOSpec, Model, TensorSpec


class ScriptedModel(Model):
    """Returns pre-set output arrays per infer() call, cycling through `outputs`.

    `outputs` is a list of dicts {output_name: ndarray}. io_spec is supplied so the
    tracker can read names/shapes. infer() ignores its inputs (geometry is driven
    entirely by the scripted outputs)."""

    def __init__(self, io_spec: IOSpec, outputs: list[dict[str, np.ndarray]]):
        self._io_spec = io_spec
        self._outputs = outputs
        self.calls = 0

    @property
    def io_spec(self) -> IOSpec:
        return self._io_spec

    def infer(self, inputs):
        out = self._outputs[self.calls % len(self._outputs)]
        self.calls += 1
        return out

    def close(self) -> None:
        self.closed = True


def siam_io(score_size: int = 17) -> IOSpec:
    return IOSpec(
        inputs=(TensorSpec("exemplar", (1, 1, 127, 127), "float32"),
                TensorSpec("search", (1, 1, 255, 255), "float32")),
        outputs=(TensorSpec("score_map", (1, 1, score_size, score_size), "float32"),))


def score_map_peaked(score_size: int, cy: int, cx: int, peak: float = 1.0) -> np.ndarray:
    m = np.zeros((1, 1, score_size, score_size), np.float32)
    m[0, 0, cy, cx] = peak
    return m
```

Create `tests/test_nn_base.py`:

```python
import numpy as np
import pytest

from edgecv.backends.base import IOSpec, TensorSpec
from edgecv.trackers.nn.base import NNTracker, Template
from tests._nn_stubs import ScriptedModel


def _stub():
    io = IOSpec(outputs=(TensorSpec("y", (1, 1), "float32"),))
    return ScriptedModel(io, [{"y": np.zeros((1, 1), np.float32)}])


def test_template_dataclass_holds_arrays():
    from edgecv.core.bbox import BoundingBox
    t = Template(arrays={"exemplar": np.zeros((1,))}, bbox=BoundingBox(0, 0, 0.1, 0.1), meta={})
    assert "exemplar" in t.arrays


def test_model_injection_bypasses_backend():
    m = _stub()
    trk = NNTracker(model=m)        # no manifest, no backend
    assert trk._model is m


def test_close_is_idempotent_and_closes_model():
    m = _stub()
    trk = NNTracker(model=m)
    trk.close()
    trk.close()
    assert getattr(m, "closed", False) is True


def test_mock_backend_resolves_via_manifest():
    trk = NNTracker("edgecv/models/manifests/yolo_generic.yaml", backend="mock")
    assert trk._model is not None
    trk.close()


def test_auto_with_no_real_backend_raises_never_mock(monkeypatch):
    import edgecv.trackers.nn.base as base
    monkeypatch.setattr(base, "available_backends", lambda: ["mock"])
    with pytest.raises(RuntimeError, match="no inference backend"):
        NNTracker("edgecv/models/manifests/yolo_generic.yaml", backend="auto")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_nn_base.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.trackers.nn.base'`

- [ ] **Step 3: Write minimal implementation**

Create `edgecv/trackers/nn/base.py`:

```python
"""NN tracker base: backend/model resolution, lifecycle, and the Template
appearance type (ARCHITECTURE.md §6.2). Trackers depend on the HAL only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgecv.backends.base import Model
from edgecv.backends.registry import available_backends, get_backend
from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.core.tracker import Tracker
from edgecv.models.manifest import ModelManifest, load_manifest


@dataclass
class Template:
    """Transferable target appearance — the NN analogue of CF FilterState."""
    arrays: dict[str, np.ndarray]
    bbox: BoundingBox
    meta: dict


def _select_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    avail = available_backends()
    for pref in ("rknn", "onnx"):
        if pref in avail:
            return pref
    raise RuntimeError(
        "no inference backend available; install edgecv[onnx], run on-device with "
        "[rknn], or pass backend='mock' for canned outputs"
    )


def resolve_model(manifest, backend: str, model: Model | None) -> Model:
    """DI seam: explicit model wins; else resolve a backend and load the manifest."""
    if model is not None:
        return model
    if manifest is None:
        raise ValueError("NNTracker needs a manifest (or an injected model=)")
    mf = manifest if isinstance(manifest, ModelManifest) else load_manifest(manifest)
    return get_backend(_select_backend(backend)).load(mf)


class NNTracker(Tracker):
    def __init__(self, manifest: ModelManifest | str | Path | None = None, *,
                 backend: str = "auto", model: Model | None = None) -> None:
        self._model: Model | None = resolve_model(manifest, backend, model)
        self._status: TrackStatus = TrackStatus.INITIALIZING
        self._seq: int = 0
        self._closed: bool = False

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:  # pragma: no cover
        raise NotImplementedError

    def update(self, frame: np.ndarray) -> TrackResult:  # pragma: no cover
        raise NotImplementedError

    def name(self) -> str:  # pragma: no cover
        raise NotImplementedError

    @property
    def status(self) -> TrackStatus:
        return self._status

    def close(self) -> None:
        if not self._closed and self._model is not None:
            self._model.close()
            self._closed = True
```

Create an empty `tests/__init__.py` only if `tests/` is not already a package — it **is** (a tracked `tests/__init__.py` exists), so `from tests._nn_stubs import ...` resolves.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_nn_base.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/base.py tests/_nn_stubs.py tests/test_nn_base.py
git commit -m "feat(nn/base): NNTracker, Template, backend resolution + DI seam"
```

---

### Task 6: `SiamFC` — `__init__` + `init` (template build)

**Files:**
- Create: `edgecv/trackers/nn/siamfc.py`
- Test: `tests/test_siamfc.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_siamfc.py`:

```python
import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.trackers.nn.siamfc import SiamFC
from tests._nn_stubs import ScriptedModel, score_map_peaked, siam_io

SS = 17  # score_size


def _frame(h=240, w=320):
    return np.zeros((h, w, 3), np.uint8)


def _box():
    return BoundingBox(x=(160 - 20) / 320, y=(120 - 20) / 240, w=40 / 320, h=40 / 240)


def _siam(maps, **kw):
    return SiamFC(model=ScriptedModel(siam_io(SS), maps), **kw)


def test_name_and_instantiation():
    t = _siam([{"score_map": score_map_peaked(SS, 8, 8)}])
    assert t.name() == "SiamFC"


def test_init_builds_127_exemplar_template():
    # 3 maps because update() runs scale_num=3 infers; init runs 0 infers.
    t = _siam([{"score_map": score_map_peaked(SS, 8, 8)}])
    t.init(_frame(), _box())
    z = t.get_template().arrays["exemplar"]
    assert z.shape == (1, 1, 127, 127)
    assert t.status == TrackStatus.LOCKED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_siamfc.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.trackers.nn.siamfc'`

- [ ] **Step 3: Write minimal implementation**

Create `edgecv/trackers/nn/siamfc.py`:

```python
"""SiamFC tracker (ARCHITECTURE.md §6.2). Single two-input graph
(exemplar, search) -> score_map; multi-scale search adapts position and size.
Reference defaults: HonglinChu/SiamTrackers."""

from __future__ import annotations

import math
import time

import numpy as np

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.trackers.cf.ops import psr
from edgecv.trackers.nn.base import NNTracker, Template
from edgecv.trackers.nn.preprocess import crop_with_context, resize_bilinear, to_input


def _hann2d(n: int) -> np.ndarray:
    h = np.hanning(n).astype(np.float32)
    win = np.outer(h, h)
    s = win.sum()
    return win / s if s > 0 else win


class SiamFC(NNTracker):
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 exemplar_size=127, search_size=255, context=0.5,
                 total_stride=8, response_up=16, scale_num=3, scale_step=1.0375,
                 scale_penalty=0.9745, scale_lr=0.59, window_influence=0.176,
                 color="gray", score_lock=8.0, score_lost=4.0) -> None:
        super().__init__(manifest, backend=backend, model=model)
        self._exemplar_size = exemplar_size
        self._search_size = search_size
        self._context = context
        self._total_stride = total_stride
        self._response_up = response_up
        self._scale_num = scale_num
        self._scale_step = scale_step
        self._scale_penalty = scale_penalty
        self._scale_lr = scale_lr
        self._window_influence = window_influence
        self._color = color
        self._score_lock = score_lock
        self._score_lost = score_lost
        out = self._model.io_spec.outputs[0]
        self._out_name = out.name
        self._score_size = out.shape[-1]
        self._up_size = self._score_size * self._response_up
        self._hann = _hann2d(self._up_size)
        self._template: Template | None = None
        self._box: BoundingBox | None = None

    def name(self) -> str:
        return "SiamFC"

    def get_template(self) -> Template:
        assert self._template is not None, "init() must run first"
        return self._template

    def set_template(self, template: Template,
                     search_box: BoundingBox | None = None) -> None:
        self._template = template
        self._box = search_box if search_box is not None else template.bbox

    def _exemplar_side(self, pix: PixelBox) -> float:
        p = self._context * (pix.w + pix.h)
        return math.sqrt((pix.w + p) * (pix.h + p))

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        h_img, w_img = frame.shape[0], frame.shape[1]
        pix = bbox.to_pixels(w_img, h_img)
        s_z = self._exemplar_side(pix)
        patch, _ = crop_with_context(frame, pix.center, (s_z, s_z),
                                     (self._exemplar_size, self._exemplar_size))
        spec_z = self._model.io_spec.inputs[0]
        z = to_input(patch, spec_z, color=self._color)
        self._template = Template(arrays={"exemplar": z}, bbox=bbox, meta={"s_z": s_z})
        self._box = bbox
        self._status = TrackStatus.LOCKED
        self._seq = 0

    def _status_from(self, value: float) -> TrackStatus:
        if value >= self._score_lock:
            return TrackStatus.LOCKED
        if value >= self._score_lost:
            return TrackStatus.COASTING
        return TrackStatus.LOST
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_siamfc.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/siamfc.py tests/test_siamfc.py
git commit -m "feat(nn/siamfc): SiamFC init + exemplar template build"
```

---

### Task 7: `SiamFC.update` — multi-scale search, status

**Files:**
- Modify: `edgecv/trackers/nn/siamfc.py`
- Test: `tests/test_siamfc.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_siamfc.py`:

```python
def test_centred_peak_keeps_centre():
    # all 3 scales return a centred peak -> no displacement, box centre unchanged
    maps = [{"score_map": score_map_peaked(SS, 8, 8)} for _ in range(3)]
    t = _siam(maps, window_influence=0.0)
    t.init(_frame(), _box())
    res = t.update(_frame())
    cx, cy = res.bbox.to_pixels(320, 240).center
    assert cx == pytest.approx(160.0, abs=2.0)
    assert cy == pytest.approx(120.0, abs=2.0)
    assert res.seq == 1


def test_offcentre_peak_moves_box_in_that_direction():
    # peak one cell to the +x of centre on every scale -> centre moves +x
    maps = [{"score_map": score_map_peaked(SS, 8, 9)} for _ in range(3)]
    t = _siam(maps, window_influence=0.0)
    t.init(_frame(), _box())
    res = t.update(_frame())
    cx, _ = res.bbox.to_pixels(320, 240).center
    assert cx > 161.0   # moved right


def test_winning_larger_scale_grows_box():
    # scales are searched ascending: [<1, 1, >1]; make the >1 scale (3rd call) win
    maps = [
        {"score_map": score_map_peaked(SS, 8, 8, peak=0.2)},
        {"score_map": score_map_peaked(SS, 8, 8, peak=0.2)},
        {"score_map": score_map_peaked(SS, 8, 8, peak=1.0)},
    ]
    t = _siam(maps, window_influence=0.0)
    t.init(_frame(), _box())
    w0 = t.get_template().bbox.w
    res = t.update(_frame())
    assert res.bbox.w > w0   # box grew toward the winning larger scale


def test_low_response_reports_lost():
    maps = [{"score_map": np.zeros((1, 1, SS, SS), np.float32)} for _ in range(3)]
    t = _siam(maps)
    t.init(_frame(), _box())
    res = t.update(_frame())
    assert res.status == TrackStatus.LOST
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_siamfc.py::test_centred_peak_keeps_centre -v`
Expected: FAIL — `NotImplementedError` (inherited `update` stub)

- [ ] **Step 3: Write minimal implementation**

Add the `update` method to the `SiamFC` class:

```python
    def update(self, frame: np.ndarray) -> TrackResult:
        assert self._template is not None and self._box is not None, "init() first"
        h_img, w_img = frame.shape[0], frame.shape[1]
        pix = self._box.to_pixels(w_img, h_img)
        cx, cy = pix.center
        s_z = self._exemplar_side(pix)
        s_x = s_z * self._search_size / self._exemplar_size
        spec_x = self._model.io_spec.inputs[1]
        z = self._template.arrays["exemplar"]

        centre = self._scale_num // 2
        scales = self._scale_step ** (np.arange(self._scale_num) - centre)
        best = None  # (idx, factor, up_norm, peak, raw_map)
        for i, f in enumerate(scales):
            side = s_x * f
            patch, _ = crop_with_context(frame, (cx, cy), (side, side),
                                         (self._search_size, self._search_size))
            x = to_input(patch, spec_x, color=self._color)
            raw = self._model.infer({"exemplar": z, "search": x})[self._out_name]
            smap = np.asarray(raw, np.float32).reshape(self._score_size, self._score_size)
            up = resize_bilinear(smap[..., None], (self._up_size, self._up_size))[..., 0]
            penalty = 1.0 if i == centre else self._scale_penalty
            peak = float(up.max()) * penalty
            if best is None or peak > best[3]:
                best = (i, float(f), up, peak, smap)

        idx, factor, up, _peak, smap = best
        total = up.sum()
        resp = up / total if total > 0 else up
        resp = (1.0 - self._window_influence) * resp + self._window_influence * self._hann
        py, px = np.unravel_index(int(resp.argmax()), resp.shape)
        disp_x = (px - (self._up_size - 1) / 2.0) * self._total_stride / self._response_up
        disp_y = (py - (self._up_size - 1) / 2.0) * self._total_stride / self._response_up
        scale_x = (s_x * factor) / self._search_size
        new_cx = cx + disp_x * scale_x
        new_cy = cy + disp_y * scale_x

        scale_factor = (1.0 - self._scale_lr) + self._scale_lr * factor
        new_w = pix.w * scale_factor
        new_h = pix.h * scale_factor
        new_pix = PixelBox(x=new_cx - new_w / 2.0, y=new_cy - new_h / 2.0, w=new_w, h=new_h)
        self._box = BoundingBox.from_pixels(new_pix, w_img, h_img)

        conf = psr(smap)
        self._status = self._status_from(conf)
        self._seq += 1
        return TrackResult(bbox=self._box, confidence=float(conf), status=self._status,
                           timestamp=time.monotonic(), seq=self._seq)
```

> Confidence/status gate on **PSR of the raw score map** (scale-free, comparable frame-to-frame);
> `psr` is reused as a generic response-map statistic. The `confidence` field carries this PSR.
> Defaults `score_lock=8.0`/`score_lost=4.0` are PSR thresholds — re-pin against the noise test if
> a real model's score map has a very different sidelobe profile (spec §13).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_siamfc.py -q`
Expected: PASS (6 tests)

> If `test_winning_larger_scale_grows_box` is borderline, confirm the scale ordering: `scales` is
> ascending, the centre index is `scale_num//2`, and only the centre scale is penalty-free. Do not
> reorder to mask a sign error — debug with superpowers:systematic-debugging.

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/siamfc.py tests/test_siamfc.py
git commit -m "feat(nn/siamfc): multi-scale update (position+scale), PSR status gate"
```

---

### Task 8: `YoloDetector.detect` — class-agnostic decode + NMS

**Files:**
- Create: `edgecv/trackers/nn/yolo.py`
- Test: `tests/test_yolo.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_yolo.py`:

```python
import numpy as np
import pytest

from edgecv.backends.base import IOSpec, TensorSpec
from edgecv.trackers.nn.yolo import YoloDetector
from tests._nn_stubs import ScriptedModel

IN = 64  # small model input for tests


def _yolo_io(nc=80):
    return IOSpec(inputs=(TensorSpec("images", (1, 3, IN, IN), "float32"),),
                  outputs=(TensorSpec("output0", (1, -1, 5 + nc), "float32"),))


def _raw(dets, nc=80):
    """dets: list of (cx, cy, w, h, obj, cls_idx) in INPUT (letterbox) px."""
    out = np.zeros((1, len(dets), 5 + nc), np.float32)
    for i, (cx, cy, w, h, obj, ci) in enumerate(dets):
        out[0, i, :4] = [cx, cy, w, h]
        out[0, i, 4] = obj
        out[0, i, 5 + ci] = 1.0
    return {"output0": out}


def _detector(raw, **kw):
    return YoloDetector(model=ScriptedModel(_yolo_io(), [raw]), input_size=IN, **kw)


def test_detect_returns_normalised_xywh_and_score():
    # one detection centred in a square input -> centred normalised box
    det = _detector(_raw([(32, 32, 16, 16, 0.9, 3)]))
    out = det.detect(np.zeros((IN, IN, 3), np.uint8))
    assert out.boxes.shape == (1, 4)
    assert out.scores[0] == pytest.approx(0.9, abs=1e-3)  # obj * max(cls)=0.9*1.0
    bx, by, bw, bh = out.boxes[0]
    assert (bx + bw / 2) == pytest.approx(0.5, abs=0.05)


def test_detect_is_class_agnostic_and_pure():
    det = _detector(_raw([(32, 32, 16, 16, 0.8, 17), (10, 10, 8, 8, 0.7, 2)]))
    img = np.zeros((IN, IN, 3), np.uint8)
    out1 = det.detect(img)
    out2 = det.detect(img)            # purity: same result, no internal mutation
    assert len(out1.scores) == 2
    np.testing.assert_array_equal(out1.boxes, out2.boxes)


def test_detect_thresholds_low_confidence():
    det = _detector(_raw([(32, 32, 16, 16, 0.1, 0)]), conf_thresh=0.25)
    out = det.detect(np.zeros((IN, IN, 3), np.uint8))
    assert len(out.scores) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_yolo.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.trackers.nn.yolo'`

- [ ] **Step 3: Write minimal implementation**

Create `edgecv/trackers/nn/yolo.py`:

```python
"""Class-agnostic YOLO detector + standalone single-object tracker
(ARCHITECTURE.md §6.2; MAFiD local-detection mode, sensors-23-07082 §3.3).

YoloDetector.detect -> DetectorOutput is the reusable primitive a future hybrid
worker calls. YoloTracker wraps it for standalone use."""

from __future__ import annotations

import time

import numpy as np

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.fusion.policy import DetectorOutput
from edgecv.trackers.nn.base import NNTracker, resolve_model
from edgecv.trackers.nn.preprocess import (
    class_agnostic_nms,
    crop_with_context,
    letterbox,
    to_input,
)


class YoloDetector:
    """Boxes in the returned DetectorOutput are (N,4) normalised xywh top-left."""

    def __init__(self, manifest=None, *, backend="auto", model=None,
                 input_size=640, color="rgb", scale=1.0 / 255.0,
                 output_format="yolov5", conf_thresh=0.25, iou_thresh=0.45,
                 class_agnostic=True) -> None:
        self._model = resolve_model(manifest, backend, model)
        self._input_size = input_size
        self._color = color
        self._scale = scale
        self._output_format = output_format
        self._conf = conf_thresh
        self._iou = iou_thresh
        self._class_agnostic = class_agnostic
        self._spec = self._model.io_spec.inputs[0]
        self._out_name = self._model.io_spec.outputs[0].name

    def detect(self, image: np.ndarray) -> DetectorOutput:
        h_img, w_img = image.shape[0], image.shape[1]
        n = self._input_size
        lb, xf = letterbox(image, (n, n))
        inp = to_input(lb, self._spec, color=self._color, scale=self._scale)
        raw = np.asarray(self._model.infer({self._spec.name: inp})[self._out_name], np.float32)
        preds = raw[0]  # (N, 5+nc)
        if preds.shape[0] == 0:
            return DetectorOutput(boxes=np.empty((0, 4), np.float32),
                                  scores=np.empty((0,), np.float32))
        if self._output_format == "yolov5":
            xywh, obj, cls = preds[:, :4], preds[:, 4], preds[:, 5:]
            score = obj * (cls.max(axis=1) if cls.shape[1] > 0 else 1.0)
        else:  # "decoded": model already emits xywh + score
            xywh, score = preds[:, :4], preds[:, 4]
        keep = score >= self._conf
        xywh, score = xywh[keep], score[keep]
        if len(score) == 0:
            return DetectorOutput(boxes=np.empty((0, 4), np.float32),
                                  scores=np.empty((0,), np.float32))
        # centre xywh (letterbox px) -> xyxy (letterbox px)
        cxs, cys, ws, hs = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        xyxy = np.stack([cxs - ws / 2, cys - hs / 2, cxs + ws / 2, cys + hs / 2], axis=1)
        kept = class_agnostic_nms(xyxy, score, self._iou)
        xyxy, score = xyxy[kept], score[kept]
        # invert letterbox -> original px -> normalised xywh top-left
        boxes = np.empty((len(kept), 4), np.float32)
        for i, b in enumerate(xyxy):
            ox1, oy1, ox2, oy2 = xf.to_orig_xyxy((b[0], b[1], b[2], b[3]))
            boxes[i] = [ox1 / w_img, oy1 / h_img, (ox2 - ox1) / w_img, (oy2 - oy1) / h_img]
        return DetectorOutput(boxes=boxes, scores=score.astype(np.float32))

    def close(self) -> None:
        self._model.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_yolo.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/yolo.py tests/test_yolo.py
git commit -m "feat(nn/yolo): class-agnostic YoloDetector.detect -> DetectorOutput"
```

---

### Task 9: `YoloTracker` — local-crop search + association

**Files:**
- Modify: `edgecv/trackers/nn/yolo.py`
- Test: `tests/test_yolo.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_yolo.py` (extend imports: `from edgecv.trackers.nn.yolo import YoloTracker`; `from edgecv.core.bbox import BoundingBox`; `from edgecv.core.result import TrackStatus`):

```python
FH, FW = 240, 320


def _tracker(maps, **kw):
    m = ScriptedModel(_yolo_io(), maps)
    return YoloTracker(model=m, input_size=IN, **kw)


def _box(cx, cy, w=40, h=40):
    return BoundingBox(x=(cx - w / 2) / FW, y=(cy - h / 2) / FH, w=w / FW, h=h / FH)


def test_yolo_tracker_name():
    assert _tracker([_raw([])]).name() == "YOLO"


def test_association_prefers_near_over_far_highscore():
    # crop is centred on prev box (160,120). Two detections inside the crop:
    # one near crop-centre with decent score, one far corner with higher score.
    near = (IN / 2, IN / 2, 16, 16, 0.6, 0)
    far = (4, 4, 8, 8, 0.95, 1)
    t = _tracker([_raw([near, far])], assoc_sigma=0.5)
    t.init(np.zeros((FH, FW, 3), np.uint8), _box(160, 120))
    res = t.update(np.zeros((FH, FW, 3), np.uint8))
    cx, cy = res.bbox.to_pixels(FW, FH).center
    assert abs(cx - 160) < 30 and abs(cy - 120) < 30   # picked the near one
    assert res.status == TrackStatus.LOCKED


def test_box_adapts_to_detection_size():
    t = _tracker([_raw([(IN / 2, IN / 2, 32, 16, 0.9, 0)])])
    t.init(np.zeros((FH, FW, 3), np.uint8), _box(160, 120, w=40, h=40))
    res = t.update(np.zeros((FH, FW, 3), np.uint8))
    # detection is 2:1 (w:h); output aspect should differ from the square init box
    assert res.bbox.w > res.bbox.h


def test_misses_coast_then_lost():
    t = _tracker([_raw([]), _raw([]), _raw([])], max_misses=2)
    t.init(np.zeros((FH, FW, 3), np.uint8), _box(160, 120))
    r1 = t.update(np.zeros((FH, FW, 3), np.uint8))
    assert r1.status == TrackStatus.COASTING
    t.update(np.zeros((FH, FW, 3), np.uint8))
    r3 = t.update(np.zeros((FH, FW, 3), np.uint8))
    assert r3.status == TrackStatus.LOST
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_yolo.py -k tracker -q`
Expected: FAIL — `ImportError: cannot import name 'YoloTracker'`

- [ ] **Step 3: Write minimal implementation**

Add to `edgecv/trackers/nn/yolo.py`:

```python
class YoloTracker(NNTracker):
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 search_factor=3.0, assoc_sigma=0.5, conf_thresh=0.25,
                 iou_thresh=0.45, max_misses=5, input_size=640,
                 color="rgb", scale=1.0 / 255.0, output_format="yolov5") -> None:
        super().__init__(manifest, backend=backend, model=model)
        self._detector = YoloDetector(
            model=self._model, input_size=input_size, color=color, scale=scale,
            output_format=output_format, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        self._search_factor = search_factor
        self._assoc_sigma = assoc_sigma
        self._max_misses = max_misses
        self._input_size = input_size
        self._box: BoundingBox | None = None
        self._misses = 0

    def name(self) -> str:
        return "YOLO"

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        self._box = bbox
        self._status = TrackStatus.LOCKED
        self._misses = 0
        self._seq = 0

    def update(self, frame: np.ndarray) -> TrackResult:
        assert self._box is not None, "init() first"
        h_img, w_img = frame.shape[0], frame.shape[1]
        pix = self._box.to_pixels(w_img, h_img)
        cx, cy = pix.center
        side = self._search_factor * max(pix.w, pix.h)
        n = self._input_size
        crop, xf = crop_with_context(frame, (cx, cy), (side, side), (n, n))
        det = self._detector.detect(crop)

        best, best_w = None, -1.0
        sigma = self._assoc_sigma * max(pix.w, pix.h) + 1e-6
        for box_n, sc in zip(det.boxes, det.scores, strict=False):
            # crop-normalised xywh -> crop-out px -> frame px (via xf.to_frame)
            ox1, oy1 = box_n[0] * n, box_n[1] * n
            ox2, oy2 = (box_n[0] + box_n[2]) * n, (box_n[1] + box_n[3]) * n
            fx1, fy1 = xf.to_frame((ox1, oy1))
            fx2, fy2 = xf.to_frame((ox2, oy2))
            dcx, dcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
            dist2 = (dcx - cx) ** 2 + (dcy - cy) ** 2
            w = float(sc) * float(np.exp(-0.5 * dist2 / (sigma * sigma)))
            if w > best_w:
                best_w, best = w, (fx1, fy1, fx2 - fx1, fy2 - fy1, float(sc))

        if best is None:
            self._misses += 1
            self._status = TrackStatus.LOST if self._misses > self._max_misses \
                else TrackStatus.COASTING
            self._seq += 1
            return TrackResult(bbox=self._box, confidence=None, status=self._status,
                               timestamp=time.monotonic(), seq=self._seq)

        fx, fy, fw, fh, score = best
        self._box = BoundingBox.from_pixels(PixelBox(x=fx, y=fy, w=fw, h=fh), w_img, h_img)
        self._misses = 0
        self._status = TrackStatus.LOCKED
        self._seq += 1
        return TrackResult(bbox=self._box, confidence=score, status=self._status,
                           timestamp=time.monotonic(), seq=self._seq)
```

> Confidence is the **detector score** — NOT on the same scale as SiamFC peaks or CF PSR; any future
> fusion must calibrate before comparing (ARCHITECTURE §8, spec §6.2). The `detect` call is reused
> verbatim, so the standalone tracker and the future hybrid worker run identical detection.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_yolo.py -q`
Expected: PASS (all YOLO tests)

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/yolo.py tests/test_yolo.py
git commit -m "feat(nn/yolo): YoloTracker local-crop search + proximity-score association"
```

---

### Task 10: RKNN `Model` wiring (device-only) + `nn` exports

**Files:**
- Modify: `edgecv/backends/rknn/__init__.py`
- Modify: `edgecv/trackers/nn/__init__.py`
- Test: `tests/test_yolo.py` (one skipped device test) — or add to `tests/test_nn_base.py`

> RKNN cannot run in CI (no NPU). Implement the wiring and a **skipped** test that documents the
> contract; on-device validation is manual. This replaces the `NotImplementedError` the adapter
> currently raises (`backends/rknn/__init__.py` defers `load` "alongside the first NN tracker").

- [ ] **Step 1: Write the failing/skipped test**

Add to `tests/test_nn_base.py`:

```python
import pytest

from edgecv.backends.registry import get_backend


def test_rknn_unavailable_off_device_is_clean():
    be = get_backend("rknn")
    if be.is_available():            # on a real device
        pytest.skip("rknn runtime present; load tested on-device manually")
    assert be.is_available() is False
```

Add to `tests/test_siamfc.py` and `tests/test_yolo.py` the import-surface check:

```python
def test_nn_package_exports():
    import edgecv.trackers.nn as nn
    assert hasattr(nn, "SiamFC")
    assert hasattr(nn, "YoloTracker")
    assert hasattr(nn, "YoloDetector")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_siamfc.py::test_nn_package_exports -q`
Expected: FAIL — `AttributeError: module 'edgecv.trackers.nn' has no attribute 'SiamFC'`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `RknnBackend.load` in `edgecv/backends/rknn/__init__.py` (keep the `_import_rknnlite` / `_INSTALL_HINT` and `is_available`):

```python
from edgecv.backends.base import IOSpec, Model, TensorSpec


def _specs(entries):
    return tuple(TensorSpec(name=e["name"], shape=tuple(e["shape"]),
                            dtype=e.get("dtype", "float32"),
                            layout=e.get("layout", "NCHW"), quant=e.get("quant"))
                 for e in entries)


class RknnModel(Model):
    """Wraps RKNNLite. Built INSIDE the using process only (ARCHITECTURE §14.7)."""

    def __init__(self, rknn, io_spec: IOSpec, output_order: list[str]):
        self._rknn = rknn
        self._io_spec = io_spec
        self._output_order = output_order

    @property
    def io_spec(self) -> IOSpec:
        return self._io_spec

    def infer(self, inputs):
        ordered = [inputs[s.name] for s in self._io_spec.inputs]
        results = self._rknn.inference(inputs=ordered)
        return dict(zip(self._output_order, results, strict=False))

    def close(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None
```

and the new `load`:

```python
    def load(self, manifest: ModelManifest) -> Model:
        try:
            rknn_lite = _import_rknnlite()
        except Exception as e:
            raise RuntimeError(_INSTALL_HINT) from e
        artifact = manifest.artifacts.get("rknn")
        if not artifact or "path" not in artifact:
            raise ValueError(f"manifest {manifest.name!r} has no rknn artifact path")
        rknn = rknn_lite()
        if rknn.load_rknn(artifact["path"]) != 0:
            raise RuntimeError(f"failed to load rknn model {artifact['path']!r}")
        core_mask = (artifact.get("npu_core") or 0)
        if rknn.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError("rknn init_runtime failed")
        io_spec = IOSpec(inputs=_specs(manifest.inputs), outputs=_specs(manifest.outputs))
        return RknnModel(rknn, io_spec, [o["name"] for o in manifest.outputs])
```

(Add the `IOSpec, Model, TensorSpec` import and the `_specs` helper to the module.)

Then make `edgecv/trackers/nn/__init__.py` export the public API:

```python
"""Dense-network (NN) trackers (ARCHITECTURE.md §6.2)."""

from edgecv.trackers.nn.base import NNTracker, Template
from edgecv.trackers.nn.siamfc import SiamFC
from edgecv.trackers.nn.yolo import YoloDetector, YoloTracker

__all__ = ["NNTracker", "SiamFC", "Template", "YoloDetector", "YoloTracker"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_nn_base.py tests/test_siamfc.py tests/test_yolo.py -q`
Expected: PASS (rknn test skips off-device; export checks pass).

- [ ] **Step 5: Commit**

```bash
git add edgecv/backends/rknn/__init__.py edgecv/trackers/nn/__init__.py tests/test_nn_base.py tests/test_siamfc.py tests/test_yolo.py
git commit -m "feat(rknn,nn): RknnModel wiring (device-only) + nn package exports"
```

---

### Task 11: onnx-backend integration test (real inference path)

**Files:**
- Create: `tests/_onnx_synth.py` (synthetic ONNX model builders — validated against onnx 1.21 / onnxruntime 1.26)
- Create: `tests/test_nn_onnx.py`

> Stubs (Tasks 6–9) drive the geometry/decode logic; this task validates the **real** path:
> manifest → `OnnxBackend.load` → io_spec parsing → `to_input` dtype/layout matching ORT's
> expectations → `infer` → decode → `TrackResult`. It builds throwaway ONNX graphs in a tmp dir (no
> trained weights — those are deferred), so it asserts **plumbing/shape correctness**, not tracking
> quality. Skips cleanly if `onnx`/`onnxruntime` are absent.

- [ ] **Step 1: Write the failing test**

Create `tests/_onnx_synth.py` (builders verified to load + run through `OnnxBackend`):

```python
"""Synthetic ONNX models for the onnx-backend integration test. No trained weights:
the graphs are shape-correct and consume their inputs so ORT is happy, but their
outputs are (deliberately) not meaningful detections/score-maps."""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_siamfc_onnx(path: str, score_size: int = 17) -> None:
    ex = helper.make_tensor_value_info("exemplar", TensorProto.FLOAT, [1, 1, 127, 127])
    se = helper.make_tensor_value_info("search", TensorProto.FLOAT, [1, 1, 255, 255])
    sc = helper.make_tensor_value_info("score_map", TensorProto.FLOAT, [1, 1, score_size, score_size])
    # AveragePool(255, k=15, s=15) -> 17x17; add scalar mean(exemplar) so both inputs are used.
    pool = helper.make_node("AveragePool", ["search"], ["pooled"],
                            kernel_shape=[15, 15], strides=[15, 15])
    rm = helper.make_node("ReduceMean", ["exemplar"], ["ex_mean"], keepdims=0)  # scalar
    add = helper.make_node("Add", ["pooled", "ex_mean"], ["score_map"])
    graph = helper.make_graph([pool, rm, add], "siamfc_stub", [ex, se], [sc])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_yolo_onnx(path: str, n: int = 64, num: int = 3, nc: int = 1) -> None:
    img = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, n, n])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, num, 5 + nc])
    dets = np.zeros((1, num, 5 + nc), np.float32)
    dets[0, 0] = [n / 2, n / 2, 16, 16, 0.9] + [1.0] * nc   # centred, high score
    dets[0, 1] = [4, 4, 8, 8, 0.95] + [1.0] * nc            # corner, higher score
    const = helper.make_node("Constant", [], ["dets_out"],
                             value=numpy_helper.from_array(dets, "dets"))
    zinit = numpy_helper.from_array(np.array(0.0, np.float32), "zero_scalar")
    rm = helper.make_node("ReduceMean", ["images"], ["img_mean"], keepdims=0)  # scalar
    mul = helper.make_node("Mul", ["img_mean", "zero_scalar"], ["zeroed"])
    add = helper.make_node("Add", ["dets_out", "zeroed"], ["output0"])  # consumes images, value unchanged
    graph = helper.make_graph([const, rm, mul, add], "yolo_stub", [img], [out],
                              initializer=[zinit])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
```

Create `tests/test_nn_onnx.py`:

```python
import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from edgecv.core.bbox import BoundingBox          # noqa: E402
from edgecv.core.result import TrackResult, TrackStatus  # noqa: E402
from edgecv.models.manifest import load_manifest  # noqa: E402
from edgecv.trackers.nn.siamfc import SiamFC       # noqa: E402
from edgecv.trackers.nn.yolo import YoloDetector, YoloTracker  # noqa: E402
from tests._onnx_synth import build_siamfc_onnx, build_yolo_onnx  # noqa: E402

FH, FW = 240, 320


def _manifest_with_artifact(yaml_path, onnx_path):
    m = load_manifest(yaml_path)
    m.artifacts["onnx"] = {"path": str(onnx_path)}
    return m


def test_siamfc_runs_through_onnx_backend(tmp_path):
    model_path = tmp_path / "siamfc.onnx"
    build_siamfc_onnx(str(model_path))
    mf = _manifest_with_artifact("edgecv/models/manifests/siamfc_generic.yaml", model_path)
    with SiamFC(mf, backend="onnx") as t:
        box = BoundingBox(x=(160 - 20) / FW, y=(120 - 20) / FH, w=40 / FW, h=40 / FH)
        t.init(np.zeros((FH, FW, 3), np.uint8), box)
        res = t.update(np.random.default_rng(0).integers(0, 256, (FH, FW, 3), np.uint8))
    assert isinstance(res, TrackResult)
    assert 0.0 <= res.bbox.w <= 1.0 and res.seq == 1
    assert isinstance(res.status, TrackStatus)


def test_yolo_detector_runs_through_onnx_backend(tmp_path):
    model_path = tmp_path / "yolo.onnx"
    build_yolo_onnx(str(model_path), n=64, num=3, nc=1)
    mf = _manifest_with_artifact("edgecv/models/manifests/yolo_generic.yaml", model_path)
    det = YoloDetector(mf, backend="onnx", input_size=64, conf_thresh=0.25)
    out = det.detect(np.zeros((64, 64, 3), np.uint8))
    det.close()
    assert out.boxes.shape[1] == 4
    assert len(out.scores) == 2          # two synthetic dets, no overlap -> both survive NMS
    assert out.boxes.min() >= 0.0


def test_yolo_tracker_runs_through_onnx_backend(tmp_path):
    model_path = tmp_path / "yolo.onnx"
    build_yolo_onnx(str(model_path), n=64, num=3, nc=1)
    mf = _manifest_with_artifact("edgecv/models/manifests/yolo_generic.yaml", model_path)
    with YoloTracker(mf, backend="onnx", input_size=64, conf_thresh=0.25) as t:
        box = BoundingBox(x=(160 - 20) / FW, y=(120 - 20) / FH, w=40 / FW, h=40 / FH)
        t.init(np.zeros((FH, FW, 3), np.uint8), box)
        res = t.update(np.zeros((FH, FW, 3), np.uint8))
    assert isinstance(res, TrackResult)
    assert res.status in (TrackStatus.LOCKED, TrackStatus.COASTING)
    assert res.seq == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_nn_onnx.py -q`
Expected: FAIL — collection/import error until `SiamFC`/`YoloTracker` exist (they do after Tasks 6–9; if running this task in isolation before them, that's the expected failure). Once the trackers exist, the first real failure is whatever the trackers/`detect` get wrong on the real backend (e.g. dtype/layout/name mismatch) — exactly what this test is here to catch.

- [ ] **Step 3: Write minimal implementation**

No production code: the trackers from Tasks 6–9 are the implementation under test. If a real mismatch surfaces (e.g. `to_input` produces a dtype ORT rejects, or `YoloDetector` reads the wrong output name), fix it in the relevant `nn/` module via TDD — do **not** weaken the integration assertions. The synthetic builders are verified working against onnx 1.21 / onnxruntime 1.26; if a newer opset rejects `ReduceMean(keepdims=0)` axes-as-attribute, bump the opset id and adjust, noting it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_nn_onnx.py -q`
Expected: PASS (3 tests). Skips entirely if `onnx`/`onnxruntime` are not installed.

- [ ] **Step 5: Commit**

```bash
git add tests/_onnx_synth.py tests/test_nn_onnx.py
git commit -m "test(nn): onnx-backend integration via synthetic models"
```

---

### Task 12: Full-suite + lint/type gate

**Files:** none (verification only)

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (prior suite + new preprocess/base/siamfc/yolo/manifest tests), only the
pre-existing skip(s) + the off-device rknn skip.

- [ ] **Step 2: Run ruff**

Run: `.venv/bin/ruff check edgecv tests`
Expected: `All checks passed!` (common fixes: import ordering `I`, line length 100, `B` rules — e.g.
use `strict=` on `zip` as already done).

- [ ] **Step 3: Run mypy**

Run: `.venv/bin/mypy edgecv`
Expected: `Success: no issues found`. If the `Model | None` attribute access in `NNTracker`
subclasses trips mypy (`self._model` is `Optional`), assert-narrow at method entry (the code already
does `assert ... first`) or annotate the subclass `_model` access; do **not** add blanket
`# type: ignore`.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore(nn): satisfy ruff and mypy"
```

(Skip if Steps 1–3 were already clean.)

---

## Self-Review

**Spec coverage** (spec → task):
- §2 module layout → Tasks 1–10 create exactly the listed files. ✓
- §3 `NNTracker` base, `Template`, DI seam, backend selection (auto never picks mock), idempotent `close` → Task 5. ✓
- §4 preprocess ops `crop_with_context`/`CropXform`, `letterbox`/`LetterboxXform`, `to_input` (color/scale/quant), `class_agnostic_nms`; edge-replicate; single inversion path → Tasks 1–3. ✓
- §5 SiamFC defaults, crop sizing (`s_z`/`s_x`), `init` template, multi-scale `update` (scale penalty + damped size update), cosine-window blend, PSR status, `get/set_template`, no filter contract → Tasks 6–7. ✓
- §6 `YoloDetector.detect` (class-agnostic `obj*max(cls)`, threshold, NMS, letterbox inversion, normalised xywh, purity); `YoloTracker` (local-crop, proximity×score association, box adapts, miss/coast/lost) → Tasks 8–9. ✓
- §7 manifests + packaging → Task 4. ✓
- §8 RKNN `RknnModel` wiring, device-only/skip → Task 10. ✓
- §9 coordinate discipline (normalised boxes, PixelBox at boundary, no clamp on output) → Tasks 6–9. ✓
- §10 test plan items 1–20 → covered across Tasks 1–9 (preprocess geometry/inversion/NMS; base resolution/DI/lifecycle; SiamFC translation/multi-scale/window/status/coords; YOLO decode/inversion/association/local-crop/adapt/miss). ✓
- **onnx-backend integration** (real path: manifest → OnnxBackend → io_spec → to_input → infer → decode → TrackResult, via synthetic models) → Task 11. ✓ *(addition beyond the original spec — requested for dev-machine testing; consider noting it in spec §10/§12.)*
- §11/§12 packaging/perf → Task 4 packaging (manifests + force-include); `onnxruntime`+`onnx` added to the `test` extra; perf is non-goal (documented), no v1 work. ✓
- Final gate (full suite + ruff + mypy) → Task 12. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every test shows assertions. ✓

**Name/type consistency:** `resolve_model`/`_select_backend`/`available_backends` shared by `NNTracker` (Task 5) and `YoloDetector` (Task 8); `ScriptedModel`/`siam_io`/`score_map_peaked` defined once in `tests/_nn_stubs.py` (Task 5) and imported by Tasks 6–9; `CropXform.to_frame` (Task 1) used by `YoloTracker` (Task 9); `LetterboxXform.to_orig_xyxy` (Task 2) used by `YoloDetector` (Task 8); `to_input` signature stable across SiamFC/YOLO; `DetectorOutput` (existing `fusion/policy.py`) is the YOLO output type. ✓

**Decision notes carried for the executor:**
- `window_influence=0.0` is set in the SiamFC displacement/scale tests to isolate the geometry from the centring prior — this is intentional, not a workaround.
- The YOLO search window is **square** in v1 (plan convention, header) — if kept, update spec §6.2's "per axis" wording at the end.
- Confidence scales differ across SiamFC (PSR), YOLO (detector score), CF (PSR); standalone status is per-tracker; fusion calibration is a hybrid-spec concern, deliberately not solved here.
- `tests/` is already a package (`tests/__init__.py` tracked), so `from tests._nn_stubs import ...` works without extra config.
```

# NanoTrack (V3) NN Tracker + Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the NanoTrack V3 single-object tracker (`edgecv/trackers/nn/nanotrack.py`) plus its manifest and a vendored ONNX-conversion adapter, following the exact SiamFC pattern.

**Architecture:** A single-graph NN tracker: inputs `(exemplar 127, search 255)` → outputs `(cls, loc)` from a MobileNetV3-small-v3 + AdjustLayer + DepthwiseBAN anchor-free head. The tracker decodes `cls`/`loc` over a point grid (scale/aspect penalty + cosine-window blend + damped size update). Conversion vendors the reference architecture and exports ONNX via the existing generic harness; real weights are deferred (validated by torch-vs-onnxruntime parity on random weights + a deterministic stub `Model` in tests).

**Tech Stack:** Python, numpy (runtime); torch + onnx + onnxruntime (host-only conversion, already wired via `tools/convert_lib`). No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-06-06-nanotrack-tracker-design.md`

**Reference (for vendoring in Task 7):** `https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack`, configv3.

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `edgecv/trackers/nn/preprocess.py` | add `points_grid(stride, size)` pure op | Modify |
| `edgecv/trackers/nn/nanotrack.py` | `NanoTrack(NNTracker)`: crop → infer → decode | Create |
| `edgecv/trackers/nn/__init__.py` | export `NanoTrack` | Modify |
| `edgecv/models/manifests/nanotrack.yaml` | single-graph 2-in/2-out manifest | Create |
| `tests/_nn_stubs.py` | `nano_io`, `cls_peaked`, `loc_const` stub helpers | Modify |
| `tests/test_nn_preprocess.py` | `points_grid` tests | Modify |
| `tests/test_nanotrack.py` | decode/penalty/window/status/lifecycle tests | Create |
| `tests/test_manifests_nn.py` | nanotrack manifest loads | Modify |
| `tools/convert_lib/adapters/nanotrack.py` | vendored arch + `build` + `register` | Create |
| `tools/convert_lib/adapters/__init__.py` | import `nanotrack` to self-register | Modify |
| `tests/test_convert_nanotrack.py` | adapter build + ONNX export + parity | Create |
| `tools/CONVERSION.md` | nanotrack recipe + caveats | Modify |
| `ARCHITECTURE.md` | note NanoTrack in §6.2 + §13 | Modify |

Run tests from the repo root with `.venv` active (the repo's `tests/conftest.py` puts `tools/` on `sys.path` so `import convert_lib` works).

---

## Task 1: `points_grid` preprocessing op

**Files:**
- Modify: `edgecv/trackers/nn/preprocess.py` (add a module-level function near `class_agnostic_nms`)
- Test: `tests/test_nn_preprocess.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_nn_preprocess.py`:

```python
def test_points_grid_shape_and_centre():
    from edgecv.trackers.nn.preprocess import points_grid
    pts = points_grid(stride=16, size=15)
    assert pts.shape == (2, 15 * 15)
    centre = (15 // 2) * 15 + (15 // 2)        # row-major index of the middle cell
    assert pts[0, centre] == 0.0               # x at centre is 0
    assert pts[1, centre] == 0.0               # y at centre is 0


def test_points_grid_spacing_and_row_major():
    from edgecv.trackers.nn.preprocess import points_grid
    pts = points_grid(stride=16, size=15)
    # index = row*size + col ; x varies with col, y varies with row.
    assert pts[0, 0] == -(15 // 2) * 16        # top-left x = ori
    assert pts[1, 0] == -(15 // 2) * 16        # top-left y = ori
    assert pts[0, 1] - pts[0, 0] == 16         # next column is +stride in x
    assert pts[1, 0] == pts[1, 14]             # whole first row shares one y
    assert pts[1, 15] - pts[1, 0] == 16        # next row is +stride in y
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_nn_preprocess.py -k points_grid -v`
Expected: FAIL with `ImportError: cannot import name 'points_grid'`.

- [ ] **Step 3: Write minimal implementation**

Add to `edgecv/trackers/nn/preprocess.py` (after `class_agnostic_nms`):

```python
def points_grid(stride: int, size: int) -> np.ndarray:
    """Anchor-free point centres for a size×size head, in search-image pixels
    centred at 0. Returns (2, size*size): row 0 = x, row 1 = y, flattened
    row-major (index = row*size + col), matching a (C, S, S)->(C, S*S) reshape.
    Mirrors NanoTrack's generate_points: ori = -(size//2)*stride."""
    ori = -(size // 2) * stride
    coords = (ori + stride * np.arange(size)).astype(np.float32)
    gx, gy = np.meshgrid(coords, coords)          # gx[r,c]=coords[c], gy[r,c]=coords[r]
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_nn_preprocess.py -k points_grid -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/preprocess.py tests/test_nn_preprocess.py
git commit -m "feat(nn/preprocess): points_grid op for anchor-free decode

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: NanoTrack test stubs

Test infrastructure (no TDD): deterministic `cls`/`loc` arrays the tracker tests script.

**Files:**
- Modify: `tests/_nn_stubs.py`

- [ ] **Step 1: Add the stub helpers**

Append to `tests/_nn_stubs.py`:

```python
def nano_io(score_size: int = 15) -> IOSpec:
    return IOSpec(
        inputs=(TensorSpec("exemplar", (1, 3, 127, 127), "float32"),
                TensorSpec("search", (1, 3, 255, 255), "float32")),
        outputs=(TensorSpec("cls", (1, 2, score_size, score_size), "float32"),
                 TensorSpec("loc", (1, 4, score_size, score_size), "float32")))


def cls_peaked(score_size: int, cy: int, cx: int, fg: float = 8.0) -> np.ndarray:
    """cls logits (1,2,S,S): bg channel 0, fg channel 1 with a high logit at (cy,cx)
    so softmax fg prob ≈ 1 there and 0.5 elsewhere."""
    m = np.zeros((1, 2, score_size, score_size), np.float32)
    m[0, 1, cy, cx] = fg
    return m


def loc_const(score_size: int, left: float, top: float,
              right: float, bottom: float) -> np.ndarray:
    """loc (1,4,S,S) with constant l,t,r,b distances at every location."""
    m = np.zeros((1, 4, score_size, score_size), np.float32)
    m[0, 0] = left
    m[0, 1] = top
    m[0, 2] = right
    m[0, 3] = bottom
    return m
```

- [ ] **Step 2: Verify the module imports**

Run: `.venv/bin/python -c "from tests._nn_stubs import nano_io, cls_peaked, loc_const; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add tests/_nn_stubs.py
git commit -m "test(nn): scripted cls/loc stubs for NanoTrack

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: NanoTrack class — construction, init, template, package export

**Files:**
- Create: `edgecv/trackers/nn/nanotrack.py`
- Modify: `edgecv/trackers/nn/__init__.py`
- Test: `tests/test_nanotrack.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nanotrack.py`:

```python
import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.models.manifest import ModelManifest
from edgecv.trackers.nn.nanotrack import NanoTrack
from tests._nn_stubs import ScriptedModel, cls_peaked, loc_const, nano_io

S = 15  # score size


def _frame(h=240, w=320):
    return np.zeros((h, w, 3), np.uint8)


def _box():
    return BoundingBox(x=(160 - 20) / 320, y=(120 - 20) / 240, w=40 / 320, h=40 / 240)


def _nano(outputs, **kw):
    return NanoTrack(model=ScriptedModel(nano_io(S), outputs), **kw)


def _out(cy, cx, l=8.0, t=8.0, r=8.0, b=8.0, fg=8.0):
    return {"cls": cls_peaked(S, cy, cx, fg), "loc": loc_const(S, l, t, r, b)}


def test_name_and_instantiation():
    t = _nano([_out(S // 2, S // 2)])
    assert t.name() == "NanoTrack"


def test_init_builds_127_exemplar_template():
    t = _nano([_out(S // 2, S // 2)])
    t.init(_frame(), _box())
    z = t.get_template().arrays["exemplar"]
    assert z.shape == (1, 3, 127, 127)
    assert t.status == TrackStatus.LOCKED


def test_set_template_round_trips():
    t = _nano([_out(S // 2, S // 2)])
    t.init(_frame(), _box())
    tmpl = t.get_template()
    sb = BoundingBox(0.1, 0.1, 0.2, 0.2)
    t.set_template(tmpl, search_box=sb)
    assert t.get_template() is tmpl


def test_manifest_preprocessing_reaches_nanotrack():
    mf = ModelManifest(name="t", task="sot_template_matching",
                       preprocessing={"window_influence": 0.99, "context": 0.7})
    t = NanoTrack(mf, model=ScriptedModel(nano_io(S), [_out(S // 2, S // 2)]))
    assert t._window_influence == 0.99
    assert t._context == 0.7


def test_explicit_kwarg_overrides_manifest():
    mf = ModelManifest(name="t", task="sot_template_matching",
                       preprocessing={"window_influence": 0.99})
    t = NanoTrack(mf, model=ScriptedModel(nano_io(S), [_out(S // 2, S // 2)]),
                  window_influence=0.1)
    assert t._window_influence == 0.1


def test_nn_package_exports_nanotrack():
    import edgecv.trackers.nn as nn
    assert hasattr(nn, "NanoTrack")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_nanotrack.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edgecv.trackers.nn.nanotrack'`.

- [ ] **Step 3: Create the tracker module (construction + init + template only)**

Create `edgecv/trackers/nn/nanotrack.py`:

```python
"""NanoTrack V3 tracker (ARCHITECTURE.md §6.2). Single two-input graph
(exemplar, search) -> (cls, loc); MobileNetV3-small-v3 + AdjustLayer + DepthwiseBAN
anchor-free head. Reference defaults: HonglinChu/SiamTrackers NanoTrack configv3."""

from __future__ import annotations

import math
import time

import numpy as np

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.trackers.nn.base import UNSET, NNTracker, Template, resolve_pp
from edgecv.trackers.nn.preprocess import crop_with_context, points_grid, to_input


def _hann2d(n: int) -> np.ndarray:
    h = np.hanning(n).astype(np.float32)
    return np.outer(h, h).reshape(-1)


def _softmax_fg(cls: np.ndarray) -> np.ndarray:
    """cls (1,2,S,S) logits -> foreground prob per location, flattened (S*S,)."""
    c = np.asarray(cls, np.float32).reshape(2, -1)
    c = c - c.max(axis=0, keepdims=True)
    e = np.exp(c)
    return (e / e.sum(axis=0, keepdims=True))[1]


class NanoTrack(NNTracker):
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 exemplar_size=UNSET, search_size=UNSET, context=UNSET,
                 stride=UNSET, base_size=UNSET, penalty_k=UNSET,
                 window_influence=UNSET, size_lr=UNSET, color=UNSET, scale=UNSET,
                 score_lock=0.6, score_lost=0.35) -> None:
        super().__init__(manifest, backend=backend, model=model)
        pp = self._preprocessing
        self._exemplar_size = resolve_pp(exemplar_size, pp, "exemplar", 127)
        self._search_size = resolve_pp(search_size, pp, "search", 255)
        self._context = resolve_pp(context, pp, "context", 0.5)
        self._stride = resolve_pp(stride, pp, "stride", 16)
        self._base_size = resolve_pp(base_size, pp, "base_size", 7)
        self._penalty_k = resolve_pp(penalty_k, pp, "penalty_k", 0.138)
        self._window_influence = resolve_pp(window_influence, pp, "window_influence", 0.455)
        self._size_lr = resolve_pp(size_lr, pp, "size_lr", 0.348)
        self._color = resolve_pp(color, pp, "color", "rgb")
        self._scale = resolve_pp(scale, pp, "scale", 1.0)
        self._score_lock = score_lock
        self._score_lost = score_lost
        names = [o.name for o in self._model.io_spec.outputs]
        self._cls_name = "cls" if "cls" in names else names[0]
        self._loc_name = "loc" if "loc" in names else names[1]
        self._score_size = self._model.io_spec.outputs[0].shape[-1]
        self._points = points_grid(self._stride, self._score_size)   # (2, S*S)
        self._hann = _hann2d(self._score_size)
        self._template: Template | None = None
        self._box: BoundingBox | None = None

    def name(self) -> str:
        return "NanoTrack"

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
        z = to_input(patch, spec_z, color=self._color, scale=self._scale)
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

- [ ] **Step 4: Export `NanoTrack` from the package**

Edit `edgecv/trackers/nn/__init__.py` to:

```python
"""Dense-network (NN) trackers (ARCHITECTURE.md §6.2)."""

from edgecv.trackers.nn.base import NNTracker, Template
from edgecv.trackers.nn.nanotrack import NanoTrack
from edgecv.trackers.nn.siamfc import SiamFC
from edgecv.trackers.nn.yolo import YoloDetector, YoloTracker

__all__ = ["NNTracker", "NanoTrack", "SiamFC", "Template", "YoloDetector", "YoloTracker"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_nanotrack.py -v`
Expected: PASS (6 passed).

- [ ] **Step 6: Commit**

```bash
git add edgecv/trackers/nn/nanotrack.py edgecv/trackers/nn/__init__.py tests/test_nanotrack.py
git commit -m "feat(nn/nanotrack): NanoTrack class, init, template, exports

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: NanoTrack `update()` — decode, penalty, window, size update, status

**Files:**
- Modify: `edgecv/trackers/nn/nanotrack.py` (add `update`)
- Test: `tests/test_nanotrack.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nanotrack.py`:

```python
def test_centred_peak_keeps_centre():
    # symmetric loc (l==r, t==b) + centred fg peak -> no displacement.
    t = _nano([_out(S // 2, S // 2)], window_influence=0.0)
    t.init(_frame(), _box())
    res = t.update(_frame())
    cx, cy = res.bbox.to_pixels(320, 240).center
    assert cx == pytest.approx(160.0, abs=2.0)
    assert cy == pytest.approx(120.0, abs=2.0)
    assert res.seq == 1


def test_offcentre_peak_moves_box_right():
    # fg peak one column to the +x of centre -> centre moves +x.
    t = _nano([_out(S // 2, S // 2 + 1)], window_influence=0.0)
    t.init(_frame(), _box())
    res = t.update(_frame())
    cx, _ = res.bbox.to_pixels(320, 240).center
    assert cx > 161.0


def test_larger_predicted_box_grows_box():
    # big symmetric distances at the centred peak -> predicted box wider than target.
    t = _nano([_out(S // 2, S // 2, l=40.0, t=40.0, r=40.0, b=40.0)],
              window_influence=0.0)
    t.init(_frame(), _box())
    w0 = t.get_template().bbox.w
    res = t.update(_frame())
    assert res.bbox.w > w0


def test_window_suppresses_far_peak():
    # one frame: a near peak and a far peak of equal fg logit; high window
    # influence makes the near peak win (centre stays near image centre).
    out = _out(S // 2, S // 2)
    out["cls"][0, 1, 0, 0] = 8.0                      # add an equal far peak at corner
    t = _nano([out], window_influence=0.9)
    t.init(_frame(), _box())
    res = t.update(_frame())
    cx, cy = res.bbox.to_pixels(320, 240).center
    assert cx == pytest.approx(160.0, abs=8.0)
    assert cy == pytest.approx(120.0, abs=8.0)


def test_high_score_locks_low_score_lost():
    locked = _nano([_out(S // 2, S // 2, fg=8.0)])
    locked.init(_frame(), _box())
    assert locked.update(_frame()).status == TrackStatus.LOCKED

    # all-zero cls logits -> fg prob 0.5 everywhere -> below score_lock(0.6) and
    # above score_lost(0.35) -> COASTING.
    coasting = _nano([{"cls": np.zeros((1, 2, S, S), np.float32),
                       "loc": loc_const(S, 8, 8, 8, 8)}])
    coasting.init(_frame(), _box())
    assert coasting.update(_frame()).status == TrackStatus.COASTING


def test_output_box_is_normalised_and_unclamped():
    # target near the right edge with a strong +x peak -> centre may exceed 1.0
    # and must be reported truthfully (no clamp).
    t = _nano([_out(S // 2, S - 1)], window_influence=0.0)
    edge = BoundingBox(x=0.95, y=0.5, w=0.1, h=0.1)
    t.init(_frame(), edge)
    res = t.update(_frame())
    assert isinstance(res.bbox, BoundingBox)
    assert res.confidence is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_nanotrack.py -k "centre or peak or grows or window or locks or normalised" -v`
Expected: FAIL — `update` is inherited from `NNTracker` and raises `NotImplementedError`.

- [ ] **Step 3: Implement `update()`**

Append to `class NanoTrack` in `edgecv/trackers/nn/nanotrack.py`:

```python
    def update(self, frame: np.ndarray) -> TrackResult:
        assert self._template is not None and self._box is not None, "init() first"
        h_img, w_img = frame.shape[0], frame.shape[1]
        pix = self._box.to_pixels(w_img, h_img)
        cx, cy = pix.center
        s_z = self._exemplar_side(pix)
        s_x = s_z * self._search_size / self._exemplar_size
        scale_z = self._exemplar_size / s_z                 # frame px -> search-crop px
        spec_x = self._model.io_spec.inputs[1]
        z = self._template.arrays["exemplar"]

        patch, _ = crop_with_context(frame, (cx, cy), (s_x, s_x),
                                     (self._search_size, self._search_size))
        x = to_input(patch, spec_x, color=self._color, scale=self._scale)
        out = self._model.infer({"exemplar": z, "search": x})

        score = _softmax_fg(out[self._cls_name])            # (S*S,)
        loc = np.asarray(out[self._loc_name], np.float32).reshape(4, -1)  # l,t,r,b
        px, py = self._points[0], self._points[1]           # search-crop px, centred at 0
        x1, y1 = px - loc[0], py - loc[1]
        x2, y2 = px + loc[2], py + loc[3]
        pred_cx, pred_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pred_w, pred_h = (x2 - x1), (y2 - y1)

        def _change(r):
            return np.maximum(r, 1.0 / r)

        def _sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        tw, th = pix.w * scale_z, pix.h * scale_z           # target size in search-crop px
        s_c = _change(_sz(pred_w, pred_h) / _sz(tw, th))
        r_c = _change((tw / th) / (pred_w / pred_h))
        penalty = np.exp(-(r_c * s_c - 1.0) * self._penalty_k)
        pscore = penalty * score
        pscore = (pscore * (1.0 - self._window_influence)
                  + self._hann * self._window_influence)
        best = int(pscore.argmax())

        lr = float(penalty[best] * score[best] * self._size_lr)
        new_cx = cx + float(pred_cx[best]) / scale_z
        new_cy = cy + float(pred_cy[best]) / scale_z
        new_w = pix.w * (1.0 - lr) + (float(pred_w[best]) / scale_z) * lr
        new_h = pix.h * (1.0 - lr) + (float(pred_h[best]) / scale_z) * lr

        new_pix = PixelBox(x=new_cx - new_w / 2.0, y=new_cy - new_h / 2.0,
                           w=new_w, h=new_h)
        self._box = BoundingBox.from_pixels(new_pix, w_img, h_img)

        conf = float(score[best])
        self._status = self._status_from(conf)
        self._seq += 1
        return TrackResult(bbox=self._box, confidence=conf, status=self._status,
                           timestamp=time.monotonic(), seq=self._seq)
```

- [ ] **Step 4: Run the full tracker test file**

Run: `.venv/bin/pytest tests/test_nanotrack.py -v`
Expected: PASS (all tests, ~12 passed).

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/nanotrack.py tests/test_nanotrack.py
git commit -m "feat(nn/nanotrack): anchor-free update decode (penalty, window, size)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: NanoTrack manifest

**Files:**
- Create: `edgecv/models/manifests/nanotrack.yaml`
- Test: `tests/test_manifests_nn.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_manifests_nn.py`:

```python
def test_nanotrack_manifest_loads():
    m = load_manifest(MANIFESTS / "nanotrack.yaml")
    assert m.name == "nanotrack"
    assert m.task == "sot_template_matching"
    assert {i["name"] for i in m.inputs} == {"exemplar", "search"}
    assert [o["name"] for o in m.outputs] == ["cls", "loc"]
    assert m.preprocessing["penalty_k"] == 0.138
    assert m.preprocessing["window_influence"] == 0.455
    assert m.artifacts["onnx"]["path"] == "nanotrack.onnx"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_manifests_nn.py -k nanotrack -v`
Expected: FAIL — `FileNotFoundError` / manifest path does not exist.

- [ ] **Step 3: Create the manifest**

Create `edgecv/models/manifests/nanotrack.yaml`:

```yaml
name: nanotrack
task: sot_template_matching
# NanoTrack V3 (HonglinChu/SiamTrackers). mobilenetv3_small_v3 + AdjustLayer +
# DepthwiseBAN anchor-free head. Trained on raw [0,255] crops (no /255, no mean/std);
# cv2/BGR channel order preserved by to_input. Single graph: re-embeds the exemplar
# each frame (two-graph template caching deferred). The cls/loc spatial side (S) is
# read from io_spec at runtime; the 15 below is nominal (configv3 OUTPUT_SIZE).
preprocessing:
  color: rgb
  scale: 1.0
  exemplar: 127
  search: 255
  context: 0.5
  stride: 16
  base_size: 7
  penalty_k: 0.138
  window_influence: 0.455
  size_lr: 0.348
io:
  inputs:
    - { name: exemplar, shape: [1, 3, 127, 127], dtype: float32 }
    - { name: search,   shape: [1, 3, 255, 255], dtype: float32 }
  outputs:
    - { name: cls, shape: [1, 2, 15, 15], dtype: float32 }
    - { name: loc, shape: [1, 4, 15, 15], dtype: float32 }
artifacts:
  onnx: { path: nanotrack.onnx }
  rknn: { path: nanotrack.rk3588.rknn, quant: int8 }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_manifests_nn.py -k nanotrack -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add edgecv/models/manifests/nanotrack.yaml tests/test_manifests_nn.py
git commit -m "feat(models): nanotrack manifest (single-graph, cls+loc)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Conversion adapter — vendor the V3 architecture

**Files:**
- Create: `tools/convert_lib/adapters/nanotrack.py`
- Modify: `tools/convert_lib/adapters/__init__.py`
- Test: `tests/test_convert_nanotrack.py`

The adapter vendors the reference architecture so the upstream repo need not be installed (as `adapters/siamfc.py` vendors AlexNetV1). Copy the reference modules **verbatim**, preserving all module/attribute names so `load_state_dict(strict=True)` works against the published `nanotrackv3.pth`.

> **Fidelity note:** weights are deferred, so the test below uses *random* weights and only validates that the graph **exports** and torch == onnxruntime. The true state_dict-key fidelity check happens when a real checkpoint is loaded (`build` uses `strict=True`, so a mismatch fails loudly then). Do **not** relax to `strict=False`.

- [ ] **Step 1: Vendor the reference source files**

Download these reference files verbatim and paste their class bodies into `tools/convert_lib/adapters/nanotrack.py` (keep class names + attribute names exactly):

- Backbone `mobilenetv3_small_v3`: `https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master/NanoTrack/nanotrack/models/backbone/mobile_v3.py` — vendor `h_sigmoid`, `h_swish`, `SELayer`, `InvertedResidual`, `MobileNetV3`, and the `mobilenetv3_small_v3()` factory. The backbone forward returns `self.features(x)` (the layer-4 feature, 96 ch).
- Neck `AdjustLayer`: `https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master/NanoTrack/nanotrack/models/neck/neck.py` — vendor `AdjustLayer` (96→96). Keep its template centre-crop behaviour.
- Head `DepthwiseBAN`: `https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master/NanoTrack/nanotrack/models/head/ban_v3.py` — vendor `DepthwiseBAN` plus its xcorr helpers (`xcorr_depthwise`, `xcorr_pixelwise`) from `https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master/NanoTrack/nanotrack/core/xcorr.py`. `forward(z_f, x_f) -> (cls, loc)`, cls first.

- [ ] **Step 2: Write the failing test**

Create `tests/test_convert_nanotrack.py`:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from convert_lib import run  # noqa: E402
from convert_lib.adapters.nanotrack import Net, build  # noqa: E402


def _save_random_ckpt(tmp_path):
    ckpt = tmp_path / "fake.pth"
    torch.save(Net().state_dict(), ckpt)
    return ckpt


def test_build_loads_strict_and_shapes(tmp_path):
    net = build(str(_save_random_ckpt(tmp_path)))
    z = torch.zeros(1, 3, 127, 127)
    x = torch.zeros(1, 3, 255, 255)
    with torch.no_grad():
        cls, loc = net(z, x)
    assert cls.shape[:2] == (1, 2)
    assert loc.shape[:2] == (1, 4)
    assert cls.shape[2:] == loc.shape[2:]        # same S×S grid


def test_run_roundtrip_parity(tmp_path):
    # run() invokes the harness, which raises SystemExit unless torch-vs-onnxruntime
    # parity holds on the first output (cls); reaching the assertions means it passed.
    ckpt = _save_random_ckpt(tmp_path)
    out = tmp_path / "nanotrack.onnx"
    run("nanotrack", str(ckpt), str(out))
    assert out.exists()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    z = rng.standard_normal((1, 3, 127, 127)).astype(np.float32)
    x = rng.standard_normal((1, 3, 255, 255)).astype(np.float32)
    cls, loc = sess.run(["cls", "loc"], {"exemplar": z, "search": x})
    assert cls.shape[:2] == (1, 2)
    assert loc.shape[:2] == (1, 4)
    assert cls.shape[2:] == loc.shape[2:]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_convert_nanotrack.py -v`
Expected: FAIL with `ImportError`/`ModuleNotFoundError` for `convert_lib.adapters.nanotrack`.

- [ ] **Step 4: Add the `Net` wrapper + `build` + `register`**

Append to `tools/convert_lib/adapters/nanotrack.py` (below the vendored modules):

```python
from convert_lib.registry import Adapter, register


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = mobilenetv3_small_v3()
        self.neck = AdjustLayer(in_channels=96, out_channels=96)
        self.head = DepthwiseBAN(in_channels=96, out_channels=96)

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        # forward arg order (z=exemplar, x=search) MUST match manifest io.inputs order.
        zf = self.neck(self.backbone(z))
        xf = self.neck(self.backbone(x))
        cls, loc = self.head(zf, xf)            # cls first, loc second
        return cls, loc


def build(checkpoint: str) -> Net:
    """Load a checkpoint state_dict into Net (strict=True) and return it in eval mode."""
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:   # tolerate wrapped checkpoints
        sd = sd["state_dict"]
    net = Net()
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


register(Adapter(name="nanotrack", build=build))
```

> If the published `nanotrackv3.pth` nests submodules differently (e.g. keys like
> `backbone.features.*`, `neck.downsample.*`, `head.cls_tower.*`), name the `Net`
> attributes to match those prefixes so `strict=True` loads cleanly. Verify against the
> real checkpoint when obtained; the random-weight test only gates export.

- [ ] **Step 5: Register the adapter on import**

Edit `tools/convert_lib/adapters/__init__.py` to:

```python
"""Importing this package registers every adapter. Add new adapters here."""

from __future__ import annotations

from . import (
    nanotrack,  # noqa: F401  (import side effect: registers the adapter)
    siamfc,  # noqa: F401  (import side effect: registers the adapter)
    yolo,  # noqa: F401  (import side effect: registers the adapter)
)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_convert_nanotrack.py -v`
Expected: PASS (2 passed). If parity fails on the dynamic-weight depthwise xcorr, confirm the harness used `dynamo=False` (it does) and that the head uses `F.conv2d` for the depthwise correlation as in the reference.

- [ ] **Step 7: Commit**

```bash
git add tools/convert_lib/adapters/nanotrack.py tools/convert_lib/adapters/__init__.py tests/test_convert_nanotrack.py
git commit -m "feat(convert): nanotrack adapter (vendored mobilenetv3_small_v3 + DepthwiseBAN)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Documentation

**Files:**
- Modify: `tools/CONVERSION.md`
- Modify: `ARCHITECTURE.md`

- [ ] **Step 1: Add the nanotrack recipe to CONVERSION.md**

In `tools/CONVERSION.md`, under "## Quick start", add:

```markdown
# NanoTrack V3: PyTorch checkpoint -> ONNX (writes models/nanotrack.onnx)
python tools/convert.py --model nanotrack --checkpoint models/nanotrackv3.pth

# ...and chain to an RK3588 INT8 RKNN (needs rknn-toolkit2 + calibration images)
python tools/convert.py --model nanotrack --checkpoint models/nanotrackv3.pth --rknn --calib calib/
```

And add a caveat paragraph near the existing "Device-path numerics caveat":

```markdown
**NanoTrack RKNN/parity caveat (untested in CI):** the DepthwiseBAN head uses a
**data-dependent conv kernel** (`xcorr_depthwise`: the exemplar feature is the conv
weight) and a **matmul** (`xcorr_pixelwise`). Both export to ONNX and pass torch-vs-
onnxruntime parity (legacy exporter, `dynamo=False`), but RKNN operator support for a
dynamic-weight grouped conv is untested on-device. Validate manually; if unsupported,
fall back to a fixed-template (two-graph) export. `to_input` feeds raw `[0,255]`
(`scale=1.0`), so configure `rknn_convert` with `mean=0, std=1` to match.
```

- [ ] **Step 2: Note NanoTrack in ARCHITECTURE.md**

In `ARCHITECTURE.md` §6.2 (NN trackers), add NanoTrack to the family description, and in §13 (directory layout) add `nanotrack.py` next to `siamfc.py` in the `nn/` listing. Example sentence for §6.2:

```markdown
NanoTrack (V3) is the lightweight anchor-free member: MobileNetV3-small-v3 backbone +
DepthwiseBAN head emitting `cls`/`loc` maps, decoded over a point grid — same
manifest-driven, HAL-only contract as SiamFC.
```

- [ ] **Step 3: Verify docs render and commit**

Run: `.venv/bin/python -c "import pathlib; print('nanotrack' in pathlib.Path('tools/CONVERSION.md').read_text())"`
Expected: prints `True`.

```bash
git add tools/CONVERSION.md ARCHITECTURE.md
git commit -m "docs: NanoTrack conversion recipe + architecture note

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Full suite + lint gate

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/pytest -q`
Expected: all tests pass (existing + new NanoTrack tests). No collection errors.

- [ ] **Step 2: Run the linter**

Run: `.venv/bin/ruff check edgecv tools tests`
Expected: no errors. Fix any (line length, import order) and re-run.

- [ ] **Step 3: Run type check (if configured)**

Run: `.venv/bin/mypy edgecv/trackers/nn/nanotrack.py`
Expected: no new errors (match the surrounding modules' typing posture; SiamFC is the reference).

- [ ] **Step 4: Commit any lint/type fixups**

```bash
git add -A
git commit -m "chore(nn/nanotrack): lint/type fixups

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

(Skip the commit if Steps 1–3 produced no changes.)

---

## Self-review notes (for the implementer)

- **Spec coverage:** Task 1 = `points_grid` (spec §4); Tasks 3–4 = `NanoTrack` tracker + decode (spec §5); Task 5 = manifest (spec §6); Task 6 = conversion adapter (spec §7); Task 7 = CONVERSION.md/ARCHITECTURE.md (spec §8) + the documented edge-replicate divergence (spec §9). Border padding stays edge-replicate (per the approved decision) — no code for mean-pad.
- **`S` is read from `io_spec`**, never hardcoded — the manifest's `15` is nominal. The convert test asserts cls/loc share `S` rather than pinning a value, so a real export of a different size still passes.
- **Harness parity only checks the first output (`cls`).** The convert test additionally asserts `loc` shape via onnxruntime; full `loc` numeric parity is not gated (matches the harness's single-output contract).
- **Confidence scale:** NanoTrack reports cls fg-prob (0–1); not comparable across trackers (ARCHITECTURE §8) — relevant only to a future hybrid.

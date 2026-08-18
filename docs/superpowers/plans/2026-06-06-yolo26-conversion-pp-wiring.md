# YOLO26 Conversion + NN Manifest-Preprocessing Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the implemented YOLO/SiamFC trackers honour `manifest.preprocessing`, add a `"yolov8"` decoder + `yolo26n`/`yolo26s` manifests, and give YOLO26 a first-class `tools/convert.py` conversion path (Ultralytics → ONNX → RKNN INT8).

**Architecture:** Trackers depend only on the HAL (manifest + `InferenceBackend`/`Model`), never a vendor runtime. Preprocessing precedence is `explicit kwarg > manifest preprocessing > hardcoded default`, via the existing `UNSET` sentinel + `resolve_pp` in `trackers/nn/base.py`. YOLO26 exports with the **one-to-many head** (`nms=False`) so the onnx (x86) and rknn (device) backends emit the same `(1, 4+nc, N)` tensor, decoded by one `"yolov8"` path. Conversion gains an optional `Adapter.export` hook for model families that ship their own exporter.

**Tech Stack:** Python 3.10+, numpy. Host-only conversion uses `ultralytics` (new `[dev]` dep) + the existing generic ONNX→RKNN step. Tests: pytest via `.venv/bin/python -m pytest`; the Ultralytics exporter and RKNN build are **mocked** (never run in CI).

**Conventions pinned for this plan (from the spec):**
- **Export head:** one-to-many (`nms=False`) for both backends → uniform `(1, 4+nc, N)`.
- **`"yolov8"` decode:** transpose `(1,4+nc,N)`→`(N,4+nc)`; `score = max(class cols)`; **no objectness**; reuse `class_agnostic_nms`.
- **`input` → `input_size` key rename:** the manifest key is `input`; the kwarg is `input_size`.
- **Hardcoded `output_format` default flips to `"yolov8"`** (the shipped-default era). Existing yolov5-layout stub tests pass `output_format="yolov5"` explicitly.
- **Validation on the export hook = `onnx.checker.check_model` only** (no torch parity — Ultralytics owns the reference); wrong-head protection is the explicit `nms=False`.

Run all tests with: `.venv/bin/python -m pytest -q`

---

## File Structure

```
edgecv/trackers/nn/yolo.py        # MODIFY — "yolov8" decode branch; UNSET+resolve_pp wiring
edgecv/trackers/nn/siamfc.py      # MODIFY — finish resolve_pp wiring (all preprocessing keys)
edgecv/models/manifests/yolo26n.yaml   # CREATE (shipped default)
edgecv/models/manifests/yolo26s.yaml   # CREATE
edgecv/models/manifests/yolo_generic.yaml  # DELETE
edgecv/models/manifests/siamfc_generic.yaml # MODIFY — update stale "documentation-only" comment

tools/convert_lib/registry.py     # MODIFY — Adapter.build optional + add Adapter.export hook
tools/convert_lib/__init__.py     # MODIFY — run() branches on adapter.export (skip torch harness)
tools/convert_lib/adapters/yolo.py     # CREATE — Ultralytics one-to-many exporter adapter
tools/convert_lib/adapters/__init__.py # MODIFY — import .yolo so it self-registers
pyproject.toml                    # MODIFY — add ultralytics to [dev]
tools/CONVERSION.md               # MODIFY — document the YOLO26 recipe via convert.py

tests/test_manifests_nn.py        # MODIFY — yolo26n/yolo26s load; output_format yolov8
tests/test_pp_precedence.py       # MODIFY — repoint yolo_generic -> yolo26n
tests/test_nn_base.py             # MODIFY — repoint yolo_generic -> yolo26n
tests/test_yolo.py                # MODIFY — "yolov8" decode + PP-precedence; make v5 helpers explicit
tests/test_siamfc.py              # MODIFY — assert manifest PP reaches SiamFC
tests/test_convert_framework.py   # CREATE — run() export-hook branch (fake adapter, mocks)
tests/test_convert_yolo.py        # CREATE — yolo adapter registered + ultralytics mocked
```

---

### Task 1: `yolo26n` + `yolo26s` manifests; retire `yolo_generic`

**Files:**
- Create: `edgecv/models/manifests/yolo26n.yaml`
- Create: `edgecv/models/manifests/yolo26s.yaml`
- Delete: `edgecv/models/manifests/yolo_generic.yaml`
- Modify: `tests/test_manifests_nn.py`, `tests/test_pp_precedence.py`, `tests/test_nn_base.py`

- [ ] **Step 1: Update the failing tests**

Replace `test_yolo_manifest_loads` in `tests/test_manifests_nn.py` with:

```python
def test_yolo26n_manifest_loads():
    m = load_manifest(MANIFESTS / "yolo26n.yaml")
    assert m.name == "yolo26n"
    assert m.task == "detection"
    assert m.preprocessing["class_agnostic"] is True
    assert m.preprocessing["output_format"] == "yolov8"
    assert m.outputs[0]["shape"] == [1, 84, -1]


def test_yolo26s_manifest_loads():
    m = load_manifest(MANIFESTS / "yolo26s.yaml")
    assert m.name == "yolo26s"
    assert m.preprocessing["output_format"] == "yolov8"
    assert m.artifacts["onnx"]["path"] == "yolo26s.onnx"


def test_yolo_generic_is_retired():
    assert not (MANIFESTS / "yolo_generic.yaml").exists()
```

In `tests/test_pp_precedence.py`, change `test_manifest_preprocessing_reads_yaml` to:

```python
def test_manifest_preprocessing_reads_yaml():
    pp = manifest_preprocessing("edgecv/models/manifests/yolo26n.yaml")
    assert pp["output_format"] == "yolov8"
```

In `tests/test_nn_base.py`, replace both occurrences of
`"edgecv/models/manifests/yolo_generic.yaml"` with
`"edgecv/models/manifests/yolo26n.yaml"` (lines 35 and 44).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_manifests_nn.py tests/test_pp_precedence.py -q`
Expected: FAIL — `FileNotFoundError` (yolo26n.yaml does not exist yet).

- [ ] **Step 3: Create the manifests and delete the old one**

Create `edgecv/models/manifests/yolo26n.yaml`:

```yaml
name: yolo26n
task: detection
# Ultralytics YOLO26n, COCO-80, class-agnostic use (the class id is discarded;
# score = max over class columns). Exported with the ONE-TO-MANY head (nms=False)
# so the onnx (x86) and rknn (device) backends emit the same (1, 4+nc, N) tensor
# (the NMS-free end-to-end head is unavailable on RKNN). See the design spec §3.
preprocessing:
  color: rgb
  input: 640
  scale: 0.00392156862745098     # 1/255
  output_format: yolov8
  class_agnostic: true
  conf_thresh: 0.25
  iou_thresh: 0.45
io:
  inputs:
    - { name: images, shape: [1, 3, 640, 640], dtype: float32 }
  outputs:
    - { name: output0, shape: [1, 84, -1], dtype: float32 }   # (1, 4+nc, N); N dynamic
artifacts:
  onnx: { path: yolo26n.onnx }
  rknn: { path: yolo26n.rk3588.rknn, quant: int8 }
```

Create `edgecv/models/manifests/yolo26s.yaml`:

```yaml
name: yolo26s
task: detection
# Ultralytics YOLO26s — accuracy variant of yolo26n. Identical I/O and preprocessing;
# only the weights (and thus the artifact paths) differ. See the design spec §5.
preprocessing:
  color: rgb
  input: 640
  scale: 0.00392156862745098     # 1/255
  output_format: yolov8
  class_agnostic: true
  conf_thresh: 0.25
  iou_thresh: 0.45
io:
  inputs:
    - { name: images, shape: [1, 3, 640, 640], dtype: float32 }
  outputs:
    - { name: output0, shape: [1, 84, -1], dtype: float32 }
artifacts:
  onnx: { path: yolo26s.onnx }
  rknn: { path: yolo26s.rk3588.rknn, quant: int8 }
```

Delete the old manifest:

```bash
git rm edgecv/models/manifests/yolo_generic.yaml
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_manifests_nn.py tests/test_pp_precedence.py tests/test_nn_base.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edgecv/models/manifests/yolo26n.yaml edgecv/models/manifests/yolo26s.yaml \
    tests/test_manifests_nn.py tests/test_pp_precedence.py tests/test_nn_base.py
git commit -m "feat(models): yolo26n/yolo26s manifests (yolov8 output), retire yolo_generic"
```

---

### Task 2: `"yolov8"` decoder branch in `YoloDetector`

**Files:**
- Modify: `edgecv/trackers/nn/yolo.py:43-74` (the `detect` method)
- Test: `tests/test_yolo.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_yolo.py` (a dedicated v8 helper set, so it is independent of the v5 helpers):

```python
def _yolo_io_v8(nc=80):
    # one-to-many head: (1, 4+nc, N) — transposed vs v5, no objectness column
    return IOSpec(inputs=(TensorSpec("images", (1, 3, IN, IN), "float32"),),
                  outputs=(TensorSpec("output0", (1, 4 + nc, -1), "float32"),))


def _raw_v8(dets, nc=80):
    """dets: list of (cx, cy, w, h, cls_score, cls_idx) in INPUT (letterbox) px.
    Builds a (1, 4+nc, N) tensor (channels-first, the YOLO26 one-to-many layout)."""
    out = np.zeros((1, 4 + nc, len(dets)), np.float32)
    for j, (cx, cy, w, h, sc, ci) in enumerate(dets):
        out[0, :4, j] = [cx, cy, w, h]
        out[0, 4 + ci, j] = sc
    return {"output0": out}


def _detector_v8(raw, **kw):
    return YoloDetector(model=ScriptedModel(_yolo_io_v8(), [raw]),
                        input_size=IN, output_format="yolov8", **kw)


def test_yolov8_decode_no_objectness_max_class():
    # class score 0.9 in column for class 5; score must be 0.9 (no obj multiply)
    det = _detector_v8(_raw_v8([(32, 32, 16, 16, 0.9, 5)]))
    out = det.detect(np.zeros((IN, IN, 3), np.uint8))
    assert out.boxes.shape == (1, 4)
    assert out.scores[0] == pytest.approx(0.9, abs=1e-3)
    bx, by, bw, bh = out.boxes[0]
    assert (bx + bw / 2) == pytest.approx(0.5, abs=0.05)


def test_yolov8_thresholds_and_is_pure():
    det = _detector_v8(_raw_v8([(32, 32, 16, 16, 0.1, 0)]), conf_thresh=0.25)
    img = np.zeros((IN, IN, 3), np.uint8)
    assert len(det.detect(img).scores) == 0          # thresholded out
    det2 = _detector_v8(_raw_v8([(32, 32, 16, 16, 0.8, 2), (10, 10, 8, 8, 0.7, 9)]))
    o1, o2 = det2.detect(img), det2.detect(img)
    np.testing.assert_array_equal(o1.boxes, o2.boxes)  # purity
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_yolo.py -k yolov8 -q`
Expected: FAIL — the `else` branch treats `raw[0]` as `(N, 5+nc)` and reads `preds[:, 4]` as a single score on a non-transposed `(4+nc, N)` tensor → wrong shapes / assertion error.

- [ ] **Step 3: Write minimal implementation**

In `edgecv/trackers/nn/yolo.py`, replace the head of `detect` (from `raw = ...` through the format branch) so the v8 tensor is transposed **before** the empty check and decoded without objectness:

```python
    def detect(self, image: np.ndarray) -> DetectorOutput:
        h_img, w_img = image.shape[0], image.shape[1]
        n = self._input_size
        lb, xf = letterbox(image, (n, n))
        inp = to_input(lb, self._spec, color=self._color, scale=self._scale)
        raw = np.asarray(self._model.infer({self._spec.name: inp})[self._out_name], np.float32)
        # v8/v26 one-to-many head is (1, 4+nc, N) channels-first -> transpose to rows;
        # v5/"decoded" are already (1, N, k).
        preds = raw[0].T if self._output_format == "yolov8" else raw[0]
        if preds.shape[0] == 0:
            return DetectorOutput(boxes=np.empty((0, 4), np.float32),
                                  scores=np.empty((0,), np.float32))
        if self._output_format == "yolov5":
            # yolov5 row layout: [cx, cy, w, h | obj | cls_0..cls_{nc-1}]
            xywh, obj, cls = preds[:, :4], preds[:, 4], preds[:, 5:]
            score = obj * (cls.max(axis=1) if cls.shape[1] > 0 else 1.0)
        elif self._output_format == "yolov8":
            # anchor-free, NO objectness: [cx, cy, w, h | cls_0..cls_{nc-1}]
            xywh, cls = preds[:, :4], preds[:, 4:]
            score = (cls.max(axis=1) if cls.shape[1] > 0
                     else np.zeros((preds.shape[0],), np.float32))
        else:  # "decoded": model already emits xywh + score
            xywh, score = preds[:, :4], preds[:, 4]
```

Leave the rest of `detect` (threshold → xyxy → NMS → letterbox inversion → return) unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_yolo.py -q`
Expected: PASS (existing v5 tests + new v8 tests).

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/yolo.py tests/test_yolo.py
git commit -m "feat(nn/yolo): yolov8 one-to-many decoder (transposed, anchor-free, class-agnostic)"
```

---

### Task 3: Wire `manifest.preprocessing` into `YoloDetector`

**Files:**
- Modify: `edgecv/trackers/nn/yolo.py` (imports + `YoloDetector.__init__`)
- Test: `tests/test_yolo.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_yolo.py` (extend the top imports with `from edgecv.models.manifest import ModelManifest`):

```python
def _pp_manifest(**pp):
    return ModelManifest(name="t", task="detection", preprocessing=pp)


def test_detector_reads_preprocessing_from_manifest():
    mf = _pp_manifest(conf_thresh=0.5, input=128, output_format="yolov8", color="gray")
    det = YoloDetector(manifest=mf, model=ScriptedModel(_yolo_io_v8(), [_raw_v8([])]))
    assert det._conf == 0.5
    assert det._input_size == 128
    assert det._output_format == "yolov8"
    assert det._color == "gray"


def test_detector_explicit_kwarg_overrides_manifest():
    mf = _pp_manifest(conf_thresh=0.5)
    det = YoloDetector(manifest=mf, model=ScriptedModel(_yolo_io(), [_raw([])]),
                       conf_thresh=0.9, output_format="yolov5")
    assert det._conf == 0.9


def test_detector_default_when_absent_from_manifest():
    det = YoloDetector(model=ScriptedModel(_yolo_io(), [_raw([])]), output_format="yolov5")
    assert det._conf == 0.25          # hardcoded default
    assert det._output_format == "yolov5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_yolo.py -k "preprocessing or override or absent" -q`
Expected: FAIL — `YoloDetector` constructor takes plain defaults, ignores the manifest; `det._conf` is `0.25` not `0.5`.

- [ ] **Step 3: Write minimal implementation**

In `edgecv/trackers/nn/yolo.py`, change the base import line to pull the helpers:

```python
from edgecv.trackers.nn.base import (
    UNSET,
    NNTracker,
    manifest_preprocessing,
    resolve_model,
    resolve_pp,
)
```

Replace `YoloDetector.__init__`:

```python
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 input_size=UNSET, color=UNSET, scale=UNSET,
                 output_format=UNSET, conf_thresh=UNSET, iou_thresh=UNSET) -> None:
        self._owns_model = model is None
        self._model = resolve_model(manifest, backend, model)
        pp = manifest_preprocessing(manifest)   # {} when a model= is injected
        self._input_size = resolve_pp(input_size, pp, "input", 640)
        self._color = resolve_pp(color, pp, "color", "rgb")
        self._scale = resolve_pp(scale, pp, "scale", 1.0 / 255.0)
        self._output_format = resolve_pp(output_format, pp, "output_format", "yolov8")
        self._conf = resolve_pp(conf_thresh, pp, "conf_thresh", 0.25)
        self._iou = resolve_pp(iou_thresh, pp, "iou_thresh", 0.45)
        self._spec = self._model.io_spec.inputs[0]
        self._out_name = self._model.io_spec.outputs[0].name
```

Now make the existing v5-layout helpers explicit (the hardcoded default is now `"yolov8"`). In `tests/test_yolo.py`, change `_detector` and `_tracker`:

```python
def _detector(raw, **kw):
    return YoloDetector(model=ScriptedModel(_yolo_io(), [raw]), input_size=IN,
                        output_format="yolov5", **kw)
```

```python
def _tracker(maps, **kw):
    m = ScriptedModel(_yolo_io(), maps)
    return YoloTracker(model=m, input_size=IN, output_format="yolov5", **kw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_yolo.py -q`
Expected: PASS (all detector + tracker tests).

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/yolo.py tests/test_yolo.py
git commit -m "feat(nn/yolo): YoloDetector reads preprocessing from manifest (resolve_pp)"
```

---

### Task 4: Wire `manifest.preprocessing` into `YoloTracker`

**Files:**
- Modify: `edgecv/trackers/nn/yolo.py` (`YoloTracker.__init__`)
- Test: `tests/test_yolo.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_yolo.py`:

```python
def test_tracker_reads_preprocessing_from_manifest():
    mf = _pp_manifest(conf_thresh=0.5, output_format="yolov8")
    t = YoloTracker(manifest=mf, model=ScriptedModel(_yolo_io_v8(), [_raw_v8([])]),
                    input_size=IN)
    assert t._detector._conf == 0.5
    assert t._detector._output_format == "yolov8"


def test_tracker_explicit_kwarg_overrides_manifest():
    mf = _pp_manifest(conf_thresh=0.5)
    t = YoloTracker(manifest=mf, model=ScriptedModel(_yolo_io(), [_raw([])]),
                    input_size=IN, conf_thresh=0.9, output_format="yolov5")
    assert t._detector._conf == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_yolo.py -k "tracker_reads or tracker_explicit" -q`
Expected: FAIL — `YoloTracker` passes plain defaults to its `YoloDetector`; `_conf` is `0.25` not `0.5`.

- [ ] **Step 3: Write minimal implementation**

In `edgecv/trackers/nn/yolo.py`, replace `YoloTracker.__init__` (the constructor only — leave `name`/`init`/`update` untouched):

```python
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 search_factor=3.0, assoc_sigma=0.5, conf_thresh=UNSET,
                 iou_thresh=UNSET, max_misses=5, input_size=UNSET,
                 color=UNSET, scale=UNSET, output_format=UNSET) -> None:
        super().__init__(manifest, backend=backend, model=model)
        pp = self._preprocessing
        input_size = resolve_pp(input_size, pp, "input", 640)
        color = resolve_pp(color, pp, "color", "rgb")
        scale = resolve_pp(scale, pp, "scale", 1.0 / 255.0)
        output_format = resolve_pp(output_format, pp, "output_format", "yolov8")
        conf_thresh = resolve_pp(conf_thresh, pp, "conf_thresh", 0.25)
        iou_thresh = resolve_pp(iou_thresh, pp, "iou_thresh", 0.45)
        self._detector = YoloDetector(
            model=self._model, input_size=input_size, color=color, scale=scale,
            output_format=output_format, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        self._search_factor = search_factor
        self._assoc_sigma = assoc_sigma
        self._max_misses = max_misses
        self._input_size = input_size
        self._box: BoundingBox | None = None
        self._misses = 0
```

(`search_factor`, `assoc_sigma`, `max_misses` stay plain — they are association policy, not model preprocessing.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_yolo.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/yolo.py tests/test_yolo.py
git commit -m "feat(nn/yolo): YoloTracker resolves preprocessing precedence to its detector"
```

---

### Task 5: Finish `SiamFC` preprocessing wiring

**Files:**
- Modify: `edgecv/trackers/nn/siamfc.py:27-45` (`__init__`)
- Modify: `edgecv/models/manifests/siamfc_generic.yaml` (stale comment)
- Test: `tests/test_siamfc.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_siamfc.py` (extend imports with `from edgecv.models.manifest import ModelManifest`):

```python
def test_manifest_preprocessing_reaches_siamfc():
    mf = ModelManifest(name="t", task="sot_template_matching",
                       preprocessing={"window_influence": 0.99, "context": 0.7})
    t = SiamFC(mf, model=ScriptedModel(siam_io(SS), [{"score_map": score_map_peaked(SS, 8, 8)}]))
    assert t._window_influence == 0.99
    assert t._context == 0.7


def test_siamfc_explicit_kwarg_overrides_manifest():
    mf = ModelManifest(name="t", task="sot_template_matching",
                       preprocessing={"window_influence": 0.99})
    t = SiamFC(mf, model=ScriptedModel(siam_io(SS), [{"score_map": score_map_peaked(SS, 8, 8)}]),
               window_influence=0.1)
    assert t._window_influence == 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_siamfc.py -k "reaches_siamfc or siamfc_explicit" -q`
Expected: FAIL — `context`/`window_influence` are plain kwargs; the manifest value never reaches them, so `t._window_influence` is `0.176` not `0.99`.

- [ ] **Step 3: Write minimal implementation**

In `edgecv/trackers/nn/siamfc.py`, replace the constructor signature and the param-assignment block (the existing `color`/`scale` already use `UNSET`+`resolve_pp`; extend the same pattern to the rest):

```python
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 exemplar_size=UNSET, search_size=UNSET, context=UNSET,
                 total_stride=UNSET, response_up=UNSET, scale_num=UNSET,
                 scale_step=UNSET, scale_penalty=UNSET, scale_lr=UNSET,
                 window_influence=UNSET, color=UNSET, scale=UNSET,
                 score_lock=8.0, score_lost=4.0) -> None:
        super().__init__(manifest, backend=backend, model=model)
        pp = self._preprocessing
        self._exemplar_size = resolve_pp(exemplar_size, pp, "exemplar", 127)
        self._search_size = resolve_pp(search_size, pp, "search", 255)
        self._context = resolve_pp(context, pp, "context", 0.5)
        self._total_stride = resolve_pp(total_stride, pp, "total_stride", 8)
        self._response_up = resolve_pp(response_up, pp, "response_up", 16)
        self._scale_num = resolve_pp(scale_num, pp, "scale_num", 3)
        self._scale_step = resolve_pp(scale_step, pp, "scale_step", 1.0375)
        self._scale_penalty = resolve_pp(scale_penalty, pp, "scale_penalty", 0.9745)
        self._scale_lr = resolve_pp(scale_lr, pp, "scale_lr", 0.59)
        self._window_influence = resolve_pp(window_influence, pp, "window_influence", 0.176)
        self._color = resolve_pp(color, pp, "color", "rgb")
        self._scale = resolve_pp(scale, pp, "scale", 1.0)
        self._score_lock = score_lock
        self._score_lost = score_lost
        out = self._model.io_spec.outputs[0]
        self._out_name = out.name
        self._score_size = out.shape[-1]
        self._up_size = self._score_size * self._response_up
        self._hann = _hann2d(self._up_size)
        self._template: Template | None = None
        self._box: BoundingBox | None = None
```

In `edgecv/models/manifests/siamfc_generic.yaml`, replace the stale comment block (lines 7–10, beginning `# Only \`color\` and \`scale\` flow into SiamFC`) with:

```yaml
  # All keys below flow into SiamFC via resolve_pp (manifest-precedence, ARCHITECTURE
  # §10.1): explicit __init__ kwarg > this manifest > hardcoded default.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_siamfc.py -q`
Expected: PASS (existing SiamFC tests + the two new wiring tests).

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/nn/siamfc.py edgecv/models/manifests/siamfc_generic.yaml tests/test_siamfc.py
git commit -m "feat(nn/siamfc): wire all manifest preprocessing keys through resolve_pp"
```

---

### Task 6: `Adapter.export` hook + `run()` branch (conversion framework)

**Files:**
- Modify: `tools/convert_lib/registry.py`
- Modify: `tools/convert_lib/__init__.py` (`run`)
- Test: `tests/test_convert_framework.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_convert_framework.py`:

```python
from pathlib import Path

import pytest

import convert_lib
from convert_lib import registry, run
from convert_lib.registry import Adapter

_MANIFEST = (
    "name: fakeyolo\ntask: detection\n"
    "io:\n"
    "  inputs: [{name: images, shape: [1, 3, 8, 8], dtype: float32}]\n"
    "  outputs: [{name: output0, shape: [1, 84, -1], dtype: float32}]\n"
    "artifacts:\n"
    "  onnx: {path: fakeyolo.onnx}\n"
    "  rknn: {path: fakeyolo.rk3588.rknn}\n"
)


def _setup(tmp_path, monkeypatch):
    pytest.importorskip("torch")   # run() imports convert_lib.adapters -> siamfc (torch)
    onnx = pytest.importorskip("onnx")
    (tmp_path / "fakeyolo.yaml").write_text(_MANIFEST)
    monkeypatch.setattr(convert_lib, "_MANIFESTS", tmp_path)
    monkeypatch.setenv("EDGECV_MODEL_DIR", str(tmp_path))   # artifact paths resolve here
    monkeypatch.setattr(onnx.checker, "check_model", lambda p: None)


def test_export_hook_runs_and_skips_torch_harness(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    seen = {}

    def fake_export(checkpoint, onnx_out, manifest):
        seen["args"] = (checkpoint, onnx_out)
        seen["imgsz"] = manifest.inputs[0]["shape"][-1]
        Path(onnx_out).write_bytes(b"onnx")
        return onnx_out

    registry.register(Adapter(name="fakeyolo", export=fake_export))
    out = run("fakeyolo", "ckpt.pt")
    assert seen["args"][0] == "ckpt.pt"
    assert seen["imgsz"] == 8
    assert Path(out).exists() and out.endswith("fakeyolo.onnx")


def test_export_hook_chains_rknn(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    import convert_lib.rknn as rknnmod

    rk = {}
    monkeypatch.setattr(rknnmod, "rknn_convert",
                        lambda onnx_p, out_p, target, calib, names: rk.update(
                            out=out_p, names=names) or out_p)
    registry.register(Adapter(
        name="fakeyolo",
        export=lambda c, o, m: Path(o).write_bytes(b"onnx") or o))
    run("fakeyolo", "ckpt.pt", rknn=True)
    assert rk["names"] == ["images"]
    assert rk["out"].endswith("fakeyolo.rk3588.rknn")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_convert_framework.py -q`
Expected: FAIL — `Adapter.__init__` has no `export` parameter (`TypeError: unexpected keyword argument 'export'`).

- [ ] **Step 3: Write minimal implementation**

In `tools/convert_lib/registry.py`, make `build` optional and add the `export` hook:

```python
@dataclass
class Adapter:
    name: str                                       # manifest model name, e.g. "siamfc_generic"
    build: Callable[[str], Any] | None = None       # torch path: checkpoint -> .eval() nn.Module
    export: Callable[..., str] | None = None         # upstream-exporter path: (ckpt, onnx_out, manifest) -> onnx_out
    dynamic_axes: dict | None = None                # optional; variable dims (e.g. YOLO det count)
```

In `tools/convert_lib/__init__.py`, replace `run` so it branches on `adapter.export` (and move the `torch` import into the torch-only branch so the export path needs no torch):

```python
def run(model: str, checkpoint: str, out: str | None = None, *,
        rknn: bool = False, target: str = "rk3588", calib: str | None = None) -> str:
    """Convert `checkpoint` for `model` to ONNX (and optionally RKNN), driven by the
    model's manifest. Writes ONNX to `out` or to the manifest's resolved artifact path.
    Two paths: a torch adapter (build + parity harness) or an upstream-exporter adapter
    (adapter.export writes the ONNX directly; we only run onnx.checker)."""
    from edgecv.models.manifest import load_manifest
    from edgecv.models.paths import resolve_artifact_path

    from . import adapters  # noqa: F401  (registers adapters)

    mf_path = _MANIFESTS / f"{model}.yaml"
    if not mf_path.exists():
        raise SystemExit(f"no manifest at {mf_path}")
    mf = load_manifest(mf_path)
    try:
        adapter = registry.get(model)
    except KeyError:
        raise SystemExit(
            f"no adapter registered for {model!r}; registered: {registry.registered_names()}"
        ) from None

    in_names = [i["name"] for i in mf.inputs]
    out_names = [o["name"] for o in mf.outputs]
    onnx_out = out or resolve_artifact_path(_artifact_path(mf, model, "onnx"))

    if adapter.export is not None:
        adapter.export(checkpoint, onnx_out, mf)
        import onnx
        onnx.checker.check_model(onnx_out)
        print(f"exported {onnx_out}  (via upstream exporter)")
    else:
        import torch

        from .harness import export_and_validate
        try:
            module = adapter.build(checkpoint)
        except (RuntimeError, OSError) as e:   # strict-load mismatch, missing/corrupt file
            raise SystemExit(f"failed to load checkpoint for {model!r}: {e}") from e
        example = tuple(torch.zeros(_concrete(i["shape"])) for i in mf.inputs)
        diff = export_and_validate(module, example, in_names, out_names, onnx_out,
                                   dynamic_axes=adapter.dynamic_axes)
        print(f"exported {onnx_out}  (parity max|delta|={diff:.2e})")

    if rknn:
        from .rknn import rknn_convert
        rk_out = resolve_artifact_path(_artifact_path(mf, model, "rknn"))
        rknn_convert(onnx_out, rk_out, target, calib, in_names)
    return onnx_out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_convert_framework.py tests/test_convert_registry.py -q`
Expected: PASS. (The SiamFC torch path is guarded by the existing `tests/test_convert_siamfc.py`, run in Task 8.)

- [ ] **Step 5: Commit**

```bash
git add tools/convert_lib/registry.py tools/convert_lib/__init__.py tests/test_convert_framework.py
git commit -m "feat(convert): optional Adapter.export hook for upstream-exporter models"
```

---

### Task 7: YOLO conversion adapter (Ultralytics → one-to-many ONNX)

**Files:**
- Create: `tools/convert_lib/adapters/yolo.py`
- Modify: `tools/convert_lib/adapters/__init__.py`
- Modify: `pyproject.toml` (add `ultralytics` to `[dev]`)
- Test: `tests/test_convert_yolo.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_convert_yolo.py`:

```python
import sys
import types
from pathlib import Path

import pytest

from convert_lib import registry


def test_yolo_adapters_registered_with_export_hook():
    pytest.importorskip("torch")   # importing convert_lib.adapters pulls siamfc (torch)
    import convert_lib.adapters  # noqa: F401  (registers all adapters)
    n, s = registry.get("yolo26n"), registry.get("yolo26s")
    assert n.export is not None and n.build is None
    assert s.export is not None and s.build is None


def test_export_invokes_ultralytics_one_to_many(tmp_path, monkeypatch):
    pytest.importorskip("torch")   # convert_lib.adapters.yolo import pulls the package (siamfc/torch)
    from edgecv.models.manifest import load_manifest

    produced = tmp_path / "src" / "yolo26n.onnx"
    produced.parent.mkdir()
    produced.write_bytes(b"onnx")
    captured = {}

    class FakeYOLO:
        def __init__(self, ckpt):
            captured["ckpt"] = ckpt

        def export(self, **kw):
            captured["kw"] = kw
            return str(produced)

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    from convert_lib.adapters.yolo import _export
    mf = load_manifest("edgecv/models/manifests/yolo26n.yaml")
    dest = tmp_path / "dest" / "yolo26n.onnx"
    out = _export("models/yolo26n.pt", str(dest), mf)

    assert captured["ckpt"] == "models/yolo26n.pt"
    assert captured["kw"]["nms"] is False          # one-to-many head (design §3)
    assert captured["kw"]["imgsz"] == 640
    assert out == str(dest)
    assert dest.exists()                            # moved into place
    assert not produced.exists()                    # source consumed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_convert_yolo.py -q`
Expected: FAIL — `KeyError: 'yolo26n'` (no adapter) / `ModuleNotFoundError: convert_lib.adapters.yolo`.

- [ ] **Step 3: Write minimal implementation**

Create `tools/convert_lib/adapters/yolo.py`:

```python
"""YOLO26 conversion adapter (ARCHITECTURE.md §11). Drives the Ultralytics exporter
to a ONE-TO-MANY (NMS-required) ONNX graph; the generic ONNX->RKNN step then handles
INT8. Unlike torch adapters (e.g. siamfc), Ultralytics owns the architecture and the
exporter, so this registers an `export` hook instead of `build` — convert_lib.run()
branches on it and skips the torch parity harness. See design spec §7.

The one-to-many head (`nms=False`) is mandatory: the NMS-free end-to-end head is
unavailable on RKNN, and we need onnx (x86) and rknn (device) to emit the same
(1, 4+nc, N) tensor for the `yolov8` decoder (design spec §3)."""

from __future__ import annotations

import shutil
from pathlib import Path

from convert_lib.registry import Adapter, register


def _export(checkpoint: str, onnx_out: str, manifest) -> str:
    from ultralytics import YOLO  # lazy: host-only [dev] dep, only needed at convert time

    imgsz = int(manifest.inputs[0]["shape"][-1])
    produced = YOLO(checkpoint).export(format="onnx", nms=False, imgsz=imgsz, opset=13)
    Path(onnx_out).parent.mkdir(parents=True, exist_ok=True)
    if str(produced) != str(onnx_out):
        shutil.move(str(produced), str(onnx_out))
    return onnx_out


register(Adapter(name="yolo26n", export=_export))
register(Adapter(name="yolo26s", export=_export))
```

In `tools/convert_lib/adapters/__init__.py`, add the import so it self-registers:

```python
"""Importing this package registers every adapter. Add new adapters here."""

from __future__ import annotations

from . import siamfc  # noqa: F401  (import side effect: registers the adapter)
from . import yolo  # noqa: F401  (import side effect: registers the adapter)
```

In `pyproject.toml`, add `ultralytics` to the `[dev]` extra:

```toml
dev = ["torch>=2.5", "onnx>=1.15", "ultralytics>=8.3"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_convert_yolo.py -q`
Expected: PASS.

> Note: `from . import yolo` must not require `ultralytics` at import time — the
> `from ultralytics import YOLO` is inside `_export`, so adapter registration works
> without the dep installed. The first test (`..._registered...`) proves this.

- [ ] **Step 5: Commit**

```bash
git add tools/convert_lib/adapters/yolo.py tools/convert_lib/adapters/__init__.py \
    pyproject.toml tests/test_convert_yolo.py
git commit -m "feat(convert): yolo26n/yolo26s adapter via Ultralytics one-to-many export"
```

---

### Task 8: Document the YOLO26 recipe + full-suite green

**Files:**
- Modify: `tools/CONVERSION.md`
- Test: the whole suite + ruff

- [ ] **Step 1: Update CONVERSION.md**

In `tools/CONVERSION.md`, under "Quick start", replace the standalone `onnx_to_rknn.py` YOLO
example with the now-first-class dispatcher recipe:

```bash
# YOLO26n: Ultralytics checkpoint -> one-to-many ONNX (writes models/yolo26n.onnx)
python tools/convert.py --model yolo26n --checkpoint models/yolo26n.pt

# ...and chain to an RK3588 INT8 RKNN (needs rknn-toolkit2 + calibration images)
python tools/convert.py --model yolo26n --checkpoint models/yolo26n.pt --rknn --calib calib/
```

Replace the "Variant: upstream already exports ONNX" section body with:

```markdown
### Variant: upstream already exports ONNX (e.g. YOLO / ultralytics)

Some model families ship their own exporter, so there is no torch `nn.Module` to own.
These register an `export` hook instead of `build` (see `adapters/yolo.py`); the
dispatcher calls it to write the ONNX directly, runs `onnx.checker`, and skips the
torch parity harness. YOLO26 uses the **one-to-many head** (`nms=False`): the NMS-free
end-to-end head is unavailable on RKNN, and both backends must emit the same
`(1, 4+nc, N)` tensor for the `yolov8` decoder.

**Device-path numerics caveat (RKNN, untested in CI):** the tracker's `to_input`
applies `scale=1/255` on the host, and `rknn_convert` configures `mean=0, std=1`
(raw-pixel passthrough). If on-device validation shows a mismatch, set the RKNN
`std_values` to `255` (let the NPU divide) **or** feed raw pixels and drop the host
scale — keep it consistent with `to_input`.
```

- [ ] **Step 2: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all tests, no regressions). The torch-dependent `test_convert_siamfc.py`
confirms the torch `build` path still routes correctly after the `run()` branch change.

- [ ] **Step 3: Run the linter**

Run: `.venv/bin/ruff check edgecv tools tests`
Expected: no errors. (Fix any import-order/line-length issues ruff flags, then re-run.)

- [ ] **Step 4: Commit**

```bash
git add tools/CONVERSION.md
git commit -m "docs(convert): document the YOLO26 dispatcher recipe + RKNN numerics caveat"
```

---

## Notes for the implementer

- **`UNSET` is a sentinel object**, not `None` — `resolve_pp` distinguishes "kwarg not passed"
  (`UNSET`) from an explicit `None`. Never default a wired kwarg to `None`.
- **`YoloDetector` is not an `NNTracker`**, so it has no `self._preprocessing`; it computes its own
  via `manifest_preprocessing(manifest)` (returns `{}` when a `model=` is injected, so injected-model
  tests fall straight through to hardcoded defaults).
- **The `input` → `input_size` rename** is the one non-1:1 mapping: `resolve_pp(input_size, pp, "input", 640)`.
- **Conversion tests never run a real model.** Ultralytics and `rknn_convert` are mocked; `onnx.checker`
  is stubbed in the framework test. The real export/RKNN build is a manual host/device step.
- **Do not reorder** the `detect` format branches or drop the pre-decode transpose — the empty-`preds`
  guard must run on the transposed `(N, 4+nc)` view for `"yolov8"`.
```

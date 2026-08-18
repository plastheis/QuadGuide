# NN Trackers (SiamFC + class-agnostic YOLO) — Design Spec

> **Status:** design spec, ready for planning. Self-contained: written to survive a context reset.
> Implements the first dense-network (NN) trackers against `ARCHITECTURE.md` §6.2 and the HAL in
> §10, as **standalone, inline** trackers. References:
> [`HonglinChu/SiamTrackers`](https://github.com/HonglinChu/SiamTrackers) for the Siamese pipeline
> and defaults; Matsuo & Yamakawa, *"High-Speed Tracking with Mutual Assistance of Feature Filters
> and Detectors (MAFiD)"*, Sensors 2023, 23, 7082 (`sensors-23-07082.pdf`) for the class-agnostic
> local YOLO detector used as a tracker.

## 1. Goal and scope

Add two **individual, inline** NN trackers plus the shared plumbing they need, so they can be used
**standalone today** and dropped into a **hybrid later** with no tracker change.

Deliverables:

1. **`NNTracker` base** (`trackers/nn/base.py`) — backend/model resolution + lifecycle + a dependency
   injection seam, a `Template` appearance abstraction, and the inline `update` scaffolding. Depends
   only on `InferenceBackend`/`Model`/`ModelManifest` (ARCHITECTURE §10), never a vendor runtime.
2. **`SiamFC`** (`trackers/nn/siamfc.py`) — the first Siamese tracker: exemplar/search cross-correlation
   score map, **multi-scale** search (position **and** scale/aspect adaptation). Matches the shipped
   `siamfc_generic` manifest in ARCHITECTURE §10.1.
3. **`YoloDetector` + `YoloTracker`** (`trackers/nn/yolo.py`) — a **class-agnostic** YOLO detector
   exposed two ways: (a) `YoloDetector.detect(image) -> DetectorOutput` — the reusable primitive a
   future hybrid detector-worker calls; (b) `YoloTracker(NNTracker)` — a standalone single-object
   tracker that runs the detector on a **local crop** around the previous box and associates by
   proximity×score (the MAFiD local-detection mode, §3.3 of the paper).
4. **Manifests** for both (`models/manifests/`), and **RKNN `Model` wiring** (device path; the rknn
   adapter currently defers `load` "alongside the first NN tracker").

### Agreed decisions (from clarification)

| Fork | Decision |
|---|---|
| Trackers in this spec | `NNTracker` base + `SiamFC` + class-agnostic YOLO. **SiamRPN/++ deferred** to a follow-up spec. |
| Weights | **Interface now, weights deferred.** Logic is validated against the `mock` backend and **purpose-built deterministic stub models** in tests; obtaining/converting real default SiamFC & YOLO weights is a separate `tools/` task (ARCHITECTURE §11). |
| YOLO standalone | **Local-crop + nearest/score association** (MAFiD local mode). |
| SiamFC scale | **Multi-scale**: 3-scale search with scale penalty + damping; box size adapts. |

### Out of scope (deliberate)

- **Hybrid / IPC / fusion wiring.** These trackers are standalone inline. The YOLO detector emits the
  existing `DetectorOutput` (`fusion/policy.py`) and `Template` mirrors `FilterState`'s shape so a
  later hybrid consumes them without redesign — but no process group, frame ring, or fusion policy is
  built here (ARCHITECTURE §6.3, §7, §8).
- **SiamRPN / SiamRPN++ / SiamFC++** — anchors + box regression, ResNet backbones. Separate spec.
- **Real default weights, ONNX export, INT8 RKNN conversion + calibration** — host-only `tools/`
  (ARCHITECTURE §11). This spec ships **manifests and the runtime path**, not `.onnx`/`.rknn` files.
- **Motion-detection cue.** Dropped from the architecture; not reintroduced here.

## 2. Module layout / files

```
edgecv/trackers/nn/
├── base.py            # NEW — NNTracker(Tracker), Template, backend resolution + lifecycle + DI seam
├── preprocess.py      # NEW — module-level numpy ops: crop_with_context, letterbox, to_input, coord inversion
├── siamfc.py          # NEW — SiamFC(NNTracker), multi-scale exemplar/search
├── yolo.py            # NEW — YoloDetector (-> DetectorOutput) + YoloTracker(NNTracker)
└── __init__.py        # EDIT — export SiamFC, YoloTracker, YoloDetector

edgecv/models/manifests/
├── siamfc_generic.yaml   # NEW
└── yolo_generic.yaml     # NEW

edgecv/backends/rknn/__init__.py   # EDIT — implement RknnBackend.load -> RknnModel (device-only; untested in CI)
pyproject.toml                     # EDIT — package the manifests (force-include, like models/profiles)

tests/
├── test_nn_preprocess.py   # NEW — geometry + inversion round-trips, letterbox, NMS, decode
├── test_nn_base.py         # NEW — backend resolution, DI seam, lifecycle/close
├── test_siamfc.py          # NEW — multi-scale plumbing with a deterministic stub Model
└── test_yolo.py            # NEW — class-agnostic decode/NMS + local-crop association with a stub Model
```

No changes to `core/`, `runtime/`, or `fusion/`. `models/manifest.py` is reused as-is.

## 3. `NNTracker` base — backend resolution, lifecycle, DI seam

`trackers/nn/base.py`. Subclasses `Tracker` (ARCHITECTURE §5.3). Owns the model so SiamFC/YOLO
share resolution + teardown and never touch a vendor runtime.

```python
@dataclass
class Template:
    """Transferable target appearance — the NN analogue of CF FilterState (ARCHITECTURE §6.1, §6.2).
    Lets a future hybrid ship a candidate appearance the way it ships a candidate filter."""
    arrays: dict[str, np.ndarray]   # e.g. {"exemplar": preprocessed_z}  (arbitrary shapes)
    bbox: BoundingBox               # normalised box the template was built for
    meta: dict                      # sizes, scales, abi tag

class NNTracker(Tracker):
    def __init__(self, manifest: ModelManifest | str | Path | None = None, *,
                 backend: str = "auto", model: Model | None = None): ...
```

**Resolution (DI seam, in priority order):**
1. `model is not None` → use it directly. This is the **test seam** and the advanced-injection path;
   no backend/manifest needed. (Tests inject a deterministic stub `Model`; §12.)
2. else `manifest` + `backend` → resolve a backend via `backends/registry.py` and `model =
   backend.load(manifest)`. `manifest` may be a `ModelManifest` or a path (`load_manifest`).

**Backend selection** (`backend=`):
- A concrete name (`"onnx"`, `"rknn"`, `"mock"`) → `registry.get_backend(name)`; error if its runtime
  isn't available.
- `"auto"` (default) → first **available** of `("rknn", "onnx")` via `registry.available_backends()`.
  **`mock` is never auto-selected** (it returns canned data — useless for real tracking) and must be
  requested explicitly. If neither is available, raise an actionable error:
  `"no inference backend available; install edgecv[onnx], run on-device with [rknn], or pass
  backend='mock' for canned outputs"`.

**Lifecycle.** `close()` calls `self._model.close()` once and is idempotent; the `Tracker`
context-manager protocol (inherited) drives it. Inline trackers own no process group / SHM, so
`close()` is just model teardown (ARCHITECTURE §5.3). Backends initialise their runtime on `load`;
for RKNN that must happen in the using process (ARCHITECTURE §14.7) — fine here since standalone is
single-process, and the future hybrid will construct the tracker **inside** its worker.

**Shared helpers** the subclasses use: `self._model`, `self._model.io_spec` (for dtype/quant/layout),
`self._status`, and a `_status_from(confidence)` overridable by subclass thresholds.

## 4. Preprocessing ops (`trackers/nn/preprocess.py`)

Module-level **pure** numpy functions (importable in a `spawn`ed worker per ARCHITECTURE §7.4 — kept
module-level now so the future hybrid reuses them unchanged). Numpy reference today; RK RGA / DMA
crop-resize can swap in **behind this boundary** later with no tracker change (ARCHITECTURE §6.2,
§16). The MOSSE `_crop_patch` / `_bilinear_sample` helpers (`trackers/cf/mosse.py`) are the
prior art; promote/generalise rather than re-derive.

```python
def crop_with_context(frame, center, size_px, out_size) -> tuple[np.ndarray, CropXform]:
    """Crop a square/rect window of `size_px` (pixels) centred at `center`, edge-replicate at
    borders, resize to `out_size`. Returns the patch and the transform needed to invert."""

@dataclass(frozen=True)
class CropXform:
    center: tuple[float, float]   # crop centre in frame px
    size_px: tuple[float, float]  # crop side(s) in frame px (pre-resize)
    out_size: tuple[int, int]     # resized output (h, w)
    def to_frame(self, out_xy) -> tuple[float, float]:   # out-image px -> frame px
        ...

def letterbox(image, out_size) -> tuple[np.ndarray, LetterboxXform]:
    """Aspect-preserving resize + symmetric pad to out_size (YOLO). Xform inverts boxes."""

def to_input(patch, spec: TensorSpec, *, color: str, scale: float = 1/255, mean=None, std=None)
    -> np.ndarray:
    """Colour-convert (gray/rgb per `color`), normalise, pack to spec.layout (NCHW), cast to
    spec.dtype, applying spec.quant {scale, zero_point} when the model input is INT8."""

def class_agnostic_nms(boxes_xyxy, scores, iou_thresh) -> np.ndarray:  # returns kept indices
```

Edge-replication on crop is mandatory (targets near frame borders must not crash and must still see a
full fixed-size window) — same rule as the CF crop. All coordinate inversion goes through the
returned `*Xform` so there is **one** place norm↔pixel and crop↔frame mapping lives (ARCHITECTURE
§5.1 calls norm/pixel mixing the #1 bug source).

## 5. SiamFC (`trackers/nn/siamfc.py`)

Faithful to the SiamFC pipeline in `HonglinChu/SiamTrackers` (defaults below are its standard
AlexNet config). The model is treated as a **single graph with two inputs** `(exemplar, search)` →
one `score_map`, matching the `siamfc_generic` manifest already in ARCHITECTURE §10.1. (Caching the
exemplar embedding to skip re-embedding it each frame is a documented optimisation needing a
two-graph manifest — **deferred**, §16.)

### 5.1 Defaults (params, exposed in `__init__`)

| Param | Default | Meaning |
|---|---|---|
| `exemplar_size` | 127 | z input side (px) |
| `search_size` | 255 | x input side (px) |
| `context` | 0.5 | context margin: `p = context*(w+h)` |
| `total_stride` | 8 | backbone stride |
| `response_up` | 16 | response upsample factor |
| `scale_num` | 3 | scales searched per frame |
| `scale_step` | 1.0375 | geometric scale ratio |
| `scale_penalty` | 0.9745 | multiplies non-best-scale responses |
| `scale_lr` | 0.59 | damping on size update |
| `window_influence` | 0.176 | cosine-window blend weight |

`score_size` is read from `model.io_spec` output shape (e.g. 17 for AlexNet@255/127), not hardcoded;
upsampled response side = `score_size * response_up`.

### 5.2 Crop sizing (exemplar + search)

```
w_c = w_px + context*(w_px + h_px);  h_c = h_px + context*(w_px + h_px)
s_z = sqrt(w_c * h_c)                       # exemplar context side, frame px
s_x = s_z * search_size / exemplar_size     # search side, frame px
```

### 5.3 `init(frame, bbox)`

Crop the exemplar window (`s_z`, centred on `bbox`, edge-padded), resize to `exemplar_size`,
`to_input` per the exemplar `TensorSpec` → store as `self._template = Template(arrays={"exemplar":
z}, bbox=bbox, meta={s_z, s_x, w_px, h_px, ...})`. Precompute the cosine (Hann) window of size
`score_size*response_up`. `self._status = LOCKED`. Reuses `_status_from`.

### 5.4 `update(frame) -> TrackResult`

For each of `scale_num` scales `f = scale_step**(i - (scale_num-1)//2)`:
1. Crop search window of `s_x * f` at the **current** centre, resize to `search_size`, `to_input`.
2. `model.infer({"exemplar": z, "search": x_i})` → `score_map_i` (`score_size²`).
3. Upsample to `score_size*response_up` (bicubic ref via the promoted bilinear/`scipy` op, behind the
   ops boundary); multiply non-centre scales by `scale_penalty`.

Pick the scale with the highest peak. On its (penalised, upsampled) response, blend the cosine window:
`resp = (1-window_influence)*resp + window_influence*hann`. `argmax(resp)` → displacement in the
upsampled response → `/response_up * total_stride` = search-image px → `/ (search_size/s_x_scaled)` =
frame px → new centre. Update size with damping:
`new = (1-scale_lr) + scale_lr*scale_step**(best-centre)`, scale `s_z, s_x, w_px, h_px` by it (so the
**box w/h adapt** — the NN tracker's advantage over fixed-scale MOSSE).

**Confidence + status.** Report `confidence = peak response value`. Compute a scale-free
**PSR on the raw (pre-window) best score map** (PSR is a generic response-map statistic; reuse the
existing `psr` op) and map it via `_status_from`: `>= score_lock → LOCKED`; between → `COASTING`;
`< score_lost → LOST` (defaults tuned in §16; SiamFC raw scores are model-dependent, so PSR is the
robust gate). Output `BoundingBox` (new centre, adapted w/h), normalised, returned in a `TrackResult`
with monotonic `seq` and `timestamp` (ARCHITECTURE §5.2, §14.10).

> **Template policy:** v1 keeps the **init exemplar fixed** (classic SiamFC, no online template
> update). Online template averaging is a documented later lever, not v1 (§16).

### 5.5 No filter contract

SiamFC is **not** a `CorrelationFilterTracker`: it implements only the `Tracker` API
(`init`/`update`/`status`/`name`→`"SiamFC"`) plus `get_template()`/`set_template()` for the future
hybrid. It does **not** implement `build_filter`/`evaluate` (those are the CF transferable-filter
contract, ARCHITECTURE §6.1, and do not apply to dense trackers).

## 6. Class-agnostic YOLO (`trackers/nn/yolo.py`)

Two objects, cleanly split so the same detector serves standalone **and** the future hybrid:

### 6.1 `YoloDetector` — the reusable primitive

```python
class YoloDetector:
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 conf_thresh=0.25, iou_thresh=0.45, class_agnostic=True): ...
    def detect(self, image: np.ndarray) -> DetectorOutput: ...   # boxes (N,4) norm xywh, scores (N,)
    def close(self) -> None: ...
```

Not a `Tracker` — a detector. Reuses the same backend resolution as `NNTracker` (factor the
resolver into a small shared free function both call). `detect`:
1. `letterbox` to the model input size; `to_input` per the input `TensorSpec` (RGB/gray, `1/255`,
   layout/dtype/quant).
2. `infer` → raw output. Decode per `manifest.preprocessing["output_format"]` (default `"yolov5"`:
   `(1, N, 5+nc)` = cxcywh + objectness + class probs). **Class-agnostic:** `score = obj * max(cls)`
   (or `obj` for a 1-class generic model); the class index is discarded entirely — any object,
   regardless of label, is a candidate.
3. Threshold by `conf_thresh`; xywh→xyxy; **single-pool `class_agnostic_nms`** (`iou_thresh`); invert
   the letterbox; return normalised `DetectorOutput(boxes, scores)`. `meta` may carry raw class ids
   for debugging but they are **not** used downstream.

`detect` is **side-effect free** (no instance mutation) so the future hybrid can call it from a
worker (parallels the CF `build_filter` purity rule, ARCHITECTURE §14.5).

> **Decode is manifest-declared, not hardcoded.** Export formats differ (raw vs. NMS-baked-in,
> v5/v8 layout). The `output_format` key selects a decode function; v1 ships `"yolov5"` and a
> `"decoded"` pass-through (model already emits boxes+scores). New formats add a decoder without
> touching the tracker.

### 6.2 `YoloTracker(NNTracker)` — standalone single-object tracker

```python
class YoloTracker(NNTracker):
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 search_factor=3.0, assoc_sigma=0.5, conf_thresh=0.25,
                 iou_thresh=0.45, max_misses=5): ...
```

`name()` → `"YOLO"`. Owns a `YoloDetector` (shares the resolved `model`).

- **`init(frame, bbox)`** — store `self._box = bbox`; `self._status = LOCKED`; `self._misses = 0`.
- **`update(frame)`** — MAFiD local-detection mode (paper §3.3):
  1. Search window = previous box expanded by `search_factor` per axis, clamped + edge-padded;
     `crop_with_context` it.
  2. `det = self._detector.detect(crop)`; map each box back to **frame-normalised** via the
     `CropXform`.
  3. **Associate (single object out of N):** weight each detection
     `w_i = score_i * exp(-0.5 * (dist_i / sigma)**2)`, `dist_i` = centre distance from the previous
     box centre, `sigma = assoc_sigma * mean(prev_w, prev_h)`. Pick `argmax(w_i)`.
  4. If a winner clears `conf_thresh`: adopt its **full box** (position **and** w/h/aspect adapt —
     YOLO returns a real box, unlike MOSSE), `self._misses = 0`, `LOCKED`, `confidence = score`.
     Else `self._misses += 1`: `COASTING` (keep previous box; the next search window may widen) until
     `max_misses`, then `LOST`.
  5. Return `TrackResult` with the box, confidence, status, monotonic `seq`, `timestamp`.

> **Confidence scale caveat (ARCHITECTURE §8):** YOLO detector scores and SiamFC peaks and CF PSR are
> **not** on the same scale. Standalone status thresholds are per-tracker; any future fusion must
> calibrate before comparing across sources. Documented here so it is not rediscovered in the hybrid.

## 7. Manifests (`models/manifests/`)

Plain `ModelManifest` YAML (existing schema, `models/manifest.py`). Shipped as package data; **no
weight files** are committed (§1, §11). `artifacts.*.path` points at where a converted model *will*
live; `mock`/stub paths are unused.

```yaml
# siamfc_generic.yaml
name: siamfc_generic
task: sot_template_matching
preprocessing: { color: gray, exemplar: 127, search: 255, context: 0.5,
                 total_stride: 8, response_up: 16,
                 scale_num: 3, scale_step: 1.0375, scale_penalty: 0.9745,
                 scale_lr: 0.59, window_influence: 0.176 }
io:
  inputs:  [ { name: exemplar, shape: [1, 1, 127, 127], dtype: float32 },
             { name: search,   shape: [1, 1, 255, 255], dtype: float32 } ]
  outputs: [ { name: score_map, shape: [1, 1, 17, 17], dtype: float32 } ]
artifacts:
  onnx: { path: siamfc_generic.onnx }
  rknn: { path: siamfc_generic.rk3588.rknn, quant: int8 }
```

```yaml
# yolo_generic.yaml
name: yolo_generic
task: detection
preprocessing: { color: rgb, input: 640, scale: 0.00392156862,
                 output_format: yolov5, class_agnostic: true,
                 conf_thresh: 0.25, iou_thresh: 0.45 }
io:
  inputs:  [ { name: images,  shape: [1, 3, 640, 640], dtype: float32 } ]
  outputs: [ { name: output0, shape: [1, -1, 85],       dtype: float32 } ]  # 5 + 80; -1 = dynamic N
artifacts:
  onnx: { path: yolo_generic.onnx }
  rknn: { path: yolo_generic.rk3588.rknn, quant: int8 }
```

Tracker `__init__` defaults are **overridable**, and `preprocessing` provides the manifest-default
values; precedence: explicit `__init__` kwarg > manifest `preprocessing` > hardcoded default.

## 8. RKNN `Model` wiring (device path)

`backends/rknn/__init__.py` currently raises `NotImplementedError` in `load`, deferring to "the first
NN tracker" — that is this work. Implement `RknnModel(Model)`:
- `load`: `RKNNLite()`, `load_rknn(artifact["path"])`, `init_runtime(core_mask=...)` (NPU core from
  the placement profile, ARCHITECTURE §7.6), build `IOSpec` from the manifest I/O (RKNN lite does not
  introspect names the way ORT does).
- `infer`: `rknn.inference(inputs=[...])`; map back to the named output dict.
- `close`: `rknn.release()`.
- Created **inside** the using process only (ARCHITECTURE §14.7).

**Not exercised in CI** (no NPU on x86). Tests cover the x86 path (stub/mock/onnx); the RKNN path is
validated on-device manually. State this in the test plan so the coverage gap is intentional, not an
oversight.

## 9. Coordinate discipline (ARCHITECTURE §5.1)

`BoundingBox` stays normalised 0–1 everywhere. Pixel work happens only inside a tracker via
`to_pixels`/`from_pixels` at the `frame.shape` boundary, and every crop↔frame inversion goes through
a `CropXform`/`LetterboxXform` (§4). SiamFC and YoloTracker report **off-frame** coordinates
truthfully (no `clamp()` on tracker output — `clamp` is lossy and is for a rendering boundary only,
per the `bbox.py` docstring). YoloTracker's box w/h come from the detection; SiamFC's adapt by the
damped scale — neither preserves the init size, which is correct for scale-adaptive trackers.

## 10. Test plan (TDD, in order)

All x86, no NPU. Real-model behavioural quality is **not** asserted (weights deferred); instead a
**deterministic stub `Model`** injected via the `model=` seam (§3) drives the geometry/decode/
association logic precisely. Watch each test fail first (Iron Law).

**`test_nn_preprocess.py`** (ops first):
1. `crop_with_context` centre/size correct; border crop edge-replicates and returns full `out_size`.
2. `CropXform.to_frame` round-trips a known out-image point back to the source frame point (±<1px).
3. `letterbox` preserves aspect, pads symmetrically; `LetterboxXform` inverts a box exactly.
4. `to_input` produces `spec.layout`/`spec.dtype`; INT8 spec applies `quant` (scale/zero_point).
5. `class_agnostic_nms` suppresses overlapping boxes across (ignored) classes; keeps the top score.

**`test_nn_base.py`:**
6. `backend="mock"` resolves and loads a manifest; `close()` calls `model.close()` and is idempotent.
7. `backend="auto"` with no onnx/rknn available raises the actionable error; **never** returns mock.
8. `model=<stub>` bypasses backend/manifest entirely (DI seam).

**`test_siamfc.py`** (stub Model returns a score map with a controlled off-centre peak):
9. Instantiation + `name()=="SiamFC"`; `init` builds a `Template` with a `127×127` exemplar.
10. **Tracks translation:** stub peak offset ⇒ `update()` centre moves by the corresponding frame px
    (within ~1px through the upsample/stride math).
11. **Multi-scale selection:** stub returns the strongest peak on a non-centre scale ⇒ box w/h change
    in the expected direction (damped by `scale_lr`).
12. **Cosine-window penalty:** a far-from-centre peak is suppressed relative to a near one
    (window blend applied).
13. **Status:** low-PSR score map ⇒ `COASTING`/`LOST` per thresholds; high ⇒ `LOCKED`.
14. **Coordinate invariants:** output box normalised; off-frame targets reported truthfully (no clamp).

**`test_yolo.py`** (stub Model returns raw `(1, N, 85)`):
15. `YoloDetector.detect` decodes class-agnostically (`obj*max(cls)`), thresholds, NMS, returns
    normalised `DetectorOutput`; **`detect` does not mutate the detector** (purity).
16. Letterbox inversion: a detection at a known input-image location maps to the right frame box.
17. `YoloTracker` association: among several stub detections, picks `argmax(score * proximity)` — a
    high-score-but-far box loses to a near, decent-score box.
18. Local-crop search: a detection only inside the expanded search window is found; outside it is not.
19. Box adapts: adopted box takes the detection's w/h/aspect (not the init size).
20. Miss handling: zero detections ⇒ `COASTING`, then `LOST` after `max_misses`; box held meanwhile.

## 11. Packaging (ARCHITECTURE §12)

No new **required** deps — core stays numpy-only and these trackers import only numpy + the HAL.
Running a real model needs an inference backend extra (`edgecv[onnx]` on x86, manual
`rknn-toolkit-lite2` on device) — already defined. Manifests are package data: add
`edgecv/models/manifests` to `[tool.hatch.build.targets.wheel.force-include]` (mirroring
`models/profiles`). Optional bicubic upsample / fast resize uses the existing `[fast]` extra
(scipy/opencv op funcs) behind the ops boundary; the numpy reference works without it.

## 12. Performance posture

Correctness first (ARCHITECTURE §10 of the MOSSE spec's posture applies). The dense work is on the
backend (NPU/ORT); Python orchestration is crop/resize/decode and is single-digit-ms on edge CPUs.
Known **later** levers, not v1: cache the SiamFC exemplar embedding (two-graph manifest) to skip
re-embedding each frame; batch the 3 scale searches into one `infer` call when the model has a batch
dim; preallocate/reuse input buffers; RK RGA hardware crop-resize behind §4. v1 stays
allocation-simple and readable.

## 13. Open implementation choices (safe to pin during planning)

- **SiamFC status thresholds** (`score_lock`/`score_lost` on PSR) — pin defaults against the
  translation/noise stub tests; SiamFC raw peaks are model-dependent, so gate on PSR not raw score.
- **Upsample kernel** — bilinear (reuse MOSSE's `_bilinear_sample`) vs bicubic (closer to reference
  SiamFC). Pin bilinear for the numpy reference; defer bicubic to the `[fast]` path if it measurably
  helps localisation.
- **YOLO `search_factor` / `assoc_sigma` / `max_misses`** — defaults (3.0 / 0.5 / 5) tuned against the
  association tests; expose all three.
- **`output_format` decoders shipped in v1** — `"yolov5"` + `"decoded"`. v8/other layouts on demand.
- **Exemplar/template online update** — off in v1 (fixed init exemplar); revisit with real weights.
- **Shared backend-resolver location** — a free function in `nn/base.py` reused by `YoloDetector`
  (which is not an `NNTracker`); pin during planning.
```

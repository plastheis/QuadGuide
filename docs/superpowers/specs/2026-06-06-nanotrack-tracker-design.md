# NanoTrack (V3) NN Tracker + Conversion — Design Spec

> **Status:** design spec, ready for planning. Self-contained: written to survive a context reset.
> Adds a third dense-network (NN) tracker, **NanoTrack V3**, against `ARCHITECTURE.md` §6.2 and the
> HAL in §10, as a **standalone, inline** tracker following the exact pattern set by `SiamFC`
> (`trackers/nn/siamfc.py`) and the conversion framework (`tools/CONVERSION.md`). Reference:
> [`HonglinChu/SiamTrackers/NanoTrack`](https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack)
> — specifically the **V3** configuration (`models/config/configv3.yaml`, `mobilenetv3_small_v3`
> backbone, `DepthwiseBAN` head).

## 1. Goal and scope

Add the **NanoTrack V3** tracker plus its conversion adapter and manifest, so it can be used
**standalone today** (against a stub/real model) and dropped into a **hybrid later** with no tracker
change — identical posture to `SiamFC`.

Deliverables:

1. **`NanoTrack`** (`trackers/nn/nanotrack.py`) — `NNTracker` subclass. MobileNetV3-small-v3 backbone
   → AdjustLayer neck → DepthwiseBAN **anchor-free** head producing a classification map (`cls`) and
   a localization map (`loc`); point-based (SiamBAN/FCOS-style) box decode with scale/aspect penalty,
   cosine-window blending, and damped size update.
2. **Manifest** (`models/manifests/nanotrack.yaml`) — single-graph two-input/two-output logical
   model, matching the conversion adapter's `forward()` I/O.
3. **Conversion adapter** (`tools/convert_lib/adapters/nanotrack.py`) — vendors the V3 architecture
   faithfully, registered for the `nanotrack` manifest; exports ONNX via the generic harness
   (torch-vs-onnxruntime parity) and chains the generic RKNN step with `--rknn`.
4. **A reusable `points_grid` op** in `trackers/nn/preprocess.py` (module-level, pure numpy) for the
   anchor-free point grid.

### Agreed decisions (from clarification)

| Fork | Decision |
|---|---|
| Variant | **V3** (`mobilenetv3_small_v3` backbone, `DepthwiseBAN` head, configv3 defaults). |
| Graph topology | **Single graph**: inputs `(exemplar, search)` → outputs `(cls, loc)`. Re-embeds the exemplar each frame; mirrors `SiamFC`, **no HAL/manifest-schema changes**. |
| Conversion scope | **Adapter + manifest + ONNX export, weights deferred.** Validated by torch-vs-ort parity (random weights) and a deterministic stub `Model` in tests, matching the SiamFC "interface now, weights deferred" pattern (ARCHITECTURE §11). |
| Border padding | **Edge-replicate** (shared `crop_with_context`), as SiamFC. Reference NanoTrack mean-pads; the divergence is frame-border-only and documented (§9), revisited with real-weight validation. |

### Out of scope (deliberate)

- **Hybrid / IPC / fusion wiring.** NanoTrack is standalone inline. `Template` mirrors SiamFC's so a
  later hybrid consumes it without redesign — but no process group, frame ring, or fusion policy is
  built here (ARCHITECTURE §6.3, §7, §8).
- **NanoTrack V1 / V2.** V2 (`DepthwiseBAN`, MobileNetV3-small) and V1 are not implemented.
- **Two-graph template caching.** The exemplar-embedding cache (separate template/track graphs) is a
  documented later optimisation needing a multi-artifact manifest — deferred (§11), as for SiamFC.
- **Real default weights, INT8 RKNN conversion + calibration.** Host-only `tools/` follow-up
  (ARCHITECTURE §11); the user runs the adapter against a real `nanotrackv3.pth` when ready. This spec
  ships the adapter, manifest, and runtime path, not `.onnx`/`.rknn` files.
- **Online template update.** v1 keeps the init exemplar fixed (classic NanoTrack inference).

## 2. Module layout / files

```
edgecv/trackers/nn/
├── nanotrack.py        # NEW — NanoTrack(NNTracker): exemplar/search -> cls,loc; anchor-free decode
├── preprocess.py       # EDIT — add points_grid(stride, size) (module-level, pure)
└── __init__.py         # EDIT — export NanoTrack

edgecv/models/manifests/
└── nanotrack.yaml      # NEW — single-graph two-in/two-out manifest

tools/convert_lib/adapters/
├── nanotrack.py        # NEW — vendored MobileNetV3-small-v3 + AdjustLayer + DepthwiseBAN v3 + register
└── __init__.py         # EDIT — import nanotrack so it self-registers

tools/CONVERSION.md     # EDIT — nanotrack recipe + RKNN/parity caveats
ARCHITECTURE.md         # EDIT — note NanoTrack in §6.2 and the directory layout (update-doc-in-same-change rule)

tests/
├── test_nanotrack.py        # NEW — decode/penalty/window/status with a deterministic stub Model
├── test_convert_nanotrack.py# NEW — adapter build + ONNX export + torch-vs-ort parity (random weights)
└── test_nn_preprocess.py    # EDIT — points_grid centres + decode round-trip
```

No changes to `core/`, `runtime/`, `fusion/`, or the backend `Model`/manifest schema. Manifests are
already force-included in the wheel (`pyproject.toml`); a new YAML in the same dir needs no packaging
change.

## 3. Inference graph (single-graph)

One graph, two inputs, two outputs:

```
inputs:  exemplar [1, 3, 127, 127], search [1, 3, 255, 255]
outputs: cls [1, 2, S, S], loc [1, 4, S, S]
```

`S` is the head's spatial output side (≈15 for V3 at 127/255). It is **read from `model.io_spec`**
at construction (as SiamFC reads `score_size`), never hardcoded — the manifest value is nominal and
confirmed when the real ONNX is produced.

The adapter's `Net.forward(z, x)`:

```python
def forward(self, z, x):
    zf = self.neck(self.backbone(z))
    xf = self.neck(self.backbone(x))
    cls, loc = self.head(zf, xf)   # cls first, loc second (matches io.outputs order)
    return cls, loc
```

Re-embeds the exemplar every frame (the single-graph tradeoff, identical to SiamFC). `forward` arg
order `(z=exemplar, x=search)` **must** match `manifest.io.inputs` order — the harness feeds inputs
positionally to torch and by name to onnxruntime (`tools/CONVERSION.md`).

## 4. `points_grid` op (`trackers/nn/preprocess.py`)

Module-level pure numpy (importable in a future `spawn`ed worker, ARCHITECTURE §7.4), alongside
`crop_with_context` / `letterbox` / `to_input` / `class_agnostic_nms`.

```python
def points_grid(stride: int, size: int) -> np.ndarray:
    """Anchor-free point centres for a size×size head, in search-image pixels centred at 0.
    Returns shape (2, size*size): row 0 = x, row 1 = y. Matches NanoTrack's Point/generate_points:
    ori = -(size // 2) * stride; coord = ori + stride * index (meshgrid, row-major flatten)."""
```

The penalty/window/argmax decode itself stays **in the tracker** (mirrors SiamFC keeping its decode
local); only the reusable grid is promoted to `preprocess.py`.

## 5. NanoTrack tracker (`trackers/nn/nanotrack.py`)

`NanoTrack(NNTracker)`, `name() == "NanoTrack"`. Subclasses the existing base (backend/model
resolution, lifecycle, `Template`). Reuses `crop_with_context`, `to_input`, `resolve_pp`,
`points_grid`.

### 5.1 Defaults (configv3, exposed in `__init__`, all `resolve_pp`-overridable)

| Param | Default | Meaning |
|---|---|---|
| `exemplar_size` | 127 | z input side (px) |
| `search_size` | 255 | x input side (px) |
| `context` | 0.5 | context margin: `p = context*(w+h)` |
| `stride` | 16 | total stride / point stride |
| `base_size` | 7 | nominal head base size (informational) |
| `penalty_k` | 0.138 | scale/aspect penalty strength |
| `window_influence` | 0.455 | cosine-window blend weight |
| `size_lr` | 0.348 | damping on size update (`LR`) |
| `color` | `rgb` | 3-channel, channel order preserved (caller feeds BGR via cv2, as SiamFC) |
| `scale` | 1.0 | raw `[0,255]` pixels (no `/255`, no mean/std) |
| `score_lock` | 0.6 | cls fg-prob ≥ → LOCKED (tunable, §10) |
| `score_lost` | 0.35 | cls fg-prob < → LOST (tunable, §10) |

`S` (score size), `cls`/`loc` output names read from `model.io_spec`.

### 5.2 Crop sizing

```
p   = context * (w_px + h_px)
s_z = sqrt((w_px + p) * (h_px + p))          # exemplar context side, frame px
s_x = s_z * search_size / exemplar_size      # search side, frame px
scale_z = exemplar_size / s_z                # frame px -> exemplar-resolution scale
```

### 5.3 `init(frame, bbox)`

Crop exemplar window (`s_z`, centred on `bbox`, edge-replicate), resize 127, `to_input` per the
exemplar `TensorSpec` → `self._template = Template(arrays={"exemplar": z}, bbox=bbox, meta={"s_z":
s_z})`. Build (once) the **non-normalised** Hann window `outer(hanning(S), hanning(S)).flatten()` and
`points = points_grid(stride, S)`. `self._status = LOCKED`; `self._box = bbox`; `self._seq = 0`.

### 5.4 `update(frame) -> TrackResult`

1. Crop search window of `s_x` at the **current** centre, resize 255, `to_input` → `x`.
2. `infer({"exemplar": z, "search": x})` → `{cls, loc}`.
3. **Score:** `score = softmax(cls, axis=channel)[fg]` flattened to `(S*S,)` (fg = channel 1).
4. **Box decode** (`loc` → `(4, S*S)`, distances `l,t,r,b`):
   ```
   x1 = px - l;  y1 = py - t;  x2 = px + r;  y2 = py + b      # px,py from points_grid
   cx,cy,w,h = corner2center(x1,y1,x2,y2)                      # search-px space, centred at 0
   ```
5. **Penalty** (`sz(w,h) = sqrt((w+pad)(h+pad))`, `pad = (w+h)/2`; `change(r)=max(r,1/r)`):
   ```
   s_c = change( sz(w, h) / sz(self.w*scale_z, self.h*scale_z) )
   r_c = change( (self.w/self.h) / (w/h) )
   penalty = exp(-(r_c * s_c - 1) * penalty_k)
   pscore  = penalty * score
   ```
6. **Window blend:** `pscore = (1 - window_influence) * pscore + window_influence * hann`.
7. **Select:** `best = argmax(pscore)`.
8. **Update:**
   ```
   lr        = penalty[best] * score[best] * size_lr
   new_cx    = prev_cx + cx[best] / scale_z          # cx centred at 0 -> displacement
   new_cy    = prev_cy + cy[best] / scale_z
   new_w     = self.w * (1 - lr) + (w[best] / scale_z) * lr
   new_h     = self.h * (1 - lr) + (h[best] / scale_z) * lr
   ```
   Build `PixelBox(new_cx - new_w/2, new_cy - new_h/2, new_w, new_h)` → `BoundingBox.from_pixels`
   (normalised, **no clamp** — off-frame reported truthfully, ARCHITECTURE §5.1 / nn-design §9).
9. **Confidence + status:** `confidence = float(score[best])` (cls fg prob, 0–1); status via
   `_status_from(score[best])` (`>= score_lock → LOCKED`; between → `COASTING`; `< score_lost →
   LOST`). Return `TrackResult` with monotonic `seq`, `timestamp` (ARCHITECTURE §5.2, §14.10).

> **Confidence scale caveat (ARCHITECTURE §8):** NanoTrack's cls fg prob, SiamFC peaks, YOLO scores,
> and CF PSR are **not** on the same scale. Standalone thresholds are per-tracker; any future fusion
> must calibrate before comparing across sources.

### 5.5 No filter contract

NanoTrack implements only the `Tracker` API (`init`/`update`/`status`/`name`→`"NanoTrack"`) plus
`get_template()`/`set_template(template, search_box=None)` for the future hybrid (same shape as
SiamFC). It does **not** implement `build_filter`/`evaluate` (CF-only contract, ARCHITECTURE §6.1).

## 6. Manifest (`models/manifests/nanotrack.yaml`)

```yaml
name: nanotrack
task: sot_template_matching
# NanoTrack V3 (HonglinChu/SiamTrackers). mobilenetv3_small_v3 + AdjustLayer + DepthwiseBAN.
# Trained on raw [0,255] crops (no /255, no mean/std); cv2/BGR channel order preserved by to_input.
preprocessing:
  color: rgb            # 3-channel, channel order preserved (caller feeds BGR via cv2)
  scale: 1.0            # raw [0,255]
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
    - { name: cls, shape: [1, 2, 15, 15], dtype: float32 }   # S nominal; read from io_spec
    - { name: loc, shape: [1, 4, 15, 15], dtype: float32 }
artifacts:
  onnx: { path: nanotrack.onnx }
  rknn: { path: nanotrack.rk3588.rknn, quant: int8 }
```

Precedence (ARCHITECTURE §10.1): explicit `__init__` kwarg > manifest `preprocessing` > hardcoded
default — via the existing `resolve_pp`.

## 7. Conversion adapter (`tools/convert_lib/adapters/nanotrack.py`)

Vendors the V3 architecture so the upstream repo need not be installed (as `adapters/siamfc.py`
vendors AlexNetV1). Three vendored components, matching the reference state_dict keys:

- **`mobilenetv3_small_v3`** — 9-block `MobileNetV3` (`InvertedResidual` with `SELayer`, `h_swish`),
  backbone-only forward returning the layer-4 feature (96 ch). `features.*` module names preserved.
- **`AdjustLayer`** (neck, 96→96) — `Conv2d + BatchNorm2d`, with the centre-crop of the **template**
  feature (reference crops `zf` to the head's nominal size when large). Module/key names preserved.
- **`DepthwiseBAN`** (head, 96→96) — `xcorr_depthwise` + `xcorr_pixelwise`, fused via `down_cls` /
  `down_reg` 1×1 conv, then `cls_tower`/`cls_logits` (→2 ch) and `bbox_tower`/`bbox_pred` (→4 ch, with
  the reference's `exp` on the bbox output). `forward(z_f, x_f) -> (cls, loc)`.

```python
def build(checkpoint: str) -> Net:
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net = Net()
    net.load_state_dict(sd, strict=True)   # strict=True -> key/shape mismatch fails loudly
    net.eval()
    return net

register(Adapter(name="nanotrack", build=build, dynamic_axes={...}))  # cls/loc spatial dims dynamic
```

Import in `adapters/__init__.py` so it self-registers. The dispatcher exports to
`resolve_artifact_path(manifest.artifacts.onnx.path)`; `--rknn` chains the generic stage 3 unchanged.

**Vendoring fidelity rule:** `load_state_dict(strict=True)`. If state_dict keys/shapes don't match
the published `nanotrackv3.pth`, the build fails loudly — fix the vendored module names, never relax
to `strict=False` (CONVERSION.md).

## 8. CONVERSION.md additions

A `nanotrack` quick-start recipe plus a caveats note:

- **RKNN/parity caveat (untested in CI):** the head's `xcorr_depthwise` uses a **data-dependent conv
  kernel** (the exemplar feature is the conv weight) and `xcorr_pixelwise` uses **matmul**. Both
  export to ONNX and pass torch-vs-ort parity, but RKNN operator support for a dynamic-weight grouped
  conv is untested on-device — validate manually, and if unsupported, fall back to the
  pixelwise-only or a fixed-template (two-graph) export. Mirrors the existing SiamFC/YOLO caveats.
- **Numerics:** `to_input` feeds raw `[0,255]` (`scale=1.0`); `rknn_convert` must use `mean=0,
  std=1` to match (same passthrough note as YOLO).

## 9. Border padding fidelity (documented divergence)

Reference NanoTrack pads out-of-frame crop regions with the frame's **per-channel mean**
(`avg_chans`). edgecv's shared `crop_with_context` **edge-replicates** (as SiamFC). v1 keeps
edge-replicate for consistency and simplicity; the difference is confined to targets near frame
borders. Revisit with real-weight validation; if it matters, add a `pad="mean"` mode to
`crop_with_context` behind the ops boundary (no tracker change). Recorded so it is not rediscovered as
a bug.

## 10. Test plan (TDD, in order)

All x86, no NPU. Real-model quality is **not** asserted (weights deferred); a **deterministic stub
`Model`** injected via the `model=` seam drives the decode/penalty/association logic precisely. Watch
each test fail first (Iron Law).

**`test_nn_preprocess.py`** (edit):
1. `points_grid(stride, size)` returns `(2, size*size)`, centred at 0, spacing `stride`, row-major.
2. Decode round-trip: with zero `loc` distances, decoded centres equal the points (×`1/scale_z` maps
   a known point to a known frame displacement within <1px).

**`test_nanotrack.py`** (stub `Model` returns `cls`/`loc` with a controlled peak):
3. Instantiation + `name() == "NanoTrack"`; `init` builds a `Template` with a `127×127` exemplar and
   reads `S`/`cls`/`loc` from a stub `io_spec`.
4. **Tracks translation:** stub peak at an off-centre point ⇒ `update()` centre moves by the
   corresponding frame px (within ~1px through the stride/scale_z math).
5. **Size adaptation:** stub `loc` encoding a larger box ⇒ box w/h grow in the expected direction,
   damped by `size_lr`.
6. **Penalty/window:** a large scale/aspect jump is suppressed (`penalty` < 1); a far-from-centre
   peak is suppressed by the cosine window relative to a near one.
7. **Status:** low cls fg prob ⇒ `COASTING`/`LOST` per thresholds; high ⇒ `LOCKED`.
8. **Coordinate invariants:** output box normalised; off-frame targets reported truthfully (no clamp).
9. `get_template`/`set_template` round-trip (template + optional `search_box`).

**`test_convert_nanotrack.py`** (mirrors `test_convert_siamfc.py`):
10. `build`-equivalent `Net()` (random weights), generic harness exports ONNX; `onnx.checker` passes;
    torch-vs-onnxruntime parity `max|Δ| < 1e-3` on both `cls` and `loc`; output shapes
    `[1,2,S,S]`/`[1,4,S,S]`. (Adapter registration is covered by the existing registry test pattern.)

The RKNN device path is **not** exercised in CI (no NPU on x86); coverage gap is intentional (§8).

## 11. Performance posture & later levers

Correctness first. Dense work is on the backend; Python orchestration is crop/resize/decode,
single-digit-ms on edge CPUs. **Later, not v1:** two-graph manifest to cache the exemplar embedding
(skip re-embedding each frame — the single-graph cost); preallocate/reuse input buffers; RK RGA
hardware crop-resize behind the `preprocess.py` boundary; mean-pad crop mode (§9). v1 stays
allocation-simple and readable, matching SiamFC.

## 12. Open implementation choices (safe to pin during planning)

- **`score_lock` / `score_lost`** on cls fg prob — pin defaults (0.6 / 0.35) against the status stub
  tests; tune with real weights.
- **`base_size`** is informational in single-graph (S comes from `io_spec`); kept in the manifest for
  parity with the reference config and a future two-graph export.
- **`AdjustLayer` template centre-crop size** — match the reference exactly during vendoring; verified
  by `strict=True` state_dict load against `nanotrackv3.pth`.
- **`dynamic_axes`** — declare cls/loc spatial dims dynamic so a different input size re-exports
  cleanly; nominal export at 127/255.
```

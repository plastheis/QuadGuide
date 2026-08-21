# YOLO26 conversion + NN manifest-preprocessing wiring — Design Spec

> **Status:** design spec, ready for planning. Self-contained: written to survive a context reset.
> Builds on the implemented NN trackers (`docs/superpowers/specs/2026-06-04-nn-trackers-design.md`)
> and the conversion framework (`docs/superpowers/specs/2026-06-05-conversion-framework-design.md`).
> Targets `ARCHITECTURE.md` §6.2 (NN trackers), §10–§10.1 (HAL + manifest), §11 (host-only
> conversion). References: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) and
> [end-to-end detection](https://docs.ultralytics.com/guides/end2end-detection).

## 1. Goal and scope

The `YoloDetector`/`YoloTracker` and `SiamFC` trackers are already implemented and green (Tasks 8–9
of the nn-trackers plan). Two gaps remain before real weights can land:

1. **NN trackers ignore `manifest.preprocessing`.** The precedence helper (`resolve_pp`) exists and is
   unit-tested, but **YOLO wires none of it** and **SiamFC wires only `color`+`scale`**. A manifest's
   `conf_thresh`/`input`/etc. silently have **zero effect** today — the constructor defaults always
   win. This bites the moment a real model ships with non-default preprocessing.
2. **No conversion path for YOLO.** `convert.py` only handles torch-`nn.Module` adapters (how SiamFC
   converts). YOLO ships its own ONNX exporter, so it has no first-class `convert.py` entry.

**Deliverables**

1. **Wire `manifest.preprocessing` precedence** through `YoloDetector`/`YoloTracker` and **finish**
   `SiamFC`, using the existing `UNSET` + `resolve_pp` mechanism (explicit kwarg > manifest > default).
2. **A `"yolov8"` decoder** in `YoloDetector` for the modern Ultralytics one-to-many output layout
   `(1, 4+nc, N)` (anchor-free, no objectness) — the format YOLO26 produces for our backends.
3. **`yolo26n` (shipped default) + `yolo26s` manifests**, replacing `yolo_generic.yaml`.
4. **A YOLO conversion adapter** wired into `tools/convert.py` via a new optional `Adapter.export`
   hook: it drives the Ultralytics exporter (one-to-many head) to ONNX, then chains the existing
   ONNX→RKNN INT8 step. `ultralytics` joins the `[dev]` extra.

### Decisions locked (from clarification)

| Fork | Decision |
|---|---|
| Model | **YOLO26n** shipped default, **YOLO26s** accuracy variant. COCO-80, class-agnostic use. |
| Export head | **One-to-many (`end2end=False`)** for both backends — see §3. |
| Decoder | New **`"yolov8"`** format (transposed `(1,4+nc,N)`, anchor-free, no objectness). |
| Manifests | **Retire `yolo_generic.yaml`**; add explicit `yolo26n.yaml` + `yolo26s.yaml`. |
| Conversion | **Add a YOLO adapter to `convert.py`** (upstream-exporter hook), not just docs. |
| PP wiring | **YOLO + finish SiamFC.** |

### Out of scope (deliberate)

- **Real weight quality / accuracy benchmarking.** Logic is validated against deterministic stub
  models and a mocked exporter; obtaining `.pt`/`.onnx`/`.rknn` blobs and on-device validation is
  manual (ARCHITECTURE §11). No weight files are committed.
- **The NMS-free end-to-end head.** Unavailable on RKNN (§3); not pursued.
- **YOLOv5 retirement.** The existing `"yolov5"` and `"decoded"` decoder branches stay for
  back-compat; this only **adds** `"yolov8"`.
- **Hybrid / fusion wiring.** `YoloDetector.detect` already emits `DetectorOutput` for a future
  hybrid; no process group is built here.

## 2. Module layout / files

```
edgecv/trackers/nn/
├── yolo.py            # MODIFY — UNSET+resolve_pp wiring; add "yolov8" decode branch
├── siamfc.py          # MODIFY — finish resolve_pp wiring (all preprocessing keys)
└── base.py            # (reuse as-is: UNSET, resolve_pp, manifest_preprocessing)

edgecv/models/manifests/
├── yolo26n.yaml       # CREATE (shipped default)
├── yolo26s.yaml       # CREATE
└── yolo_generic.yaml  # DELETE

tools/convert_lib/
├── registry.py        # MODIFY — add optional Adapter.export hook
├── __init__.py        # MODIFY — run() uses adapter.export when present (skip torch harness)
└── adapters/
    ├── yolo.py        # CREATE — Ultralytics exporter adapter (one-to-many ONNX)
    └── __init__.py    # MODIFY — import adapters.yolo so it self-registers

pyproject.toml         # MODIFY — add ultralytics to [dev]
tools/CONVERSION.md    # MODIFY — document the YOLO26 recipe via convert.py

tests/
├── test_yolo.py          # MODIFY — add "yolov8" decode + PP-precedence cases
├── test_siamfc.py        # MODIFY — assert manifest PP now reaches SiamFC
├── test_pp_precedence.py # MODIFY — repoint yolo_generic -> yolo26n
├── test_manifests_nn.py  # MODIFY — yolo26n/yolo26s load; assert output_format yolov8
├── test_nn_base.py       # MODIFY — repoint manifest path references
└── test_convert_yolo.py  # CREATE — adapter registered; export hook + RKNN chain (ultralytics mocked)
```

No changes to `core/`, `runtime/`, `fusion/`, or the backend ABCs.

## 3. The export-head decision (load-bearing)

YOLO26 has a dual head. Its **default** ONNX export is **NMS-free end-to-end**: `(1, 300, 6)` =
`[x1, y1, x2, y2, conf, class_id]` (xyxy, deduplicated, threshold-only). But **RKNN is explicitly a
format that falls back to the one-to-many head** `(1, nc+4, 8400)` because the end-to-end ops are
unsupported there ([end2end docs](https://docs.ultralytics.com/guides/end2end-detection)).

If we exported the default head, the **onnx backend (x86/CI)** and the **rknn backend (device)** would
emit **different tensors for the same logical model**, forcing the tracker to branch decode by
backend — a direct violation of the HAL contract (ARCHITECTURE §10: the tracker depends on the
manifest, not the backend).

**Decision: export with `end2end=False` (one-to-many head) for both backends.** Both then emit
`(1, 4+nc, N)`; one decoder, one NMS path (our existing `class_agnostic_nms`), full parity. The
NMS-free path is simply not used.

## 4. The `"yolov8"` decoder (`yolo.py`)

The one-to-many output is the modern Ultralytics layout, shared by v8/v9/v11/**v26**:
`(1, 4+nc, N)`, **transposed** relative to v5, **anchor-free**, **no objectness**. New branch in
`YoloDetector.detect`, selected by `output_format == "yolov8"`:

```
raw = infer(...)                      # (1, 4+nc, N)
preds = raw[0].T                      # (N, 4+nc)
xywh = preds[:, :4]                   # box in input (letterbox) px, centre xywh
score = preds[:, 4:].max(axis=1)      # class-agnostic: max over class cols; NO objectness
```

then the **existing** shared tail runs unchanged: threshold by `conf_thresh` → centre-xywh→xyxy →
`class_agnostic_nms(iou_thresh)` → invert letterbox via `LetterboxXform` → normalised
`DetectorOutput`. The `"yolov5"` (with objectness) and `"decoded"` branches are untouched.

Edge cases to preserve from the current code: empty `preds` and post-threshold-empty both return the
empty `DetectorOutput`. `detect` stays **side-effect free** (purity test still holds).

> A 1-class generic export (`nc==1`) still works: `max` over a single class col = that score.

## 5. Manifests (`models/manifests/`)

Replace `yolo_generic.yaml` with two explicit manifests. COCO-80, so `4 + 80 = 84`.

```yaml
# yolo26n.yaml  (shipped default)
name: yolo26n
task: detection
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

`yolo26s.yaml` is identical except `name: yolo26s` and `artifacts.*.path` (`yolo26s.onnx`,
`yolo26s.rk3588.rknn`). Output layout/preprocessing are scale-independent. No weight files committed
(§1). Manifests are already packaged via the `force-include` glob — no `pyproject.toml` change for
packaging.

## 6. Manifest-preprocessing wiring (`yolo.py`, `siamfc.py`)

Use the established pattern (`base.py`: `UNSET`, `resolve_pp`, `manifest_preprocessing`; already
consumed by SiamFC for `color`/`scale`). `NNTracker.__init__` already populates
`self._preprocessing`. `YoloDetector` is **not** an `NNTracker`, so it computes its own preprocessing
dict from the manifest with `manifest_preprocessing(manifest)` (a no-op `{}` when a `model=` is
injected).

**`YoloDetector.__init__`** — switch each preprocessing kwarg default to `UNSET` and resolve:

| kwarg | manifest key | hardcoded default |
|---|---|---|
| `input_size` | `input` | 640 |
| `color` | `color` | `"rgb"` |
| `scale` | `scale` | `1/255` |
| `output_format` | `output_format` | `"yolov8"` |
| `conf_thresh` | `conf_thresh` | 0.25 |
| `iou_thresh` | `iou_thresh` | 0.45 |

Note the **`input` → `input_size` key rename**: the manifest key is `input`, the kwarg is
`input_size`. `resolve_pp(input_size, pp, "input", 640)`.

**`YoloTracker.__init__`** passes its resolved preprocessing through to the `YoloDetector` it owns.
Tracker-only knobs — `search_factor`, `assoc_sigma`, `max_misses` — are **not** manifest-driven and
keep plain defaults (they describe the association policy, not the model's preprocessing).

**`SiamFC.__init__`** — extend the existing partial wiring (`color`, `scale`) to every preprocessing
key the `siamfc_generic` manifest declares: `exemplar`→`exemplar_size`, `search`→`search_size`,
`context`, `total_stride`, `response_up`, `scale_num`, `scale_step`, `scale_penalty`, `scale_lr`,
`window_influence`. Each becomes `UNSET` + `resolve_pp` with its current default. `score_lock`/
`score_lost` are status thresholds, not manifest preprocessing — leave plain.

Precedence everywhere: **explicit kwarg > manifest `preprocessing` > hardcoded default** (the
documented contract, ARCHITECTURE §10.1).

## 7. Conversion adapter (`tools/`)

The existing `run()` (convert_lib/`__init__.py`) does: `adapter.build(ckpt) -> nn.Module` →
`export_and_validate` (torch.onnx.export + parity). YOLO has no torch module we own — Ultralytics
owns the architecture and the exporter. So extend the `Adapter` abstraction with an **optional
export hook**:

```python
@dataclass
class Adapter:
    name: str
    build: Callable[[str], Any] | None = None              # torch path (SiamFC)
    export: Callable[..., str] | None = None               # upstream-exporter path (YOLO)
    dynamic_axes: dict | None = None
```

`run()` branch: **if `adapter.export` is set**, call it with `(checkpoint, onnx_out, manifest)` to
write the ONNX directly and **skip** `build`+`export_and_validate`; validate with `onnx.checker` +
an output rank/shape check against the manifest (there is no separately-ownable torch reference to do
full numerical parity against — documented as an intentional difference from the SiamFC path).
Otherwise the existing torch path runs unchanged. The `--rknn` chain to `rknn_convert` is shared by
both paths.

**`adapters/yolo.py`** registers one adapter per scale and drives Ultralytics **in-process**:

```python
def _export(checkpoint, onnx_out, manifest):
    from ultralytics import YOLO                  # lazy; only needed at convert time
    imgsz = manifest.inputs[0]["shape"][-1]       # 640 from the manifest
    path = YOLO(checkpoint).export(format="onnx", nms=False, imgsz=imgsz, opset=13)
    # move/rename the emitted file to onnx_out (resolved manifest artifact path)
    ...
    return onnx_out

register(Adapter(name="yolo26n", export=_export))
register(Adapter(name="yolo26s", export=_export))
```

`nms=False` selects the one-to-many head (§3). `ultralytics` is added to the `[dev]` extra. Then:

```bash
python tools/convert.py --model yolo26n --checkpoint models/yolo26n.pt          # -> models/yolo26n.onnx
python tools/convert.py --model yolo26n --checkpoint models/yolo26n.pt --rknn --calib calib/
```

> **Device-path numerics caveat (RKNN, untested in CI):** the tracker's `to_input` applies
> `scale=1/255` on the host. `rknn_convert` currently configures `mean=0, std=1` (raw-pixel
> passthrough), which matches a model fed already-scaled input. If on-device validation shows a
> mismatch, set the RKNN `std_values` to `255` (let the NPU do the divide) **or** feed raw pixels and
> drop the host scale — must stay consistent with `to_input`. This is a manual on-device step; flag
> it in CONVERSION.md, do not block CI on it.

## 8. Test plan (TDD, x86 only, no NPU)

Real-model quality is **not** asserted (weights deferred). Deterministic stub `Model`s drive the
decoder; the Ultralytics exporter is **mocked**. Watch each test fail first (Iron Law).

**`test_yolo.py` (decoder + wiring):**
1. `"yolov8"` decode: stub emits `(1, 4+nc, N)`; assert transpose, `score = max(cls)` (no objectness
   term), correct normalised `DetectorOutput`. Mirror the existing `_raw` helper for the new layout.
2. `"yolov8"` thresholding + class-agnostic NMS behave as the `"yolov5"` cases do.
3. `detect` purity holds under `"yolov8"` (two calls, equal output).
4. **PP precedence (detector):** construct `YoloDetector(manifest=yolo26n, model=stub)` →
   `conf_thresh`/`input_size`/`output_format` come from the manifest; an explicit kwarg overrides it;
   absent-from-manifest falls to the hardcoded default. (Manifest passed for its `preprocessing`;
   `model=` injected so no backend is needed.)
5. **PP precedence (tracker):** same, surfaced through `YoloTracker`.
6. Existing `"yolov5"` tests still pass (no regression).

**`test_siamfc.py`:**
7. A manifest preprocessing value (e.g. `context` or `window_influence`) now **reaches** SiamFC and
   changes behaviour vs the hardcoded default; explicit kwarg still overrides.

**`test_manifests_nn.py`:**
8. `yolo26n.yaml` and `yolo26s.yaml` load; `output_format == "yolov8"`, output shape `[1, 84, -1]`;
   `yolo_generic.yaml` is gone (update/replace the old assertions).

**`test_pp_precedence.py`, `test_nn_base.py`:** repoint `yolo_generic` references to `yolo26n`.

**`test_convert_yolo.py` (NEW, ultralytics mocked):**
9. `yolo26n`/`yolo26s` adapters are registered with an `export` hook (no `build`).
10. `run("yolo26n", ckpt)` calls the export hook with `nms=False` + `imgsz=640`, writes to the
    manifest's resolved ONNX artifact path, and runs `onnx.checker` (checker mocked or fed a tiny
    valid ONNX) — `build`/torch harness are **not** invoked.
11. `run(..., rknn=True)` chains `rknn_convert` with the manifest's rknn path + input names
    (`rknn_convert` mocked; assert args).
12. The existing SiamFC torch path still routes through `build`+harness (no regression from the
    `run()` branch).

**Coverage gap (intentional, stated):** the real Ultralytics export and the RKNN build run only on a
host with those toolkits; CI exercises the x86 stub/mock path and the branching logic, never a real
YOLO26 graph or `.rknn`.

## 9. Packaging (ARCHITECTURE §12)

Core stays numpy-only. `ultralytics` is **host-only**, added to `[dev]` (alongside `torch`, `onnx`).
No new runtime dependency. Manifests ship via the existing `force-include` glob.

## 10. Coordinate / contract discipline (unchanged)

`DetectorOutput.boxes` stays `(N,4)` normalised xywh top-left. All letterbox inversion goes through
`LetterboxXform` (ARCHITECTURE §5.1). `detect` stays pure. No `clamp()` on output (ARCHITECTURE §9).
The `"yolov8"` branch only changes how raw rows become `(xywh, score)`; the normalised-output
contract and the shared NMS/inversion tail are identical to `"yolov5"`.

## 11. Open implementation choices (safe to pin during planning)

- **Ultralytics export filename handling.** `.export()` returns the written path (next to the `.pt`);
  the adapter must move/rename it to the manifest's resolved artifact path. Pin the exact move logic
  in planning.
- **opset.** Pin `opset=13` (matches the SiamFC harness default and RKNN-toolkit2 compatibility);
  bump only if a YOLO26 op requires it.
- **`onnx.checker` in the export-hook validation.** Pin whether validation also asserts the output
  tensor **rank** (3) and trailing channel (`4+nc`) against the manifest, or just runs the checker.
  Recommend the shape assertion — it catches a wrong-head export (`(1,300,6)`) immediately.
- **Single shared `_export` vs per-scale closures.** One `_export` reused by both `register(...)`
  calls (scale is implied by the checkpoint + manifest). Pin in planning.

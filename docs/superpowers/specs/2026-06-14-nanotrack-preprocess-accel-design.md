# NanoTrack Preprocess Acceleration — Design Spec

> **Status:** design spec, ready for planning. Self-contained: written to survive a context reset.
> Removes the dominant per-frame cost in the NanoTrack (and SiamFC) inference loop on RK3588: the
> pure-numpy bilinear crop+resize in `edgecv/trackers/nn/preprocess.py`. Targets `ARCHITECTURE.md`
> §6.2 ("Preprocessing … is a candidate for hardware acceleration … start with numpy and swap in
> fast paths later **without changing the tracker**") and §16. Consumed by QuadGuide
> (`~/QuadGuide`) via `quadguide.perception.edgecv_adapter:EdgeCVTracker`.

## 1. Problem and evidence

NanoTrack on RK3588 is **CPU-bound on preprocessing, not NPU-bound**. Measured on-device (2026-06-14):

| Stage | A76 big core | A55 little core |
|---|---|---|
| `crop_with_context` (255×255 bilinear resize, `preprocess.py:97`) | **11.0 ms** | **51.8 ms** |
| `to_input` (transpose + int8 quant, `preprocess.py:126`) | 0.8 ms | — |
| backbone INT8 + head FP16 (two 1 MB RKNN models) | a few ms each | — |

The two RKNN models run in a handful of ms on the 6-TOPS NPU; the numpy resize per `update()`
dominates. Root cause: `resize_bilinear`/`_sample_clamped` (`preprocess.py:17-42`) build a 255×255
meshgrid and do four fancy-indexed gathers into the full 1080p frame in interpreted numpy, every
frame.

**QuadGuide baseline** (last trace `~/QuadGuide/quadguide-trace/20260611-084906/tracker_NanoTrack.jsonl`,
1300 `lat` records): tracker **stage latency p50 = 20.4 ms, p95 = 23.3 ms, max = 54 ms**; loop period
p50 = 32.6 ms (~30 FPS). The ~11 ms resize is ≈ half the stage time, and the worker is already on big
cores (else stage would exceed the 52 ms little-core figure). So the resize is the lever; core
placement is not.

## 2. Goal and scope

Cut the crop+resize from ~11 ms to <1 ms by adding an **optional cv2-accelerated fast path** behind
the existing preprocessing boundary, with the current numpy implementation retained as a bit-faithful
fallback. **No tracker, manifest, or HAL change** — `crop_with_context` keeps its signature and
`CropXform` inversion semantics exactly.

Deliverables:

1. A resize/crop **dispatcher** in `edgecv/trackers/nn/preprocess.py` that uses `cv2.warpAffine`
   (INTER_LINEAR + BORDER_REPLICATE) when cv2 is importable, else the existing numpy path.
2. Bit-faithful sampling parity: the cv2 affine maps output pixel `(ox, oy)` to frame coord
   `(cx − sw/2) + (ox+0.5)/ow·sw` (same half-pixel convention as the numpy grid), so `CropXform.to_frame`
   coordinate inversion (`preprocess.py:51`) remains valid unchanged.
3. cv2 listed as an optional accelerator (already in the `fast` extra, `pyproject.toml:21`); detection
   is lazy and import-guarded so EdgeCV CI (no cv2) and the device both work.
4. A parity test (cv2 path vs numpy path, max abs diff within interpolation tolerance) and a
   benchmark assertion that the fast path is materially faster when cv2 is present.

### Agreed decisions

| Fork | Decision |
|---|---|
| Fast-resize backend | **cv2 first** (`cv2.warpAffine`, INTER_LINEAR, BORDER_REPLICATE). Hard dep of QuadGuide (`opencv-python>=4.10`), already an EdgeCV `fast` extra. ~50–100× faster than numpy, SIMD on CPU. |
| Hardware RGA | **Deferred.** `librga.so.2` is on the device but has no Python binding. Dispatcher is structured so an RGA backend can drop in later (ARCHITECTURE §16). |
| Sampling semantics | **Preserved exactly.** cv2 affine constructed to match the numpy half-pixel grid; edge-replicate ↔ `BORDER_REPLICATE`, bilinear ↔ `INTER_LINEAR`. `CropXform` math untouched. |
| Layout transposes (NCHW↔NHWC at model boundaries) | **Out of scope.** Real but marginal (~µs on a 96×16×16 tensor) and invasive (manifest layout + backend `data_format` + tracker crop axes + possible recompile). Not worth the risk vs the 11 ms resize. |
| FP16 head | **Out of scope.** ~2× the NPU cost of an INT8 head, but a deliberate accuracy decision (INT8 head fails to lock; see memory `nanotrack-int8-head-broken`). Separate effort if revisited. |
| Worker core placement | **Verify only.** QuadGuide already runs the tracker on big cores per the baseline; no change proposed here. |

### Out of scope (deliberate)

- RGA / DMA hardware crop-resize implementation (design for it; don't build it — no binding).
- Any change to `to_input`, quantization, the RKNN backend, manifests, or model artifacts.
- The QuadGuide free-running-vs-gated loop question (separate, see `scripts/diagnose_latency.py`).

## 3. Design

### 3.1 Dispatcher

Add a module-level, import-guarded cv2 probe and a single private entry point used by
`crop_with_context` (and, optionally, `letterbox`/`resize_bilinear` for YOLO/SiamFC):

```python
try:
    import cv2 as _cv2
except Exception:               # not installed (EdgeCV CI) → numpy fallback
    _cv2 = None

def _crop_resize(frame, center, size_px, out_size):
    """Edge-replicate bilinear crop+resize. cv2 fast path, numpy fallback.
    Both honour the SAME half-pixel sampling grid so CropXform inversion holds."""
```

cv2 path — separable affine (frame = a·ox + b, c·oy + d):

```
a = sw/ow;  b = (cx - sw/2) + 0.5*a
c = sh/oh;  d = (cy - sh/2) + 0.5*c
M = [[a, 0, b], [0, c, d]]                       # dst→src (WARP_INVERSE_MAP)
out = cv2.warpAffine(frame, M, (ow, oh),
                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                     borderMode=cv2.BORDER_REPLICATE)
```

This reproduces `crop_with_context`'s grid exactly: at output column `ox`, src x = `a·ox + b =
(cx − sw/2) + (ox+0.5)/ow·sw`, matching `preprocess.py:108`. Grayscale (2-D frame) and 3-channel both
supported by warpAffine.

`crop_with_context` becomes a thin wrapper: call `_crop_resize`, return the patch + the **unchanged**
`CropXform(center, (sh, sw), (oh, ow))`. The numpy branch is the current body verbatim.

### 3.2 Why inversion stays correct

`CropXform.to_frame` (`preprocess.py:51-60`) derives frame coords from output indices using the same
`(ox+0.5)/ow·sw` formula independently of how the patch was sampled. As long as the sampler honours
that grid (both branches do), every downstream coordinate (box decode in `nanotrack.py:234-241`) is
unaffected. This is the invariant the parity test pins.

## 4. Validation

1. **Parity test** (`tests/test_nn_preprocess.py`): on a synthetic frame, `_crop_resize` cv2 vs numpy
   max-abs diff ≤ small tol (e.g. 1.0 on a 0–255 scale; interpolation rounding only). Skipped if cv2
   absent. A no-cv2 path test asserts the numpy fallback still runs.
2. **Tracker-level**: existing `tests/test_nanotrack.py` decode/lifecycle tests pass unchanged
   (semantics preserved). RKNN tests remain skipped off-device.
3. **On-device benchmark**: micro-bench `_crop_resize` cv2 vs numpy on the RK3588 (expect <1 ms vs
   ~11 ms big core).
4. **End-to-end (QuadGuide)**: re-run the webcam inference loop, dump a new trace under
   `~/QuadGuide/quadguide-trace/`, and compare tracker **stage** latency against the baseline
   (`20260611-084906`: p50 20.4 ms / p95 23.3 ms). Target: stage p50 ≈ 9–11 ms (≈ the 11 ms resize
   removed).

## 5. QuadGuide integration notes (verified good, with one minor item)

`quadguide/perception/edgecv_adapter.py` usage is correct: lazy EdgeCV imports, built in the forked
child (RKNN context in the using process), BGR→RGB conversion, confidence clamp, structural protocol
mapping. **No change required for this spec.** One minor, optional follow-up (not in scope): the
adapter flips the **whole** 1080p frame BGR→RGB with `np.ascontiguousarray(frame[..., ::-1])` every
`update()` (`edgecv_adapter.py:139-142`); cropping first then converting the 255×255 patch would shave
a ~1–2 ms full-frame copy. Tracked as a separate optimization.

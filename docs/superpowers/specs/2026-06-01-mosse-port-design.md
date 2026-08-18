# MOSSE Port — Design Spec

> **Status:** design spec, ready for planning. Self-contained: written to survive a context reset.
> Implements the first concrete CF tracker against the contract in `ARCHITECTURE.md` §5–§6.1 and
> the shared ops in `edgecv/trackers/cf/ops/`. Reference: Bolme, Beveridge, Draper, Lui,
> *"Visual Object Tracking using Adaptive Correlation Filters"* (CVPR 2010).

## 1. Goal and scope

Port MOSSE as an **individual, inline** CF tracker that:

1. Subclasses `CorrelationFilterTracker` and implements the **full transferable-filter contract**
   (`build_filter` / `evaluate` / `get_filter` / `set_filter` / `response_map` / `psr`), not just
   `init` / `update`. This is what makes it usable as a hybrid component later (§6.1 of ARCHITECTURE).
2. Is built **entirely on the `trackers/cf/ops/` layer** — it is the proof that the ops layer is a
   sufficient foundation, and it becomes the **template the KCF/DSST/Staple ports copy**.
3. Is faithful to Bolme 2010 on the parts that define MOSSE, plus two universally-adopted, zero-cost
   refinements (subpixel peak, power-of-2 FFT sizing).

**Agreed fidelity decisions** (from brainstorming):

| Fork | Decision |
|---|---|
| Fidelity | **Full Bolme**: log+z-score preprocessing, **seeded** random-affine init augmentation, online numerator/denominator (A/B) update with learning rate η, PSR failure detection. |
| ROI sizing | **Fixed template, no scale.** `padding` is a parameter (default 1.0, paper-faithful); template rounded up to an efficient FFT size. Per-axis (template may be non-square). |
| Peak localization | **Parabolic subpixel** refinement with fftshift wrap handling. |

### Out of scope (deliberate)

- **Scale adaptation** — MOSSE has none; that is DSST/Staple territory (separate ports).
- **Padded-context discrimination windows** (1.5–2.5×) — that is KCF territory.
- **Hybrid/IPC wiring, MotionPredictor** — MOSSE here is standalone inline. It only *satisfies* the
  pure-op contract so a future hybrid can drive it; it does not spin up processes.
- **Color/HOG features** — MOSSE is grayscale raw pixels.

## 2. Dependencies on the ops layer

Reuses existing ops: `extract_raw`, `cos_window`, `fft2`, `ifft2`, `psr`.

**Two ops additions** (each TDD'd as part of this work, before the tracker):

1. `gaussian2d_labels(size: tuple[int,int], sigma: float) -> np.ndarray` — new module
   `ops/labels.py`. The desired correlation output: a 2D Gaussian peaked at the **center** of a
   `(h, w)` map, std `sigma`, float32. Used to form `G = fft2(gaussian2d_labels(...))`.
   - Tests: shape == size; global max at center `(h//2, w//2)`; value 1.0 at center (peak-normalised);
     larger `sigma` ⇒ larger effective support (e.g. count of cells above 0.5 increases); all values
     in (0, 1]; symmetric.
2. `fft_size(n: int) -> int` — added to `ops/fft.py`. Smallest efficient transform length ≥ `n`.
   Default: next power of two. (Optional refinement: defer to `scipy.fft.next_fast_len` when the
   scipy backend is active; power-of-two is the numpy-reference behavior.)
   - Tests: `fft_size(1)==1` (or 2 — pick and pin), monotonic, `fft_size(64)==64`, `fft_size(65)==128`,
     result ≥ input, result is a power of two under the numpy reference.

Both are exported from `ops/__init__.py` and added to its `__all__`.

## 3. The MOSSE math (exact, with conventions)

All transforms run over the spatial axes via `ops.fft2` / `ops.ifft2`. Arrays are 2D here (single
grayscale channel). Complex storage is **complex64** (halves IPC payload size; matches the
`FilterState` example in `ARCHITECTURE.md` §5.1). `*` denotes complex conjugate; `⊙` elementwise.

**Desired output.** `g = gaussian2d_labels(template_size, sigma)`, `G = fft2(g)`. Constant per
filter; recomputable from `meta` (`template_size`, `sigma`).

**Preprocessing of a cropped patch** `p` (the standard MOSSE pipeline; reuses ops):
```
gray = extract_raw(p)[..., 0]          # gray in [0,1], (th, tw)
x    = log(gray + 1.0)
x    = (x - mean(x)) / (std(x) + 1e-5)
x    = x * cos_window(template_size)    # separable Hann from ops
F    = fft2(x)
```

**Filter (closed form, accumulated over training samples i):**
```
A = Σ_i  G ⊙ conj(F_i)      # numerator
B = Σ_i  F_i ⊙ conj(F_i)    # denominator
H* = A / (B + λ)            # λ = regularization (meta["lambda"])
```

**Detection** on a new patch `F` at the current search center:
```
R = real( ifft2( F ⊙ H* ) )   # response map, shape == template_size
```
Peak location → target displacement (see §4 for subpixel + wrap handling). `psr(R)` → confidence.

**Online update** (running average, learning rate η = `meta["eta"]`), performed **after** detection,
on a patch re-cropped at the *new* center:
```
A ← η·(G ⊙ conj(F)) + (1−η)·A
B ← η·(F ⊙ conj(F)) + (1−η)·B
```

**Init augmentation.** From the first patch, accumulate A and B over the identity plus
`n_warps` small random **affine** perturbations (rotation/scale jitter within a few percent /
degrees) of the patch — each preprocessed and FFT'd. This generalises a one-example filter.
Driven by a **seeded** RNG (`np.random.default_rng(rng_seed)`) so `build_filter` is reproducible and
its purity is testable. (Augmentation is init-only: zero per-frame cost.)

## 4. Peak localization (called out — classic bug surface)

1. Integer peak `(py, px) = unravel_index(argmax(R), R.shape)`.
2. **Parabolic subpixel** per axis over the 3 neighbours, with denominator guard:
   ```
   dx = 0.5*(R[py,px+1]-R[py,px-1]) / (R[py,px-1]-2*R[py,px]+R[py,px+1] + eps)
   dy = 0.5*(R[py+1,px]-R[py-1,px]) / (R[py-1,px]-2*R[py,px]+R[py+1,px] + eps)
   ```
   Skip the refinement on an axis where the peak is on the border (no neighbours).
3. **fftshift wrap handling.** The response is circularly indexed: a peak index past the half-size
   wraps to a negative displacement. Convert before applying:
   ```
   sy = (py + dy);  if sy > th/2:  sy -= th
   sx = (px + dx);  if sx > tw/2:  sx -= tw
   ```
   `(sy, sx)` is the center displacement **in template pixels**. Because the template is cropped at
   native resolution (no resize), this maps **directly** to the pixel center move — no scale factor.

## 5. Data structures

```python
@dataclass  # already exists in trackers/cf/base.py
class FilterState:
    arrays: dict[str, np.ndarray]   # {"A": complex64[th,tw], "B": complex64[th,tw]}
    bbox: BoundingBox               # normalised; the box (center) the filter currently tracks
    meta: dict                      # see below

# meta keys (the transferable description; also the ABI surface):
#   "template_size": (th, tw)   "padding": float   "sigma": float
#   "eta": float   "lambda": float
#   "feature": "raw"   "preproc": "log_zscore"
#   "abi": "mosse-1"            # bump on any layout/semantics change (ARCHITECTURE §7.5)
```

`EvalResult(bbox, response_map, psr)` is reused unchanged from `base.py`.

## 6. Class API and behavior

```python
class Mosse(CorrelationFilterTracker):
    def __init__(self, *, padding=1.0, sigma=2.0, eta=0.125, lmbda=1e-3,
                 n_warps=8, psr_lock=7.0, psr_lost=5.0, rng_seed=0): ...
```

`name() -> "MOSSE"`.

**Single source of truth for "where to look next":** the working `self._state.bbox`. Its center is
the crop center; its `w,h` are the (constant, no-scale) output dimensions. There is **no** separate
`self._center`/`self._size` — that redundancy is what desyncs CF trackers. Pixel conversion happens
per call from the current `frame.shape`; `template_size` (pixels) lives in `meta` and is frame-size
independent.

**`init(frame, bbox)`** — build the initial filter via `build_filter(frame, bbox)` (which itself
derives `template_size` and the warp-augmented A,B and populates `meta`); store it as `self._state`;
cache `self._G` from `meta`; set `self._status = LOCKED`, `self._response = None`, `self._psr = 0.0`,
`self._seq = 0`.

**`update(frame) -> TrackResult`** (the mutating loop):
1. `er = self.evaluate(frame, self._state)` — crops at `self._state.bbox` center, returns new
   center + response + PSR. (Reusing `evaluate` keeps detection on the *same* code path — the
   "same-engine" guarantee, ARCHITECTURE §14.6.)
2. `self._response, self._psr = er.response_map, er.psr`; `self._status = _status_from(er.psr)`.
3. Set `self._state.bbox = er.bbox` (new center; `evaluate` preserved `w,h`) — this is what moves the
   next crop.
4. **If `er.psr >= psr_lost`**: re-crop at the new center (`self._state.bbox`), preprocess→`F`, and
   apply the online A/B update **in place** on `self._state.arrays`. **Else (LOST): freeze** — do not
   update (Bolme failure handling; prevents learning background).
5. `self._seq += 1`; return `TrackResult(bbox=er.bbox, confidence=er.psr, status=self._status,
   timestamp=now(), seq=self._seq)`.

**`build_filter(frame, bbox) -> FilterState`** — **PURE** (no `self` mutation). Derive
`template_size = (fft_size(round(h*padding)), fft_size(round(w*padding)))` from `bbox`; crop the
window at `bbox` center; build `G` locally; accumulate A,B (complex64) over identity + `n_warps`
seeded affine warps; return a fresh `FilterState` whose `meta` records `template_size` and the params
(§5). Reads config attributes (allowed); writes none. Safe to call in a detector worker.

**`evaluate(frame, state) -> EvalResult`** — **PURE**. Crop the window of `state.meta["template_size"]`
at `state.bbox` center; preprocess→`F`; `H* = A/(B+λ)`; `R = real(ifft2(F⊙H*))`; subpixel+wrap peak
(§4) → new center → output `BoundingBox` (new center, **same w/h** as `state.bbox`); `psr(R)`. No
mutation; deterministic.

**`get_filter() -> FilterState`** — return the current `self._state` (the evolving A,B).

**`set_filter(state, search_box=None) -> None`** — adopt `state` as `self._state`; recompute
`self._G` from its meta. When `search_box` is given, write its center into the working
`self._state.bbox` (keeping `state`'s `w,h`) so the next crop looks at the predicted/current position
(ARCHITECTURE §6.1); otherwise the crop center stays at `state.bbox`. Used by the hybrid inject path
and by get/set round-trips.

**`response_map` / `psr` properties** — last `self._response` / `self._psr`.

**`status` property** — `self._status`. `_status_from(psr)`: `>= psr_lock → LOCKED`;
`psr_lost <= psr < psr_lock → COASTING`; `< psr_lost → LOST`.

### 6.1 Cropping & borders (helper)

A private `_crop(frame, center, size) -> np.ndarray` returns a fixed `size` patch centered at
`center`, **edge-replicating** (`np.pad`, mode `"edge"`) when the window crosses the frame boundary —
so the FFT always sees the full fixed size and targets near edges don't crash. Pure numpy (no cv2 in
core). Flagged as a **promotion candidate** to `ops/` once KCF needs the same crop.

## 7. Coordinate discipline (ARCHITECTURE §5.1)

`BoundingBox` stays normalised 0–1 at all times. Pixel work happens only inside the tracker, via the
explicit `PixelBox` conversion at the `frame.shape` boundary. Output boxes keep input `w/h` exactly
(no scale) and only move their center. No raw pixel tuple ever masquerades as a `BoundingBox`.

## 8. Test plan (TDD, in order)

Ops first, then the tracker. `tests/test_cf_ops.py` gains the `gaussian2d_labels` and `fft_size`
tests (§2). New `tests/test_mosse.py`:

1. **Contract / instantiation** — `Mosse()` instantiates (all abstract methods present); `name()=="MOSSE"`.
2. **init builds a valid filter** — after `init`, `get_filter().arrays` has complex64 `A,B` of shape
   `template_size`; `status == LOCKED`; output box dims preserved.
3. **`build_filter` purity** — snapshot `get_filter()` arrays; call `build_filter`; assert `self`
   state unchanged (arrays identical object/values).
4. **`evaluate` purity** — `evaluate` does not mutate `self._state`; returns `EvalResult` with
   `response_map.shape == template_size` and finite `psr`.
5. **`build_filter` determinism** — two calls with the same `rng_seed` produce **identical** A,B
   (seeded warps reproducible).
6. **Tracks translation (core behavioral test)** — synthetic frame with a bright Gaussian blob on
   noise; translate the blob by known `(dx,dy)` across frames; assert `update()` center follows within
   ~1 px, and `status == LOCKED`.
7. **Subpixel beats integer** — blob shifted by 0.5 px ⇒ reported center moves ~0.5 px (tolerance
   tighter than 1 px), demonstrating subpixel refinement.
8. **Failure detection + freeze** — feed pure noise; `psr` drops, `status == LOST`, and A,B are
   **unchanged** versus the pre-noise filter (frozen, not corrupted).
9. **`set_filter` honours `search_box`** — after `set_filter(state, search_box=...)`, the next
   `evaluate`/`update` crops at the search_box center (verify the track relocates accordingly).
10. **get/set round-trip** — `t.set_filter(t.get_filter())` preserves tracking on the next frame.
11. **Coordinate invariants** — output `BoundingBox` stays in [0,1]; `w,h` equal the init box.
12. **Border safety** — target initialized near a frame edge: `init`/`update` run without error and
    return a valid box.

All tests use real synthetic frames (no mocks). Watch each fail before implementing (Iron Law).

## 9. Files

```
edgecv/trackers/cf/
├── mosse.py                 # NEW — Mosse(CorrelationFilterTracker)
└── ops/
    ├── labels.py            # NEW — gaussian2d_labels
    ├── fft.py               # EDIT — add fft_size()
    └── __init__.py          # EDIT — export gaussian2d_labels, fft_size
tests/
├── test_cf_ops.py           # EDIT — add gaussian2d_labels + fft_size tests
└── test_mosse.py            # NEW — tracker tests
```

No changes to `base.py`, `core/`, runtime, or fusion. (Optional, only if trivial: register `Mosse`
in `trackers/cf/__init__.py` for import convenience.)

## 10. Performance posture

Correctness first; optimize only if profiling on the RK3588 demands it (per the C-core analysis: the
Python orchestration share is expected to be single-digit %). Known later levers, **not** v1 work:
preallocate the per-frame patch / FFT buffers and reuse them; `set_fft_backend("scipy")` or pyFFTW
plans for the fixed template size; `gc.disable()` in the loop (ARCHITECTURE §14.9). v1 stays
allocation-simple and readable.

## 11. Open implementation choices (safe to pin during planning)

- `fft_size(1)` return value (1 vs 2) — pin in the op test.
- Exact warp magnitude (rotation ° / scale %) for init augmentation — small (~±2°, ±2%); pin a
  default, expose if cheap.
- Default `psr_lock` / `psr_lost` (7.0 / 5.0) — reasonable for grayscale MOSSE; tune against the
  translation test.

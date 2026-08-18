# NanoTrack Preprocess Acceleration — Implementation Plan

> **For agentic workers:** implement task-by-task; steps use checkbox (`- [ ]`) syntax for tracking.
> Each task is independently testable. Keep the numpy path as the bit-faithful fallback throughout.

**Goal:** Cut the NanoTrack/SiamFC per-frame crop+resize from ~11 ms (numpy, big core) to <1 ms by
adding an optional cv2 fast path behind `edgecv/trackers/nn/preprocess.py`, with **no tracker,
manifest, or HAL change**. Validate against the QuadGuide trace baseline.

**Architecture:** `crop_with_context` keeps its signature and returns the same `CropXform`. A new
import-guarded dispatcher `_crop_resize` uses `cv2.warpAffine` (INTER_LINEAR + BORDER_REPLICATE) when
cv2 is importable, else the existing numpy `_sample_clamped` body. The cv2 affine is constructed to
honour the identical half-pixel sampling grid, so coordinate inversion (`CropXform.to_frame`) and box
decode are unaffected.

**Tech Stack:** Python, numpy (runtime, required); opencv-python (optional accelerator — hard dep in
QuadGuide, EdgeCV `fast` extra). No new required EdgeCV dependency.

**Spec:** `docs/superpowers/specs/2026-06-14-nanotrack-preprocess-accel-design.md`

**Baseline to beat:** `~/QuadGuide/quadguide-trace/20260611-084906/tracker_NanoTrack.jsonl` — tracker
stage latency **p50 20.4 ms / p95 23.3 ms**, ~30 FPS loop.

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `edgecv/trackers/nn/preprocess.py` | add guarded `cv2` import + `_crop_resize` dispatcher; route `crop_with_context` through it | Modify |
| `tests/test_nn_preprocess.py` | cv2-vs-numpy parity + CropXform-inversion invariant + fallback test | Modify |
| `pyproject.toml` | (verify) `opencv-python` stays in `fast` extra; note as preprocess accelerator | Verify |
| `docs/superpowers/specs/2026-06-14-nanotrack-preprocess-accel-design.md` | spec | (done) |

No changes to: `nanotrack.py`, manifests, the RKNN backend, model artifacts, or QuadGuide source.

---

## Task 1 — Dispatcher with numpy fallback (semantics-preserving) — [x] DONE

- [ ] In `preprocess.py`, add a module-level guarded import:
      `try: import cv2 as _cv2  except Exception: _cv2 = None`.
- [ ] Add `_crop_resize(frame, center, size_px, out_size) -> np.ndarray` that:
      - **numpy branch** (when `_cv2 is None`): the current `crop_with_context` body
        (`fx/fy` grid → `_sample_clamped`), returning the patch only.
      - **cv2 branch**: build the separable affine `a=sw/ow, b=(cx−sw/2)+0.5*a, c=sh/oh,
        d=(cy−sh/2)+0.5*c`; `M = [[a,0,b],[0,c,d]]` (float32); call
        `cv2.warpAffine(frame, M, (ow, oh), flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE)`. Cast result to float32 to match the numpy branch.
- [ ] Refactor `crop_with_context` to call `_crop_resize`, keep the `frame.ndim == 2` reshape, and
      return `patch, CropXform(center, (sh, sw), (oh, ow))` **unchanged**.
- [ ] Confirm 2-D (gray) and 3-D (rgb) frames both work in the cv2 branch (warpAffine handles both;
      keep the existing reshape for the gray case).

**Test:** existing `tests/test_nanotrack.py` and any SiamFC tests pass unchanged (run without cv2 to
exercise the fallback).

## Task 2 — Parity + invariant tests — [x] DONE (19 passed; cv2 4.13 via QuadGuide venv)

- [ ] In `tests/test_nn_preprocess.py`, add a cv2-vs-numpy parity test: synthetic gradient frame,
      a handful of `(center, size_px, out_size)` cases incl. off-centre and border-overlapping crops;
      assert `max(|cv2 − numpy|) <= 1.0` (0–255 scale). `@pytest.mark.skipif(cv2 missing)`.
- [ ] Add an inversion-invariant test: for several output pixels, `CropXform.to_frame` returns the
      same frame coord regardless of which branch produced the patch (pins the spec invariant).
- [ ] Add a fallback test: force `_cv2 = None` (monkeypatch) and assert `crop_with_context` still
      returns a correct-shape float32 patch.

**Test:** `pytest tests/test_nn_preprocess.py` green (parity skipped if cv2 absent in CI).

## Task 3 — On-device micro-benchmark — [x] DONE (cv2 1.9 ms vs numpy 18 ms, ~9x, 1080p→255²)

- [ ] On the RK3588, time `_crop_resize` cv2 vs numpy (1080p frame → 255×255), warm + N=50. Record
      both in the task notes. Expect cv2 <1 ms vs numpy ~11 ms (big core).
- [ ] Sanity-check the cv2 path output visually/numerically against numpy on a real frame.

## Task 4 — End-to-end QuadGuide validation against baseline — [x] DONE

> **Result (trace `20260614-082203` vs baseline `20260611-084906`):** restricted to the
> locked/inference window (`health==nominal`; the overall p50 is diluted by ~half idle `no_lock`
> frames that early-return without infer), tracker **stage p50 21.6 ms → 12.8 ms (~41% faster)**,
> p95 24.1 → 14.0 ms, tight distributions. Lock behaviour unchanged. Remaining ~12.8 ms is
> inference-bound (2× NPU + adapter full-frame BGR→RGB copy + to_input + decode), not preprocessing.

- [ ] Ensure QuadGuide's runtime venv has cv2 (it requires `opencv-python>=4.10`) and picks up the
      edited EdgeCV checkout (editable install or `edgecv.pth`).
- [ ] Run the webcam inference loop; dump a fresh trace into `~/QuadGuide/quadguide-trace/<new>/`.
- [ ] Compare with the baseline using the QuadGuide tool:
      `scripts/diagnose_latency.py trace quadguide-trace/<new>` and the stage-latency stats
      (parse `lat` records: `stage_ms = (t − in)/1e6`).
- [ ] **Pass criteria:** tracker stage p50 drops from ~20.4 ms toward ~9–11 ms; no regression in
      health/lock behaviour (`state` records still reach `nominal`).

## Task 5 — Docs + memory — [ ]

- [ ] Note in `preprocess.py` module docstring that the cv2 fast path is active when available and the
      numpy path is the reference/fallback (ARCHITECTURE §6.2/§16).
- [ ] Update memory `nanotrack-cpu-preprocess-bottleneck` with the post-fix numbers once Task 4 lands.

---

## Risks / notes

- **Sampling drift:** the only correctness risk is the cv2 affine not matching the numpy grid. Task 2's
  parity + inversion tests gate this; the half-pixel `+0.5` term in `b`/`d` is the crux — do not drop it.
- **dtype:** keep the cv2 output float32 so `to_input` (which does `patch.astype(float32)`) and the
  int8 quant path are unchanged. warpAffine on a uint8 frame returns uint8 — cast explicitly.
- **Out of scope here** (separate efforts): RGA hardware path (no Python binding for `librga.so.2`
  yet), NCHW↔NHWC transpose elimination (marginal + invasive), FP16→INT8 head (accuracy), and the
  QuadGuide full-frame BGR→RGB copy (`edgecv_adapter.py:139`).

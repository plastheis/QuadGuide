# YoloKalmanTracker — YOLO + Kalman tracking-by-detection (design)

> **Status:** implemented. Two trackers landed:
> **(a)** inline `edgecv/trackers/nn/yolo_kalman.py:YoloKalmanTracker` — the
> simple synchronous building block; **(b)** async
> `edgecv/trackers/hybrid/acquire_kalman_track.py:AcquireKalmanTrack` — the
> production tracker, wired into QuadGuide as `tracker: kalman_track`. Owns the
> *policy* for a new single-object tracker; the detector it composes
> (`trackers/nn/yolo.py:YoloDetector`, the `yolo11n` P2/P3/P4 head) and the YOLO
> worker it reuses (`trackers/hybrid/acquire_workers.py`) are existing and
> unchanged. Per ARCHITECTURE §6.3/§17 a tracker gets its own spec; this is it.
> Sibling of [[2026-06-14-acquire-track-design]] (which hands off YOLO→NanoTrack);
> this one uses **only** YOLO + an in-parent motion filter.

---

## 1. Goal

A single-object tracker that is **just the `yolo11n` P2/P3/P4 detector plus a
Kalman-filter associator**: run the detector every frame, keep one
constant-velocity Kalman filter over the target box, and on each frame
**predict → associate → correct**. The filter bridges detection gaps (coast on
the prediction) and smooths detector jitter. It exposes the ordinary `Tracker`
ABC so it slots into QuadGuide via the existing `edgecv_adapter` like every other
EdgeCV tracker.

This is the **single-object specialisation of SORT** (Bewley et al., *Simple
Online and Realtime Tracking*, ICIP 2016). SORT's pipeline is: Kalman predict →
IoU-cost assignment (Hungarian) → Kalman update → track lifecycle. With exactly
one target there is no assignment problem — we just pick the best detection — so
all that remains is a per-target Kalman filter and an association gate.

### 1.1 Why this over the existing `YoloTracker`

`trackers/nn/yolo.py:YoloTracker` already does YOLO + a *static* proximity
association (`score · exp(-d²/2σ²)` against the **last** box, on a crop). It has
**no motion model**: it cannot anticipate where a fast target will be next frame,
and when a detection is missed it simply repeats the stale box. Adding a Kalman
filter buys two concrete things:

| | `YoloTracker` (proximity) | `YoloKalmanTracker` (this) |
|---|---|---|
| Inter-frame motion | none (associates to last box) | constant-velocity **prediction** |
| Missed detection | repeat stale box, COASTING | **extrapolate** along velocity, COASTING |
| Reported box | raw detection | **filtered** (smoothed) estimate |
| Fast motion / re-lock | proximity to last box only | IoU to *predicted* box + Mahalanobis gate |

---

## 2. Research — lightweight Kalman tracking-by-detection

### 2.1 The SORT lineage

- **SORT** (Bewley 2016): the canonical lightweight tracking-by-detection
  baseline. Per track: a linear Kalman filter with a **constant-velocity** model;
  association by **IoU** between predicted boxes and detections; Hungarian
  assignment; tracks born from unmatched detections, killed after `T_lost`
  missed frames. Deliberately ignores appearance — pure motion + geometry, which
  is exactly why it runs at thousands of Hz on a CPU. State in the original:
  `[u, v, s, r, u̇, v̇, ṡ]` (centre, scale=area, aspect — aspect has *no*
  velocity).
- **DeepSORT** (Wojke 2017): adds an appearance embedding and a **Mahalanobis
  gate** (χ² on the Kalman innovation) alongside IoU. Its filter is the design we
  borrow the *noise model* from: 8-dim state `[cx, cy, a, h, v…]`, with process
  and measurement noise **scaled by target height** so the filter self-adapts to
  scale (`std_weight_position = 1/20`, `std_weight_velocity = 1/160`). We keep the
  motion filter + Mahalanobis gate and **drop the appearance branch** (single
  object, edge budget).
- **OC-SORT / ByteTrack** (2022): refinements for *crowded multi-object* video
  (observation-centric re-update, low-score detection recovery). Not needed for a
  single target, but ByteTrack's "keep low-score detections during association"
  idea maps onto our **Mahalanobis fallback** (§4.3) — when a confident detection
  is missing, a weaker nearby one can still sustain the track.

### 2.2 What "lightweight" means on RK3588

The detector dominates cost: `yolo11n_p2p3p4` is ~55 ms (INT8) / ~73 ms (FP16) on
one NPU core. The Kalman filter is **8×8 linear algebra per frame — microseconds**
on the A55. So the filter is free; the design question is purely *how the detector
is scheduled* (§6): inline (block ~55 ms/update) vs. async worker (parent runs the
free predict at full frame-rate, corrections arrive late). MVP = inline.

### 2.3 Design decisions taken (and why)

| Decision | Choice | Rationale |
|---|---|---|
| State | 8-dim `[cx,cy,w,h,vcx,vcy,vw,vh]`, all box params have velocity | Fully linear, `z` = the box directly; simpler than SORT's area/aspect and lets w,h drift smoothly. |
| Motion | constant velocity, `dt = 1` frame | SORT convention; per-frame is deterministic and frame-rate-agnostic for association. |
| Noise | DeepSORT size-relative (∝ height), `σ_p=1/20`, `σ_v=1/160` | Self-adapts to target scale; no per-scene retuning. |
| Association | IoU gate (primary) + Mahalanobis centre gate (fallback) | IoU is SORT-faithful and intuitive; Mahalanobis recovers fast motion / size jumps where IoU→0. |
| Lifecycle | `max_age` coast budget → LOST | Single-object SORT; coasting feeds guidance the extrapolated box. |
| Coordinates | normalised 0–1 throughout (state + I/O) | Matches `BoundingBox`; frame-size-independent. |

---

## 3. The Kalman filter (`KalmanBoxState`)

Pure numpy, reusable (an async worker variant predicts in the parent, corrects on
async detections — §6).

- **State** `x ∈ ℝ⁸ = [cx, cy, w, h, vcx, vcy, vw, vh]ᵀ`, normalised; velocities
  per frame. **Measurement** `z = [cx, cy, w, h]ᵀ` (linear `H = [I₄ | 0]`).
- **Transition** `F` = constant velocity: `x' = x + v·dt`.
- **Predict** `x ← Fx`, `P ← F P Fᵀ + Q`.
- **Update** standard gain step `K = P Hᵀ S⁻¹`, `S = H P Hᵀ + R`.
- **Noise** (height-scaled, `h = current height`):
  - `Q = diag([σ_p·h]×4, [σ_v·h]×4)²`
  - `R = diag([σ_p·h]×4)²`
  - init `P = diag([2σ_p·h]×4, [10σ_v·h]×4)²` (loose on the unknown birth velocity).
- **Gating distance** = squared Mahalanobis of a box centre vs. the projected
  prediction (2 dof), for the fallback gate.

## 4. Per-frame algorithm (`YoloKalmanTracker.update`)

1. **Predict** the filter one frame → predicted box `B̂`.
2. **Detect** full-frame YOLO → `(boxes, scores)`.
3. **Associate** (`_associate`, §4.3) → best detection or `None`.
4. **Correct or coast:**
   - matched → `kf.update(box)`; `misses = 0`; **LOCKED**; report filtered box,
     confidence = detection score.
   - none → `misses += 1`; **COASTING** while `misses ≤ max_age`, else **LOST**;
     report the predicted (extrapolated) box, confidence `None`.

### 4.1 init / reset

`init(frame, bbox)` seeds a fresh `KalmanBoxState` from `bbox` (operator lock-on
via the adapter), status LOCKED, velocity zero. There is no separate `reset()`:
the adapter's `_initialized` gate stops `update()` before the next `init`, exactly
as for the inline `YoloTracker`/`NanoTrack` (only `AcquireTrack` needs its own
`reset()`).

### 4.2 Coordinates & detection region

MVP detects on the **full frame** every update — `yolo11n_p2p3p4` is a
small-object (stride 4/8/16) head designed for full-frame input, and full-frame
detection makes re-lock after a miss automatic (a detection reappearing near the
coasted prediction re-associates). *Optimisation (deferred):* crop a search ROI
around the predicted box (à la `YoloTracker.search_factor`) to cut pixels and
raise small-target resolution; the prediction already gives the ROI centre. Trade
-off: a cropped detector can't see a target that left the ROI, so re-lock then
needs a full-frame escalation like AcquireTrack's REACQ.

### 4.3 Association policy

```
primary  (SORT IoU): eligible = { det : score ≥ min_score ∧ IoU(B̂, det) ≥ iou_min }
                     pick argmax IoU (score breaks ties)
fallback (Mahalanobis): if primary empty ∧ use_maha_fallback
                     pick det with min gating_distance < χ²₂(0.95)=5.99, score ≥ min_score
```

IoU handles the common case; the Mahalanobis fallback catches fast motion / abrupt
size change where the predicted and detected boxes don't yet overlap but the
centre is statistically consistent with the filter's uncertainty.

### 4.4 Parameters (defaults; all constructor kwargs)

| Param | Default | Meaning |
|---|---|---|
| `iou_min` | `0.3` | IoU gate for primary association (SORT default). |
| `min_score` | `0.25` | Min YOLO score for an eligible detection. |
| `max_age` | `30` | Consecutive misses to coast before LOST (~1 s at 30 fps). |
| `use_maha_fallback` | `True` | Enable Mahalanobis centre gate when IoU finds nothing. |
| `dt` | `1.0` | Time step per update (frames). |
| `std_position` | `1/20` | Position/measurement noise weight (× height). |
| `std_velocity` | `1/160` | Velocity process-noise weight (× height). |

Detector preprocessing (`conf_thresh`, `iou_thresh`, `input_size`, `strides`,
`reg_max`, …) resolve through the manifest exactly as in `YoloDetector` — for
`yolo11n.yaml` that means `output_format: rknn_dfl`, `strides: [4,8,16]`.

---

## 5. Async architecture (`AcquireKalmanTrack`) — the production tracker

The Kalman filter makes an **async** architecture especially clean, and it is the
real payoff for a real-time consumer like QuadGuide:

```
 parent (every frame, ~free):  kf.predict() → report extrapolated box   (full rate)
 YOLO worker (async, ~55 ms):  detect on control.crop → publish boxes+scores+src_ts
 parent (when a result lands): associate → kf.update(matched det)       (correction)
```

The parent emits a smoothed, full-frame-rate estimate by **predicting** every
frame; the slow detector only supplies **corrections**, and detection latency is
bridged by the motion model (the same insight as MAFiD's filter-injection, but via
a Kalman predictor instead of a CF filter). It reuses the existing
`acquire_workers.YoloWorker` / `_yolo_main`, `FrameRing`, `AcquireControlChannel`,
and `PayloadChannel` **verbatim** — there is **one** spawned worker (YOLO); the
Kalman filter (`KalmanBoxState`, written pure/standalone for exactly this reuse)
runs inline in the parent. No NanoTrack worker, no `NanoResultChannel`.

### 5.1 Acquire-before-lock state machine (mirrors AcquireTrack)

`AcquireKalmanTrack` adopts AcquireTrack's behavioural contract, replacing the
NanoTrack *track* stage with the in-parent Kalman associator:

| State | Worker crop (`control.crop`) | Parent does | Health | bbox |
|---|---|---|---|---|
| `ACQUIRE` | fixed central crop (`acquire_crop`) | report best YOLO det in crop as a candidate | `INITIALIZING` (non-driving) | best candidate / none |
| `LOCKED` | ROI around prediction (`search_factor`) | predict every frame; associate + correct on each new det | `LOCKED`, → `COASTING` after `coast_locked_frames` w/o correction | filtered (predicted/corrected) box |
| `REACQ` | full frame | coast (predict); re-lock on a confident det | `COASTING` | extrapolated box |
| `LOST` | full frame | coast; re-lock; reset after `search_timeout_frames` | `LOST` | extrapolated box |

- **Lock (operator-gated).** `init(frame, bbox)` with a non-zero box = "commit
  now": seed the Kalman from the current best in-crop candidate (padded by
  `lock_pad`), or from the sent box if none. Zero box = `reset()` → `ACQUIRE`.
  Identical to AcquireTrack's lock semantics, so the existing `LockOnCmd` wire
  format and lock-button-sends-crop-box contract carry over unchanged.
- **Drop.** `drop_frames` consecutive *new YOLO results* that fail to associate
  (IoU + Mahalanobis both reject) → `REACQ`. Note misses are counted per
  detection result, not per frame, because YOLO runs slower than the frame rate.
- **Re-lock.** `REACQ`/`LOST` re-seed the filter on the most confident full-frame
  detection (`_best_above`, no spatial gate) — a target that reappears anywhere
  recovers (AcquireTrack parity).

### 5.2 Tracker contract & adapter

Implements the standard `Tracker` ABC, so **nothing structural changes** in
QuadGuide. Status → health uses the existing `_HEALTH_BY_STATUS`
(LOCKED→nominal, COASTING→uncertain, INITIALIZING→acquiring, LOST→lost).
Confidence is the raw YOLO score (already 0–1). Like AcquireTrack it is
`_always_update` + `_async` in the adapter: `update()` runs every frame
(acquisition runs pre-lock), and the estimate's lineage timestamp is the
just-captured frame's capture time — the Kalman predicts *to now*, so added
latency is ~0 (cf. AcquireTrack, which reports the NanoTrack source frame's older
ts). Wired one-branch in `edgecv_adapter._build`:

```python
# edgecv_adapter.py _build, alongside the acquire_track branch
if name == "kalman_track":
    from edgecv.trackers.hybrid import AcquireKalmanTrack
    self._always_update = True
    self._async = True
    return AcquireKalmanTrack(
        yolo_manifest=self._manifests_dir() / "yolo11n.yaml",
        backend=resolved, **params)
```

```yaml
# QuadGuide config
tracker:
  import: quadguide.perception.edgecv_adapter:EdgeCVTracker
  params:
    tracker: kalman_track
    backend: rknn
    model_dir: /home/radxa/EdgeCV/models
    acquire_crop: 0.5      # central scan crop before lock
    search_factor: 2.0     # LOCKED detect-ROI = predicted box × this
    drop_frames: 5         # failed associations before re-acquire
    lost_timeout_frames: 90
```

### 5.3 The inline building block (`YoloKalmanTracker`)

`edgecv/trackers/nn/yolo_kalman.py:YoloKalmanTracker` is the simple synchronous
variant: it owns a `YoloDetector`, detects full-frame every `update()`, and runs
predict→associate→correct inline (blocks ~55 ms/update). No acquisition state
machine — like `YoloTracker` it stays silent until `init()`. Kept as the
zero-IPC reference and reused unit-test surface; `AcquireKalmanTrack` is the one
to deploy. Both share `KalmanBoxState` and `associate_detection`.

---

## 7. Files

```
edgecv/
├── trackers/nn/
│   ├── yolo_kalman.py            # NEW: KalmanBoxState + associate_detection + YoloKalmanTracker (inline)
│   └── __init__.py               # export YoloKalmanTracker, KalmanBoxState
├── trackers/hybrid/
│   ├── acquire_kalman_track.py   # NEW: AcquireKalmanTrack (async; Tracker ABC + state machine)
│   ├── acquire_workers.py        # REUSED verbatim (YoloWorker / _yolo_main / build_yolo_detector)
│   └── __init__.py               # export AcquireKalmanTrack
├── models/manifests/
│   └── yolo11n.yaml              # EXISTING (rknn_dfl P2/P3/P4 head); reused as-is

# QuadGuide side
src/quadguide/perception/edgecv_adapter.py   # +kalman_track branch (_always_update/_async)
configs/rk3588.yaml                          # tracker: kalman_track   (set when deploying)
```

No new SHM structs and no `ABI_VERSION` bump: the async tracker rides the existing
`AcquireControl` + `PayloadChannel` (mode always `YOLO`; `lock_gen`/`lock_bbox`
unused, published as zeros).

---

## 8. Testing (x86, no NPU)

- **Filter math** (done, no model): constant-velocity target → the filter learns
  velocity (≈0.0186 vs. true 0.02 after 7 steps), prediction anticipates motion,
  IoU/gating helpers correct.
- **State flow** (done, stubbed detector): locks on detections, coasts while
  extrapolating (+0.028/frame), → LOST after `max_age` misses.
- **To add (`tests/test_yolo_kalman.py`):** association precedence (IoU beats
  Mahalanobis when both apply); `min_score`/`iou_min` gates reject weak/distant
  detections; re-lock after a coast when a detection reappears near the
  prediction; smoothing (filtered output lower-variance than noisy measurements).
  Use the `mock` backend or a stubbed detector (as in the smoke checks above).
- **On-device smoke (manual):** lock → track a moving target → brief occlusion →
  coast → re-lock; confirm the reported box leads the raw detections slightly
  (velocity) and is visibly smoother.

---

## 9. Defaults chosen for you (override in config)

- State 8-dim `[cx,cy,w,h,+v]`, constant velocity, `dt = 1` frame.
- Noise height-scaled (DeepSORT): `σ_p = 1/20`, `σ_v = 1/160`.
- Association: IoU ≥ `0.3` primary, Mahalanobis χ²₂(0.95) fallback on.
- Coast budget `max_age = 30` (~1 s at 30 fps) before LOST.
- Full-frame detection every update (search-ROI crop deferred, §4.2).
- Inline/synchronous (async worker variant specced in §6, deferred).
```

# AcquireTrack — concurrent YOLO lock verification (design)

> **Status:** implemented as `VerifiedAcquireTrack`, a subclass of AcquireTrack
> (not an in-place edit of AcquireTrack — base behaviour is untouched). Extends
> the existing AcquireTrack hybrid
> ([2026-06-14-acquire-track-design.md](2026-06-14-acquire-track-design.md)) with a
> drift-detection mechanism for the LOCKED state. New file
> `trackers/hybrid/verified_acquire_track.py`; `Mode.BOTH` added to
> `runtime/shm/control_channel.py` and the two workers in
> `trackers/hybrid/acquire_workers.py` switched to membership tests (backward
> compatible). NanoTrack, YOLO, and the runtime are otherwise unchanged. Sections
> below describing changes to `acquire_track.py` are realised as overrides in the
> subclass.

---

## 1. Problem

In LOCKED, AcquireTrack runs **NanoTrack only** (`_publish_control` sets
`Mode.NANO`; the YOLO worker idles because `snap.mode != Mode.YOLO`). The sole
loss signal is NanoTrack's own confidence, gated by `drop_score` / `drop_frames`
(`_tick_locked`).

This fails for a common single-object-tracker failure mode: **NanoTrack drifts
off the target onto background/clutter while keeping high confidence.** The
correlation/NN response peaks on whatever it has latched (an edge, a texture, a
fixed object), so `conf` never falls below `drop_score` and re-acquire never
fires. The tracker confidently reports the wrong box, which then drives guidance.
No threshold tuning can fix this — there is no second opinion during LOCKED.

Observed in a QuadGuide bench trace (2026-06-16): a locked track held `nominal`
with conf 0.9–1.0 while visibly sitting on clutter; the only transitions were
threshold flapping, never a true loss.

**Scope (from requirements):** the distractor is **background / clutter** — not a
same-class object. That makes a class-level detector check sufficient: if
NanoTrack's box is not supported by any YOLO detection, it has drifted. Same-class
distractor disambiguation (appearance ReID / motion prior) is an explicit
**non-goal** here and is noted as future work in §9.

## 2. Goal / approach

Run **YOLO concurrently with NanoTrack during LOCKED** (both already pinned to
their own NPU core, both keep RKNN contexts warm — so concurrency is free of
re-init cost and free of core contention). Use YOLO as an independent verifier:

> A locked NanoTrack box is **supported** if it overlaps a YOLO detection. If it
> goes **unsupported** for `verify_miss_frames` consecutive checks, declare drift
> and re-acquire — anchored on the **last verified-good position**, not the
> current (drifted) box.

This adds detector verification without changing the operator-gated lock flow, the
coast/LOST behaviour, or the existing confidence-drop path (the two loss signals
are complementary: confidence catches fast loss, verification catches confident
drift).

## 3. Control channel: concurrent operation

`runtime/shm/control_channel.py` `Mode` is a single selector
(`IDLE=0, YOLO=1, NANO=2`); workers run iff `snap.mode == <their mode>`. Add a
concurrent mode:

```python
class Mode(IntEnum):
    IDLE = 0
    YOLO = 1
    NANO = 2
    BOTH = 3   # NanoTrack tracks + YOLO verifies (LOCKED verification)
```

Worker activation predicate changes from equality to membership
(`acquire_workers.py`):

- YOLO worker infers when `snap.mode in (Mode.YOLO, Mode.BOTH)`.
- NanoTrack worker infers when `snap.mode in (Mode.NANO, Mode.BOTH)`.

No new shared-memory fields: the existing `crop` carries the YOLO search region
(full frame for verification — a drifted box can be anywhere), and `lock_bbox` /
`lock_gen` are unchanged. The header layout and seqlock are untouched, so this is
backward compatible with any reader that only knows the old modes (they simply
never see `BOTH`).

## 4. State machine changes (`acquire_track.py`)

### 4.1 `_publish_control`

LOCKED publishes `Mode.BOTH` with the YOLO crop = full frame (verification must
see the whole image), instead of `Mode.NANO`:

```python
if self._state == State.LOCKED:
    mode = Mode.BOTH if self._verify else Mode.NANO
    self._control.publish(mode=mode, crop=_FULL_CROP,
                          lock_gen=self._lock_gen, lock_bbox=self._lock_bbox)
```

All other states are unchanged (re-acquire/LOST already run YOLO via `Mode.YOLO`).

### 4.2 `_tick_locked`

After the existing confidence/drop logic, when a fresh YOLO result is available
and the verification cadence is due, run the support check. Pseudocode (additions
marked `+`):

```python
def _tick_locked(self):
    sample = self._read_nano_new()
    if sample is None:
        return
    conf, bbox = sample.confidence, sample.bbox
    self._last_bbox = bbox

    # (existing) confidence-drop path → REACQ_CROP, unchanged
    if conf < self._drop_score:
        ...                       # unchanged
        return
    self._miss = 0

+   # (new) concurrent YOLO verification
+   if self._verify:
+       yres = self._read_yolo_new()
+       if yres is not None:
+           _seq, boxes, scores, _ss, _st = yres
+           if self._supported(bbox, boxes, scores):
+               self._verify_miss = 0
+               self._last_good_bbox = bbox      # anchor for re-acquire
+           else:
+               self._verify_miss += 1
+               if self._verify_miss >= self._verify_miss_frames:
+                   self._enter_reacq_crop_from(self._last_good_bbox)
+                   self._set_out(self._last_good_bbox, conf,
+                                 TrackStatus.COASTING, sample.src_seq, sample.src_ts)
+                   return

    self._set_out(bbox, conf, TrackStatus.LOCKED, sample.src_seq, sample.src_ts)
```

Key points:

- **Anchor on last-good, not last.** `_last_good_bbox` is the NanoTrack box at the
  most recent *supported* verification. Re-acquire centers its crop on that
  position (`_enter_reacq_crop_from`), because `_last_bbox` is by now the drifted
  box — re-acquiring there would just re-lock the clutter. `_enter_reacq_crop_from`
  is `_enter_reacq_crop` plus setting `self._last_bbox = anchor` so the existing
  `_reacq_crop()` geometry and `_try_relock` association use the good position.
- **Reuses the existing re-acquire path.** Drift enters `REACQ_CROP` exactly like
  a confidence drop, so escalation (crop→full), coast, LOST, and re-lock all work
  unchanged. The only difference is the trigger and the anchor.

### 4.3 `_supported(bbox, boxes, scores)`

NanoTrack box is supported iff some detection above `verify_min_score` overlaps it
by at least `verify_min_iou`. Use **IoU** (symmetric) so a hugely oversized
detection can't trivially "contain" a drifted box:

```python
def _supported(self, bbox, boxes, scores):
    if boxes.shape[0] == 0:
        return False                       # no detections → unsupported (see §6)
    keep = scores >= self._verify_min_score
    if not keep.any():
        return False
    return self._max_iou(bbox, boxes[keep]) >= self._verify_min_iou
```

`_max_iou` is a small vectorised helper (normalised xywh → corners → IoU); add it
next to the existing geometry helpers.

### 4.4 New state fields (in `__init__`)

```python
self._verify            = verify
self._verify_min_iou    = verify_min_iou
self._verify_min_score  = verify_min_score
self._verify_miss_frames = verify_miss_frames
self._verify_miss       = 0
self._last_good_bbox    = None    # last verified-good NanoTrack box
```

`_last_good_bbox` is seeded to the lock box in `_relock` (a fresh lock is good by
definition) and reset there along with `_verify_miss = 0`.

## 5. Config / constructor params

Add to `AcquireTrack.__init__` (keyword-only, defaults chosen conservative so the
feature is safe to leave on):

| Param | Default | Meaning |
|---|---|---|
| `verify` | `True` | Run YOLO concurrently during LOCKED and check support. |
| `verify_min_iou` | `0.2` | Min IoU of the NanoTrack box with a detection to count as supported. |
| `verify_min_score` | `0.25` | Min YOLO score for a detection to count toward support. |
| `verify_miss_frames` | `5` | Consecutive unsupported checks before declaring drift. |

These flow through QuadGuide with **no adapter change** — `EdgeCVTracker` passes
`**params` straight to `AcquireTrack` (verified: unknown keys would `TypeError`).
So they are set in `configs/*.yaml` under `tracker.params` exactly like
`drop_score`, e.g.:

```yaml
    verify: true
    verify_min_iou: 0.2
    verify_min_score: 0.25
    verify_miss_frames: 5
```

Setting `verify: false` restores the current `Mode.NANO`-only behaviour exactly.

## 6. Edge cases

- **Occlusion / target legitimately gone.** If YOLO sees *no* detections, the box
  is unsupported and `_verify_miss` climbs → re-acquire → coast → LOST. This is
  the desired behaviour: a confident NanoTrack box over a vanished target is
  itself a drift. `verify_miss_frames` (default 5 ≈ 0.15 s @30fps of YOLO cadence)
  provides hysteresis so a single dropped detection frame doesn't trip it.
- **Verification vs confidence-drop ordering.** Confidence drop is checked first
  and short-circuits; verification only runs on otherwise-healthy (`conf ≥
  drop_score`) frames. The two counters (`_miss`, `_verify_miss`) are independent.
- **YOLO cadence < NanoTrack cadence.** `_read_yolo_new` already returns `None`
  when there's no new YOLO result, so verification naturally checks only on fresh
  detections; NanoTrack keeps publishing every frame. `verify_miss_frames` counts
  YOLO checks, not NanoTrack frames — document this (it is YOLO-paced).
- **Multiple valid detections (e.g. two clutter objects + target).** Support only
  requires overlap with *one* detection. Since the scope is clutter (not
  same-class), any target-class detection overlapping the box means NanoTrack is
  on a real object near the locked target; that is acceptable for this milestone.
  Same-class disambiguation is §9.
- **Re-lock loop safety.** If re-acquire keeps re-locking onto the same clutter
  (because YOLO also detects it as the target class), `verify` will immediately
  flag it again. This is acceptable (it degrades to the current behaviour for that
  pathological case) and is bounded by the existing LOST/`search_timeout_frames`
  reset. Genuinely fixing it needs §9.

## 7. Performance / NPU

- YOLO and NanoTrack already own separate RK3588 NPU cores and keep contexts warm;
  `BOTH` simply stops gating YOLO off during LOCKED, so there is **no extra model
  init** and **no core contention** — only YOLO's steady-state inference power
  during LOCKED (previously idle). Expect higher sustained NPU utilisation and
  power draw during tracking; thermals should be checked on the target SBC.
- If power/thermal is a concern, a follow-up can throttle verification cadence
  (run YOLO every K frames during LOCKED via a control-word countdown) without
  changing the verification logic. Not in this milestone — start at full cadence
  for maximum drift sensitivity, measure, then throttle if needed.

## 8. Testing

`tests/` (hermetic, no NPU — drive the state machine with a `spawn_workers=False`
AcquireTrack and fed YOLO/NanoTrack channel writes, as the existing AcquireTrack
tests do):

1. **Supported → stays LOCKED.** NanoTrack box overlaps a detection each check →
   `_verify_miss` stays 0, state LOCKED, `_last_good_bbox` tracks the box.
2. **Unsupported → drift → REACQ_CROP after N.** Feed a confident NanoTrack box
   with YOLO detections elsewhere (IoU 0) for `verify_miss_frames` checks → enters
   REACQ_CROP, anchored on `_last_good_bbox` (assert the re-acq crop is centered on
   the last good position, not the drifted one).
3. **No detections → drift.** Empty YOLO boxes for N checks → drift (occlusion
   path).
4. **`verify=False` → legacy.** LOCKED publishes `Mode.NANO`; verification never
   runs even with unsupported boxes.
5. **`_supported` unit:** IoU threshold and `verify_min_score` filtering, empty
   detection array, oversized-detection IoU (no false support).
6. **Mode membership:** YOLO worker activation predicate true for `YOLO` and
   `BOTH`; NanoTrack for `NANO` and `BOTH` (worker-level unit).

## 9. Future work (out of scope)

- **Same-class distractor disambiguation.** YOLO verifies class, not identity. To
  reject drift onto a *same-class* object, add either (a) an appearance embedding
  of the locked target compared each verification (needs another NPU model), or
  (b) a motion/position prior that re-locks to the detection nearest the predicted
  target position. Either layers on top of this verification (replace the
  "supported = overlaps any detection" rule with "supported = overlaps the
  *identity-matched* detection").
- **Adaptive verification cadence** (§7) for power/thermal.

## 10. Rollout

1. `control_channel.py`: add `Mode.BOTH` (+ const).
2. `acquire_workers.py`: membership activation for both workers.
3. `acquire_track.py`: params + state fields, `_publish_control` BOTH in LOCKED,
   `_supported` / `_max_iou` helpers, `_tick_locked` verification, re-acquire
   anchored on `_last_good_bbox`.
4. Tests (§8).
5. QuadGuide: add the `verify_*` params to `configs/rk3588.yaml` `tracker.params`
   (no code change). Trace a bench run and confirm drift onto clutter now produces
   a `nominal → coasting → (re-lock | lost)` transition (visible in the QuadGuide
   handoff timeline) instead of a stuck-confident `nominal`.
```

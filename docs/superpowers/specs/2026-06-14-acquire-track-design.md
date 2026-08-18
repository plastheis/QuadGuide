# AcquireTrack — YOLO-acquire → NanoTrack-track hybrid (design)

> **Status:** design spec, pre-implementation. Owns the *policy* for a new hybrid
> tracker; the EdgeCV runtime (`runtime/`, §7 of ARCHITECTURE.md), the HAL
> (`backends/`), and the NN trackers (`trackers/nn/yolo.py`, `nanotrack.py`) it
> composes are existing and unchanged except where called out. Per ARCHITECTURE
> §6.3/§17 a hybrid gets its own spec; this is it.

---

## 1. Goal

A single-object tracker that **acquires** a target with a YOLO detector and then
**tracks** it with NanoTrack, handing off between the two and re-acquiring on loss
— while exposing the ordinary `Tracker` API so it slots into QuadGuide (or any
caller) like every other EdgeCV tracker. The user supplies a single-class
`yolo11n` model (anchor-free `yolov8`-format head — identical decode path to the
existing `yolo26n`); NanoTrack is the shipped split backbone+head.

YOLO and NanoTrack each run on their **own RK3588 NPU core, in their own spawned
worker process**. They are **mutually exclusive**: exactly one is doing inference
at any moment; the idle one keeps its RKNN context warm so handoff costs no model
re-init.

### 1.1 Behavioural contract (from requirements)

1. **Acquire.** Before lock, YOLO runs continuously on a **fixed central crop** of
   the frame. Detections are reported out (for the HUD) but do **not** drive
   guidance. The crop is a static, configurable centered region.
2. **Lock (operator-gated).** The operator watches the HUD; when a YOLO detection
   sits inside the crop they issue an **init command**. On that command the
   tracker takes the **current best YOLO detection** in the crop, pads it, and
   initialises NanoTrack on it. YOLO then goes idle.
3. **Track.** NanoTrack runs. YOLO idle.
4. **Drop.** When NanoTrack confidence stays below a threshold (with hysteresis),
   NanoTrack goes idle and YOLO re-activates on a **crop centered on NanoTrack's
   last bbox**.
5. **Escalate.** If YOLO finds nothing in that crop within a window, it escalates
   to the **full frame**.
6. **Re-lock.** On a confident YOLO re-detection (associated to the last-known
   position), NanoTrack is re-initialised and YOLO goes idle — back to state 3.
7. While re-acquiring (4–6) the tracker **coasts**: it reports the last-known bbox
   as `COASTING` so guidance keeps flying toward it, until a configurable timeout
   elapses with no re-lock, at which point it reports `LOST`.

---

## 2. State machine

```
                 init cmd (lock current YOLO det in crop)
   ┌──────────┐  ───────────────────────────────────────►  ┌──────────┐
   │ ACQUIRE  │                                             │  LOCKED  │
   │ YOLO on  │  ◄───────────────────────────────────────  │ NanoTrack│
   │ centre   │        re-lock (confident YOLO det)         │  on full │
   │ crop     │                                             │  frame   │
   └──────────┘                                             └────┬─────┘
        ▲  ▲                                                     │
        │  │                                          conf < score_lost
        │  │                                          for N consecutive
        │  │                                                     ▼
        │  │  no det in crop      ┌──────────────┐  ┌──────────────────┐
        │  └──── for W_crop ◄──── │ REACQ_FULL   │  │   REACQ_CROP     │
        │        frames           │ YOLO on full │◄─│ YOLO on crop     │
        │                         │ frame        │  │ around last bbox │
   lost timeout                   └──────┬───────┘  └────────┬─────────┘
   (report LOST,                         │ re-lock           │ re-lock
    stay searching                       ▼                   ▼
    or reset)                        (→ LOCKED)          (→ LOCKED)
```

| State | Active worker | Crop fed to YOLO | Reported health | bbox reported |
|---|---|---|---|---|
| `ACQUIRE` | YOLO | fixed central crop | `INITIALIZING`→ adapter maps to a **non-driving** state (see §5) | best current YOLO det (for HUD), else none |
| `LOCKED` | NanoTrack | — | `LOCKED` / `COASTING` per NanoTrack score | NanoTrack bbox |
| `REACQ_CROP` | YOLO | crop around last bbox | `COASTING` | **last-known** bbox (held) |
| `REACQ_FULL` | YOLO | full frame | `COASTING` | **last-known** bbox (held) |
| `LOST` | YOLO (full) | full frame | `LOST` | last-known (not drawn, ignored by guidance) |

Transitions are evaluated **inline in the parent** on each `update()`, driven by
the latest worker result read non-blocking from the result channels.

### 2.1 Tunable transition parameters (defaults; all config-overridable)

| Param | Default | Meaning |
|---|---|---|
| `acquire_crop` | `0.5` | central crop side as a fraction of the *shorter* frame dimension (square, centered). |
| `lock_pad` | `1.15` | multiplier applied to the chosen YOLO box (w,h) before seeding NanoTrack. |
| `lock_min_score` | `0.35` | min YOLO score for a box to be lockable / shown as a candidate. |
| `drop_score` | `0.35` | NanoTrack fg-prob below which a frame counts as a "miss" (reuse NanoTrack `score_lost`). |
| `drop_frames` | `5` | consecutive misses before `LOCKED → REACQ_CROP` (hysteresis; prevents flicker). |
| `reacq_crop_factor` | `3.0` | re-acq crop side = factor × max(last_w, last_h) of the last NanoTrack bbox. |
| `reacq_crop_frames` | `15` | frames searching the crop before `REACQ_CROP → REACQ_FULL`. |
| `reacq_assoc_sigma` | `0.5` | proximity gate for choosing a re-acq detection near the last bbox (as in `YoloTracker`). |
| `lost_timeout_frames` | `90` | frames in `REACQ_*` with no re-lock before reporting `LOST` (≈3 s at 30 fps). |
| `search_timeout_frames` | `300` | after `LOST`, frames to keep auto-searching full-frame (auto re-lock, no operator cmd) before giving up → reset to `ACQUIRE` (≈10 s at 30 fps). `0` = search forever. |

---

## 3. Architecture — process group

This is a **detector + NN-tracker handoff**, not the CF-filter-injection fusion the
existing `trackers/hybrid/` (MAFiD) scaffolding implements. It **reuses the runtime
primitives** (Orchestrator, FrameRing, seqlock channels) but defines its own
worker bodies and its own (simpler) payload contract — no `FilterState`, no
`build_filter`/`evaluate`, no `FusionPolicy`.

```
 Caller process (QuadGuide tracker_worker — itself forked from run.py)
 ┌───────────────────────────────────────────────────────────────────────┐
 │  AcquireTrack  (the Tracker object; orchestrator + state machine)        │
 │    update(frame):                                                        │
 │      1. frame_ring.publish(frame, seq, ts)        # single writer        │
 │      2. control.publish(mode, crop_roi, lock_gen) # parent → workers     │
 │      3. read latest result from the ACTIVE worker's result channel       │
 │      4. run state-machine transitions inline                             │
 │      5. return TrackResult(bbox, confidence, status, seq=source_seq)     │
 └───────┬───────────────────────────────────┬────────────────▲────────────┘
   frame ring (latest-only)            control channel          │ result channels
   + control                       (seqlock, fixed struct)      │ (seqlock)
         ▼                                   ▼                   │
 ┌──────────────────────────┐     ┌──────────────────────────────┴──────────┐
 │ YOLO worker  (spawn)     │     │ NanoTrack worker  (spawn)                 │
 │  NPU core 0, A76 core    │     │  NPU core 1, A76 core                     │
 │  loop:                   │     │  loop:                                    │
 │   read frame+control     │     │   read frame+control                      │
 │   if mode != YOLO: idle  │     │   if mode != NANO: idle                   │
 │   crop = control.roi     │     │   on new lock_gen: init(frame, lock_bbox) │
 │   det = YoloDetector     │     │   res = NanoTrack.update(frame)           │
 │   publish DetResult      │     │   publish TrackResult(bbox, conf, status) │
 └──────────────────────────┘     └───────────────────────────────────────────┘
```

### 3.1 Why multi-process (the chosen model)

- Matches ARCHITECTURE §7.4/§14.7: **`spawn`, not `fork`; each RKNN context is
  created inside the process that uses it.** The parent (and QuadGuide's
  `run.py`) never load a model. Two models → two NPU contexts → two children is
  the natural fit.
- Each model gets a dedicated NPU **and** CPU core. Per the memory note
  *[[nanotrack-cpu-preprocess-bottleneck]]*, NanoTrack's cost is the CPU crop/resize
  (~11 ms on an A76, ~52 ms on an A55), not the NPU — so **both NN workers pin to
  A76 big cores** and should use `cv2`/RGA resize, not numpy. The light parent
  (state machine + SHM I/O) can sit on an A55.
- Mutual exclusion is enforced by the **control channel `mode`**: the inactive
  worker reads frames but skips inference and sleeps, keeping its context warm at
  ~zero NPU cost. This honours "YOLO stops when NanoTrack runs" without paying
  re-init latency on handoff.

### 3.2 Channels (all single-writer, wait-free seqlock reads — §7.2/§7.3)

| Channel | Writer | Reader(s) | Payload |
|---|---|---|---|
| `frame_ring` (`runtime/shm/frame_ring.py`, existing) | parent | both workers | latest frame + `(seq, ts)` |
| `control` (**new** fixed struct, like `SearchROIControl`) | parent | both workers | `mode ∈ {IDLE,YOLO,NANO}`, `crop_roi (x,y,w,h)`, `lock_gen`, `lock_bbox` |
| `yolo_result` (`PayloadChannel`, existing — variable boxes/scores) | YOLO worker | parent | `boxes (N,4)`, `scores (N,)`, source `seq` |
| `nano_result` (**new** fixed struct, like a TrackResult control word) | NanoTrack worker | parent | `bbox (x,y,w,h)`, `confidence`, `status`, source `seq` |

- The **control channel** generalises `SearchROIChannel`: same seqlock+struct
  pattern, but the struct also carries `mode`, a monotone `lock_gen` counter, and
  the `lock_bbox`. The NanoTrack worker re-initialises its template whenever it
  observes a new `lock_gen` (this is exactly the `request_refresh` mechanism the
  existing `NanoTrackDetectorAdapter` already uses, generalised to carry the bbox).
- `yolo_result` reuses `PayloadChannel` because the detection count is variable.
  `nano_result` is a fixed struct (always one box) so it uses the cheaper
  `SearchROIChannel`-style control word.
- New structs land in `runtime/shm/structs.py` and **bump `ABI_VERSION`** (§7.5/§14.4).

### 3.3 Lifecycle

`AcquireTrack` owns an `Orchestrator` (`runtime/orchestrator.py`), creates+unlinks
all segments, spawns both workers in `__init__` (or lazily on first `update`),
and tears them down in `close()`. It is a context manager (§5.3 of ARCHITECTURE).
`PR_SET_PDEATHSIG` (via `request_death_with_parent`) ties workers to the parent.
Buffer-export release rules (§7.4) apply to the new structs.

---

## 4. Latency / seq association (§6.3, §14.10)

`update(frame)` publishes frame `seq` then reads the **latest available** worker
result, which corresponds to an **earlier** `seq` (inference is a few frames
behind at 30 fps). Therefore:

- Worker results carry the **source frame `seq`**; the parent returns
  `TrackResult.seq = source_seq` and `TrackResult.timestamp = the capture ts of
  that source frame` (looked up from the frame the worker stamped), **not** the
  just-submitted frame's time.
- This matters for QuadGuide's latency lineage: its `tracker_worker` currently
  stamps `origin_ns = frame_ts` of the frame *it* read. With an async tracker that
  over-states freshness by the inference lag. **Recommended contract tweak
  (small):** have the adapter expose the EdgeCV result's source-frame timestamp and
  let `tracker_worker` use it as `origin_ns` when present. MVP can skip this and
  accept a ~1–2 frame lineage error; document it.
- The handoff bbox (the YOLO det that seeds NanoTrack) is from frame N; NanoTrack
  inits on the latest frame N+k. At 30 fps with k≈1–2 the exemplar offset is
  negligible. If it ever matters, the constant-velocity predictor
  (`fusion/predict.py`) can advance the lock bbox by k·dt — hook is available, not
  required for MVP.

---

## 5. The `Tracker` contract & the QuadGuide adapter

`AcquireTrack` implements the standard EdgeCV `Tracker` ABC (`init`, `update`,
`status`, `name`, `close`) so **nothing structural changes** in QuadGuide's
`tracker_worker`. The behavioural mapping is what's new, and it lands almost
entirely in the **adapter** (`edgecv_adapter.py`) — the designated impedance match.

### 5.1 "Update runs before lock" — adapter change

QuadGuide's `tracker_worker` already calls `tracker.update(frame)` **every tick
unconditionally** (`tracker_worker.py:179`); only `EdgeCVTracker.update` short-
circuits to `_NO_LOCK` before `_initialized`. **Change:** for `AcquireTrack`, the
adapter must call through on every frame so YOLO acquisition runs pre-lock. Cleanest
form: `AcquireTrack` self-manages state and is "running" from construction; the
adapter drops the `_initialized` gate for this tracker (or `AcquireTrack.update`
is simply always valid before `init`). No `tracker_worker` edit required.

### 5.2 "Init command locks the current YOLO detection" — no wire change

QuadGuide lock-on is `LockOnCmd(seq, bbox)` → `tracker.init(frame, bbox)`, and a
**zero-size bbox already means `reset()`** (`tracker_worker.py:213`). So the
zero-size sentinel is taken. Mapping that avoids any message-format change:

- **Any non-zero `init(frame, bbox)` = "commit/lock now."** `AcquireTrack.init`
  ignores the precise bbox and locks the **current best YOLO detection inside the
  crop** (padded by `lock_pad`). The operator's HUD button sends a lock-on whose
  bbox is just the (already known, static) crop rectangle — non-zero, harmless.
  - Fallback (**resolved**): if there is **no** qualifying detection at the
    instant of the command, `init` **seeds NanoTrack directly from the passed
    crop bbox** (manual override). So `init` *always* locks: prefer the best YOLO
    detection in the crop, else fall back to the crop box itself.
- **Zero-size bbox** keeps its meaning: `reset()` → back to `ACQUIRE`.

This preserves the 26-byte `LockOnCmd` wire format and the existing lock-on flow
(ARCHITECTURE §4.2). The only operator-facing change is that the HUD's lock button
no longer needs a hand-drawn box — it sends the crop rectangle.

### 5.3 Health mapping — keep pre-lock boxes from driving flight

QuadGuide guidance **acts on `NOMINAL` and `UNCERTAIN`** and **ignores `NO_LOCK`
and `LOST`** (`guidance/worker.py:58`). The overlay **draws only `NOMINAL`
(orange) / `UNCERTAIN` (yellow)** and nothing for `NO_LOCK`/`LOST`
(`overlay.py:19`). Consequences for the mapping:

| AcquireTrack state | EdgeCV `TrackStatus` | adapter → QuadGuide health | guidance? | drawn? |
|---|---|---|---|---|
| `ACQUIRE` (det present) | `INITIALIZING` | **must be non-driving** | no | want: yes (candidate) |
| `ACQUIRE` (no det) | `INITIALIZING` | `no_lock` | no | no |
| `LOCKED` (conf≥lock) | `LOCKED` | `nominal` | yes | yes (orange) |
| `LOCKED`/reacq (coasting) | `COASTING` | `uncertain` | yes (coasts to last bbox) | yes (yellow) |
| `LOST` (timeout) | `LOST` | `lost` | no | no |

The wrinkle: pre-lock we want the candidate box **shown but not driving**. The two
existing non-driving healths (`no_lock`, `lost`) are **not drawn**, and the one
drawn health an acquiring box could use (`uncertain`) **drives guidance**. There is
no existing health that is *drawn but not driving*. Resolution options (QuadGuide-
side, pick one at implementation):

1. **Add a `TrackerHealth.ACQUIRING`** (append to the byte enum — backward-safe).
   Guidance adds it to its ignore set; overlay draws it in a distinct colour
   (e.g. cyan). Carries the single best candidate box in the normal estimate
   bbox. **Recommended** — smallest change, no new topic.
2. **Side channel for acquisition.** Report `no_lock` on `target/estimate`
   pre-lock (nothing drives, nothing drawn), and stream the crop rectangle +
   *all* candidate boxes to the HUD via a new field in the SSE telemetry. More
   work; needed only if you want **multiple** candidate boxes drawn at once.

The **static crop rectangle** the operator sees is a fixed config value (`acquire_crop`)
and is best drawn **client-side** by the HUD (it never changes), independent of either
option above.

> The tracker spec only requires that pre-lock states report a **non-driving**
> health and carry the best candidate bbox. Which HUD mechanism shows it is a
> QuadGuide presentation choice; option 1 is the minimal path and is assumed below.

---

## 6. Models, manifests, placement

### 6.1 YOLO manifest (user supplies `yolo11n`)

`yolo11n` has the same anchor-free `yolov8`-format head as the shipped `yolo26n`,
so the existing `YoloDetector` decode path is reused verbatim. Create
`edgecv/models/manifests/yolo11n.yaml` mirroring `yolo26n.yaml`
(`output_format: yolov8`, `class_agnostic: true` — the single class id is
discarded, score = max over class columns, which for one class is just that
column). Convert ONNX→RKNN with the existing `tools/convert.py` (INT8, NMS-free
one-to-many head). Add the model's NPU core in the artifact:

```yaml
artifacts:
  onnx: { path: yolo11n.onnx }
  rknn: { path: "yolo11n.{target}.rknn", quant: int8, npu_core: 1 }   # RKNNLite NPU_CORE_0
```

The RKNN backend already reads `artifact["npu_core"]` and passes it to
`init_runtime(core_mask=...)` (`backends/rknn/__init__.py:109`) — **no runtime
change needed for per-model core placement.**

### 6.2 NanoTrack core placement

NanoTrack is split (backbone INT8 + head FP16, per *[[nanotrack-int8-head-broken]]*).
Assign both halves to a second NPU core via `npu_core` in each artifact entry of
`nanotrack.yaml`. Default split: **YOLO → core_mask `NPU_CORE_0`, NanoTrack →
`NPU_CORE_1` (head may share core 1 or take core 2).** `{target}` token selection
is unchanged (*[[nanotrack-rknn-target-selection]]*).

> RKNNLite `core_mask` encoding (informational): `AUTO=0`, `NPU_CORE_0=1`,
> `NPU_CORE_1=2`, `NPU_CORE_2=4`, `NPU_CORE_0_1=3`, `…_ALL=7`. The manifest stores
> the integer mask. Validate the exact constants against the installed
> `rknnlite` on-device.

### 6.3 Placement profile

Extend the board-profile pattern (`runtime/placement.py`, `models/profiles/rk3588.yaml`)
with the two NN workers:

```yaml
board: rk3588
processes:
  caller:     { cpu_affinity: [0],    sched: {policy: FIFO, priority: 80} }  # light state machine, A55
  yolo:       { cpu_affinity: [4, 5], npu_core: 1, backend: rknn }           # A76
  nanotrack:  { cpu_affinity: [6, 7], npu_core: 2, backend: rknn }           # A76
```

(CPU affinity for the workers is applied inside each child via
`ProcessPlacement.apply()`; NPU core is carried in the manifest artifact as above.
Keep the two sources consistent — NPU core in the manifest is authoritative for
`init_runtime`.)

---

## 7. Construction / config

`AcquireTrack` is built like other NN trackers — manifest(s) + backend — but takes
**two** model sources plus the state-machine params. Proposed signature:

```python
AcquireTrack(
    yolo_manifest, nanotrack_manifest, *,
    backend="auto",
    # acquisition
    acquire_crop=0.5, lock_pad=1.15, lock_min_score=0.35, lock_requires_detection=True,
    # track / drop
    drop_score=0.35, drop_frames=5,
    # re-acquire
    reacq_crop_factor=3.0, reacq_crop_frames=15, reacq_assoc_sigma=0.5,
    lost_timeout_frames=90,
    # passthrough
    yolo_kwargs=None, nanotrack_kwargs=None,
    profile=None,                  # placement BoardProfile; default = shipped rk3588
)
```

QuadGuide wiring (extends the existing `EdgeCVTracker` `tracker:` selector):

```yaml
tracker:
  import: quadguide.perception.edgecv_adapter:EdgeCVTracker
  params:
    tracker: acquire_track            # NEW value alongside mosse|nanotrack|siamfc|yolo
    backend: rknn
    model_dir: /home/radxa/EdgeCV/models
    acquire_crop: 0.5
    lock_pad: 1.15
    drop_frames: 5
    reacq_crop_factor: 3.0
    lost_timeout_frames: 90
```

The adapter's `_build` gains an `acquire_track` branch that loads both manifests
(`yolo11n.yaml`, `nanotrack.yaml`) and constructs `AcquireTrack`. All EdgeCV
imports stay lazy (built in the child), so `run.py`'s parent never loads RKNN
(unchanged invariant).

---

## 8. Files

```
edgecv/
├── trackers/hybrid/
│   ├── acquire_track.py        # NEW: AcquireTrack (Tracker ABC) — orchestrator + state machine
│   ├── acquire_workers.py      # NEW: _yolo_main, _nanotrack_main worker bodies (spawned)
│   └── __init__.py             # export AcquireTrack
├── runtime/shm/
│   ├── structs.py              # +AcquireControl, +NanoResult structs; BUMP ABI_VERSION
│   ├── control_channel.py      # NEW: mode+crop+lock_gen+lock_bbox channel (SearchROI-style)
│   └── nano_result.py          # NEW: single-box result channel (SearchROI-style)
├── models/manifests/
│   └── yolo11n.yaml            # NEW (user model; yolov8 head, class-agnostic, npu_core)
└── models/profiles/
    └── rk3588.yaml             # add yolo/nanotrack worker placement rows

# QuadGuide side
src/quadguide/perception/edgecv_adapter.py   # +acquire_track branch; always-update; init=commit
src/quadguide/core/messages.py               # OPTIONAL: +TrackerHealth.ACQUIRING (HUD option 1)
src/quadguide/guidance/worker.py             #   ignore ACQUIRING; src/.../ground/overlay.py draw it
docs/ (QuadGuide)                            # note the lock-button-sends-crop-box contract
```

---

## 9. Testing (x86, no NPU — mock/onnx backends per §15)

- **State machine, mocked workers.** Drive `AcquireTrack` with a mock detector
  that returns scripted boxes/scores; assert the full transition table:
  acquire→lock on command, lock→reacq_crop after `drop_frames` misses,
  reacq_crop→reacq_full after `reacq_crop_frames`, reacq→lock on a confident
  associated detection, reacq→lost after `lost_timeout_frames`.
- **Lock semantics.** `init` with a non-zero bbox while a qualifying detection
  exists → LOCKED seeded from the (padded) detection; with no detection and
  `lock_requires_detection=True` → stays ACQUIRE; zero-size bbox → reset to ACQUIRE.
- **Mutual exclusion.** Assert that in every state exactly one worker is in
  non-idle `mode` (control-channel inspection).
- **seq association.** `TrackResult.seq` matches the source frame the active
  worker computed from, and is monotonic; lineage timestamp is the source frame's.
- **Health mapping (adapter).** ACQUIRE→non-driving health; LOCKED→nominal;
  coasting→uncertain; timeout→lost. (Guidance-ignore + overlay-draw behaviour is
  covered in QuadGuide's tests.)
- **Lifecycle.** Context-manager teardown releases all SHM exports before close
  (§7.4); `close()` reaps both workers; ABI header validates on attach.
- **On-device smoke (manual).** Acquire→lock→track→occlude→reacq_crop→reacq_full
  →re-lock loop; confirm per-state NPU core occupancy and that idle-worker NPU use
  is ~0; confirm NanoTrack worker uses cv2/RGA resize (not numpy) per the
  preprocess-bottleneck note.

---

## 10. Defaults chosen for you (override in config)

- Acquisition crop: central square, `0.5 × min(W,H)`.
- Lock target selection: **highest-score** detection inside the crop above
  `lock_min_score`.
- Re-acq target selection: proximity-weighted to the last-known centre
  (`score · exp(-d²/2σ²)`, σ from `reacq_assoc_sigma`) — same association as
  `YoloTracker.update`, so weak/distant detections are rejected.
- Drop hysteresis: `drop_frames=5` consecutive sub-`drop_score` frames.
- Coast budget: `lost_timeout_frames=90` (~3 s at 30 fps) before `LOST`.
- NPU cores: YOLO core 0, NanoTrack core 1 (head may share). Both NN workers on
  A76 cores; parent on an A55.
- HUD: option 1 (new `ACQUIRING` health, drawn cyan, ignored by guidance); crop
  rectangle drawn client-side from `acquire_crop`.

---

## 11. Resolved decisions (2026-06-14)

1. **HUD acquisition display** — **option 1**: new `TrackerHealth.ACQUIRING`,
   single best candidate box carried on `target/estimate`, drawn distinctly,
   ignored by guidance.
2. **Post-`LOST` behaviour** — keep auto-searching full-frame (auto re-lock, no
   operator command) until `search_timeout_frames` elapses, then reset to
   `ACQUIRE` (drop to crop, await a fresh init command). `0` = search forever.
3. **Manual override on lock** — lock with no qualifying detection **seeds
   NanoTrack from the sent crop box**. `init` always locks.
4. **NanoTrack core sharing** — head **shared with the backbone** (both on
   NanoTrack's core). Current `nanotrack.yaml` sets no `npu_core` → runs `AUTO`
   (mask 0); switch both halves to the explicit NanoTrack core. Measure on-device
   and revisit only if the head wants its own core.
5. **Latency lineage** — **adopt** the source-frame `origin_ns` adapter tweak (§4):
   workers stamp source `seq`+capture ts; the parent returns it; the adapter
   surfaces it; `tracker_worker` uses it as `origin_ns` when present.
```

# AcquireTrack — YOLO-acquire → NanoTrack-track hybrid: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `AcquireTrack` — a hybrid tracker that acquires a target with YOLO (single-class `yolo11n`) on a fixed central crop, locks NanoTrack onto it on an operator init command, and re-acquires with YOLO (crop → full-frame) when NanoTrack drops. YOLO and NanoTrack run mutually-exclusive, each in its own spawned worker on its own RK3588 NPU core. It implements the standard EdgeCV `Tracker` ABC and slots into QuadGuide via the existing `EdgeCVTracker` adapter.

**Architecture:** Parent (the `AcquireTrack` object) owns an `Orchestrator`, a `FrameRing`, a control channel (parent→workers: mode/crop/lock), a YOLO result channel (`PayloadChannel`, variable boxes), and a NanoTrack result channel (fixed single-box struct). The state machine (`ACQUIRE / LOCKED / REACQ_CROP / REACQ_FULL / LOST`) runs inline in `update()`; workers free-run reading the latest frame and infer only when their `mode` is active. Spawn-only (RKNN contexts never survive fork); per-model NPU core via the manifest `npu_core` the rknn backend already honours.

**Tech Stack:** Python, numpy. Reuses `runtime/` SHM primitives, `backends/`, `trackers/nn/yolo.py` (`YoloDetector`) and `trackers/nn/nanotrack.py` (`NanoTrack`). No new runtime dependencies. Tests run on x86 with `mock`/`onnx` backends — no NPU.

**Spec:** `docs/superpowers/specs/2026-06-14-acquire-track-design.md`

**Resolved decisions (spec §11):** HUD = new `ACQUIRING` health (single best candidate, drawn, non-driving); post-LOST auto-search full-frame until `search_timeout_frames` then reset to ACQUIRE; lock with no detection seeds NanoTrack from the crop box; NanoTrack head shares the backbone's NPU core; adopt the source-frame `origin_ns` lineage tweak.

---

## File structure

### EdgeCV

| File | Responsibility | Action |
|---|---|---|
| `edgecv/runtime/shm/structs.py` | `+AcquireControl`, `+NanoResult` structs; **bump `ABI_VERSION` 2→3** | Modify |
| `edgecv/runtime/shm/control_channel.py` | `AcquireControlChannel`: mode + crop_roi + lock_gen + lock_bbox (SearchROI-style seqlock) | Create |
| `edgecv/runtime/shm/nano_result.py` | `NanoResultChannel`: single bbox + conf + status + source seq/ts | Create |
| `edgecv/runtime/worker.py` | wire `detach_resource_tracker` into `child_main` for attaching workers | Modify |
| `edgecv/trackers/hybrid/acquire_workers.py` | `_yolo_main`, `_nanotrack_main` spawned worker bodies | Create |
| `edgecv/trackers/hybrid/acquire_track.py` | `AcquireTrack(Tracker)` — orchestrator + state machine | Create |
| `edgecv/trackers/hybrid/__init__.py` | export `AcquireTrack` | Modify |
| `edgecv/models/manifests/yolo11n.yaml` | single-class yolov8-head manifest + `npu_core` | Create |
| `edgecv/models/manifests/nanotrack.yaml` | add `npu_core` to backbone+head artifacts | Modify |
| `edgecv/models/profiles/rk3588.yaml` | add `yolo` / `nanotrack` worker placement rows | Modify |
| `tests/_acquire_stubs.py` | scripted mock YOLO/NanoTrack `Model`s + control-channel helpers | Create |
| `tests/test_acquire_channels.py` | control + nano-result channel round-trip / seqlock / lifecycle | Create |
| `tests/test_acquire_state_machine.py` | full transition table with mocked workers | Create |
| `tests/test_acquire_lifecycle.py` | spawn/teardown, ABI, buffer-export release | Create |
| `tests/test_manifests_nn.py` | yolo11n manifest loads | Modify |
| `ARCHITECTURE.md` | note AcquireTrack hybrid in §6.3 + reference this spec | Modify |

### QuadGuide (`~/QuadGuide`)

| File | Responsibility | Action |
|---|---|---|
| `src/quadguide/core/messages.py` | `+TrackerHealth.ACQUIRING` (append to byte enum) | Modify |
| `src/quadguide/guidance/worker.py` | add `ACQUIRING` to the ignore set | Modify |
| `src/quadguide/ground/overlay.py` | draw `ACQUIRING` in a distinct colour | Modify |
| `src/quadguide/perception/edgecv_adapter.py` | `acquire_track` branch; always-update; `init`=commit; `ACQUIRING`/source-`origin_ns` mapping | Modify |
| `src/quadguide/perception/tracker_worker.py` | use tracker-provided `origin_ns`/source ts when present | Modify |
| `src/quadguide/ground/static/index.html` | draw the static central crop rectangle (client-side from `acquire_crop`); lock button sends crop box | Modify |
| `configs/rk3588.yaml` | add an `acquire_track` tracker preset (commented or active) | Modify |
| `tests/...` | guidance ignores ACQUIRING; overlay draws it; adapter mapping | Modify |

Run EdgeCV tests from repo root with `.venv` active (`conftest.py` puts `tools/` on `sys.path`).

---

## Phase A — EdgeCV SHM channels (no NPU)

## Task 1: `AcquireControl` + `NanoResult` structs, ABI bump

**Files:** Modify `edgecv/runtime/shm/structs.py`; Test `tests/test_acquire_channels.py`

- [ ] **Step 1: Failing test** — assert both structs begin with `magic, abi_version, seq, seqlock` and that `ABI_VERSION == 3`; assert `ctypes.sizeof` is stable (record the value).
- [ ] **Step 2: Implement** — add:
  - `AcquireControl`: `magic, abi_version, seq, seqlock (u64)`, `mode (u32: 0=IDLE,1=YOLO,2=NANO)`, `lock_gen (u64)`, crop `cx,cy,cw,ch (double, normalised)`, lock `lx,ly,lw,lh (double)`, `timestamp (double)`.
  - `NanoResult`: `magic, abi_version, seq, seqlock (u64)`, `x,y,w,h (double)`, `confidence (double)`, `status (u32 = TrackStatus int)`, `src_seq (u64)`, `src_ts (double)`.
  - Bump `ABI_VERSION = 3`. Document the change in the module docstring per §7.5.
- [ ] **Step 3: Verify** — `python -c` size check + grep that no other ABI consumer hardcodes `2`.

## Task 2: `AcquireControlChannel`

**Files:** Create `edgecv/runtime/shm/control_channel.py`; Test `tests/test_acquire_channels.py`

- [ ] **Step 1: Failing test** — `create()` → `publish(mode, crop_bbox, lock_gen, lock_bbox)` → `attach()` → `read_latest()` returns the same values; `read_latest()` before first publish returns a default (`mode=IDLE, lock_gen=0`); torn-read safety under a concurrent writer thread (reuse the seqlock test pattern from `tests/test_search_roi*` if present).
- [ ] **Step 2: Implement** — mirror `SearchROIChannel` exactly (owner sets magic/abi/zeroes; `validate_header` on attach; `SeqLock` at `AcquireControl.seqlock.offset`; `publish` does `write_begin/…/write_end`; `read` via `self._seqlock.read(snapshot)`). `close(unlink)` drops the header view, `seqlock.release()`, `gc.collect()`, `shm.close()`, owner-only `unlink`.
- [ ] **Step 3: Verify** — round-trip + lifecycle (`close()` raises no `BufferError` on CPython ≥3.12, §7.4).

## Task 3: `NanoResultChannel`

**Files:** Create `edgecv/runtime/shm/nano_result.py`; Test `tests/test_acquire_channels.py`

- [ ] **Step 1: Failing test** — publish a `(BoundingBox, confidence, TrackStatus, src_seq, src_ts)` → read back identical; pre-publish read returns `None`.
- [ ] **Step 2: Implement** — same SeqLock+struct pattern as Task 2 over `NanoResult`. Reader returns a small dataclass/namedtuple `NanoSample(bbox, confidence, status, src_seq, src_ts)`.
- [ ] **Step 3: Verify** — round-trip + buffer-export release.

---

## Phase B — Worker bodies (mockable, no NPU)

## Task 4: YOLO worker body

**Files:** Create `edgecv/trackers/hybrid/acquire_workers.py`; Test `tests/test_acquire_state_machine.py` (uses `tests/_acquire_stubs.py`)

Worker signature mirrors `trackers/hybrid/worker.py:_detector_main` (factory + SHM names + stop_event), constructs everything **inside the child**.

- [ ] **Step 1: Failing test** — with a stub `YoloDetector` (injected `Model`) and in-process SHM, run one loop iteration: when control `mode==YOLO`, it reads the latest frame, crops to control `crop_roi`, runs detect, and publishes boxes+scores+`src_seq` to the YOLO `PayloadChannel`; when `mode!=YOLO` it publishes nothing and sleeps.
- [ ] **Step 2: Implement** `_yolo_main(detector_factory, det_cfg, fr_name, ctrl_name, result_name, max_h/w/c, slots, payload_cap, stop_event)`:
  - `request_death_with_parent()`, attach FrameRing / `AcquireControlChannel` / `PayloadChannel` (attach-only, `detach_resource_tracker`).
  - Build `YoloDetector` from `det_cfg` (manifest+backend) in-child.
  - Loop: read control; if `mode != YOLO`, `sleep(1ms); continue`. Read latest frame; skip if `seq <= last`. Crop to `crop_roi` (use `preprocess.crop_with_context`; full-frame when `crop_roi` covers the frame). `detect()`, map boxes back to full-frame normalised (reuse the `YoloDetectorAdapter.detect` mapping logic). Publish `{boxes, scores}` with `src_seq`, plus `src_ts`.
  - `finally`: detector.close(); close channels `unlink=False`.
- [ ] **Step 3: Verify** — stub-driven single-iteration + idle-skip tests pass.

## Task 5: NanoTrack worker body

**Files:** Modify `edgecv/trackers/hybrid/acquire_workers.py`; Test `tests/test_acquire_state_machine.py`

- [ ] **Step 1: Failing test** — stub `NanoTrack` (injected backbone/head `Model`s): when a **new `lock_gen`** appears, the worker calls `nanotrack.init(frame, lock_bbox)`; on subsequent `mode==NANO` frames it calls `update()` and publishes the result (bbox/conf/status/src_seq) to `NanoResultChannel`; `mode!=NANO` → idle (no publish).
- [ ] **Step 2: Implement** `_nanotrack_main(nano_factory, nano_cfg, fr_name, ctrl_name, result_name, …, stop_event)`:
  - Attach FrameRing / control / `NanoResultChannel`; build `NanoTrack.from_manifest(...)` in-child.
  - Track `last_lock_gen`. Loop: read control; if `mode != NANO`, idle. If control `lock_gen != last_lock_gen`: read latest frame, `nanotrack.init(frame, lock_bbox)`, set `last_lock_gen`, continue. Else read latest frame (skip stale), `res = nanotrack.update(frame)`, publish `NanoResult(res.bbox, res.confidence, res.status, src_seq, src_ts)`.
  - `finally`: nanotrack.close(); close channels.
- [ ] **Step 3: Verify** — init-on-new-lock_gen + update-publish + idle tests pass.

---

## Phase C — The tracker (state machine)

## Task 6: `AcquireTrack` skeleton, lifecycle, ACQUIRE state

**Files:** Create `edgecv/trackers/hybrid/acquire_track.py`; Modify `edgecv/trackers/hybrid/__init__.py`; Test `tests/test_acquire_lifecycle.py`, `tests/test_acquire_state_machine.py`

- [ ] **Step 1: Failing test** — construct `AcquireTrack` with stub factories + `mp_context="spawn"`; assert: it is a context manager; `name()=="AcquireTrack"`; after construction the control channel `mode==YOLO` with `crop_roi` = the fixed central crop (`acquire_crop`); `update(frame)` publishes the frame and returns a `TrackResult` with `status==INITIALIZING` and bbox = best YOLO candidate (or `None` when no detection); `close()` reaps workers and raises no `BufferError`.
- [ ] **Step 2: Implement**:
  - `__init__(yolo_manifest, nanotrack_manifest, *, backend="auto", <params from spec §7>, profile=None, mp_context="spawn")`. Create `Orchestrator`, `FrameRing`, `AcquireControlChannel`, YOLO `PayloadChannel`, `NanoResultChannel` (parent owns+unlinks). Add two `WorkerSpec`s; `start()`. Apply placement (`profile` or shipped rk3588) — affinity inside children.
  - State enum `ACQUIRE/LOCKED/REACQ_CROP/REACQ_FULL/LOST`; start in `ACQUIRE`, publish control `mode=YOLO, crop=central`.
  - `update(frame)`: `self._seq+=1`; `frame_ring.publish`; `self._tick()` (state machine, Task 7+); return the latest `TrackResult` (seq = source seq, timestamp = source ts — §4).
  - ACQUIRE handling: read YOLO result; pick best in-crop detection above `lock_min_score`; cache it as the lock candidate; return `INITIALIZING` with that bbox (or `None`).
  - `close()` / `__exit__`: control `mode=IDLE`; `orchestrator.close()`; close+unlink all segments (drop views first, §7.4).
- [ ] **Step 3: Verify** — lifecycle + ACQUIRE tests pass; `Orchestrator` enforces spawn.

## Task 7: Lock (`init`) → LOCKED, and the LOCKED→drop transition

**Files:** Modify `acquire_track.py`; Test `tests/test_acquire_state_machine.py`

- [ ] **Step 1: Failing test**:
  - `init(frame, bbox)` with a qualifying YOLO candidate present → state `LOCKED`, control `mode=NANO`, `lock_gen` incremented, `lock_bbox` = padded candidate (`lock_pad`). With **no** candidate → `lock_bbox` = the passed crop bbox (seed-from-crop), still `LOCKED`. Zero-size bbox → `reset()` → back to `ACQUIRE`.
  - In LOCKED, feeding NanoTrack results with confidence `< drop_score` for `drop_frames` consecutive ticks → transition to `REACQ_CROP` (control `mode=YOLO`, crop = `reacq_crop_factor × last bbox` centred on last bbox); a single recovered frame resets the miss counter.
- [ ] **Step 2: Implement** — `init()` sets the lock candidate→`lock_bbox`, bumps `lock_gen`, publishes control `mode=NANO`; `reset()` → `mode=YOLO`, central crop, state ACQUIRE. LOCKED tick: read `NanoResultChannel`; map status/confidence; maintain `miss_count` hysteresis; on threshold cross, compute the re-acq crop and switch control to YOLO.
- [ ] **Step 3: Verify** — lock/seed/reset + drop-hysteresis tests pass.

## Task 8: Re-acquisition (CROP → FULL → re-lock → LOST → search timeout)

**Files:** Modify `acquire_track.py`; Test `tests/test_acquire_state_machine.py`

- [ ] **Step 1: Failing test** — drive scripted YOLO results:
  - `REACQ_CROP`: report `COASTING` with the **held last-known bbox**; on a confident detection associated near the last centre (`score·exp(-d²/2σ²) ≥` gate) → re-lock (`mode=NANO`, new `lock_gen`) → `LOCKED`. No detection for `reacq_crop_frames` ticks → `REACQ_FULL` (crop = full frame).
  - `REACQ_FULL`: report `COASTING` (held bbox); confident associated detection → re-lock. No re-lock for `lost_timeout_frames` (counted from drop) → report `LOST` (still searching full-frame).
  - After `LOST`, a confident detection → auto re-lock (no operator cmd). No re-lock for `search_timeout_frames` after entering LOST → `reset()` to `ACQUIRE` (central crop). `search_timeout_frames==0` → never give up.
- [ ] **Step 2: Implement** — re-acq association reuses `YoloTracker.update`'s proximity weighting (`reacq_assoc_sigma`, scaled by last bbox size). Maintain frame counters for the three timeouts. Re-lock path mirrors `init()` (bump `lock_gen`, `mode=NANO`).
- [ ] **Step 3: Verify** — full transition table green; assert **exactly one** worker non-idle in every state (control-channel inspection).

---

## Phase D — Models, manifests, placement

## Task 9: `yolo11n` manifest + nanotrack/profile NPU cores

**Files:** Create `edgecv/models/manifests/yolo11n.yaml`; Modify `edgecv/models/manifests/nanotrack.yaml`, `edgecv/models/profiles/rk3588.yaml`, `tests/test_manifests_nn.py`

- [ ] **Step 1: Failing test** — `load_manifest(yolo11n.yaml)` loads; `preprocessing.output_format=="yolov8"`, `class_agnostic` true; both nanotrack artifacts now carry `npu_core`.
- [ ] **Step 2: Implement**:
  - `yolo11n.yaml` mirrors `yolo26n.yaml` (yolov8 anchor-free head, `input: 640`, `scale 1/255`, conf/iou defaults). Artifact: `onnx: {path: yolo11n.onnx}`, `rknn: {path: "yolo11n.{target}.rknn", quant: int8, npu_core: 1}`. (User supplies the trained ONNX; convert via `tools/convert.py`.)
  - `nanotrack.yaml`: add `npu_core: 2` to **both** backbone and head rknn artifacts (head shares the backbone's core per decision 4).
  - `rk3588.yaml` profile: `yolo {cpu_affinity:[4,5], npu_core:1}`, `nanotrack {cpu_affinity:[6,7], npu_core:2}`, `caller {cpu_affinity:[0]}` (light state machine on A55).
- [ ] **Step 3: Verify** — manifest/profile load tests pass. Note: convert the user's `yolo11n.onnx` on-device/host separately; validate RKNNLite `core_mask` constants against the installed runtime.

---

## Phase E — QuadGuide integration (contract)

## Task 10: `TrackerHealth.ACQUIRING` + guidance/overlay

**Files (QuadGuide):** Modify `core/messages.py`, `guidance/worker.py`, `ground/overlay.py`; Tests under `tests/`

- [ ] **Step 1: Failing test** — `ACQUIRING` packs/unpacks (byte-enum append, no wire-size change); guidance `continue`s (no accel) on `ACQUIRING`; overlay draws a rectangle for `ACQUIRING` in a distinct colour (e.g. cyan) and still nothing for `NO_LOCK`/`LOST`.
- [ ] **Step 2: Implement** — append `ACQUIRING = "acquiring"` to `TrackerHealth` (after `NO_LOCK`; `_byte_enum` ordinals are append-safe). Add it to guidance's ignore tuple (`worker.py:58`). Add a colour + draw branch in `overlay.py`. Confirm `_ord`/`_from_ord` round-trip and that the 37-byte estimate is unchanged.
- [ ] **Step 3: Verify** — guidance/overlay tests pass.

## Task 11: Adapter `acquire_track` branch + always-update + `init`=commit + origin lineage

**Files (QuadGuide):** Modify `perception/edgecv_adapter.py`, `perception/tracker_worker.py`; Tests under `tests/`

- [ ] **Step 1: Failing test** — adapter with `tracker: acquire_track`:
  - `update()` calls through to the EdgeCV tracker **before** any `init` (no `_initialized` short-circuit for this tracker); ACQUIRE result maps to health `acquiring` with the candidate bbox.
  - A non-zero `init(bbox)` forwards to the EdgeCV `init` (commit/lock); zero-size bbox → `reset()`.
  - LOCKED→`nominal`, COASTING→`uncertain`, LOST→`lost`, INITIALIZING(no det)→`acquiring` with zero bbox or `no_lock` when truly nothing.
  - The adapter surfaces the EdgeCV result's **source-frame capture ts**; `tracker_worker` uses it as `origin_ns` when present (else falls back to `frame_ts`).
- [ ] **Step 2: Implement**:
  - `_NN_MANIFESTS` / `_build`: add `acquire_track` → load `yolo11n.yaml` + `nanotrack.yaml`, construct `AcquireTrack(...)` forwarding params; all imports lazy (child).
  - Add `_status_to_health` mapping incl. `INITIALIZING→acquiring`. Drop the `_initialized` gate for this tracker (or make `AcquireTrack.update` valid pre-init and let the worker's unconditional `update` flow).
  - Surface `res.timestamp`/source seq from the EdgeCV `TrackResult`; expose an `origin_ns` on the adapter's output namedtuple. In `tracker_worker.run()`, set `origin_ns = out.origin_ns if getattr(out,'origin_ns',0) else frame_ts`.
- [ ] **Step 3: Verify** — adapter mapping + lineage tests pass.

## Task 12: HUD crop rectangle + lock button + config preset

**Files (QuadGuide):** Modify `ground/static/index.html`, `configs/rk3588.yaml`

- [ ] **Step 1:** HUD draws the static central crop rectangle client-side from a config-exposed `acquire_crop` (via SSE telemetry or a served constant); the **lock button** issues a `POST /lockon` carrying the crop box (non-zero) — no hand-drawn box required. Reset button keeps sending the zero-size box.
- [ ] **Step 2:** `configs/rk3588.yaml`: add an `acquire_track` tracker preset (params from spec §7) — commented next to the existing `nanotrack` preset, or behind a profile.
- [ ] **Step 3: Verify** — manual: HUD shows crop + candidate boxes pre-lock; lock button locks; reset returns to acquire.

---

## Phase F — Docs & on-device validation

## Task 13: Docs

- [ ] Update EdgeCV `ARCHITECTURE.md` §6.3 to name `AcquireTrack` as a non-MAFiD hybrid (NN→NN handoff, mutually-exclusive, two NPU cores) and link this spec. Add a short note to QuadGuide `ARCHITECTURE.md` §11 (EdgeCV trackers) on the `acquire_track` value, the `init`=commit contract, and the `ACQUIRING` health.

## Task 14: On-device smoke (manual, RK3588 — not CI)

- [ ] Convert the user's `yolo11n.onnx` → `yolo11n.<soc>.rknn` (`tools/convert.py`, int8). Place under `models/`.
- [ ] Run `tools/track_webcam.py` (or QuadGuide `dev_ground_perception.py --log`) with `acquire_track`. Verify: acquire→lock→track→occlude→reacq_crop→reacq_full→re-lock→lost→search-timeout→reset.
- [ ] Confirm per-state NPU core occupancy; idle-worker NPU ≈ 0; NanoTrack worker uses cv2/RGA resize, not numpy (preprocess-bottleneck note). Capture the `--log` latency trace; confirm `origin_ns` lineage reflects inference lag. **Decide** whether the NanoTrack head wants its own core (decision 4 revisit).

---

## Invariants to honour (ARCHITECTURE §14)

- `spawn`, never `fork`; backends built in-child (Orchestrator enforces; workers build models in their body).
- Single writer per shared region: parent writes control + frame ring; each worker writes only its own result channel.
- Wait-free seqlock reads; contended reads yield (reuse existing `SeqLock`).
- Bump `ABI_VERSION` (Task 1); headers carry magic+abi, validated on attach.
- Release every buffer export before `close()` (drop ctypes/numpy views, `seqlock.release()`, `gc.collect()`).
- `seq`/`timestamp` travel with every frame and result; the parent returns source-frame seq/ts (§4).
- No new runtime deps; tests pass on x86 with mock/onnx (no NPU).

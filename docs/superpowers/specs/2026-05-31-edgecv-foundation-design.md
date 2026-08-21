# edgecv — Foundation (dependencies) design

> **Date:** 2026-05-31
> **Status:** approved for planning
> **Companion:** `ARCHITECTURE.md` (authoritative). This spec scopes the *first build*:
> everything trackers depend on, with **no concrete trackers**. Where this spec and
> `ARCHITECTURE.md` disagree, `ARCHITECTURE.md` wins and this spec should be corrected.

## 1. Goal

Stand up `edgecv` as a pip-installable package with a **working foundation**: the core
data types, the runtime/IPC stack, the hardware-abstraction backends, the fusion/predictor
abstractions, the model-manifest layer, and packaging — all CI-runnable on x86 with no NPU.

**Explicitly not in this build:** concrete CF/NN/hybrid trackers, CF `ops/` implementations,
the reference PSR-gate fusion policy, the constant-velocity predictor, and the hybrid
`update()` backbone. These are deferred to later specs.

## 2. Settled parameters

| Parameter | Decision |
|---|---|
| Completeness | Working foundation (real, tested impls), no concrete trackers |
| Backends | `mock` full, `onnx` full, `rknn` lazy adapter |
| Fusion / predictor | **ABCs only** — no reference policy or predictor impl |
| Min Python | **3.10+** |
| Build backend | hatchling |
| Layout | flat `edgecv/` at repo root (matches `ARCHITECTURE.md` §13) |
| YAML | PyYAML as a **core** dependency (manifests + board profiles are YAML) |
| Seqlock | pure-Python ctypes over `shared_memory`, §7.3 caveat documented in-code |
| CI | minimal GitHub Actions: pytest + ruff + mypy on x86 (mock/onnx only) |

## 3. Deliverables

### 3.1 Packaging

- `pyproject.toml` (hatchling, PEP 621). Core deps: `numpy`, `PyYAML`.
  - Extras:
    - `[onnx]` → `onnxruntime`
    - `[rknn]` → **marker only**; `rknn-toolkit-lite2` is not on PyPI and is installed
      manually on-device — document this. The extra exists to signal intent / register the
      backend, not to pull the runtime.
    - `[dev]` → host-side conversion/training tooling per `ARCHITECTURE.md` §11
      (documented; several deps are not cleanly pip-installable, so kept minimal + noted).
  - A separate lint/test dependency group (`pytest`, `ruff`, `mypy`).
  - Entry-point group `edgecv.backends` registering `mock`, `onnx`, `rknn`.
  - Package data: shipped board profile(s).
- `.gitignore`, short `README.md`, CI workflow (`.github/workflows/ci.yml`).
- `LICENSE` left to the user (flagged, not chosen here).

### 3.2 `edgecv/core/` — full

- `bbox.py` — `BoundingBox` (normalised 0–1), `PixelBox`, `to_pixels`/`from_pixels`,
  validation/clamping. Norm-vs-pixel separation enforced by type (§5.1).
- `result.py` — `TrackStatus(IntEnum)`, `TrackResult` (§5.2).
- `tracker.py` — `Tracker` ABC: `init`/`update`/`status`/`name`/`close` + context manager (§5.3).

### 3.3 `edgecv/backends/`

- `base.py` — `InferenceBackend`, `Model`, `IOSpec`, `Handle` ABCs (§10).
- `registry.py` — entry-point (`edgecv.backends`) registry with **lazy import**;
  `get_backend(name)`, `list_backends()`, availability checks (§10 registry note).
- `mock/` — full. `MockBackend`/`MockModel` produce canned outputs shaped by `io_spec`.
  Lets the runtime/IPC stack run with no model and no accelerator.
- `onnx/` — full. Wraps `onnxruntime` (CPU), **lazy import**; derives `io_spec` from the
  ONNX model.
- `rknn/` — lazy adapter. `is_available()` returns `False` when `rknnlite` is unimportable;
  `load`/`infer` raise a clear, actionable error if used without the runtime. Registered via
  entry point. Initialised **inside the worker process** only.

### 3.4 `edgecv/runtime/` — full

- `shm/structs.py` — single source of truth for shared layouts. `MAGIC` + `ABI_VERSION`,
  ctypes structs for control words and payload headers, validated on attach (§7.5).
- `shm/seqlock.py` — `SeqLock` over a shared `uint64`: `write_begin`/`write_end` (odd→even),
  wait-free read-with-retry helper. In-code caveat re: memory barriers (§7.3).
- `shm/frame_ring.py` — N fixed-size slots sized for max resolution; zero-copy numpy views;
  **latest-only** consumer semantics; triple-or-more buffering for recycling (§7.1).
- `shm/payload.py` — variable-shape numpy payload channel: max-size byte buffer + header
  (`magic, abi_version, seq, n_arrays, [name, dtype, shape, offset]…`) under a seqlock (§7.2).
- `orchestrator.py` — spawns workers via **spawn/forkserver (never fork)**; **centralised SHM
  ownership** (parent creates + unlinks, children attach only); heartbeat reap/optional
  restart; `close()` + context-manager lifecycle (§7.4).
- `worker.py` — child entrypoint: attach-only SHM, detach `resource_tracker` for attached
  segments, `PR_SET_PDEATHSIG`, **backend initialised in-child** (§7.4).
- `placement.py` — `BoardProfile` dataclass + YAML loader + shipped `rk3588` default; applies
  CPU affinity (`sched_setaffinity`) and optional `SCHED_FIFO`; records NPU core mask for the
  backend to consume (§7.6). No placement hardcoded in trackers.

### 3.5 `edgecv/fusion/` — ABCs only

- `policy.py` — `FusionPolicy` ABC + `DetectorOutput`, `FusionDecision` dataclasses (§8).
- `predict.py` — `MotionPredictor` ABC (§9).
- **Deferred:** `psr_gate.py` (reference policy) and the constant-velocity predictor.

### 3.6 `edgecv/trackers/` — contracts + skeleton only

- `cf/base.py` — `CorrelationFilterTracker` ABC + `FilterState`, `EvalResult` dataclasses:
  the **mandatory transferable-filter contract** (`update`, pure `build_filter`/`evaluate`,
  `get_filter`/`set_filter`, `response_map`/`psr`) per §6.1 / §14.5.
- `cf/ops/`, `nn/`, `hybrid/` — empty package skeletons (`__init__.py`) only.
- **Deferred:** all concrete trackers, all CF ops implementations, the hybrid §6.3 backbone.

### 3.7 `edgecv/models/` — full

- `manifest.py` — `ModelManifest` schema (name, task, preprocessing, I/O spec, per-backend
  artifacts), YAML loader, validation (§10.1).
- Shipped `rk3588` board profile as package data (consumed by `placement.py`).

### 3.8 `tools/` — directory + README only

Host-only conversion/training helpers per §11. **Not a runtime dependency.** No
implementation in this build.

### 3.9 `tests/` — runs on x86, no NPU

- `bbox` / `result` round-trips and validation.
- `seqlock` under concurrent read/write (torn-read detection).
- `frame_ring` latest-only + slot recycling.
- `payload` variable-shape round-trip + ABI validation.
- `registry` lazy import + availability.
- `mock` backend I/O.
- `onnx` backend (`skipif` onnxruntime missing).
- `manifest` loader + validation.
- `placement` profile load (affinity application guarded for CI portability).

## 4. Invariants honoured (from `ARCHITECTURE.md` §14)

Single-writer per shared region; wait-free seqlock reads; latest-only for trackers; ABI
discipline (`magic + abi_version`, bump on layout change); pure `build_filter`/`evaluate`
contract (enforced at the ABC + test level); `spawn` not `fork` with backends init-in-child;
centralised SHM ownership; `seq`/`timestamp` travel with every frame/result/payload.

## 5. Out of scope (deferred to later specs)

Concrete CF trackers (MOSSE/CSK/KCF/DSST/Staple…), CF `ops/` implementations, NN trackers
(SiamFC/RPN…), hybrid `update()` backbone + MAFiD-style tracker, reference PSR-gate policy,
constant-velocity predictor, shipped model artifacts + fetch logic, host conversion/training
tooling, motion-cue worker.

## 6. Open items (non-blocking)

- `LICENSE` choice (user).
- Model-artifact packaging strategy (§11) — deferred until the first NN tracker.
- Whether the seqlock control word eventually needs a tiny C extension (§7.3) — revisit only
  if a correctness issue is observed.

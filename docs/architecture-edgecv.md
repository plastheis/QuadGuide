# edgecv — Architecture

> **Status:** living document. Serves two audiences: (1) humans designing and reviewing the
> library, and (2) coding agents that need authoritative context before editing. When a change
> contradicts this document, update the document in the same change.

> **Note:** EdgeCV was merged into QuadGuide on 2026-08-17 and now lives at
> `src/edgecv/`. This document describes the tracking library; the system it
> runs inside is described in [`ARCHITECTURE.md`](../ARCHITECTURE.md). Paths
> below that read `edgecv/...` are now `src/edgecv/...`.

`edgecv` is a pip-installable Python library of **single-object visual trackers** scoped to
**real-time deployment on edge hardware**. It houses individual trackers (conventional
correlation-filter trackers and dense-network trackers such as Siamese trackers) and hybrid
"fusion" trackers that compose them across processes. Trackers run efficiently by offloading
dense work to neural accelerators on SoMs/SBCs (e.g. the Rockchip RK3588 NPU via RKNN tooling),
behind an abstraction that hides the chip-specific runtime.

---

## 1. Scope and non-goals

**In scope**
- Single-object tracking (SOT). One target in, one bounding box out per `update()`.
- An OpenCV-like tracker API: importable classes with `init()` / `update()`.
- Correlation-filter (CF) trackers built from fundamental ops orchestrated in Python — **not**
  OpenCV *tracker classes*. The exclusion is specifically `cv2.Tracker*` (KCF, CSRT, …): they are
  sealed boxes that expose no filter, no PSR, and no build/evaluate split, so the build-elsewhere /
  evaluate-here / swap contract (§6.1) is impossible on top of them. It is **not** a ban on
  optimized *primitives*: the underlying ops (FFT, image crop/resize/colour-convert, HOG) may use
  the best aarch64-available library — numpy, scipy.fft, pyFFTW, numba, or `cv2`'s op functions —
  behind a swappable backend (§6.1). What edgecv owns is the filter state and its transferable
  composition, never the box. The ports in
  [`pyCFTrackers`](https://github.com/fengyang95/pyCFTrackers) are the reference for **per-tracker
  algorithm and parameters** (faithful ports of the official MATLAB code), but note: their filters
  live as private instance attributes with no transferable contract, and they depend on `cv2` +
  compiled HOG. edgecv therefore reimplements the feature/FFT ops in `trackers/cf/ops/` and **adds**
  the build/evaluate/get/set filter contract itself — neither is borrowable from pyCFTrackers.
- Dense-network trackers (Siamese family) backed by a hardware-abstracted inference layer.
- A backend (runtime + IPC + fusion primitives) that makes **hybrid trackers** buildable, with
  Rockchip NPUs as the first concrete target and other vendors addable behind the same interface.

**Out of scope (for now)**
- Frame capture / camera I/O. **The caller owns frames** and feeds an `np.ndarray` into
  `update()`. Trackers only consume frames.
- Multi-object tracking and track-ID association.
- Motion-detection / background-subtraction cues. The fusion runtime is designed so a motion cue
  *could* be added later as another async worker, but it is **not** implemented now.
- A synchronous / deterministic stepped execution mode. Hybrids are **always parallel-async**
  (see §6.3).

---

## 2. Design principles

1. **One public API, two execution models.** Individual trackers run **inline** (no processes):
   `update(frame)` does the numpy/NPU work and returns. Hybrid trackers own a process group
   internally, but expose the *same* non-blocking `update(frame)` signature; the multiprocessing
   complexity is hidden.
2. **Hardware is abstracted, Rockchip is first.** Trackers depend on a logical model + a backend
   interface, never on `rknn` directly. RKNN is one backend; ONNX (CPU/dev) and a mock backend
   ship for development and CI.
3. **CF model state is transferable and pure.** CF trackers expose pure, side-effect-free filter
   build/evaluate ops plus filter get/set. This is what lets a hybrid build a filter in one
   process and inject it into the CF tracker running in another. This is a **mandatory base-class
   contract**, not a MAFiD-specific add-on.
4. **Non-blocking IPC.** Communication between hybrid processes uses shared memory with
   single-writer, wait-free reads (seqlock). No hybrid process ever blocks on another.
5. **Placement is fully configurable.** Which CPU core / NPU core each process lands on is
   declared in a board profile with shipped defaults and full user override. Nothing is
   hardcoded per tracker.
6. **Testable without an NPU.** The mock backend + ONNX CPU backend make the whole runtime/IPC/
   fusion stack exercisable in CI on x86 with no accelerator.

---

## 3. Glossary

| Term | Meaning |
|---|---|
| **CF tracker** | Correlation-filter tracker (MOSSE, CSK, KCF, DSST, Staple, …), numpy-based. |
| **NN tracker** | Dense-network tracker (Siamese family), backend-backed. |
| **Hybrid tracker** | A tracker that composes CF + NN/detector components across processes. |
| **FilterState** | Transferable CF model state: arbitrary-shape numpy arrays + the ROI they were built for + metadata. |
| **PSR** | Peak-to-Sidelobe Ratio of a CF response map; used as both confidence and lock signal. |
| **Backend** | A vendor inference runtime adapter (`rknn`, `onnx`, `mock`). |
| **Manifest** | Declarative description of a logical model: per-backend artifacts + preprocessing + I/O spec. |
| **Frame ring** | Shared-memory ring buffer of recent frames, zero-copy, latest-only. |
| **Payload channel** | Shared-memory channel carrying variable-shape numpy payloads (detector output + candidate FilterState) under a seqlock. |
| **Placement profile** | Declarative mapping of each process to CPU core(s) / NPU core / backend. |

---

## 4. High-level architecture

```
            ┌─────────────────────────────────────────────────────────────┐
            │  Caller process  (owns frames, calls tracker.update(frame))  │
            │                                                              │
  frame ──► │  Individual tracker:  CF or NN  → TrackResult  (fully inline)│
            │                                                              │
            │  Hybrid tracker:                                             │
            │    • runs CF + fusion INLINE (fast path, sets output rate)   │
            │    • publishes frame to the frame ring                       │
            │    • polls payload channel (non-blocking)                    │
            └───────────────┬───────────────────────────▲──────────────────┘
                            │ frame ring (latest-only)   │ payload channel (seqlock)
                            ▼                             │  detector boxes+scores
            ┌───────────────────────────────────────────┴──────────────────┐
            │  Detector worker process  (async, free-running, slower)        │
            │    reads newest frame  →  NPU inference (via Backend)          │
            │    → build_filter(...)  →  publishes candidate FilterState     │
            └────────────────────────────────────────────────────────────────┘
```

- **Individual trackers** are inline and synchronous. They match OpenCV semantics exactly.
- **Hybrid trackers** spin up a process group. The CF step and the fusion decision run **inline in
  the caller's process** (CF is ~1–2 ms; running fusion inline guarantees both filters are
  evaluated by the *same* CF engine, which is required for a fair PSR comparison and avoids an
  extra latency hop). The only async worker (now that motion detection is dropped) is the
  **detector** on the NPU, which free-runs slower and injects opportunistically.
- The inline CF path is the **rate limit**: the caller's `update()` cadence defines the system
  output rate. The detector runs in parallel at its own (slower) rate.

---

## 5. Core data types and the tracker contract

All in `edgecv/core/`. These are the load-bearing public types; treat changes here as breaking.

### 5.1 BoundingBox — normalised 0–1

```python
@dataclass
class BoundingBox:
    x: float  # top-left x, normalised 0–1
    y: float  # top-left y, normalised 0–1
    w: float  # width,  normalised 0–1
    h: float  # height, normalised 0–1

    def to_pixels(self, width: int, height: int) -> "PixelBox": ...
    @classmethod
    def from_pixels(cls, box: "PixelBox", width: int, height: int) -> "BoundingBox": ...
```

Normalisation makes boxes resolution-independent and compact for IPC. **Norm-vs-pixel mixing is
the #1 bug source in tracker code**: keep `BoundingBox` always normalised, use the explicit
`PixelBox` helper type at the pixel boundary, and never let a raw pixel tuple masquerade as a
`BoundingBox`.

### 5.2 TrackResult and TrackStatus

```python
class TrackStatus(IntEnum):
    INITIALIZING = 0   # workers warming up / no lock yet
    LOCKED       = 1   # confident track
    COASTING     = 2   # low confidence, extrapolating / awaiting correction
    LOST         = 3   # track lost

@dataclass
class TrackResult:
    bbox: Optional[BoundingBox]   # None when no estimate is available
    confidence: Optional[float]   # None when the tracker genuinely has no score
    status: TrackStatus
    timestamp: float              # monotonic seconds, source-frame time
    seq: int                      # frame sequence number this result corresponds to
```

`confidence` is `None` where a tracker has no meaningful score. For CF trackers it is the PSR.
Note that CF PSR and NN classifier scores are **not** on the same scale — fusion must calibrate
before comparing (see §8).

### 5.3 Tracker base class

```python
class Tracker(ABC):
    @abstractmethod
    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None: ...

    @abstractmethod
    def update(self, frame: np.ndarray) -> TrackResult:
        """Non-blocking. For hybrids this publishes the frame and returns the
        latest fused estimate; early calls may return status=INITIALIZING."""

    @property
    @abstractmethod
    def status(self) -> TrackStatus: ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable tracker name, e.g. "MOSSE", "SiamFC", "MAFiD".
        Identifies the implementation for logging, benchmarking, and config."""

    def close(self) -> None:
        """Tear down any owned process group / shared memory. No-op for inline trackers."""

    def __enter__(self): ...
    def __exit__(self, *exc): self.close()
```

Hybrids **must** be usable as context managers; their process groups and shared-memory segments
have explicit lifecycle (see §7.4).

---

## 6. Tracker families

### 6.1 CF trackers (`edgecv/trackers/cf/`)

Built from fundamental ops orchestrated in Python — **not** OpenCV tracker classes. Shared ops live
in `edgecv/trackers/cf/ops/`: FFT helpers (`fft.py`), feature extractors (`features.py`: `raw`
pixels, `hog`, `colornames`), cosine/Hann windows (`window.py`), and PSR (`psr.py`). Concrete
trackers (`mosse.py`, `csk.py`, `kcf.py`, `dsst.py`, `staple.py`, …) compose these and expose their
parameters and the filter itself.

**Ops have a numpy reference + optional fast backend.** Every op ships a pure-numpy implementation
that always works (the reference). FFT and HOG — the two hotspots — additionally expose optional
accelerated backends selected behind a stable signature: `set_fft_backend(...)` picks
numpy/scipy/pyFFTW (`auto` prefers scipy), and `feature_backends()` reports `numba` when present as
the drop-in point for a jitted HOG. Nothing imports an optional lib at module load, so the base
wheel stays numpy-only and a device build opts into faster paths with **no tracker change**. Two
rules: ops must be module-level functions (importable in a `spawn`ed worker, §7.4), and
`build_filter` and `evaluate` must go through the *same* ops module so a candidate filter is never
penalised by cross-backend numerical drift (this extends the same-engine rule, §14.6). The CF-core
language stays Python deliberately: the per-frame math already runs in compiled kernels (pocketfft,
numba, BLAS), so a C/C++ core buys only ~1.1–1.5× on a path that already fits the frame budget
~10–20× over, while it would break filter transferability and editability — accelerate one op behind
this layer if profiling demands it, never the whole core.

**Mandatory transferable-filter contract.** Every CF tracker subclasses
`CorrelationFilterTracker` and implements both the online (mutating) loop *and* the pure ops:

```python
@dataclass
class FilterState:
    arrays: dict[str, np.ndarray]  # e.g. {"H": ..., "A": ..., "B": ...} — arbitrary shapes
    bbox: BoundingBox              # ROI the filter was built for
    meta: dict                     # feature type, window params, scale/aspect info, abi tag

@dataclass
class EvalResult:
    bbox: BoundingBox
    response_map: np.ndarray
    psr: float

class CorrelationFilterTracker(Tracker):
    # --- online, mutating (the normal tracking loop) ---
    def update(self, frame: np.ndarray) -> TrackResult: ...

    # --- pure ops: MUST NOT mutate self ---
    def build_filter(self, frame: np.ndarray, bbox: BoundingBox) -> FilterState: ...
    def evaluate(self, frame: np.ndarray, state: FilterState) -> EvalResult: ...

    # --- state access ---
    def get_filter(self) -> FilterState: ...
    def set_filter(self, state: FilterState,
                   search_box: Optional[BoundingBox] = None) -> None: ...

    @property
    def response_map(self) -> np.ndarray: ...
    @property
    def psr(self) -> float: ...
```

Why this contract is mandatory: hybrids do **build-elsewhere / evaluate-here / swap**. A worker
builds a `FilterState` from a frame and bbox (pure `build_filter`, runnable in another process),
ships it over the payload channel, and the caller evaluates incumbent vs candidate on the *current*
frame (`evaluate`) and swaps with `set_filter` if the candidate wins. `build_filter` and `evaluate`
**must be free of side effects** so they are safe to call across processes and in arbitrary order.

`set_filter` accepts a `search_box` because the filter's correct search window is the target's
**current/predicted** position, not the (now-stale) position it was built at — see §9 on the
motion predictor.

### 6.2 NN trackers (`edgecv/trackers/nn/`)

Siamese family (SiamFC / SiamRPN / ++ style). Each depends on a **logical model manifest** and an
`InferenceBackend`, never on a vendor runtime. Trackers carry a target-appearance/template
abstraction so they can be template-conditioned. Preprocessing (crop/resize/colour-convert) is a
candidate for hardware acceleration (e.g. RK RGA) behind the backend boundary; start with numpy and
swap in fast paths later without changing the tracker.

NanoTrack (V3) is the lightweight anchor-free member: MobileNetV3-small-v3 backbone +
DepthwiseBAN head emitting `cls`/`loc` maps, decoded over a point grid — same
manifest-driven, HAL-only contract as SiamFC.

### 6.3 Hybrid trackers (`edgecv/trackers/hybrid/`)

A hybrid composes a CF tracker + a detector (and, later, a motion cue) across processes. It owns a
process group and exposes the standard `Tracker` API. **Execution is always parallel-async**;
there is no synchronous mode. The inline CF path is the rate limit.

**The `update()` flow** (this is the generic backbone all hybrids reuse; specific hybrids like the
MAFiD-style tracker layer their own fusion policy on top and are specified separately):

```python
def update(self, frame: np.ndarray) -> TrackResult:
    self._seq += 1
    self._frame_ring.publish(frame, self._seq, now())      # single-writer, for async workers

    cf_result = self._cf.update(frame)                      # inline, mutating, rate-limiting

    candidate = self._payload.try_read_candidate()          # non-blocking seqlock read
    if candidate is None:
        return cf_result                                    # filter injection not ready → plain CF

    # injection ready → confidence-gate fusion, both filters evaluated by the SAME engine
    incumbent_eval = self._cf.evaluate(frame, self._cf.get_filter())
    candidate_eval = self._cf.evaluate(frame, candidate.filter_state)
    decision = self._fusion.fuse(incumbent_eval, candidate_eval, candidate.detector_out)

    if decision.take_candidate:
        # search window = current/predicted position, NOT the stale detection position
        self._cf.set_filter(candidate.filter_state, search_box=incumbent_eval.bbox)
        chosen = candidate_eval
    else:
        chosen = incumbent_eval

    return TrackResult(bbox=chosen.bbox, confidence=chosen.psr,
                       status=self._status_from(chosen.psr),
                       timestamp=now(), seq=self._seq)
```

Behaviour the caller sees, matching the agreed contract: `update()` returns the **plain CF
response** while filter injection isn't ready, and once a candidate is ready it returns the CF
result with the candidate filter injected **only if** the confidence (PSR) gate selects it;
otherwise it keeps the incumbent.

**Latency desync awareness.** A detection that returns N frames late must be associated with its
source `seq`/`timestamp`, not blindly applied to the current frame — this is why results and
payloads carry `seq`. Bridging the gap to the current position is the predictor's job (§9). The
reference hybrid (MAFiD-style) is documented in its own spec; this library provides the runtime,
the CF contract, the fusion abstractions, and the predictor hook it needs.

**Not every hybrid is CF+detector fusion.** `AcquireTrack`
(`trackers/hybrid/acquire_track.py`; spec `docs/superpowers/specs/2026-06-14-acquire-track-design.md`)
is an **NN→NN handoff**: a YOLO detector acquires a target on a fixed central crop, an operator
init command locks NanoTrack onto the current best detection, and on confidence drop YOLO
re-acquires (crop → full frame) and re-locks. The two NN models are **mutually exclusive** (only one
infers at a time) and run in their own spawned workers, each pinned to its own NPU core via the
manifest `npu_core` (`backends/rknn` reads it into `init_runtime(core_mask)`). It reuses the §7
primitives — `Orchestrator`, `FrameRing`, seqlock channels — but defines its own thin contract
rather than the CF one: a parent→workers control channel
(`runtime/shm/control_channel.py`: mode + crop + `lock_gen` + lock bbox), a YOLO `PayloadChannel`,
and a single-box NanoTrack result channel (`runtime/shm/nano_result.py`). No `FilterState`,
`build_filter`/`evaluate`, or `FusionPolicy`. The state machine runs inline in `update()`; workers
free-run and infer only when their control `mode` is active. It still honours the §14 invariants
(spawn-only, single-writer seqlock, ABI discipline, source-`seq`/`timestamp` lineage).

---

## 7. Runtime and IPC (`edgecv/runtime/`)

The hard part. Everything here is **single-writer** with **wait-free reads**, so no hybrid process
ever blocks on another.

### 7.1 Frame ring (`runtime/shm/frame_ring.py`)

- N fixed-size slots sized for the max supported resolution, stored **zero-copy**
  (`np.ndarray(shape, dtype, buffer=shm.buf, offset=slot*slot_size)`).
- The producer (caller) writes the next slot, then publishes `(slot, seq, timestamp, h, w, c,
  dtype)` in a small control word.
- **Latest-only semantics** for trackers: a consumer that fell behind jumps to the newest `seq`
  instead of draining. Freshest frame wins — correct for real-time. (A future every-frame consumer
  mode can be added for cues that need continuity.)
- Slot recycling is handled by triple-or-more buffering + latest-only reads, avoiding refcounts
  (which would reintroduce contention).
- **`read_latest` returns a decoupled copy, not a live view.** Slot-data safety comes from N-buffer
  recycling, not the seqlock — the seqlock guards only the control word, and the producer writes the
  *data* before bumping it. A returned zero-copy view would therefore tear once the producer cycled
  back to that slot, so the read copies out under the resolved `(slot, h, w, c)`. The copy sits
  *outside* the seqlock retry `fn` deliberately: retrying it would buy nothing (it cannot detect a
  later recycle anyway) and would re-copy on every contended retry. This differs from
  `payload.try_read`, which copies *inside* the `fn` because its data lives in one seqlock-guarded
  region rather than in recycled slots.

### 7.2 Payload channel (`runtime/shm/payload.py`)

Carries **variable-shape numpy payloads** — detector boxes+scores **and** the candidate
`FilterState` (a complex-valued array whose shape depends on ROI size). Layout: a max-size byte
buffer + a header carrying `(magic, abi_version, seq, n_arrays, [name, dtype, shape, offset]...)`,
all guarded by a seqlock. A fixed ctypes struct is **insufficient** for the result payload because
the filter is variable-shape; only small scalar control words use plain structs.

### 7.3 Seqlock (`runtime/shm/seqlock.py`)

ctypes/shared-memory writes are **not atomic across processes** — there is no GIL protecting a
multi-field write, so a reader can catch a half-written payload. The fix is a **seqlock**: a
`uint64 seq` word that the writer bumps odd → writes payload → bumps even; the reader retries while
`seq` is odd or changed across the read. Reads are wait-free and never block the writer, satisfying
the non-blocking requirement.

**Contended reads MUST yield, not busy-spin (mandatory for liveness).** When a reader sees the seq
odd (writer mid-write) or changed across its read, it must yield the CPU/GIL (`os.sched_yield()`)
before retrying — never `continue` in a tight pure-Python loop. This is wait-free in spirit (the
reader takes no lock and never blocks the writer) but is **required**, not cosmetic: under a *shared
interpreter* (a writer thread, e.g. tests or a same-process producer) or on a single core, a
pure-Python spin holds the GIL and the writer can never finish its odd→even transition — the reader
then exhausts its retry budget and raises *spuriously*, and any non-daemon writer it abandoned leaks.
The cross-process case (the real deployment) tolerates a naive spin because the OS preempts the
spinner, but the yield keeps the primitive correct under *both* models and costs nothing on the
uncontended fast path (even seq, unchanged → immediate return). This was a real livelock in the
foundation build; do not "optimise" the yield away.

> **Honest caveat for implementers:** pure Python has no explicit memory barriers, so this is
> "correct in practice for aligned word-size stores on ARM64/x86" rather than provably correct. If
> stronger guarantees are needed, the control word can be backed by a tiny C extension (or a
> microsecond-held lock on *only* the control word, keeping the data path lock-free). Document the
> choice where it lands.

### 7.4 Orchestrator, workers, lifecycle (`runtime/orchestrator.py`, `runtime/worker.py`)

- **`spawn` (or `forkserver`), never `fork`.** NPU runtime contexts (RKNN) do not survive `fork`
  and must be created in the process that uses them. Backends are initialised **inside** each
  child; the parent never loads the model.
- **Centralised SHM ownership.** The orchestrator (parent) creates and `unlink`s all segments;
  children only attach and never `unlink`. Detach the child `resource_tracker` for attached
  segments to avoid `multiprocessing.shared_memory` double-unlink / leak warnings.
- **Death propagation.** Set `PR_SET_PDEATHSIG` so children die with the parent; the orchestrator
  also reaps and (optionally) restarts workers on a heartbeat timeout.
- **Explicit lifecycle.** `close()` / context-manager protocol tears down workers and segments
  deterministically.
- **Release buffer exports before `close()` (mandatory on CPython ≥ 3.12).** A `SharedMemory`
  segment cannot be closed while any object still *exports* a pointer into its buffer —
  `multiprocessing.shared_memory.SharedMemory.close()` raises `BufferError: cannot close exported
  pointers exist`. Every `ctypes.*.from_buffer(shm.buf, ...)` view (control structs, the seqlock
  word, payload descriptors) and every `np.ndarray(buffer=shm.buf, ...)` view counts as an export.
  Therefore each SHM component's `close()` must **drop all such views first** (set them to `None` /
  let them go out of scope, `gc.collect()` to be safe) and only then call `shm.close()`. The seqlock
  exposes `release()` for exactly this; frame ring and payload channel call it plus drop their own
  structs. Transient views taken inside `publish`/`read` must not be retained past the call. Callers
  that build their own views on a segment (e.g. a test) are equally responsible for releasing them —
  including any closures that capture them — before the owner closes.

### 7.5 ABI versioning

Every shared struct/payload header begins with `magic + abi_version`, validated on attach. The
single source of truth is `runtime/shm/structs.py`. **If you change a shared layout, bump
`abi_version` and update both the producer and consumer** — this is the most common way an agent
edit silently breaks cross-process reads.

### 7.6 Placement (`runtime/placement.py`)

Process→hardware mapping is **entirely configurable** via a declarative board profile, with shipped
defaults for RK3588 and full user override. No placement is hardcoded in a tracker.

```yaml
# example board profile (shipped default for rk3588, user-overridable)
board: rk3588
processes:
  caller:                     # CF + fusion run inline here; sets output rate
    cpu_affinity: [4, 5, 6, 7]          # A76 big cores
    sched: { policy: FIFO, priority: 80 }   # optional; needs CAP_SYS_NICE
  detector:                   # the async NPU worker
    cpu_affinity: [0, 1]                # A55 little cores for pre/post-processing
    npu_core: 0                         # one of the RK3588's three NPU cores
    backend: rknn
```

Applied via `sched_setaffinity`, NPU core masks, and optional `SCHED_FIFO`. The RK3588's tri-core
NPU means multiple NN workers can be pinned to different NPU cores; the profile is where that is
expressed.

---

## 8. Fusion framework (`edgecv/fusion/`)

The library ships the **abstractions** hybrids need, not specific hybrid trackers.

```python
class FusionPolicy(ABC):
    @abstractmethod
    def fuse(self,
             incumbent: EvalResult,
             candidate: Optional[EvalResult],
             detector_out: Optional[DetectorOutput]) -> FusionDecision: ...
```

- A reference **confidence-gate (PSR) policy** selects between incumbent and candidate by comparing
  their PSR on the current frame — both produced by the same CF engine, so the comparison is fair.
- Policies own **score calibration**: CF PSR and NN classifier scores live on different scales and
  must be normalised before any cross-source comparison.
- A policy may return `take_candidate=False` even when detection succeeded — e.g. when the target's
  appearance changed between detection and injection, the detector-derived filter often matches
  worse than the incumbent. The PSR gate catches this, which also makes INT8/quantisation noise in
  NPU-derived filters mostly self-correcting.

---

## 9. Motion predictor (`edgecv/fusion/predict.py`)

```python
class MotionPredictor(ABC):
    @abstractmethod
    def predict(self, history: Sequence[Tuple[float, BoundingBox]], dt: float) -> BoundingBox: ...
```

Edge cameras run at ~30–60 FPS, not 500+. During a detection cycle the target moves **far**, so a
candidate filter cannot simply be applied at its build-time position. The predictor (default:
constant-velocity) supplies the **search window** for `set_filter`, bridging detection latency.
This generalises the high-speed-camera assumption that the object "barely moved" into something
that holds at edge frame rates. The hook is optional per hybrid but provided in the backend so
hybrids don't reinvent it.

---

## 10. Hardware Abstraction Layer (`edgecv/backends/`)

Trackers depend on a **logical model** + these interfaces, never on a vendor runtime.

```python
class InferenceBackend(ABC):
    name: str
    def is_available(self) -> bool: ...
    def load(self, manifest: "ModelManifest") -> "Model": ...

class Model(ABC):
    @property
    def io_spec(self) -> "IOSpec": ...   # input/output names, shapes, dtypes, quant params, layout
    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...
    def infer_async(self, inputs: dict[str, np.ndarray]) -> "Handle": ...  # optional; NPUs pipeline
    def close(self) -> None: ...
```

**Backends**
- `backends/rknn/` — first concrete target; uses `rknn-toolkit-lite2` on-device. Initialised inside
  the worker process (never the parent).
- `backends/onnx/` — ONNXRuntime CPU backend for development on x86 and CI fallback.
- `backends/mock/` — canned outputs; lets the entire runtime/IPC/fusion stack run with no model and
  no accelerator. Critical for CI and for agentic validation.

**Registry & lazy imports** (`backends/registry.py`): backends register via an entry-point group
(`edgecv.backends`) and are **lazily imported**, so a missing `rknn-toolkit-lite2` only errors when
the `rknn` backend is actually used, and future vendors register without touching core.

### 10.1 Model manifest (`edgecv/models/manifest.py`)

A manifest maps one logical model to per-backend artifacts plus preprocessing — this is what makes
the HAL real. A Siamese tracker depends on the manifest, not on any `.rknn` file.

```yaml
name: siamfc_generic
task: sot_template_matching
preprocessing: { color: gray, exemplar: 127, search: 255, ... }
io: { inputs: [exemplar, search], outputs: [score_map] }
artifacts:
  onnx: { path: siamfc_generic.onnx }
  rknn: { path: siamfc_generic.rk3588.rknn, quant: int8 }
```

---

## 11. Models: shipped defaults, conversion, training (`edgecv/models/`, `tools/`)

- **Shipped defaults.** Trackers that require a model ship a default, pre-trained on generic
  object-tracking datasets and ported to supported backends, addressed via the manifest. The
  tracker works out of the box on a supported board.
- **Packaging note (decide at implementation):** `.rknn` artifacts are large and platform-specific
  while `.onnx` is portable. Options are bundling defaults as package data, shipping them as a
  backend extra, or fetching on first use. Whichever is chosen, the manifest indirection means
  trackers are unaffected.
- **Host-side tooling (`tools/`).** Conversion (e.g. ONNX → RKNN with INT8 calibration via
  `rknn-toolkit2`) and training are **host-only** and **not runtime dependencies**. These are
  largely helper scripts + docs that wrap external libraries — notably
  [`pytracking`](https://github.com/visionml/pytracking) — to produce new ports and train weights.
  Conversion/calibration runs offline on x86; the device only ever runs the lite runtime.

---

## 12. Packaging and extras

Core stays numpy-only so the base wheel is universal and the CF trackers run with nothing exotic.
The "no OpenCV" rule is about *tracker classes*, not primitives (§1): optimized op libraries
(scipy.fft, numba, pyFFTW, `opencv-python`'s op functions) are allowed but **optional** — each is
lazily imported inside the relevant op, with a numpy reference fallback, so the base install never
requires them. Accelerator inference backends and host tooling are likewise optional extras.

| Install | Contents |
|---|---|
| `pip install edgecv` | Core: numpy CF trackers, runtime/IPC, fusion abstractions, `mock` backend. |
| `pip install edgecv[fast]` | Optional CF-ops accelerators (scipy, numba, pyFFTW, opencv-python). Reference numpy ops work without it; this just selects faster FFT/HOG/image paths. |
| `pip install edgecv[onnx]` | ONNXRuntime CPU/dev backend (NN trackers on x86). |
| `pip install edgecv[rknn]` | Registers the RKNN backend. **`rknn-toolkit-lite2` is installed manually on the device** — it is not on PyPI, so the extra cannot pull it; document the manual step. |
| `pip install edgecv[dev]` | Host-side conversion/training helpers and their heavy deps (pytracking, etc.). |

---

## 13. Directory layout

```
edgecv/
├── core/
│   ├── bbox.py            # BoundingBox (0–1), PixelBox, conversions
│   ├── result.py          # TrackResult, TrackStatus
│   └── tracker.py         # Tracker ABC
├── trackers/
│   ├── cf/
│   │   ├── base.py        # CorrelationFilterTracker + FilterState + EvalResult (the contract)
│   │   ├── mosse.py csk.py kcf.py dsst.py staple.py ...
│   │   └── ops/           # fft, features (raw/hog/colornames), windows, psr
│   ├── nn/
│   │   └── siamfc.py siamrpn.py nanotrack.py ...     # backend-backed, manifest-driven, template-conditioned
│   └── hybrid/            # fusion trackers (e.g. MAFiD-style) — own process groups; own specs
├── backends/
│   ├── base.py            # InferenceBackend, Model, IOSpec ABCs
│   ├── registry.py        # entry-point plugin registry, lazy import
│   ├── rknn/  onnx/  mock/
├── runtime/
│   ├── shm/
│   │   ├── frame_ring.py  # zero-copy frame ring, latest-only
│   │   ├── payload.py     # variable-shape numpy payload channel
│   │   ├── seqlock.py     # wait-free reads
│   │   └── structs.py     # single source of truth for shared layouts + abi_version
│   ├── orchestrator.py    # spawn workers, own SHM lifecycle, heartbeat
│   ├── worker.py          # child entrypoint; initialises backend in-process
│   └── placement.py       # board profiles, affinity, NPU core masks, sched
├── fusion/
│   ├── policy.py          # FusionPolicy ABC + DetectorOutput, FusionDecision
│   ├── psr_gate.py        # reference confidence-gate policy
│   └── predict.py         # MotionPredictor ABC + constant-velocity default
└── models/
    ├── manifest.py        # manifest schema + loader
    └── (shipped default artifacts or fetch logic)

tools/                     # HOST-ONLY: conversion (rknn-toolkit2), training (pytracking) helpers + docs
tests/                     # runs on x86 with mock/onnx backends, no NPU required
```

---

## 14. Concurrency & correctness invariants

Agents and reviewers: treat these as hard rules.

1. **Single writer per shared region.** Each frame-ring slot control word and each payload channel
   has exactly one writer.
2. **Wait-free reads via seqlock.** Readers retry on odd/changed `seq`; never take a blocking lock
   on the data path. A contended reader **yields** the CPU/GIL (`os.sched_yield()`) before each
   retry — a pure-Python busy-spin starves a same-interpreter/single-core writer and livelocks (§7.3).
3. **Latest-only for trackers.** Consumers jump to newest `seq`; do not drain stale frames.
4. **ABI discipline.** Any change to a shared layout bumps `abi_version` in `structs.py` and updates
   both sides; headers carry `magic + abi_version`, validated on attach.
5. **`build_filter` / `evaluate` are pure.** No mutation of `self`; safe across processes and call
   orders.
6. **Same-engine PSR.** Incumbent and candidate filters are always evaluated by the *same* CF
   engine instance, inline in the caller. Never compare PSR across engines.
7. **`spawn`, not `fork`; backends init in-child.** No NPU/model handle is created in the parent.
8. **Centralised SHM ownership.** Parent creates+unlinks; children attach only.
9. **No per-frame allocation in hot loops.** Preallocate and reuse numpy buffers; consider
   `gc.disable()` in the inline CF loop to avoid GC jitter.
10. **`seq`/`timestamp` travel with every frame, result, and payload.** Late detections are
    associated by `seq`, never applied blindly to the current frame.
11. **Release buffer exports before closing a segment.** Drop every `ctypes.from_buffer` / numpy
    view into `shm.buf` (and any closure capturing them) before `SharedMemory.close()`, or it raises
    `BufferError` on CPython ≥ 3.12 (§7.4). The seqlock provides `release()`; owners call it and drop
    their own structs.

---

## 15. Testing strategy

- The `mock` backend + ONNX CPU backend make the runtime/IPC/fusion stack runnable on x86 in CI
  with no accelerator.
- Unit-test the seqlock under concurrent read/write (torn-read detection), the frame ring
  (latest-only + recycling), and the CF purity contract (`build_filter`/`evaluate` do not mutate).
- Integration-test a hybrid end-to-end with the mock detector: assert plain-CF responses before
  injection, and correct gate behaviour (take/keep) after.
- Property: a hybrid's output `seq` is monotonic and each `TrackResult.seq` matches a frame the
  caller submitted.

---

## 16. Open questions / future work

- **Motion cue (deferred).** Re-introducible as a second async worker (frame-diff / optical-flow /
  none — swappable, since background subtraction assumes a static camera). The runtime already
  supports adding another worker + payload producer without structural change.
- **Zero-copy fast paths.** RK RGA for hardware crop/resize/colour-convert and DMA-buf zero-copy
  into the NPU, hidden behind `Model.infer` / preprocessing. Start numpy-in/numpy-out; add later
  without changing tracker code.
- **MOT.** Out of scope now; would change the result data model (track IDs, association).
- **Additional vendor backends.** Hailo, Coral, etc., register via the same entry-point group.

---

## 17. Notes for contributors and agents

- **Add a CF tracker:** subclass `CorrelationFilterTracker` in `trackers/cf/`, implement the online
  `update` *and* the pure `build_filter`/`evaluate`/`get_filter`/`set_filter`/`response_map`/`psr`.
  Reuse `trackers/cf/ops/`. Do not depend on OpenCV.
- **Add an NN tracker:** write a manifest, depend on `InferenceBackend`/`Model`, never import a
  vendor runtime directly.
- **Add a backend:** subclass `InferenceBackend`/`Model`, register via the `edgecv.backends`
  entry-point group, keep heavy imports lazy.
- **Add a hybrid:** reuse the §6.3 `update()` backbone, the §8 `FusionPolicy`, and the §9 predictor;
  give the specific tracker its own spec document. This file owns the *backend*; hybrid specs own
  the *policy*.
- **Touching shared memory?** Re-read §7 and §14, and bump `abi_version`.

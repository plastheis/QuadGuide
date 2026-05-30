# Tracker Worker Refactor — Design Spec

Date: 2026-05-29

---

## Overview

Quadguide's perception layer collapses from three processes (CCV worker + NCV
worker + fusion worker) into one generic `tracker_worker`. The new worker owns
the bus subscription to `lockon/cmd`, the SHM frame read, and the
`target/estimate` publish — but owns no tracking algorithm. The tracking
algorithm is chosen at startup from a single config string `tracker.import`
and built once. From the loop's point of view the tracker is opaque; same
calls regardless of whether the underlying object is an OpenCV `cv2.TrackerKCF`
wrapped in a small built-in adapter, or an externally-imported hybrid/fusion
tracker library that the operator authors in a separate repo.

Net effect: roughly 28 files and on the order of 1500 lines deleted from
quadguide, ~200 lines added. The NPU runtime abstraction (`inference/`) and the entire
`perception/fusion/` package go away. NPU runtime selection becomes the
external library's concern; quadguide stops branching on NPU type.

---

## Goals

- One generic `tracker_worker` process in place of `ccv_tracker_worker`,
  `ncv_tracker_worker`, and `fusion/worker`.
- Single config field `tracker.import` selects the tracker by module path.
- Built-in OpenCV adapter handles `cv2:*` specs (≈40 lines, inside quadguide).
- External tracker libraries plug in by satisfying a structural protocol
  (duck-typed `init`/`update`/`name`/`reset`/`close`) — no quadguide imports
  required on the library side.
- Bus topic graph simplifies: `ccv_tracker/estimate` and `ncv_tracker/estimate`
  deleted; `target/estimate` carries `TrackerEstimate` directly.
- `TargetEstimate` and `ActiveTracker` enum deleted from `core/messages.py`.

## Non-goals

- The hybrid tracker library itself. This spec defines the protocol the
  library satisfies; its internal architecture is the library's concern.
- Bringing back per-tracker tuning of cv2's KCF/MOSSE params (the existing
  `tracker.kcf.{sigma, lambda_, detect_thresh}` knobs become unreachable
  after the refactor). Listed under Known Limitations.
- HIL, link, control, guidance algorithm, or bus implementation changes.
- Updating `architecture.md` is required but the wording is left to the
  implementation plan; this spec describes the destination state of the code,
  not the prose.

---

## Architecture

### Process count

Drops from 8 to 6. The system retains "one process per resource" — the
tracker process becomes the single owner of perception state including, when
applicable, the NPU (owned via the external library's internals).

```
camera ──► tracker ──► guidance ──► control ──► link ──► FC
                                    ▲           ▲
ground ◄────── all topics ──────────┘           │
                                                │
link (rx) ──► fc/attitude, fc/imu ──────────────┘
```

Deleted processes: `ccv_tracker`, `ncv_tracker`, `fusion`.

### Topic graph

| Topic | Producer | Consumers | Payload (after) | Change |
|---|---|---|---|---|
| `frame_buffer` (shm) | camera | tracker | `np.ndarray` | unchanged |
| `target/estimate` | **tracker** | guidance, control watchdog, ground | **`TrackerEstimate`** | producer + payload changed |
| `lockon/cmd` | ground | tracker | `LockOnCmd` | consumer changed |
| `system/health` | all | ground | `HealthReport` | one fewer producer process |
| `fc/attitude`, `fc/imu`, `arm/cmd`, `guidance/accel`, `control/cmd` | — | — | — | unchanged |

Deleted topics: `ccv_tracker/estimate`, `ncv_tracker/estimate`.

### Files

**Added:**
- `src/quadguide/perception/tracker_worker.py` (≈200 lines)

**Deleted (whole packages or files):**
- `src/quadguide/inference/` (whole package — `base.py`, `factory.py`,
  `onnx_cpu.py`, `onnx_cuda.py`, `rknn.py`, `__init__.py`)
- `src/quadguide/perception/fusion/` (whole package — `worker.py`, `fusion.py`,
  `algorithms/{base,__init__,_helpers,passthrough,confidence_weighted,iou_gated}.py`)
- `src/quadguide/perception/nanotrack/` (whole package —
  `tracker.py`, `preprocess.py`, `postprocess.py`, `worker.py`, `__init__.py`)
- `src/quadguide/perception/kcf/` (whole package)
- `src/quadguide/perception/mosse/` (whole package)
- `src/quadguide/perception/ccv_tracker_worker.py`
- `src/quadguide/perception/ncv_tracker_worker.py`
- `src/quadguide/perception/tracker_factories.py`

Approximately 28 files deleted in total.

**Deleted (code blocks within retained files):**
- `core/messages.py`: `TargetEstimate` dataclass + `FMT_TARGET_ESTIMATE` +
  `_ST_TARGET_ESTIMATE`; `ActiveTracker` enum and its `_byte_enum` decoration;
  all three names removed from `__all__`.
- `core/config.py`: `InferenceConfig`, `FusionConfig`, `KCFConfig`,
  `NanotrackConfig`, `MOSSEConfig` dataclasses; the `inference` field on
  `PlatformConfig`; the `inference=...` construction line in `cfg_platform`;
  the nested-config logic in `cfg_tracker`.
- `configs/config.yaml`: `platform.inference` subsection; the entire
  `tracker.{ccv, ncv, kcf, nanotrack, mosse, fusion}` block; rename
  `realtime.kcf_cpu_core` → `realtime.tracker_cpu_core`.
- Bus topic pre-declaration site: remove `ccv_tracker/estimate` and
  `ncv_tracker/estimate` from the topic list.

---

## Tracker protocol (structural)

The worker treats the tracker as opaque and reads it duck-typed. The library
defines its own types and never imports from quadguide.

### Required methods

| Method | Signature | Lifecycle |
|---|---|---|
| `name` | `() -> str` | Static. Called at construction to derive worker process name. Same return for the instance's lifetime. |
| `init` | `(frame: np.ndarray, bbox) -> None` | Called per lock-on (each new `LockOnCmd.seq`). May be called multiple times. Bbox is duck-typed `.x/.y/.w/.h` normalized 0–1. |
| `update` | `(frame: np.ndarray) -> object` | Per worker tick. Returns an object satisfying the structural output protocol below. |
| `reset` | `() -> None` | Called on zero-size lockon bbox (operator clears lock). After reset, next `update()` returns `health="no_lock"`. Does not release hardware. |
| `close` | `() -> None` | Called once in SIGTERM handler before process exit. Releases hardware and tears down internal subprocesses. |

### Update output (structural)

The object returned by `update()` must expose:

| Attribute | Type | Range |
|---|---|---|
| `.bbox` | object with `.x`, `.y`, `.w`, `.h` floats | normalized 0–1 |
| `.confidence` | float | [0, 1] |
| `.health` | str | `"nominal"` \| `"uncertain"` \| `"lost"` \| `"no_lock"` |

Attribute names are read directly. The library may use any concrete type
(`@dataclass`, `namedtuple`, `attrs`, `SimpleNamespace`). Field names
chosen to match a generic tracker library convention (`health`, not
`tracker_health`) — quadguide does the renaming on its side.

### Constructor

`__init__(self, **params)` — accepts arbitrary keyword arguments. The worker
calls `cls(**config["tracker"]["params"])`. The library documents its own
params. Type/value errors at construction fall out as ordinary Python
exceptions and kill the worker (run.py surfaces the exit).

### Lifecycle invariants

1. `update()` may be called before `init()` — must return `health="no_lock"`
   and a zero bbox.
2. `init()` may be called repeatedly without intervening `close()`.
3. `close()` is called exactly once. No methods after `close()`.
4. `reset()` is a soft variant of "back to pre-init"; must not release
   hardware that `init()` would expect to re-acquire.

### What the protocol omits

- No frame format negotiation. Frames are HxWx3 uint8 BGR (whatever
  `FrameBuffer.read_latest()` returns). Trackers convert internally if needed.
- No timestamps in. Worker stamps `timestamp_ns` and `latency_ns` from the SHM
  frame timestamp.
- No bus access. Tracker sees the bus only via `init()`/`update()` arguments.
- No async or generator variants. `update()` is synchronous.

### Library example (in a separate repo)

```python
# myhybrid/tracker.py — zero quadguide imports
from dataclasses import dataclass

@dataclass(frozen=True)
class BBox:
    x: float; y: float; w: float; h: float

@dataclass(frozen=True)
class TrackerOutput:
    bbox: BBox
    confidence: float
    health: str

class HybridTracker:
    def __init__(self, **params): ...     # may spawn internal subprocesses / shm
    def name(self): return "hybrid"
    def init(self, frame, bbox): ...      # reads bbox.x/.y/.w/.h structurally
    def update(self, frame) -> TrackerOutput: ...
    def reset(self): ...
    def close(self): ...                  # tears down internals; critical for NPU release
```

---

## Loader and OpenCV adapter

Both live in `perception/tracker_worker.py`.

### Loader

```python
import importlib

def load_tracker(config: dict):
    """Construct the tracker selected by config['tracker']['import']."""
    tcfg = config["tracker"]
    spec = tcfg["import"]
    params = tcfg.get("params") or {}

    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"tracker.import must be 'module:Class', got {spec!r}")

    if module_name == "cv2":
        return OpenCVTrackerAdapter(class_name, params)

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**params)
```

Errors are not caught:

- malformed spec → `ValueError` at startup
- missing module → `ImportError`
- missing class name → `AttributeError`
- bad params → whatever the tracker raises

All surfaced by `run.py` as process exit. Configuration errors are not
silently degraded.

### OpenCV adapter

```python
from collections import namedtuple
from quadguide.core.messages import BoundingBox, TrackerEstimate, TrackerHealth

_TrackerOutput = namedtuple("_TrackerOutput", "bbox confidence health")
_BBox          = namedtuple("_BBox",          "x y w h")

def _resolve_cv2_factory(class_name: str):
    import cv2
    if hasattr(cv2, class_name):
        return getattr(cv2, class_name).create
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, class_name):
        return getattr(cv2.legacy, class_name).create
    raise AttributeError(
        f"cv2 has no tracker named {class_name!r} on cv2 or cv2.legacy"
    )

class OpenCVTrackerAdapter:
    """Wraps cv2 trackers (pixel tuples, success bools) into the structural
    output protocol the worker reads."""

    def __init__(self, class_name: str, params: dict):
        self._factory = _resolve_cv2_factory(class_name)
        self._params = params  # held but unused; reserved for a future cv2 params bridge
        self._name = class_name.lower().removeprefix("tracker")
        self._tracker = None
        self._initialized = False

    def name(self): return self._name

    def init(self, frame, bbox):
        h, w = frame.shape[:2]
        self._tracker = self._factory()
        self._tracker.init(frame, (
            int(bbox.x * w),
            int(bbox.y * h),
            max(1, int(bbox.w * w)),
            max(1, int(bbox.h * h)),
        ))
        self._initialized = True

    def update(self, frame):
        if not self._initialized:
            return _TrackerOutput(_BBox(0.0, 0.0, 0.0, 0.0), 0.0, "no_lock")
        h, w = frame.shape[:2]
        ok, bbox_px = self._tracker.update(frame)
        if ok:
            x, y, bw, bh = bbox_px
            return _TrackerOutput(_BBox(x/w, y/h, bw/w, bh/h), 1.0, "nominal")
        return _TrackerOutput(_BBox(0.0, 0.0, 0.0, 0.0), 0.0, "lost")

    def reset(self):
        self._initialized = False
        self._tracker = None

    def close(self):
        pass
```

### Worked examples

| `tracker.import` | What happens |
|---|---|
| `cv2:TrackerKCF` | adapter wraps `cv2.TrackerKCF.create()`, name `"kcf"` |
| `cv2:TrackerMOSSE` | adapter finds it on `cv2.legacy`, name `"mosse"` |
| `cv2:TrackerNano` | adapter wraps `cv2.TrackerNano.create()`, name `"nano"` |
| `myhybrid.tracker:HybridTracker` | `importlib.import_module("myhybrid.tracker")`, calls `.HybridTracker(**params)` |
| `kcf` (no colon) | `ValueError` at startup |
| `cv2:DoesNotExist` | `AttributeError` at startup |

---

## Worker IPC loop

`TrackerWorker` is a thin loop that owns one tracker and three IPC surfaces
(frame SHM read, `lockon/cmd` poll, `target/estimate` publish + periodic
`system/health`).

```python
import dataclasses, os, signal
from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import (
    BoundingBox, HealthReport, LockOnCmd, ProcessState,
    TrackerEstimate, TrackerHealth,
)

_HEALTH_EVERY = 50   # publish HealthReport every N iterations

class TrackerWorker:
    def __init__(self, tracker, bus: Bus, frame_buffer: FrameBuffer,
                 cpu_core: int | None = None, config: dict | None = None):
        self._tracker   = tracker
        self._bus       = bus
        self._fb        = frame_buffer
        self._cpu_core  = cpu_core
        self._config    = config or {}
        self._last_seq: int | None = None
        self._stop      = False
        self._proc_name = f"tracker_{tracker.name()}"

    def run(self):
        log = setup_logging(self._proc_name, self._config)
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        if self._cpu_core is not None:
            try:
                os.sched_setaffinity(0, {self._cpu_core})
            except (AttributeError, OSError):
                pass

        log.info(f"{self._proc_name}: started")
        i = 0
        while not self._stop:
            self._check_lockon()
            frame, frame_ts = self._fb.read_latest()
            if frame is not None:
                out     = self._tracker.update(frame)
                now_ns  = monotonic_ns()
                latency = min(now_ns - frame_ts, 0xFFFF_FFFF) if frame_ts > 0 else 0
                est = TrackerEstimate(
                    timestamp_ns=now_ns,
                    bbox=BoundingBox(out.bbox.x, out.bbox.y, out.bbox.w, out.bbox.h),
                    confidence=float(out.confidence),
                    tracker_health=TrackerHealth(out.health),
                    latency_ns=latency,
                )
                self._bus.publish("target/estimate", est)

            i += 1
            if i % _HEALTH_EVERY == 0:
                self._bus.publish(
                    "system/health",
                    HealthReport(monotonic_ns(), self._proc_name, ProcessState.OK, ""),
                )

        self._tracker.close()    # SIGTERM cleanup — last chance to release NPU
        self._bus.detach()
        log.info(f"{self._proc_name}: stopped")

    def _check_lockon(self):
        cmd: LockOnCmd | None = self._bus.latest("lockon/cmd")
        if cmd is None or cmd.seq == self._last_seq:
            return
        self._last_seq = cmd.seq
        if cmd.bbox.w == 0.0 and cmd.bbox.h == 0.0:
            self._tracker.reset()
            return
        frame, _ = self._fb.read_latest()
        if frame is not None:
            self._tracker.init(frame, cmd.bbox)

    def _handle_sigterm(self, sig, frame):
        self._stop = True


def run_from_config(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    """Build the tracker selected by config.tracker.import and run it."""
    from quadguide.core.config import cfg_platform
    pcfg = cfg_platform(config)
    tracker = load_tracker(config)
    TrackerWorker(
        tracker, bus, frame_buffer,
        cpu_core=pcfg.realtime.tracker_cpu_core,
        config=config,
    ).run()
```

### Behaviors and non-behaviors

- **No try/except around `tracker.update()`.** A misbehaving tracker that
  raises kills the worker; `run.py` surfaces the exit; control's watchdog
  fails safe to level flight on `target/estimate` staleness.
- **No artificial rate limit.** The loop runs as fast as the slower of frame
  arrival and `tracker.update`. Guidance's own 50 Hz `RateLimiter` decouples
  downstream consumers from tracker rate.
- **No `subscribe_any`.** With fusion gone, no worker needs blocking
  multi-topic wait. Tracker worker polls `lockon/cmd` via non-blocking
  `bus.latest`.
- **CPU affinity is optional.** If `realtime.tracker_cpu_core` is null/absent,
  `os.sched_setaffinity` is skipped. External libraries that spawn their own
  subprocesses are responsible for setting affinity on their own children.
- **Health-publish cadence is fixed at 50 iterations.** At 200 Hz that's 4 Hz;
  at 30 Hz that's ~0.6 Hz — `target/estimate` itself serves as a fresher
  liveness signal for downstream watchdogs. Not a config knob.

---

## Config schema

### `configs/config.yaml`

**After:**

```yaml
platform:
  name: orange_pi5
  camera:
    backend: v4l2
    pipeline: "..."
    width: 640
    height: 480
    fps: 60
  serial:
    port: /dev/ttyAMA0
    baud: 420000
    rx_pin: "GPIO15"
    tx_pin: "GPIO14"
  # inference: subsection deleted
  realtime:
    tracker_cpu_core: 1       # renamed from kcf_cpu_core; optional
    control_cpu_core: 3
    control_sched_fifo: true
    control_fifo_prio: 80

tracker:
  import: cv2:TrackerKCF      # or e.g. "myhybrid.tracker:HybridTracker"
  params: {}                  # passed to tracker constructor as **kwargs
```

Other top-level sections (`airframe`, `guidance`, `watchdog`, `link`,
`mission`, `logging`, `bus`) unchanged.

### `core/config.py`

**Deleted dataclasses:** `InferenceConfig`, `FusionConfig`, `KCFConfig`,
`NanotrackConfig`, `MOSSEConfig`.

**Deleted from `PlatformConfig`:** the `inference: InferenceConfig` field.

**Deleted from `cfg_platform`:** the `inference=InferenceConfig(...)`
construction lines.

**Replaced `TrackerConfig`:**

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class TrackerConfig:
    import_spec: str                                          # YAML "import"
    params: dict[str, Any] = field(default_factory=dict)
```

(`import_spec` instead of `import` because `import` is a Python keyword. YAML
key remains `import`.)

**Replaced `cfg_tracker`:**

```python
def cfg_tracker(d: dict) -> TrackerConfig:
    t = d["tracker"]
    return TrackerConfig(
        import_spec=t["import"],
        params=dict(t.get("params") or {}),
    )
```

KeyError on missing `tracker.import` is the intended failure mode.

**Changed `RealtimeConfig`:**

```python
@dataclass(frozen=True)
class RealtimeConfig:
    tracker_cpu_core: int | None      # was kcf_cpu_core; now Optional
    control_cpu_core: int
    control_sched_fifo: bool
    control_fifo_prio: int
```

`cfg_platform` reads `realtime.tracker_cpu_core` and tolerates absence
(returns `None`).

---

## Message schema

### Deleted from `core/messages.py`

- `TargetEstimate` dataclass
- `FMT_TARGET_ESTIMATE` format string
- `_ST_TARGET_ESTIMATE` precompiled `struct.Struct`
- `ActiveTracker` enum and its `_byte_enum` decoration
- All three names removed from `__all__`

### Retained unchanged

- `TrackerEstimate` — same fields, same `FMT_TRACKER_ESTIMATE = "!QfffffBI"`,
  33 bytes on the wire. Now the sole tracker-side bus type.
- `BoundingBox`, `TrackerHealth`, `LockOnCmd`, `HealthReport`, `AttitudeState`,
  `IMUFrame`, `AccelCmd`, `ControlCmd`, `ArmCmd` — all unchanged.

### Not added

- `TrackerOutput` is **not** added to `core/messages.py`. The library defines
  its own. Exporting one would invite library authors to import it "for
  convenience" and defeat the decoupling.

---

## Downstream consumer changes

### `guidance/base.py`

```diff
-from quadguide.core.messages import IMUFrame, LockOnCmd, TargetEstimate
+from quadguide.core.messages import IMUFrame, LockOnCmd, TrackerEstimate

 class GuidanceMethod(Protocol):
     def compute(
         self,
-        est: TargetEstimate,
+        est: TrackerEstimate,
         imu: IMUFrame,
         lockon_cmd: LockOnCmd | None,
         now_ns: int,
     ) -> tuple[float, float]: ...
     def name(self) -> str: ...
```

### `guidance/pronav.py`

```diff
-from quadguide.core.messages import IMUFrame, LockOnCmd, TargetEstimate
+from quadguide.core.messages import IMUFrame, LockOnCmd, TrackerEstimate

     def compute(self, est, imu, lockon_cmd, now_ns):
-        los_r = self._los.update(est.centroid_norm, imu, lockon_cmd, now_ns)
+        cx = (est.bbox.x + est.bbox.w * 0.5 - 0.5) * 2.0
+        cy = (est.bbox.y + est.bbox.h * 0.5 - 0.5) * 2.0
+        los_r = self._los.update((cx, cy), imu, lockon_cmd, now_ns)
         v_c   = self._cv.update(est.bbox, now_ns, self._cfg)
         return pronav(los_r, v_c, self._cfg.N)
```

### `guidance/pure_pursuit.py`

```diff
-from quadguide.core.messages import IMUFrame, LockOnCmd, TargetEstimate
+from quadguide.core.messages import IMUFrame, LockOnCmd, TrackerEstimate

     def compute(self, est, imu, lockon_cmd, now_ns):
-        cx, cy = est.centroid_norm
+        cx = (est.bbox.x + est.bbox.w * 0.5 - 0.5) * 2.0
+        cy = (est.bbox.y + est.bbox.h * 0.5 - 0.5) * 2.0
         return self._K * cx * self._scale_x, self._K * cy * self._scale_y
```

The `(bbox) → (cx, cy)` calc appears in three places (pronav, pure_pursuit,
ground server). The implementation plan should pull it into a small helper
in `core/messages.py` next to `BoundingBox`:

```python
def bbox_centroid_norm(bbox: BoundingBox) -> tuple[float, float]:
    return ((bbox.x + bbox.w * 0.5 - 0.5) * 2.0,
            (bbox.y + bbox.h * 0.5 - 0.5) * 2.0)
```

### `guidance/worker.py`, `guidance/los.py`, `guidance/closing_vel.py`

No changes. `est.tracker_health` access works on `TrackerEstimate` (unchanged
field). `LOSRateEstimator.update` takes the centroid tuple by argument.
`ClosingVelEstimator.update` takes the bbox directly.

### `control/watchdog.py`, `control/worker.py`

No changes. Watchdog uses topic-string only; reads `bus.latest(topic).timestamp_ns`
which is structurally compatible across `TargetEstimate` and `TrackerEstimate`.
Control worker does not import either.

### `ground/overlay.py`

```diff
-from quadguide.core.messages import TargetEstimate, TrackerHealth
+from quadguide.core.messages import TrackerEstimate, TrackerHealth

-def draw_overlay(frame: np.ndarray, estimate: TargetEstimate | None) -> bytes:
+def draw_overlay(frame: np.ndarray, estimate: TrackerEstimate | None) -> bytes:
```

Logic unchanged — already only reads `estimate.bbox` and `estimate.tracker_health`.

### `ground/server.py`

The SSE feed simplifies. Deleted reads:

```python
ccv = app.state.bus.latest("ccv_tracker/estimate")
ncv = app.state.bus.latest("ncv_tracker/estimate")
```

Deleted SSE fields: `active_tracker`, `centroid_x`, `centroid_y`, `ccv_algo`,
`ccv_health`, `ccv_conf`, `ncv_algo`, `ncv_health`, `ncv_conf`.

The two-prefix algo derivation collapses to one:

```diff
-ccv_algo = next((k[4:] for k in app.state.process_health if k.startswith("ccv_")), None)
-ncv_algo = next((k[4:] for k in app.state.process_health if k.startswith("ncv_")), None)
+tracker_algo = next((k[8:] for k in app.state.process_health if k.startswith("tracker_")), None)
```

Added SSE field: `tracker_algo`. The HUD computes centroid from
`bbox_x/y/w/h` in JS if it wants the crosshair.

### `ground/static/index.html`

Cosmetic; not strictly required by the refactor. Anywhere the HUD references
`centroid_x` / `centroid_y` should compute from bbox; anywhere it shows
`ccv_algo` / `ncv_algo` should display `tracker_algo`. Until updated, those
HUD widgets render `undefined`. Implementation plan lists this as a small
follow-up.

### `scripts/run.py`

```diff
-def _ncv_run(config, bus, frame_buffer):
-    from quadguide.inference.factory import get_runtime
-    from quadguide.perception.tracker_factories import get_ncv_tracker
-    from quadguide.perception.ncv_tracker_worker import NCVTrackerWorker
-    runtime = get_runtime(config)
-    tracker = get_ncv_tracker(config, runtime)
-    NCVTrackerWorker(tracker, bus, frame_buffer, config=config).run()
-
 def _start_workers(config, bus, frame_buffer, *, ground=True):
-    from quadguide.core.config import cfg_tracker
     from quadguide.perception.camera.worker import run_from_config as camera_run
-    from quadguide.perception.ccv_tracker_worker import run_from_config as ccv_run
-    from quadguide.perception.fusion.worker import run as fusion_run
+    from quadguide.perception.tracker_worker import run_from_config as tracker_run
     from quadguide.link.worker import run as link_run
     ...

-    tcfg = cfg_tracker(config)
-
     entries = [
         ("camera",   camera_run,   (config, bus, frame_buffer)),
+        ("tracker",  tracker_run,  (config, bus, frame_buffer)),
     ]
-    if tcfg.ccv is not None:
-        entries.append(("ccv_tracker", ccv_run,  (config, bus, frame_buffer)))
-    if tcfg.ncv is not None:
-        entries.append(("ncv_tracker", _ncv_run, (config, bus, frame_buffer)))
-    if tcfg.ccv is not None or tcfg.ncv is not None:
-        entries.append(("fusion",      fusion_run, (config, bus, frame_buffer)))
     entries += [
         ("link",     link_run,     (config, bus)),
         ("guidance", guidance_run, (config, bus, frame_buffer)),
         ("control",  control_run,  (config, bus, frame_buffer)),
     ]
     ...
```

### `scripts/bench_tracker.py`, `scripts/dev_ground_perception.py`

Currently walk the CCV + NCV + fusion path explicitly. Rewrite to load one
tracker via `load_tracker(config)` and drive it directly. Same pattern as
`run.py`. These are dev-only and may land in the same PR or a follow-up;
the runtime refactor does not depend on them.

### `systemd/`

Replace `qg-kcf.service`, `qg-nano.service`, `qg-fusion.service` with one
`qg-tracker.service`. Set `TimeoutStopSec=2s` (or higher if the configured
library's `close()` is known to be slow tearing down internal procs).

---

## Migration plan — single hard cutover

The refactor lands as one coordinated change rather than a feature-flagged
transition. All files added, deleted, and edited in the same PR. The HIL
rig is the validation gate — the PR does not merge until tracking works
end-to-end with `tracker.import: cv2:TrackerKCF` in HIL.

**Order of work within the PR:**

1. **Edit `core/messages.py`**: delete `TargetEstimate`, `FMT_TARGET_ESTIMATE`,
   `_ST_TARGET_ESTIMATE`, `ActiveTracker`. Update `__all__`. Add the
   `bbox_centroid_norm` helper next to `BoundingBox`.
2. **Edit `core/config.py`**: delete the obsolete dataclasses; replace
   `TrackerConfig`; replace `cfg_tracker`; remove the `inference` field on
   `PlatformConfig` and its construction in `cfg_platform`; rename
   `kcf_cpu_core` → `tracker_cpu_core` in `RealtimeConfig`.
3. **Edit `configs/config.yaml`**: delete `platform.inference`; replace the
   `tracker` block with `import` + `params`; rename
   `realtime.kcf_cpu_core` → `realtime.tracker_cpu_core`.
4. **Add `perception/tracker_worker.py`**: full file per Sections "Loader
   and OpenCV adapter" and "Worker IPC loop" above.
5. **Update consumer files**: guidance (base, pronav, pure_pursuit),
   ground (overlay, server, optionally static/index.html), `scripts/run.py`.
6. **Update bus topic pre-declaration site**: drop `ccv_tracker/estimate` and
   `ncv_tracker/estimate`. The pre-declaration list is created at startup
   in the parent process before workers fork (per architecture.md §2.3,
   §2.4); the implementation plan locates the exact call site by grepping
   for those two topic strings.
7. **Delete files**: the 20 files listed under "Files → Deleted" above.
8. **Update systemd units**: replace three service files with one.
9. **Update `scripts/bench_tracker.py`, `scripts/dev_ground_perception.py`**
   (in the same PR — easier to validate as a single change).
10. **Smoke-test paths**:
    - SWIL mode (`mission.mode: swil`) with `tracker.import: cv2:TrackerKCF`
      — full loop, simulated FC, virtual camera, expect to see `target/estimate`
      flowing and the control worker out of failsafe within the arm-dwell window.
    - bench_hil mode with `tracker.import: cv2:TrackerKCF` — same with real
      bench rig.
    - Manual lock-on via ground UI — confirm `tracker_algo: "kcf"` shows up.
11. **Update `architecture.md`**: process count, topic table, file inventory,
   limitations section. This is the documentation-only sweep.

The PR is not split because the cross-file edits (message schema + consumers
+ producer + bus declaration + run.py + deletions) are tightly coupled — any
intermediate state where some are merged and others aren't breaks the build.
Single PR keeps the bisect surface clean too.

---

## Known limitations

These are real trade-offs introduced by the refactor; the implementation
should add a corresponding section to `architecture.md` §11.

1. **Loss of intra-perception fault isolation.** Today a NanoTrack crash
   leaves KCF publishing; guidance gets degraded but valid data. After the
   refactor, any tracker-process crash takes the whole tracking path down,
   control's watchdog trips on `target/estimate` staleness, and the system
   fails safe to level flight. This is graceful degradation, not a flight-
   safety regression — but operational restarts now affect the entire
   perception path.
2. **NPU handle leak on SIGKILL is now the library's problem.** Quadguide
   guarantees that `tracker.close()` is called on SIGTERM. Under SIGKILL,
   `close()` does not run — same caveat as today's NCV worker. The library
   must document its SIGKILL behavior and any required device-reset steps.
3. **cv2 tracker params are not configurable.** The `tracker.params` field
   exists but the OpenCV adapter ignores it (cv2's typed `Params` object
   doesn't map cleanly to a YAML dict). The old `tracker.kcf.{detect_thresh,
   sigma, lambda_}` knobs become unreachable. Deferred until tuning actually
   requires it; the operational tracker is expected to be the external
   hybrid library, with cv2 trackers serving as a fallback / smoke-test path.
4. **No multi-tracker redundancy from quadguide's side.** If the hybrid
   library wants ensembles, it spawns them internally.
5. **bench_tracker.py and dev_ground_perception.py are broken until rewritten
   in the same PR.** Spec lists them as required updates.

---

## Out of scope

- The hybrid tracker library itself (gets its own design doc in its own repo).
- HIL changes (HIL composes with `run.py` once `run.py` is updated).
- CRSF / link / control / guidance algorithm changes.
- Bus implementation changes (topic table shrinks; mechanism unchanged).
- Adding a cv2 params bridge (deferred; listed under limitations).
- Per-tracker watchdog tunings (the existing `watchdog.target_estimate_ms`
  is reused — it's already gated on the same topic name).

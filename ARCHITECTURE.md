# quadguide — Architecture Blueprint

> This document is the single source of truth for the quadguide project structure,
> process model, data flow, and communication contracts. It is intended as both a
> developer reference and as context for AI-assisted development.
> Every design decision made here has a reason — those reasons are documented inline.

---

## 1. Project Overview

**quadguide** is an SBC-resident flight guidance stack for a manual lock-on target
tracking quadcopter. It runs on a companion computer (initially Raspberry Pi 4,
target RK3576 or RK3588 board) mounted on the airframe. It receives camera frames, runs two
parallel object trackers, fuses their outputs, computes proportional navigation
guidance commands, and sends roll/pitch setpoints to an ESP-FC flight controller
over UART using the MSP v2 protocol.

### Companion repo

**quadtrack** (git submodule, pinned) is a separate research workspace used to
develop and validate tracker algorithms. quadguide does NOT import quadtrack as a
library. Tracker algorithm code is ported manually from quadtrack into
`src/quadguide/perception/` when it is stable.

### Hardware stack

```
┌─────────────────────────────────┐
│  SBC                            │
│  quadguide running as systemd   │
│  services                       │
│                                 │
│  Camera ──→ perception workers  │
│  perception ──→ guidance        │
│  guidance ──→ control           │
│  control ──→ UART ──→ ESP-FC    │
│  ESP-FC ──→ UART ──→ link       │
│  link ──→ bus (attitude/IMU)    │
└─────────────────────────────────┘
```

The camera is oriented along the drone's +Z axis and is not gimbalized.
The image centre is the projection of the +Z axis onto the image plane.
The centroid error vector (image centre → target centroid) as well as
transmitted attitude predictions from FC are the primary guidance inputs.

---

## 2. Design Principles

### 2.1 One process per resource

Each OS process owns exactly one external resource or one logical responsibility.
No two processes share a file descriptor. This eliminates locking at the hardware
level and means any process can crash and be restarted by systemd without
corrupting another process's hardware state.


| Process          | Owned resource                |
| ------------------ | ------------------------------- |
| camera worker    | `/dev/video0` or CSI pipeline |
| kcf worker       | CPU core 1                    |
| nanotrack worker | `/dev/rknpu0` (NPU)           |
| fusion worker    | none — pure computation      |
| link worker      | UART serial port              |
| guidance worker  | none — pure computation      |
| control worker   | CPU core 3 (SCHED_FIFO)       |
| ground worker    | TCP port 8080                 |

### 2.2 Python multiprocessing, not threading

Python's GIL prevents true parallelism between threads. The two tracker workers
must run simultaneously — KCF at ~200 Hz on CPU and NanoTrack at ~30 Hz on the
NPU. Using `multiprocessing.Process` gives each worker its own interpreter with
its own GIL, allowing genuine concurrent execution on separate CPU cores.

asyncio is NOT used at the top level. Individual workers may use asyncio
internally for managing multiple I/O sources (e.g. the link worker managing
simultaneous UART read and write), but the process model is always
multiprocessing.

### 2.3 Shared memory for frames, structured bus for messages

Camera frames are large (e.g. 640×480×3 = 921 KB). Passing them through a pipe
or queue on every frame would saturate IPC bandwidth. Instead, the camera worker
writes frames into a shared memory ring buffer. Tracker workers read the latest
frame directly from shared memory with zero copy.

All other inter-process data (estimates, commands, telemetry) are small
dataclasses (< 200 bytes). These travel through the message bus — also shared
memory backed, but structured as a per-topic ring of serialised dataclass
instances packed with `struct`.

All topics in the bus are **pre-declared** at startup from the IPC table (Section 7).
This allows per-topic `os.pipe()` pairs to be created before any worker is spawned,
which is a hard requirement for the pipe-based blocking API (`subscribe_one` /
`subscribe_any`). Attempting to create synchronisation primitives after fork and
pass them to already-running processes is not supported.

### 2.4 Platform portability via config, not code branches

Adding a new SBC requires:

1. Adding one entry to `platform/factory.py`'s `PLATFORMS` dict
2. Adding one entry to `inference/factory.py`'s `RUNTIMES` dict
3. Recompiling ONNX models for the new NPU if applicable
4. Writing a new platform section in `configs/config.yaml`

No other source files change. Platform-specific behaviour is expressed as
configuration values, not `if platform == "rpi4"` branches scattered through the
codebase.

### 2.5 Failsafe is a first-class citizen

The control worker runs a watchdog on every input topic. If any of the following
go stale beyond their timeout, the control worker immediately switches to
`FailsafeState.LEVEL` and commands zero roll, zero pitch, zero closing velocity:

- `target/estimate` — perception pipeline dead or target lost
- `fc/attitude` — link worker dead or FC disconnected
- `guidance/accel` — guidance worker dead

The failsafe does not disarm the FC. It commands level flight and holds until
inputs recover or the operator intervenes.

---

## 3. Repository Layout

```
quadguide/
│
├── pyproject.toml              # package metadata, entry points, optional deps
├── README.md                   # setup, wiring, HIL quickstart
├── .gitmodules                 # quadtrack pinned as submodule (weights source only)
│
├── configs/
│   └── config.yaml             # single unified config file (see Section 5)
│
├── models/                     # compiled inference artefacts — NOT tracked in git
│   ├── nanotrack_backbone.onnx # source ONNX — used for CPU/CUDA runtime
│   ├── nanotrack_head.onnx
│   ├── nanotrack_backbone.rknn # compiled for OPi5 NPU by scripts/convert_rknn.py
│   └── nanotrack_head.rknn
│
├── src/quadguide/
│   ├── core/                   # shared primitives — no imports from other modules
│   ├── platform/               # SBC hardware abstraction
│   ├── inference/              # NPU/GPU runtime abstraction
│   ├── perception/             # four workers: camera, kcf, nanotrack, fusion
│   │   ├── camera/
│   │   ├── kcf/
│   │   ├── nanotrack/
│   │   └── fusion/
│   ├── link/                   # UART ↔ ESP-FC bridge
│   ├── guidance/               # proportional navigation
│   ├── control/                # attitude command + real-time loop
│   ├── hil/                    # hardware-in-the-loop harness
│   └── ground/                 # web-based operator interface
│
├── tests/
│   ├── unit/                   # pure logic, no hardware
│   ├── integration/            # multi-worker, no flight hardware
│   └── hil/                    # scenario-based pass/fail
│
├── scripts/                    # dev and deploy utilities
└── systemd/                    # service unit files for production
```

---

## 4. Process Architecture and Data Flow

### 4.1 Full data flow diagram

```
                    ┌─────────────────────────────────────────────────────┐
                    │  SHARED MEMORY                                      │
                    │                                                     │
                    │  frame_buffer (shm ring, 4 slots, ~1MB each)        │
                    │  bus topics (shm rings, small structs):             │
                    │    kcf/estimate   nano/estimate   target/estimate   │
                    │    fc/attitude    fc/imu          guidance/accel    │
                    │    control/cmd    system/health   lockon/cmd        │
                    └─────────────────────────────────────────────────────┘
                           ↑ write            ↓ read (zero-copy for frames)

[camera worker]
  open camera (V4L2 / GStreamer CSI / virtual)
  loop:
    frame = camera.read()
    frame_buffer.write_frame(frame)        → shm frame ring

                    ┌────────────────────────────┐
                    │ both read frame_buffer      │
                    ↓                            ↓

[kcf worker]                        [nanotrack worker]
  CPU core 1                          owns /dev/rknpu0
  loop (rate: ~200Hz):                loop (rate: ~30Hz):
    f = frame_buffer.read_latest()      f = frame_buffer.read_latest()
    est = kcf.update(f)                 est = nanotrack.update(f)
    bus.publish(                        bus.publish(
      "kcf/estimate", est)               "nano/estimate", est)

                    ↓                            ↓
                    └──────────┬─────────────────┘
                               ↓

[fusion worker]
  loop:
    topic, msg = bus.subscribe_any(
      ["kcf/estimate", "nano/estimate"])
    update latest_{kcf,nano}
    estimate = fuse(latest_kcf, latest_nano)
    bus.publish("target/estimate", estimate)

[link worker]                               [ground worker]
  rx loop:                                    subscribe all topics
    parse MSP frames from UART                serve web UI on :8080
    bus.publish("fc/attitude", att)           handle POST /lockon
    bus.publish("fc/imu", imu)                  → bus.publish("lockon/cmd", cmd)
  tx loop:                                    stream annotated MJPEG
    cmd = bus.latest("control/cmd")
    write MSP_SET_RAW_RC to UART

                    ↓ target/estimate
                    ↓ fc/attitude

[guidance worker]
  loop (50Hz):
    est = bus.latest("target/estimate")
    att = bus.latest("fc/attitude")
    los_r = los.los_rate(
      est.centroid_norm, att.body_rates, fov)
    v_c = closing_vel.estimate(est)
    accel = pronav.pronav(los_r, v_c, N)
    bus.publish("guidance/accel", accel)

                    ↓ guidance/accel
                    ↓ fc/attitude

[control worker]   ← SCHED_FIFO prio 80, CPU core 3
  loop (100Hz, RateLimiter):
    watchdog.check_all()           # failsafe if any input stale
    accel = bus.latest("guidance/accel")
    att   = bus.latest("fc/attitude")
    cmd   = attitude_cmd.compute(accel)
    cmd   = limiter.apply(cmd, prev_cmd)
    bus.publish("control/cmd", cmd)
    # link worker reads control/cmd and writes to UART
```

### 4.2 Lock-on flow

The lock-on command originates from the operator clicking a target in the ground
station web UI. The flow:

```
operator clicks bbox in browser
  → POST /lockon {"x":0.4,"y":0.3,"w":0.1,"h":0.1}
  → ground/server.py
  → bus.publish("lockon/cmd", LockOnCmd(bbox, timestamp_ns, seq=next_seq()))
  → kcf/worker.py reads lockon/cmd
      if cmd.seq != last_lockon_seq:
          kcf.init(frame_buffer.read_latest(), cmd.bbox)
          last_lockon_seq = cmd.seq
  → nanotrack/worker.py reads lockon/cmd
      if cmd.seq != last_lockon_seq:
          nanotrack.init(frame_buffer.read_latest(), cmd.bbox)
          last_lockon_seq = cmd.seq
  → both trackers now tracking
  → fusion worker begins producing TargetEstimate with tracker_health=TrackerHealth.NOMINAL
```

The `seq` counter on `LockOnCmd` is monotonically increasing. Each tracker
worker tracks the last `seq` it processed. This prevents silent skips if two
lock-on commands arrive close together (e.g. operator double-clicks): the
`latest()` read would otherwise return only the second command, and the first
could be silently dropped. With `seq`, any `seq != last_seen_seq` triggers a
reinit, regardless of whether intermediate commands were missed.

Until a lock-on command is received, both tracker workers run their update loops
but publish estimates with `tracker_health=TrackerHealth.NO_LOCK`. The fusion
worker propagates this through and the control worker watchdog suppresses commands
while no lock exists.

---

## 5. Configuration (configs/config.yaml)

Single file, four top-level sections. The `config.py` loader merges these and
provides typed accessors. CLI overrides are supported via `--set key=value`.

```yaml
platform:
  name: orange_pi5           # key into platform/factory.py PLATFORMS dict
  camera:
    backend: gstreamer       # "v4l2" | "gstreamer" | "virtual"
    pipeline: "..."          # GStreamer pipeline string for CSI cameras
    width: 640
    height: 480
    fps: 60
  serial:
    port: /dev/ttyS0
    baud: 115200
  inference:
    device: rknn             # "cpu" | "cuda" | "rknn" | "tensorrt"
    backbone: models/nanotrack_backbone.rknn
    head: models/nanotrack_head.rknn
  realtime:
    kcf_cpu_core: 1
    control_cpu_core: 3
    control_sched_fifo: true
    control_fifo_prio: 80

airframe:
  name: flix_micro
  mass_kg: 0.18
  # inertia tensor (kg·m²)
  inertia: [0.0012, 0.0012, 0.0020]
  control_limits:
    max_roll_deg: 35
    max_pitch_deg: 35
    max_roll_rate_dps: 200
    max_pitch_rate_dps: 200

tracker:
  kcf:
    detect_thresh: 0.5
    sigma: 0.2
    lambda_: 0.0001
  nanotrack:
    exemplar_sz: 127
    instance_sz: 255
    score_threshold: 0.7
  fusion:
    confidence_gate: 0.7      # use nanotrack above this, kcf below
    iou_divergence_thresh: 0.3
    nano_staleness_ms: 100    # ignore nano estimate if older than this

guidance:
  N: 4.0                      # proportional navigation gain (3–5 typical)
  closing_vel_fallback: 2.0   # m/s, used when bbox size rate is unreliable
  fov_horizontal_rad: 1.047   # camera horizontal FoV (~60°)

watchdog:
  target_estimate_ms: 150
  fc_attitude_ms: 50
  guidance_accel_ms: 100

mission:
  mode: bench_hil             # "flight" | "bench_hil" | "swil"
  hil:
    target_model: constant_velocity
    initial_offset_m: [2.0, 0.0, 0.0]
    target_speed_mps: 1.5

logging:
  level: INFO
  dir: /var/log/quadguide
  max_bytes: 10485760         # 10MB per file
  backup_count: 3
```

---

## 6. Source Modules — File by File

### 6.1 core/

Zero imports from other quadguide modules. Everything else imports from here.
Never the reverse.

**`core/messages.py`**
All inter-process data structures as frozen dataclasses. Every message carries
`timestamp_ns: int` (monotonic) set at production time. This is the single source
of truth for what crosses process boundaries.

Each dataclass is accompanied by a `struct` format string constant (prefixed
`FMT_`) that fully describes its wire layout. Health states are expressed as
a `TrackerHealth` enum (a `str` subclass so it serialises without extra logic)
rather than bare strings — this eliminates typo-class bugs and makes exhaustive
matching possible.

```python
from enum import Enum

class TrackerHealth(str, Enum):
    NOMINAL   = "nominal"
    UNCERTAIN = "uncertain"
    LOST      = "lost"
    NO_LOCK   = "no_lock"

@dataclass(frozen=True)
class BoundingBox:
    x: float        # top-left x, normalised 0–1
    y: float        # top-left y, normalised 0–1
    w: float        # width, normalised 0–1
    h: float        # height, normalised 0–1

# Wire: timestamp(Q=u64) + bbox(4f) + confidence(f) + health(B) = 29 bytes
FMT_TRACKER_ESTIMATE = "!QffffB"

@dataclass(frozen=True)
class TrackerEstimate:
    timestamp_ns: int
    bbox: BoundingBox
    confidence: float           # 0–1
    tracker_health: TrackerHealth

# Wire: timestamp(Q) + bbox(4f) + centroid(2f) + confidence(f) + health(B) + tracker(B) = 38 bytes
FMT_TARGET_ESTIMATE = "!QffffffBB"

@dataclass(frozen=True)
class TargetEstimate:
    timestamp_ns: int
    bbox: BoundingBox
    centroid_norm: tuple[float, float]  # (-1,1) range, (0,0) = image centre
    confidence: float
    tracker_health: TrackerHealth
    active_tracker: str                 # "kcf" | "nano" | "fused"

# Wire: timestamp(Q) + 6×float = 32 bytes
FMT_ATTITUDE_STATE = "!Qffffff"

@dataclass(frozen=True)
class AttitudeState:
    timestamp_ns: int
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    roll_rate_rps: float
    pitch_rate_rps: float
    yaw_rate_rps: float

# Wire: timestamp(Q) + 6×float = 32 bytes
FMT_IMU_FRAME = "!Qffffff"

@dataclass(frozen=True)
class IMUFrame:
    timestamp_ns: int
    ax: float; ay: float; az: float     # m/s²
    gx: float; gy: float; gz: float     # rad/s

# Wire: timestamp(Q) + 2×float = 16 bytes
FMT_ACCEL_CMD = "!Qff"

@dataclass(frozen=True)
class AccelCmd:
    timestamp_ns: int
    ax: float       # body-frame lateral accel command (m/s²)
    ay: float       # body-frame longitudinal accel command (m/s²)

# Wire: timestamp(Q) + 4×float = 24 bytes
FMT_CONTROL_CMD = "!Qffff"

@dataclass(frozen=True)
class ControlCmd:
    timestamp_ns: int
    roll_deg: float
    pitch_deg: float
    yaw_rate_dps: float
    throttle_norm: float    # 0–1

# Wire: timestamp(Q) + seq(H) + bbox(4f) = 26 bytes
# seq is a monotonically increasing counter. Tracker workers record the last
# seq they processed; if current seq != last_seen_seq, they reinitialise.
# This prevents silent skip of a lock-on if two arrive close together.
FMT_LOCKON_CMD = "!QHffff"

@dataclass(frozen=True)
class LockOnCmd:
    timestamp_ns: int
    seq: int                # monotonic lock-on sequence counter
    bbox: BoundingBox       # operator-selected initial bounding box

# Wire: timestamp(Q) + process(16s) + state(B) = 25 bytes; detail is logged only
FMT_HEALTH_REPORT = "!Q16sB"

@dataclass(frozen=True)
class HealthReport:
    timestamp_ns: int
    process: str
    state: str              # "ok" | "degraded" | "failsafe" | "dead"
    detail: str             # human-readable, not on the wire
```

**`core/bus.py`**
Shared memory message bus. One ring buffer per topic. Each ring holds the last N
messages serialised as packed bytes via `struct` (see `core/messages.py` for
per-message format strings). `pickle` is explicitly NOT used on the hot path —
`struct.pack/unpack` is a single C call with a precompiled format string, giving
microsecond-level IPC overhead appropriate for a 200 Hz tracker loop.

All topics are **pre-declared** at bus init time (see IPC table in Section 7).
Lazy creation of per-topic pipe pairs after process spawn is not supported —
pre-declaration eliminates the lazy-creation vs. pre-spawn-primitive contradiction
and costs nothing since the full topic set is known at design time.

Blocking is implemented via **`os.pipe()` per topic** created at bus init.
On `publish`, the writer drains any stale wakeup byte from the read end
(non-blocking), then writes one byte to the write end. On `subscribe_one`,
the reader calls `os.read(r_fd, 1)` which blocks with zero CPU until a
publish occurs. `subscribe_any` uses `select.select([t.r_fd for t in topics])`
— the kernel wakes the caller on whichever topic fires first. This is the only
design that gives zero-CPU blocking and multi-topic wakeup without a manager
process.

Provides:

- `Bus.publish(topic: str, msg: dataclass) → None`
- `Bus.latest(topic: str) → dataclass | None` — returns the most recent message,
  non-blocking
- `Bus.subscribe_one(topic: str) → dataclass` — blocks until a new message
  arrives on this topic (pipe-backed, zero CPU)
- `Bus.subscribe_any(topics: list[str]) → tuple[str, dataclass]` — blocks until
  any of the listed topics receives a new message; returns `(topic, msg)`
  (implemented via `select.select` over topic pipe read-ends)

Ring size per topic is configured in `config.yaml` under `bus.ring_depth`
(default 8). The bus object is initialised once in `scripts/run.py` and passed
to every worker at spawn time via the `multiprocessing` shared memory handle.

**`core/frame_buffer.py`**
Shared memory frame ring for zero-copy camera frame delivery. Distinct from the
bus because frames are numpy arrays (~1MB), not small structs.

- **6 slots** by default (not 4), each sized `width × height × channels` bytes.
  At 60 fps the ring wraps every ~100ms. KCF at 200 Hz reads every 5ms — well
  within one slot lifetime. NanoTrack at ~30 Hz reads every ~33ms — safe under
  normal conditions, but NPU inference can spike to 50ms+ on cold start or during
  RKNN model warm-up. With only 4 slots (wrap at ~67ms) a 50ms inference spike
  brings NanoTrack within one slot of the write head. 6 slots provides a
  comfortable margin and costs ~3MB additional shared memory.
- An atomic integer (via `multiprocessing.Value`) holds the index of the most
  recently completed write
- `FrameBuffer.write_frame(arr: np.ndarray) → None` — advances slot index,
  writes, marks ready
- `FrameBuffer.read_latest() → tuple[np.ndarray, int]` — returns a view of the
  latest ready slot and its timestamp_ns; does NOT copy
- Reader processes must consume or copy the frame before the camera worker
  overwrites the slot

**`core/clock.py`**

- `monotonic_ns() → int` — `time.monotonic_ns()` wrapper
- `RateLimiter(hz: float)` — call `RateLimiter.sleep()` to enforce a fixed loop
  rate; accounts for loop execution time
- `sleep_until(target_ns: int) → None` — precise sleep using `clock_nanosleep`
  via ctypes on Linux

**`core/health.py`**

- `Watchdog(topic: str, timeout_ms: int, bus: Bus)` — raises `HealthFault` if
  `bus.latest(topic)` timestamp is older than `timeout_ms`
- `FailsafeState` enum: `NOMINAL`, `LEVEL`, `DISARMED`
- `HealthReport` publishing helper used by every worker's main loop

**`core/config.py`**

- `load_config(path: str, overrides: dict) → dict` — loads YAML, applies
  CLI overrides, validates required keys
- Typed accessor helpers: `cfg_platform(config)`, `cfg_tracker(config)`, etc.

**`core/logging.py`**

- `setup_logging(process_name: str, config: dict) → logging.Logger`
- Configures a `RotatingFileHandler` per process writing to
  `/var/log/quadguide/{process_name}.log`
- Log record format includes `timestamp_ns`, process name, level, message

---

### 6.2 platform/

**`platform/adapter.py`**
Single `PlatformAdapter` class. Constructed with config dict. Methods:

- `open_serial() → serial.Serial` — returns configured, opened serial port
- `set_realtime(core: int, prio: int) → None` — sets CPU affinity and
  `SCHED_FIFO` priority via `os.sched_setaffinity` and `os.sched_setscheduler`;
  no-op if `config.realtime.control_sched_fifo` is false (dev machine)
- `gpio() → GPIOInterface | None` — returns GPIO handle if available, else None

**`platform/factory.py`**
Registry dict mapping platform name string to capability flags. `get_platform(config)`
reads `config["platform"]["name"]`, looks up the dict, returns a configured
`PlatformAdapter`. To add a new SBC: add one dict entry here.

```python
PLATFORMS = {
    "rpi4":        {"sched_fifo": True,  "gpio": "RPi.GPIO"},
    "orange_pi5":  {"sched_fifo": True,  "gpio": "OPi.GPIO"},
    "jetson_orin": {"sched_fifo": True,  "gpio": None},
    "dev_pc":      {"sched_fifo": False, "gpio": None},
}
```

---

### 6.3 inference/

**`inference/base.py`**
`NPURuntime` Protocol (structural subtyping via `typing.Protocol`):

```python
class NPURuntime(Protocol):
    def load(self, path: str) -> Any: ...
    def infer(self, model: Any, inputs: dict[str, np.ndarray]
              ) -> dict[str, np.ndarray]: ...
    def close(self) -> None: ...
```

All tracker inference code calls `runtime.infer(model, inputs)`. No tracker
file imports RKNN, TensorRT, or ONNX directly.

**`inference/onnx_cpu.py`**
`OnnxCPURuntime` — uses `onnxruntime.InferenceSession` with
`CPUExecutionProvider`. Universal fallback on any platform.

**`inference/onnx_cuda.py`**
`OnnxCUDARuntime` — uses `CUDAExecutionProvider`. Used on dev machine with RTX
3070 for development and benchmark.

**`inference/rknn.py`**
`RKNNRuntime` — uses `rknnlite.api.RKNNLite` when running on device. Auto-detects
whether it is on-device (import succeeds) or on x86 sim (uses `rknn.api.RKNN`
from rknn-toolkit2 for simulation). The nanotrack worker never needs to know which.

**`inference/tensorrt.py`**
`TensorRTRuntime` — loads `.engine` file, allocates CUDA input/output buffers,
runs `context.execute_async_v2()`. For Jetson Orin target.

**`inference/factory.py`**
`get_runtime(config) → NPURuntime`. Reads `config["platform"]["inference"]["device"]`,
returns the correct runtime instance. To add a new NPU: add one entry here and
one implementation file.

```python
RUNTIMES = {
    "cpu":      OnnxCPURuntime,
    "cuda":     OnnxCUDARuntime,
    "rknn":     RKNNRuntime,
    "tensorrt": TensorRTRuntime,
}
```

---

### 6.4 perception/

Four subdirectories, each containing a `worker.py` (process entry point) and the
algorithm code that worker uses. Algorithm files are pure functions/classes with
no bus or IPC dependencies.

#### perception/camera/

**`camera/worker.py`** — process entry point

```
open camera source (selected by config.platform.camera.backend)
loop:
    frame, ts = camera.read()
    frame_buffer.write_frame(frame, ts)
    bus.publish("system/health", HealthReport("camera", "ok"))
on SIGTERM:
    camera.close()
    exit cleanly
```

**`camera/sources.py`** — `CameraSource` ABC and implementations

- `CameraSource` ABC: `open()`, `read() → (np.ndarray, int)`, `close()`,
  `__iter__`
- `USBCamera(CameraSource)` — V4L2 via `cv2.VideoCapture`
- `CSICamera(CameraSource)` — GStreamer pipeline string from config
- `VirtualCamera(CameraSource)` — reads from `hil/virtual_source.py`; used in
  HIL mode; selected when `config.mission.mode != "flight"`

#### perception/kcf/

**`kcf/worker.py`** — process entry point

```
platform.set_realtime(core=config.kcf_cpu_core, prio=0)
kcf = KCFTracker(config.tracker.kcf)
loop:
    check bus for lockon/cmd → if new: kcf.init(frame_buffer.read_latest(), cmd.bbox)
    frame, ts = frame_buffer.read_latest()
    est = kcf.update(frame)
    bus.publish("kcf/estimate", est)
```

Rate is not artificially limited — KCF runs as fast as the CPU allows (~200Hz
typical). This is intentional: KCF is the high-rate fallback tracker.

**`kcf/tracker.py`**

- `KCFTracker(config)` — wraps `cv2.TrackerKCF_create()`
- `init(frame: np.ndarray, bbox: BoundingBox) → None`
- `update(frame: np.ndarray) → TrackerEstimate` — returns confidence 0 if
  tracking fails

#### perception/nanotrack/

**`nanotrack/worker.py`** — process entry point

```
runtime = get_runtime(config)
backbone_model = runtime.load(config.inference.backbone)
head_model = runtime.load(config.inference.head)
tracker = NanoTracker(runtime, backbone_model, head_model, config.tracker.nanotrack)
loop:
    check bus for lockon/cmd → if new: tracker.init(frame_buffer.read_latest(), cmd.bbox)
    frame, ts = frame_buffer.read_latest()
    est = tracker.update(frame)
    bus.publish("nano/estimate", est)
on SIGTERM:
    runtime.close()     # release NPU handle — critical, else NPU wedges
    exit cleanly
```

Rate is bottlenecked by NPU inference time (~30Hz on OPi5). This is expected —
NanoTrack is the high-accuracy, low-rate tracker.

**`nanotrack/tracker.py`**

- `NanoTracker(runtime, backbone, head, config)`
- `init(frame, bbox)` — runs backbone on exemplar crop, stores template features
- `update(frame) → TrackerEstimate` — runs backbone on search region, head to
  score and regress bbox, returns confidence from score map peak

**`nanotrack/preprocess.py`**

- `get_exemplar_crop(frame, bbox, exemplar_sz) → np.ndarray`
- `get_search_crop(frame, bbox, instance_sz) → np.ndarray`
- `normalise(crop) → np.ndarray` — ImageNet mean/std normalisation

**`nanotrack/postprocess.py`**

- `decode_response(score_map, bbox_map, stride) → (BoundingBox, float)`
  — finds peak in score map, reads regression at that location, maps back to
  image coordinates

#### perception/fusion/

**`fusion/worker.py`** — process entry point

```
latest_kcf: TrackerEstimate | None = None
latest_nano: TrackerEstimate | None = None
loop:
    topic, msg = bus.subscribe_any(["kcf/estimate", "nano/estimate"])
    if topic == "kcf/estimate":  latest_kcf = msg
    else:                        latest_nano = msg
    estimate = fuse(latest_kcf, latest_nano, config.tracker.fusion)
    bus.publish("target/estimate", estimate)
```

Fusion runs on every new arrival from either tracker. It does not wait for both.
This means guidance always has the freshest possible estimate at the rate of
whichever tracker last updated (~200 Hz dominated by KCF; NanoTrack contributes
at ~30 Hz).

The `subscribe_any` call is the only use of blocking multi-topic wait in the
entire system. All other workers use `bus.latest()` (non-blocking poll).

**`fusion/fusion.py`**
`fuse(kcf: TrackerEstimate | None, nano: TrackerEstimate | None, cfg) → TargetEstimate`

Logic:

1. If neither has a lock → return `tracker_health=TrackerHealth.NO_LOCK`
2. Staleness check: compare `nano.timestamp_ns` to `monotonic_ns()` **at the
   time of this fuse() call** (not to the KCF timestamp). If the delta exceeds
   `cfg.nano_staleness_ms` → treat nano as None. Using call-time monotonic_ns
   is critical: NPU inference time can vary (e.g. cold start, RKNN queue depth),
   so the gap between when NanoTrack published and when fusion evaluates can
   differ from the gap between NanoTrack and KCF timestamps.
3. Confidence gate: if `nano.confidence > cfg.confidence_gate`
   → use nano bbox as primary, label `active_tracker="nano"`
4. IoU divergence check: if both present and `iou(kcf.bbox, nano.bbox) < cfg.iou_divergence_thresh`
   → confidence is penalised, health = `TrackerHealth.UNCERTAIN`
5. Otherwise: weighted average of bboxes by respective confidence scores,
   label `active_tracker="fused"`
6. Compute `centroid_norm` from fused bbox: `((x + w/2 - 0.5) * 2, (y + h/2 - 0.5) * 2)`

---

### 6.5 link/

**`link/worker.py`** — process entry point
Two concurrent loops (asyncio internally, since this is pure I/O):

- RX loop: read bytes from serial, feed to MSP parser, on complete frame call
  `espfc.parse_attitude()` or `espfc.parse_imu()`, publish to bus
- TX loop: every 10ms read `bus.latest("control/cmd")`, encode as
  `MSP_SET_RAW_RC`, write to serial

The link worker publishes `HealthReport("link", ...)` at 5 Hz regardless of
UART state. This is the direct health signal for the link process. Note that
`fc/attitude` staleness in the control watchdog is an *indirect* signal of link
health — it only fires once the FC stops sending MSP frames, which may lag
behind the underlying serial fault. The direct `system/health` from the link
worker catches the fault earlier and lets the ground station display the correct
cause.

On serial disconnect: log error, attempt reconnect every 500ms, publish
`HealthReport("link", "degraded")` during outage.

**`link/msp.py`**
MSP v2 protocol implementation.

- `encode_cmd(cmd_id: int, payload: bytes) → bytes` — builds `$X<` framed packet
  with CRC8-DVB-S2
- `MSPParser` — stateful byte-by-byte parser; calls callback on complete frame
- `MSPFrame` dataclass: `cmd_id`, `payload`, `timestamp_ns`

**`link/espfc.py`**
ESP-FC specific knowledge:

- `parse_attitude(frame: MSPFrame) → AttitudeState` — `MSP_ATTITUDE` (cmd 108)
- `parse_imu(frame: MSPFrame) → IMUFrame` — `MSP_RAW_IMU` (cmd 102)
- `encode_rc(cmd: ControlCmd) → bytes` — maps roll/pitch/yaw/throttle to 8
  RC channel values (1000–2000 µs) for `MSP_SET_RAW_RC` (cmd 200)
- RC channel mapping: ch1=roll, ch2=pitch, ch3=throttle, ch4=yaw;
  ch5-8 are held at mid (1500) unless arming logic requires otherwise

**`link/serial_port.py`**

- `SerialPort(port, baud)` — opens with `pyserial`
- Non-blocking read with configurable timeout
- Write queue: `enqueue(data)` is non-blocking; background coroutine drains it
- `is_connected → bool`
- Reconnect logic on `serial.SerialException`

---

### 6.6 guidance/

**`guidance/worker.py`** — process entry point

```
rate = RateLimiter(hz=50)
loop:
    rate.sleep()
    est = bus.latest("target/estimate")
    att = bus.latest("fc/attitude")
    if est is None or att is None: continue
    if est.tracker_health in ("lost", "no_lock"): continue
    los_r = los.los_rate(est.centroid_norm, att, config.guidance.fov_horizontal_rad)
    v_c   = closing_vel.estimate(est)
    accel = pronav.pronav(los_r, v_c, config.guidance.N)
    bus.publish("guidance/accel", AccelCmd(monotonic_ns(), accel[0], accel[1]))
```

**`guidance/los.py`**
`los_rate(centroid_norm, attitude: AttitudeState, fov_rad) → tuple[float, float]`

Computes the line-of-sight rate vector in body frame. The image-plane centroid
error is a direct measurement of LOS angle (for small angles). LOS rate is
estimated by differencing centroid position between the current and previous
estimate, divided by elapsed time, then correcting for body rotation using
attitude body rates from the FC.

```
los_rate = (centroid_now - centroid_prev) / dt - body_rates_projected
```

**`guidance/pronav.py`**
`pronav(los_rate: tuple, closing_vel: float, N: float) → tuple[float, float]`

Proportional navigation law:

```
a_cmd = N × V_c × los_rate
```

Returns `(ax, ay)` — lateral body-frame acceleration commands in m/s².
`N` is the navigation gain, typically 3–5. Higher N = more aggressive pursuit.

**`guidance/closing_vel.py`**
`estimate(est: TargetEstimate) → float`

Estimates closing velocity from the rate of change of bounding box area.
Growing bbox → target is getting closer → positive closing velocity.
Falls back to `config.guidance.closing_vel_fallback` if the estimate is
noisy or bbox area change is below a minimum threshold.

**Important:** whenever the fallback constant is used, the function must emit a
`log.debug("closing_vel: using fallback")` at debug level. The fallback is a
meaningful diagnostic signal — if it fires continuously during a tracking run,
the PN gain is scaling against a wrong velocity and the guidance output is
suspect. `scripts/bench_tracker.py` should count fallback activations in its
CSV output.

---

### 6.7 control/

**`control/worker.py`** — process entry point, SCHED_FIFO, CPU core 3

```
platform.set_realtime(core=3, prio=80)
rate    = RateLimiter(hz=100)
watchdog = Watchdog(topics=[
    ("target/estimate", cfg.watchdog.target_estimate_ms),
    ("fc/attitude",     cfg.watchdog.fc_attitude_ms),
    ("guidance/accel",  cfg.watchdog.guidance_accel_ms),
], bus=bus)
prev_cmd = None
state = FailsafeState.NOMINAL

loop:
    rate.sleep()
    try:
        watchdog.check_all()
        state = FailsafeState.NOMINAL
    except HealthFault as e:
        state = FailsafeState.LEVEL
        bus.publish("control/cmd", limiter.failsafe_level())
        log.warning(f"Failsafe: {e}")
        continue

    accel = bus.latest("guidance/accel")
    att   = bus.latest("fc/attitude")
    cmd   = attitude_cmd.compute(accel, att)
    cmd   = limiter.apply(cmd, prev_cmd, cfg.airframe.control_limits)
    bus.publish("control/cmd", cmd)
    prev_cmd = cmd
```

**`control/attitude_cmd.py`**
`compute(accel: AccelCmd, att: AttitudeState) → ControlCmd`

Small-angle mapping from body-frame acceleration to attitude setpoints:

```
roll_deg  =  degrees(accel.ay / g)
pitch_deg = -degrees(accel.ax / g)
```

Yaw rate command is zero (yaw hold). Throttle is held constant at a config
value during tracking (altitude hold is delegated to the FC's baro loop if
available, or held open-loop).

**`control/limiter.py`**

- `saturate(cmd, limits) → ControlCmd` — clamps roll/pitch to
  `±max_roll_deg` / `±max_pitch_deg`
- `slew_rate(cmd, prev, max_dps, dt) → ControlCmd` — limits rate of change
  between consecutive commands
- `failsafe_level() → ControlCmd` — returns zero roll, zero pitch, zero yaw
  rate, mid throttle

**`control/watchdog.py`**
Per-topic staleness check called inside the control loop on every iteration.
Reads `bus.latest(topic)`, compares `timestamp_ns` to `monotonic_ns()`.
If delta > timeout: raises `HealthFault(topic)`.

---

### 6.8 hil/

Loaded only when `config.mission.mode` is `"bench_hil"` or `"swil"`. In flight
mode these files are never imported.

**`hil/orchestrator.py`** — HIL entry point
Spawns all normal workers but replaces the camera source with `VirtualCamera`.
Runs the target dynamics simulation and feeds it into `virtual_source.py`.
In `swil` mode, also replaces the link worker with a simulated FC dynamics model.

**`hil/virtual_source.py`**
`VirtualCamera(CameraSource)` — registered in `camera/sources.py` under the key
`"virtual"`. On each `read()` call, queries the current sim state from
`hil/projector.py` and renders a synthetic frame with the target drawn as a
coloured rectangle.

**`hil/target_models.py`**

- `ConstantVelocity` — target moves at fixed velocity vector; `step(dt) → Pose3D`
- `Maneuvering` — periodic step changes in velocity direction
- `Stationary` — fixed position

**`hil/projector.py`**
`project_target(pose_3d: Pose3D, attitude: AttitudeState, K: np.ndarray) → BoundingBox`
Projects a 3D target position into image-plane pixel coordinates using the camera
intrinsic matrix K (from calibration) and the current drone attitude. Returns a
normalised bounding box.

**`hil/dynamics.py`**
6-DoF rigid body integrator for the quad (used in `swil` mode only). Takes
`ControlCmd` from the bus, integrates equations of motion, publishes synthetic
`AttitudeState` and `IMUFrame` to the bus. Bypasses the real FC entirely.

**`hil/scenario.py`**
`load_scenario(path) → Scenario` — loads a YAML scenario file defining initial
conditions, target trajectory, duration, and pass/fail criteria (e.g. mean IoU
over run > 0.5).

---

### 6.9 ground/

**`ground/worker.py`** — process entry point
Subscribes to all bus topics. Runs `ground/server.py` FastAPI app. Feeds
annotated frames to the MJPEG stream. Handles lock-on POST requests by
publishing to `lockon/cmd`.

**`ground/server.py`**
FastAPI endpoints:

- `GET /stream` — MJPEG stream of annotated camera frames at ~15Hz
- `GET /telemetry` — Server-Sent Events; pushes latest `TargetEstimate`,
  `AttitudeState`, and all `HealthReport` messages as JSON every 100ms
- `POST /lockon` — body `{"x","y","w","h"}` normalised; publishes `LockOnCmd`
- `GET /health` — returns JSON summary of all process health states

**`ground/overlay.py`**
`draw_overlay(frame, estimate, attitude, health) → bytes`
Draws bounding box, centroid crosshair, confidence bar, attitude HUD, and
health indicator onto the frame. Returns JPEG bytes. Runs inside the ground
worker process — does not affect tracker frame timing.

**`ground/static/index.html`**
Single-file web UI (vanilla JS, no framework):

- Left pane: `<img>` tag pulling MJPEG stream
- Click handler on image: maps click coordinates to normalised bbox, POSTs to
  `/lockon`
- Right pane: telemetry dials (roll, pitch, confidence, LOS error)
- Bottom bar: health indicators per process, colour-coded

---

## 7. Inter-Process Communication Summary


| Topic             | Type                  | Producer         | Consumers                            | Approx rate      |
| ------------------- | ----------------------- | ------------------ | -------------------------------------- | ------------------ |
| frame_buffer      | shm ring (np.ndarray) | camera worker    | kcf worker, nanotrack worker         | 30–60 Hz        |
| `kcf/estimate`    | TrackerEstimate       | kcf worker       | fusion worker                        | ~200 Hz          |
| `nano/estimate`   | TrackerEstimate       | nanotrack worker | fusion worker                        | ~30 Hz           |
| `target/estimate` | TargetEstimate        | fusion worker    | guidance, control (watchdog), ground | ~200 Hz          |
| `fc/attitude`     | AttitudeState         | link worker      | guidance, control (watchdog), ground | 100–200 Hz      |
| `fc/imu`          | IMUFrame              | link worker      | ground                               | 200+ Hz          |
| `guidance/accel`  | AccelCmd              | guidance worker  | control worker                       | 50 Hz            |
| `control/cmd`     | ControlCmd            | control worker   | link worker                          | 100 Hz           |
| `lockon/cmd`      | LockOnCmd             | ground worker    | kcf worker, nanotrack worker         | event-driven     |
| `system/health`   | HealthReport          | all workers      | ground worker                        | 5 Hz per process |

---

## 8. Startup and Shutdown

### Startup sequence (managed by systemd)

```
1. qg-camera.service     starts first — frame_buffer shm created here
2. qg-kcf.service        After=qg-camera
3. qg-nano.service       After=qg-camera
4. qg-fusion.service     After=qg-kcf, qg-nano
5. qg-link.service       no camera dependency — starts in parallel
6. qg-guidance.service   After=qg-fusion, qg-link
7. qg-control.service    After=qg-guidance, qg-link
8. qg-ground.service     After=network.target (independent)
```

### Shutdown sequence

SIGTERM is sent to all workers by systemd. Each worker:

1. Finishes the current loop iteration
2. Publishes a final `HealthReport("dead")`
3. Releases owned resources (camera, NPU handle, serial port)
4. Exits with code 0

The nanotrack worker MUST release the NPU handle on exit. If it is killed with
SIGKILL (e.g. timeout), the `/dev/rknpu0` handle may be left open and the NPU
will require a reboot or driver reload to recover. The systemd `TimeoutStopSec`
for `qg-nano.service` should be set to at least 2s to allow clean shutdown.

### Manual startup (development)

```bash
python scripts/run.py --config configs/config.yaml
```

`run.py` reads config, creates the bus and frame_buffer in shared memory, then
spawns each worker as a `multiprocessing.Process`. Handles SIGINT/SIGTERM by
sending SIGTERM to all children and waiting for them to exit.


---

## 9. Scripts

**`scripts/run.py`**
Main entry point. Usage:

```bash
python scripts/run.py --config configs/config.yaml [--set platform.name=rpi4]
```

**`scripts/calibrate.py`**
Camera intrinsic calibration using a printed checkerboard. Writes `camera.K`
and `camera.dist_coeffs` into `configs/config.yaml`. Run once per camera/mount
combination.

```bash
python scripts/calibrate.py --config configs/config.yaml --board 9x6
```

**`scripts/bench_tracker.py`**
Offline benchmark of the full perception pipeline (camera → kcf → nanotrack →
fusion) against a recorded video file. Outputs per-frame IoU, confidence, and
inference latency as CSV.

```bash
python scripts/bench_tracker.py --video path/to/test.mp4 --config configs/config.yaml
```

**`scripts/convert_rknn.py`**
Converts ONNX models to RKNN format using rknn-toolkit2. Must be run on an x86
machine with rknn-toolkit2 installed. Writes output to `models/`.

```bash
python scripts/convert_rknn.py --backbone models/nanotrack_backbone.onnx \
                                --head models/nanotrack_head.onnx \
                                --platform rk3588s
```

**`scripts/deploy.py`**
Rsyncs source, configs, and models to the SBC over SSH. Restarts systemd services.

```bash
python scripts/deploy.py --host orangepi5.local --user plas
```

---

## 10. Adding a New SBC

1. Add a platform entry to `platform/factory.py`:
   ```python
   "my_new_sbc": {"sched_fifo": True, "gpio": "MyGPIOLib"}
   ```
2. Add an inference entry to `inference/factory.py` if using a new NPU.
   Write the corresponding `inference/my_npu.py` implementing `NPURuntime`.
3. Add a config section to `configs/config.yaml` under `platform:`.
4. Compile ONNX models for the new NPU if applicable (`scripts/convert_rknn.py`
   or equivalent).
5. Set `platform.name: my_new_sbc` in config and run.

No other source files change.

---

## 11. Known Constraints and Limitations

- **Altitude hold** is not implemented in quadguide. Throttle is held open-loop.
  The FC's barometer loop (if enabled in ESP-FC) handles altitude stability
  independently.
- **Yaw** is not controlled by the guidance system. Yaw hold is delegated to the
  FC heading hold mode.
- **Only one target at a time.** The lock-on command replaces any existing target.
- **No re-acquisition.** If the target is lost (`tracker_health="lost"`), the
  system enters failsafe level flight. Re-acquisition requires a new lock-on
  command from the operator.
- **MSP_SET_RAW_RC rate** is capped at ~100Hz by ESP-FC. The control loop runs
  at 100Hz to match. Faster commands will queue in the serial buffer.
- **NPU handle leak on SIGKILL.** Clean shutdown (SIGTERM) is required for the
  nanotrack worker. See Section 8.
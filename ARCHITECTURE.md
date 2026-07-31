# quadguide — Architecture Blueprint

> Single source of truth for project structure, process model, data flow, and
> communication contracts. Every design decision documented inline with its
> reason.

---

## 1. Project Overview

**quadguide** is an SBC-resident flight guidance stack for a manual lock-on
target tracking quadcopter. It runs on a companion computer (initially
Raspberry Pi 4, target RK3576 / RK3588) mounted on the airframe. It receives
camera frames, runs one configurable object tracker, computes proportional-
navigation or pure-pursuit guidance commands, and sends roll/pitch attitude setpoints
to an ArduPilot flight controller (H743) over UART using MAVLink2
(SET_ATTITUDE_TARGET, GUIDED_NOGPS).

The perception layer is now a single generic process (`tracker_worker`) that
loads its tracking algorithm at startup from `tracker.import` in config. Built
in: OpenCV trackers (`cv2:TrackerKCF`, `cv2:TrackerMOSSE`, `cv2:TrackerNano`,
…) via a small adapter. Pluggable: any external library that satisfies the
structural protocol (see §6.4) — typically a hybrid CCV+NCV+fusion tracker
that owns its own NPU and internal subprocesses.

### Hardware stack

```
┌─────────────────────────────────┐
│  SBC                            │
│  quadguide running as systemd   │
│  services                       │
│                                 │
│  Camera ──→ tracker             │
│  tracker ──→ guidance           │
│  guidance ──→ control           │
│  control ──→ UART ──→ FC        │
│  FC ──→ UART ──→                │
│  link ──→ bus (attitude + IMU)  │
└─────────────────────────────────┘
```

The camera is oriented along the drone's **+Z body axis and is not gimbalized.
When the quad is level, +Z points up — the camera looks at the sky, not the
ground.** The image centre is the projection of the +Z axis onto the image
plane. The centroid error vector (image centre → target centroid), together
with body-rate and acceleration data from the FC's MAVLink ATTITUDE
(#30) and RAW_IMU (#27) telemetry, are the primary guidance inputs.

> **Orientation note.** Because the bore-sight points up, the engagement
> geometry is the inverse of a conventional downward/forward-looking seeker.
> To pursue a target the quad tilts its thrust vector toward the target;
> tilting also swings the +Z axis (and therefore the bore-sight) in the same
> direction, which drives the centroid back toward image centre. Per-axis
> sign mapping is in `control/attitude_cmd.py`.

---

## 2. Design Principles

### 2.1 One process per resource

Each OS process owns exactly one external resource or one logical
responsibility. No two processes share a file descriptor. Any process can
crash and be restarted by systemd without corrupting another process's
hardware state.

| Process          | Owned resource                                          |
| ---------------- | ------------------------------------------------------- |
| camera worker    | `/dev/video0` or CSI pipeline                           |
| tracker worker   | CPU core 1 (configurable); NPU if used by the library   |
| link worker      | UART serial port                                        |
| guidance worker  | none — pure computation                                 |
| control worker   | CPU core 3 (SCHED_FIFO)                                 |
| ground worker    | TCP port 8080                                           |

Six processes total (seven with the optional ground UI). The tracker process
is the sole owner of all perception state — including, when the configured
library uses it, the NPU. Quadguide does not branch on NPU type; runtime
selection is the library's concern.

### 2.2 Python multiprocessing, not threading

Python's GIL prevents true parallelism between threads. `multiprocessing`
gives each worker its own interpreter; CPU-bound work (tracker.update,
control loop) runs concurrently on separate cores.

asyncio is not used at the top level. Individual workers may use asyncio
internally for I/O multiplexing (e.g. the link worker's simultaneous UART
read/write).

### 2.3 Shared memory for frames, structured bus for messages

Camera frames are large (640×480×3 ≈ 921 KB). The camera writes into a shared
memory ring buffer (`FrameBuffer`); the tracker reads the latest frame with
zero copy. All other inter-process data (estimates, commands, telemetry) are
small dataclasses (< 50 B on the wire) that travel through the bus —
also shared memory backed, structured as per-topic rings of dataclass
instances packed with `struct`.

The bus topic registry (`core/bus.py:TOPICS`) is created once in the parent
process before workers fork. Workers inherit the shared memory handles and
pipe fds across fork; no per-worker re-registration.

### 2.4 Platform portability via config, not code branches

A new SBC adds a row in `platform/factory.py` for its camera backend and a
preset YAML — never an `if platform == "rpi": …` in feature code. A new
tracker is just a new value for `tracker.import` in YAML; the library lives
in a separate repo with zero quadguide imports.

### 2.5 Failsafe is a first-class citizen

Every loop that drives the FC has a deadline. If `target/estimate` goes stale
(`watchdog.target_estimate_ms`), control commands level flight and holds
throttle. If `fc/imu` or `fc/attitude` go stale, control commands neutral
sticks. The link worker streams SET_ATTITUDE_TARGET at a constant 50 Hz regardless of
upstream health so the FC holds its GUIDED_NOGPS setpoint instead of timing out.

---

## 3. Repository Layout

```
quadguide/
│
├── pyproject.toml
├── README.md
│
├── configs/
│   └── config.yaml             # single unified config file (§5)
│
├── src/quadguide/
│   ├── core/                   # shared primitives — no cross-module imports
│   ├── platform/               # SBC hardware abstraction
│   ├── perception/             # camera worker + tracker worker
│   │   ├── camera/
│   │   └── tracker_worker.py   # generic tracker process (loader + adapter + loop)
│   ├── link/                   # MAVLink2 ↔ UART/TCP bridge
│   ├── guidance/               # pronav, pure_pursuit, LOS, closing-vel
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

### 4.1 Data flow diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │  SHARED MEMORY                                  │
                    │                                                 │
                    │  frame_buffer (shm ring, BGR HxWx3)             │
                    │  bus topics (shm rings of struct-packed msgs):  │
                    │    target/estimate                              │
                    │    fc/attitude    fc/imu          guidance/accel│
                    │    control/cmd    system/health   lockon/cmd    │
                    │    arm/cmd                                      │
                    └─────────────────────────────────────────────────┘
                          ↑ write           ↓ read (zero-copy frames)

[camera worker]                       [tracker worker]
  open camera                           load tracker from
  loop:                                 config.tracker.import
    frame = camera.read()               loop:
    frame_buffer.write_frame(frame)       check lockon/cmd
                                          f = frame_buffer.read_latest()
                                          out = tracker.update(f)
                                          bus.publish("target/estimate",
                                                      TrackerEstimate(...))

[link worker]                           [guidance worker]
  rx loop:                                loop (50 Hz RateLimiter):
    decode ATTITUDE → fc/attitude          est = bus.latest("target/estimate")
    decode RAW_IMU → fc/imu               imu = bus.latest("fc/imu")
  tx loop (50 Hz):                         ax, ay = method.compute(est, imu, …)
    encode SET_ATTITUDE_TARGET              bus.publish("guidance/accel", …)
      quat from roll/pitch + yaw latch
      thrust = throttle_norm
    arm via MAV_CMD_ARM_DISARM           [control worker]  CPU core 3, SCHED_FIFO
                                           loop (100 Hz):
                                             watchdogs on target/estimate,
                                                          fc/attitude, fc/imu
                                             accel = bus.latest("guidance/accel")
                                             cmd   = attitude_cmd(accel, att)
                                             bus.publish("control/cmd", cmd)

[ground worker]  (optional)
  serve web UI on :8080
  POST /lockon  → bus.publish("lockon/cmd", LockOnCmd)
  POST /arm     → bus.publish("arm/cmd", ArmCmd)
  GET /telemetry (SSE) → reads all topics
  GET /stream  (MJPEG) → frame_buffer + overlay(target/estimate)
```

### 4.2 Lock-on flow

1. Operator drags a bbox in the ground UI HUD.
2. Ground server publishes `LockOnCmd(seq=N+1, bbox=…)` on `lockon/cmd`.
3. Tracker worker sees a new `seq` on its next iteration, reads the latest
   frame, and calls `tracker.init(frame, bbox)`.
4. Subsequent iterations call `tracker.update(frame)` and publish
   `TrackerEstimate` on `target/estimate`.
5. Control's watchdog clears (`target/estimate` is fresh again) and the loop
   exits failsafe within the arm-dwell window.

A zero-size bbox in `LockOnCmd` triggers `tracker.reset()` — soft state clear,
no hardware release; next `update()` returns `health="no_lock"`.

---

## 5. Configuration (configs/config.yaml)

```yaml
platform:
  name: orange_pi5
  camera: { backend, pipeline, width, height, fps }
  serial: { port, baud, rx_pin, tx_pin }
  realtime:
    tracker_cpu_core: 1            # optional; null/absent → no affinity
    control_cpu_core: 3
    control_sched_fifo: true
    control_fifo_prio: 80

tracker:
  import: cv2:TrackerKCF           # "module:Class"; cv2:* uses built-in adapter
  params: {}                       # passed to constructor as **kwargs

guidance:
  method: pure_pursuit             # "pronav" | "pure_pursuit"
  fov_horizontal_rad: 0.972
  throttle_hold: 0.4
  pronav: { N, closing_vel_fallback, closing_vel_ema_alpha, … }
  pure_pursuit: { K }

watchdog:
  target_estimate_ms: 200
  fc_attitude_ms: 250
  fc_imu_ms: 250
  guidance_accel_ms: 100

link:
  tx_rate_hz: 50
  diff_lowpass_alpha: 0.3
  channels: { roll, pitch, throttle, yaw, arm, flight_mode, … }

mission:
  mode: bench_hil                  # "flight" | "bench_hil" | "swil"
  hil: { target_model, initial_offset_m, target_speed_mps }

logging: { level, dir, max_bytes, backup_count }
bus:     { ring_depth }
diag:    { trace, trace_dir, trace_max_rows }   # latency trace (§13); off by default, set via --log
```

Overrides: `python scripts/run.py --set guidance.pure_pursuit.K=15`.

---

## 6. Source Modules — File by File

### 6.1 core/

- **`messages.py`** — Wire-format dataclasses with `pack`/`unpack` against
  fixed `struct` formats. The complete topic-payload type list:
  `TrackerEstimate`, `BoundingBox`, `TrackerHealth`, `AttitudeState`,
  `IMUFrame`, `AccelCmd`, `ControlCmd`, `LockOnCmd`, `HealthReport`, `ArmCmd`,
  `ProcessState`. Format strings are the source of truth for wire layout.
- **`bus.py`** — `Bus` (shm rings + pipe wakeups) and `TOPICS` registry.
  Constructor builds all topics; child workers inherit on fork. Methods:
  `publish`, `latest`, `subscribe_one`, `detach`, `close`.
- **`frame_buffer.py`** — `FrameBuffer` — shm-backed multi-slot frame ring.
  Single writer (camera), single reader (tracker). Zero-copy reads.
- **`config.py`** — YAML loader + typed accessors. Dataclasses: `BusConfig`,
  `DiagConfig`, `LoggingConfig`, `MissionConfig`, `WatchdogConfig`, `PronavConfig`,
  `PurePursuitConfig`, `GuidanceConfig`, `TrackerConfig`, `AirframeConfig`,
  `RealtimeConfig`, `SerialConfig`, `CameraConfig`, `PlatformConfig`,
  `HILConfig`, `ControlLimitsConfig`. `cfg_*` functions narrow `dict` → typed
  config. Dot-notation overrides via `load_config(path, overrides)`.
- **`clock.py`** — `monotonic_ns()` wrapper (one source of time across the
  stack — never `time.time()` for deltas).
- **`logging.py`** — `setup_logging(name, config)` rotating-file logger per
  worker process.
- **`health.py`** — small helpers for HealthReport authoring.
- **`diagtrace.py`** — `DiagTrace`, the per-process diagnostic trace used by the
  `--log` latency capture (§13). No-op when disabled; RAM-buffered and flushed to
  one JSONL file per process at shutdown when enabled. `resolve_trace_dir()`
  picks/creates the timestamped output dir.

### 6.2 platform/

`factory.py` maps `platform.name` to a `(camera_factory,
serial_factory)` tuple. Add a new SBC by adding a row here and a preset
YAML — no edits to perception/control code.

### 6.3 perception/

- **`camera/`** — `sources.py` (`USBCamera`, `CSICamera`, `VirtualCamera`),
  `worker.py` (loop that pulls frames and writes to `FrameBuffer`; a read that
  fails because SIGTERM interrupted a blocking capture mid-shutdown is treated as
  teardown, not a fault). `USBCamera.open()` calls `_force_constant_framerate()`,
  which disables the UVC `exposure_dynamic_framerate` control — without it,
  auto-exposure lengthens exposure in low light and silently drops the frame rate
  (a Logitech C920 fell from 30 → 24 fps). Note the C920's hardware max is 30 fps
  in every format; there is no 60 fps mode.
- **`tracker_worker.py`** — the entire generic tracker process in ~200 lines:
  - `_resolve_cv2_factory(class_name)` — finds a cv2 tracker class on `cv2`
    or `cv2.legacy`.
  - `OpenCVTrackerAdapter` — wraps a cv2 tracker (pixel tuples + success
    bool) in the structural output protocol the worker reads.
  - `load_tracker(config)` — parses `tracker.import` (`module:Class`). For
    `cv2:*` returns the adapter; otherwise `importlib.import_module(module)`
    and `cls(**params)`. Errors (malformed spec, missing module, missing
    class, bad params) surface as ordinary Python exceptions and kill the
    worker — `run.py` surfaces the exit. No silent degradation.
  - `TrackerWorker` — IPC loop. Owns: lockon/cmd subscription, SHM frame
    read, `target/estimate` publish, periodic `system/health` publish.
    Tracker is opaque; no per-implementation branching. On SIGTERM, calls
    `tracker.close()` exactly once (last chance for NPU release / subprocess
    teardown). **New-frame gate:** the loop processes each frame exactly once —
    it skips (and briefly sleeps, `_IDLE_POLL_S`) when `frame_ts` is unchanged
    rather than re-running on a stale frame. Without this gate the loop free-ran
    at ~80 kHz, re-timestamping each frame thousands of times and producing a
    capture→track latency *sawtooth* that the 10 Hz HUD aliased into a rhythmic
    8–30 ms wobble (see §13). It also stamps `est.origin_ns = frame_ts`.
  - `run_from_config(config, bus, frame_buffer)` — entry point used by
    `scripts/run.py`; constructs the tracker and the worker, sets the
    optional CPU affinity, and calls `worker.run()`.

  #### Tracker protocol (structural)

  Trackers — built-in or external — expose:

  | Method   | Signature                                | Notes |
  | -------- | ---------------------------------------- | ----- |
  | `name`   | `() -> str`                              | Static; used to form the worker process name `tracker_<name>`. |
  | `init`   | `(frame: np.ndarray, bbox) -> None`      | Called per lock-on. bbox is duck-typed `.x/.y/.w/.h`, normalized 0–1. May be called repeatedly without `close()`. |
  | `update` | `(frame: np.ndarray) -> object`          | Per tick. Returns an object with `.bbox.x/y/w/h` (normalized 0–1), `.confidence` ∈ [0,1], `.health` ∈ `{"nominal","uncertain","lost","no_lock"}`. |
  | `reset`  | `() -> None`                             | Soft back-to-pre-init; must not release hardware. |
  | `close`  | `() -> None`                             | Called once in SIGTERM handler. Releases hardware, tears down internal subprocesses. |

  No quadguide imports required on the library side. Library defines its
  own bbox / output types (dataclass, namedtuple, attrs, SimpleNamespace —
  attribute names are read directly).

### 6.4 link/

MAVLink2 codec + UART worker for an ArduPilot FC. `mavlink_codec.py` builds a
codec-mode pymavlink object (`file=None`) and holds `euler_to_quaternion` +
message-id/mask constants; `fc.py` maps messages ⇄ bus dataclasses; `worker.py`
runs the RX/TX/heartbeat/stream-setup/health loops over the transport-agnostic
`SerialPort`/`TCPSerialPort`. RX decodes `ATTITUDE` (#30, native body rates) →
`fc/attitude` and `RAW_IMU` (#27) → `fc/imu`, and tracks armed/mode from
`HEARTBEAT`. TX streams `SET_ATTITUDE_TARGET` at a **constant** 50 Hz (roll/pitch
from `control/cmd`, yaw held by a latched heading baked into the quaternion,
thrust = `throttle_norm`); arming is edge-triggered via
`MAV_CMD_COMPONENT_ARM_DISARM`. The pilot's RC switch owns the RC↔GUIDED_NOGPS
toggle — quadguide never changes mode. Drop the 50 Hz stream and the FC abandons
the GUIDED setpoint. Requires FC params `SERIALn_PROTOCOL=2` and `GUID_OPTIONS`
bit 3 (direct thrust).

### 6.5 guidance/

- **`base.py`** — `GuidanceMethod` Protocol (`compute(est, imu, lockon_cmd,
  now_ns) -> (ax, ay)`).
- **`_centroid.py`** — `bbox_centroid_norm(bbox)` helper that converts
  `BoundingBox` (top-left, 0–1) to image-centre-relative centroid (-1..+1)
  on each axis. Used by pronav and pure_pursuit.
- **`pronav.py`** — Proportional navigation: `a = N · V_c · LOS_rate`. Uses
  `LOSRateEstimator` (image-plane derivative with body-rate derotation) and
  `ClosingVelEstimator` (bbox area growth → m/s).
- **`pure_pursuit.py`** — Simplest homing law: `a = K · LOS_angle`. No LOS
  rate, no closing velocity, no derotation. Centroid → LOS angle via FoV.
- **`los.py`**, **`closing_vel.py`** — pure computation.
- **`worker.py`** — 50 Hz `RateLimiter` loop; reads `target/estimate` +
  `fc/imu` + `lockon/cmd`; publishes `guidance/accel`.

### 6.6 control/

- **`watchdog.py`** — gates on `target/estimate`, `fc/attitude`, `fc/imu`
  staleness; flips to failsafe outputs on miss.
- **`attitude_cmd.py`** — maps `(ax, ay)` and current attitude to
  `(roll_deg, pitch_deg, yaw_rate_dps, throttle)`. Includes the
  bore-sight-up sign convention.
- **`worker.py`** — 100 Hz loop on CPU core 3, SCHED_FIFO. Reads
  `guidance/accel` + `fc/attitude` + `fc/imu`; publishes `control/cmd`.

### 6.7 hil/

`worker.py` — synthetic FC + target dynamics for `mission.mode: swil`
(software-in-the-loop) and `bench_hil`. Composes with `run.py`.

### 6.8 ground/

- **`server.py`** — FastAPI app: `GET /` (HUD), `GET /stream` (MJPEG),
  `GET /telemetry` (SSE), `POST /lockon`, `POST /reset_lockon`, `POST /arm`.
  The SSE payload includes `tracker_algo` (the tracker's `.name()` exposed
  via the worker's process name).
- **`overlay.py`** — draws the bbox over each MJPEG frame based on the
  latest `target/estimate`.
- **`worker.py`** — uvicorn launcher.
- **`static/index.html`** — HUD. Centroid for the crosshair is computed in
  JS from `bbox_x/y/w/h`.

### 6.9 Target-loss disarm failsafe

When `failsafe.disarm_on_lost` is set, the control worker debounces
`target/estimate.tracker_health == LOST` for `failsafe.lost_hold_ms` and, while
armed, publishes a latching `failsafe/disarm` (FailsafeCmd). The link worker
computes `effective_armed = arm/cmd AND NOT failsafe/disarm` and commands the FC
to DISARM via the existing arm path. The latch is sticky until the operator
re-arms (cycle the ground arm switch off→on). `tracker_health` is already the
NanoTrack confidence gate (`tracker.params.score_lock`/`score_lost`); this
feature only debounces and disarms. Note that re-arming while the target is
still `LOST` restarts the debounce and re-disarms after `lost_hold_ms`; a clean
re-arm requires the tracker to have re-acquired (health no longer `LOST`).

---

## 7. Inter-Process Communication Summary

| Topic              | Producer        | Consumers                          | Payload          | Wire |
| ------------------ | --------------- | ---------------------------------- | ---------------- | ---- |
| `frame_buffer` (shm) | camera        | tracker, ground                    | `np.ndarray`     | shm  |
| `target/estimate`  | tracker         | guidance, control watchdog, ground | `TrackerEstimate` | 37 B |
| `lockon/cmd`       | ground          | tracker                            | `LockOnCmd`      | 26 B |
| `fc/attitude`      | link            | control, ground                    | `AttitudeState`  | 32 B |
| `fc/imu`           | link            | guidance, control, ground          | `IMUFrame`       | 32 B |
| `guidance/accel`   | guidance        | control, ground                    | `AccelCmd`       | 24 B |
| `control/cmd`      | control         | link, ground                       | `ControlCmd`     | 32 B |
| `arm/cmd`          | ground          | link                               | `ArmCmd`         |  9 B |
| `failsafe/disarm`  | control         | link                               | `FailsafeCmd`    |  9 B |
| `system/health`    | all             | ground                             | `HealthReport`   | 25 B |

**Latency lineage (`origin_ns`).** `target/estimate`, `guidance/accel` and
`control/cmd` each carry an `origin_ns` field: the monotonic capture timestamp
(`frame_ts`) of the frame the message ultimately derives from. The tracker stamps
it; guidance and control copy it forward verbatim. Any stage can then compute the
true end-to-end "glass→here" age as `now − origin_ns` along the real consumed
lineage (see §13). `origin_ns == 0` means "no lineage yet" (pre-lock-on, or
control with no upstream accel). This is why those three payloads grew by 8 B
(`target/estimate` also swapped its old 4 B `latency_ns` delta for the 8 B
absolute `origin_ns`). The diagnostic trace (§13) does **not** use the bus — it
writes per-process files, because the bus is a latest-value transport, not a
lossless log.

---

## 8. Startup and Shutdown

### Startup (systemd, production)

A **single** unit, `quadguide.service`, runs `scripts/run.py` — the same
orchestrator used in development. systemd keeps that one parent alive; the
parent forks and supervises all six workers and owns `Bus`/`FrameBuffer`
lifecycle. A per-worker unit split is **not** possible: the bus's
`multiprocessing.Lock`/`Value` and anonymous `os.pipe()` wakeups are created
once in the parent and inherited across `fork()` (`core/bus.py`), so the
workers must share one parent process. Install with `scripts/install_sbc.sh`
(which also sets up the WiFi AP). Unit policy: `Restart=always` (keep
recovering in flight), `KillMode=mixed` (lets `run.py`'s ordered shutdown run),
`LimitRTPRIO=99` (control worker SCHED_FIFO), runs as `root`. See the operator
runbook `docs/sbc-setup.md`.

### Startup (development)

`python scripts/run.py --config configs/config.yaml [--no-ground]` forks all
workers from a single parent process. The parent owns `Bus.close()` and
`FrameBuffer.unlink()` for the lifetime of the run; workers `bus.detach()`
on SIGTERM.

### Shutdown

Any worker exit (clean or error) triggers `_shutdown(procs)` in `run.py`:
SIGTERM to all, 5 s grace, SIGKILL stragglers, then parent unlinks shm.

---

## 9. Scripts

- **`run.py`** — flight orchestrator (above). `--log` enables the diagnostic
  trace (§13) for every worker.
- **`dev_ground_perception.py`** — camera + tracker + ground only (no
  link/guidance/control). Useful for tracker tuning over the HUD, and for a
  latency baseline with no FC connected. Also accepts `--log`.
- **`diagnose_latency.py`** — offline latency analysis (§13). `trace <dir>`
  ingests a `--log` dump and reports per-stage / cumulative latency + a spectral
  view; `sim` reproduces the free-running sawtooth with no hardware.
- **`bench_tracker.py`** — frame-by-frame tracker benchmark from a recorded
  video; emits a CSV of `bbox / confidence / health / latency_ns`.
- **`calibrate.py`**, **`convert_rknn.py`**, **`deploy.py`**,
  **`test_link*.py`** — utilities.

---

## 10. Adding a New SBC

1. Add a `(camera_factory, serial_factory)` row to `platform/factory.py`.
2. Add a YAML preset under `configs/`.
3. If the SBC has an NPU you want to use, point `tracker.import` at a
   library that targets that NPU. Nothing in quadguide branches on NPU type.

---

## 11. Adding a New Tracker

In quadguide: edit `configs/config.yaml`:

```yaml
tracker:
  import: myhybrid.tracker:HybridTracker
  params: { model_path: models/hybrid.rknn, … }
```

In the external library: any class with `name`/`init`/`update`/`reset`/
`close` whose `update()` returns an object exposing `.bbox.{x,y,w,h}`,
`.confidence`, `.health`. The library may own its own NPU, spawn internal
subprocesses, and define its own types. **Zero quadguide imports required.**

### EdgeCV trackers

[EdgeCV](../EdgeCV) is the reference external library. Its `Tracker` contract is
close but not identical to the §6.4 protocol — `update()` returns a
`TrackResult(bbox|None, confidence|None, status: TrackStatus)`, it has no
`reset()`, and its trackers take a manifest + backend rather than plain kwargs.
`perception/edgecv_adapter.py:EdgeCVTracker` is the impedance match (analogous
to `OpenCVTrackerAdapter` for cv2): it converts BGR→RGB, maps `TrackStatus`→the
health strings, normalises confidence to 0–1 via the tracker's calibrator, and
implements `reset()` as a soft clear. EdgeCV stays quadguide-free; all EdgeCV
imports are lazy so the orchestrator parent never loads RKNN before fork (the
tracker child builds it). Wire it via config:

```yaml
tracker:
  import: quadguide.perception.edgecv_adapter:EdgeCVTracker
  params:
    tracker: nanotrack            # mosse | nanotrack | siamfc | yolo
    backend: rknn                 # auto | onnx | rknn | mock
    model_dir: /home/radxa/EdgeCV/models   # resolves *.rknn artifacts
```

`configs/rk3588.yaml` ships this preset (NanoTrack on the RK3588 NPU). MOSSE
needs no model and runs on CPU — handy for a no-NPU dry run.

#### AcquireTrack (`tracker: acquire_track`)

EdgeCV's `AcquireTrack` is a YOLO-acquire → NanoTrack-track hybrid that owns its
own process group (a YOLO worker on one NPU core, NanoTrack on another). It adds
two small contract points the adapter handles, with **no wire-format change**:

- **It runs before lock-on.** YOLO scans a fixed central crop continuously and the
  adapter calls `update()` every frame (the `_always_update` path), so pre-lock
  detection candidates flow even before any `init`. These report
  `TrackerHealth.ACQUIRING` — a new health that the **HUD draws (cyan) but guidance
  ignores** (`guidance/worker.py:_NON_DRIVING_HEALTH`), so a candidate box never
  drives flight. The HUD also draws the static crop guide (`index.html`,
  `ACQUIRE_CROP`).
- **The lock-on command commits, it doesn't pick.** Any **non-zero** `LockOnCmd`
  bbox means "lock the current best YOLO detection"; the existing crosshair box is
  fine as the carrier (its exact value is only a seed fallback when no detection is
  present). A **zero-size** bbox still means `reset()`.

`AcquireTrack` is async (workers infer behind the caller), so the adapter forwards
the result's **source-frame `origin_ns`** and `tracker_worker` prefers it over its
own `frame_ts`, keeping the §13 latency lineage honest about inference lag. See the
EdgeCV spec `docs/superpowers/specs/2026-06-14-acquire-track-design.md`. The
`acquire_track` preset (commented) is in `configs/rk3588.yaml`; it needs
`yolo11n.<soc>.rknn` in `model_dir`.

---

## 12. Known Constraints and Limitations

1. **SET_ATTITUDE_TARGET cadence is fixed at 50 Hz.** Drop it and the FC
   abandons its GUIDED_NOGPS setpoint — independent of quadguide's watchdogs.
   ArduPilot also needs `GUID_OPTIONS` bit 3 set so `thrust` is direct (0–1),
   not climb rate.

---

## 13. Latency Model & Diagnostics

### 13.1 What "latency" means here

This is a closed homing loop (§4.1), so the number that governs stability is the
**glass→actuation age**: how old a frame is when the control action it produced
reaches the FC. Every stage runs on the SBC as a `fork()`ed child sharing one
`CLOCK_MONOTONIC`, so timestamps stamped in one process and subtracted in another
are directly comparable — no clock sync.

Two complementary quantities:

- **Cumulative age** `now − origin_ns` — the true age of the data lineage at a
  given stage. `origin_ns` (the capture `frame_ts`) is stamped by the tracker and
  copied verbatim through `target/estimate → guidance/accel → control/cmd` (§7),
  so it follows the *actual consumed* messages even though every consumer reads
  `bus.latest()` and skips intermediate ones. This is the trustworthy end-to-end
  number.
- **Per-stage latency** `now − input.timestamp_ns` — how stale the freshest input
  was when a stage ran (its rate-limit wait + upstream age). Useful for finding
  the bottleneck stage; these do **not** telescope to the cumulative total,
  because they are asynchronous snapshots of different lineages.

The HUD shows the cumulative glass→control age (`control/cmd.timestamp_ns −
origin_ns`, which excludes SSE polling delay). The control→link/TX hop is visible
only in the trace (§13.3), since the link writes to UART and publishes no bus
message the HUD can read.

### 13.2 Why the bus is not used for raw latency capture

The bus is a **latest-value** transport (`bus.latest()` returns only the newest
slot; `subscribe_one()` also returns `latest()`; ring depth 8; the only reader is
the 10 Hz ground server; external processes can't attach to the inherited shm).
Pushing raw per-iteration samples through it would lose most of them and re-alias
the rest. So raw capture uses files, not topics.

### 13.3 The `--log` trace

`run.py --log` (or `dev_ground_perception.py --log`) sets `diag.trace` for every
worker and resolves a timestamped output dir (`{logging.dir}/trace/<ts>`, falling
back to `./quadguide-trace/<ts>` when the log dir isn't writable). Each worker
holds a `core/diagtrace.py:DiagTrace` that, when enabled, buffers records in RAM
and flushes one `{process}.jsonl` file at shutdown — **never** writing inside a
loop, so the SCHED_FIFO control loop takes no I/O. Record kinds: `lat` (raw
`t`/`in`/`org` timestamps), `state` (periodic worker state), `health`. Stats are
computed offline from the raw timestamps. Trace mode is for bench/diagnosis runs,
not production flight (unbounded RAM unless `diag.trace_max_rows` is set).

### 13.4 Analysis & a worked example

`scripts/diagnose_latency.py trace <dir>` derives per-stage and cumulative latency
and runs a spectral check that pins any rhythmic beat; `sim` reproduces the
mechanism with no hardware.

The original symptom — the HUD latency "bouncing 8–30 ms rhythmically, even with
no lock-on" — was the free-running tracker (§6.3) re-timestamping a stale frame
into a sawtooth at the camera frame interval, aliased by the 10 Hz SSE. Measured
on hardware before/after the new-frame gate (no-lock): tracker loop **83 kHz →
~30 Hz**, capture→track latency **p50 20.3 → 1.6 ms, p95 39.7 → 2.8 ms, std 11.5
→ 0.7 ms**; the sawtooth vanished. Locked, the trace then exposed the genuine
NanoTrack inference cost (~21.5 ms p50, well under the 33 ms frame interval) — a
number the old sawtooth had masked.

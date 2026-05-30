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
navigation or pure-pursuit guidance commands, and sends roll/pitch setpoints
to a madflight flight controller over UART using the CRSF protocol (420000
baud, bidirectional).

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
with raw body-rate and acceleration data from the FC over the CRSF `0x80` IMU
frame, are the primary guidance inputs.

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
sticks. The link worker keeps the CRSF uplink at a constant 50 Hz regardless
of upstream health so the FC never enters its own RX failsafe.

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
│   ├── link/                   # UART ↔ FC bridge (CRSF)
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
  rx loop (CRSF):                         loop (50 Hz RateLimiter):
    parse 0x1E → fc/attitude               est = bus.latest("target/estimate")
    parse 0x80 → fc/imu                    imu = bus.latest("fc/imu")
  tx loop (50 Hz):                         ax, ay = method.compute(est, imu, …)
    ctrl = bus.latest("control/cmd")       bus.publish("guidance/accel", …)
    arm  = bus.latest("arm/cmd")
    write CRSF channels                  [control worker]  CPU core 3, SCHED_FIFO
                                           loop (250 Hz):
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
  `LoggingConfig`, `MissionConfig`, `WatchdogConfig`, `PronavConfig`,
  `PurePursuitConfig`, `GuidanceConfig`, `TrackerConfig`, `AirframeConfig`,
  `RealtimeConfig`, `SerialConfig`, `CameraConfig`, `PlatformConfig`,
  `HILConfig`, `ControlLimitsConfig`. `cfg_*` functions narrow `dict` → typed
  config. Dot-notation overrides via `load_config(path, overrides)`.
- **`clock.py`** — `monotonic_ns()` wrapper (one source of time across the
  stack — never `time.time()` for deltas).
- **`logging.py`** — `setup_logging(name, config)` rotating-file logger per
  worker process.
- **`health.py`** — small helpers for HealthReport authoring.

### 6.2 platform/

`factory.py` maps `platform.name` to a `(camera_factory,
serial_factory)` tuple. Add a new SBC by adding a row here and a preset
YAML — no edits to perception/control code.

### 6.3 perception/

- **`camera/`** — `sources.py` (`V4L2Camera`, `GStreamerCamera`,
  `VirtualCamera`), `worker.py` (loop that pulls frames and writes to
  `FrameBuffer`).
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
    teardown).
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

CRSF parser/serializer + UART worker. RX path decodes `0x1E` (ATTITUDE),
`0x80` (custom IMU) into `fc/attitude` and `fc/imu`. TX path runs at a fixed
50 Hz, reading `control/cmd` and `arm/cmd`, mapping to channel ticks via
the configured `link.channels` table. The 50 Hz cadence is **constant**
regardless of upstream health — drop it and the FC enters its own RX
failsafe.

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
- **`worker.py`** — 250 Hz loop on CPU core 3, SCHED_FIFO. Reads
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

---

## 7. Inter-Process Communication Summary

| Topic              | Producer        | Consumers                          | Payload          | Wire |
| ------------------ | --------------- | ---------------------------------- | ---------------- | ---- |
| `frame_buffer` (shm) | camera        | tracker, ground                    | `np.ndarray`     | shm  |
| `target/estimate`  | tracker         | guidance, control watchdog, ground | `TrackerEstimate` | 33 B |
| `lockon/cmd`       | ground          | tracker                            | `LockOnCmd`      | 26 B |
| `fc/attitude`      | link            | control, ground                    | `AttitudeState`  | 32 B |
| `fc/imu`           | link            | guidance, control, ground          | `IMUFrame`       | 32 B |
| `guidance/accel`   | guidance        | control, ground                    | `AccelCmd`       | 16 B |
| `control/cmd`      | control         | link, ground                       | `ControlCmd`     | 24 B |
| `arm/cmd`          | ground          | link                               | `ArmCmd`         |  9 B |
| `system/health`    | all             | ground                             | `HealthReport`   | 25 B |

---

## 8. Startup and Shutdown

### Startup (systemd, production)

`quadguide.target` Wants/After: `qg-camera`, `qg-tracker`, `qg-link`,
`qg-guidance`, `qg-control`, `qg-ground`. Each unit is a thin invocation of
the matching worker entry point. The tracker unit has `After=qg-camera`.

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

- **`run.py`** — flight orchestrator (above).
- **`dev_ground_perception.py`** — camera + tracker + ground only (no
  link/guidance/control). Useful for tracker tuning over the HUD.
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

---

## 12. Known Constraints and Limitations

1. **CRSF TX cadence is fixed at 50 Hz.** Drop it and the FC enters its own
   RX failsafe — independent of quadguide's watchdogs.

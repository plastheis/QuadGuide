# Design Spec: Config + IPC Framework
**Date:** 2026-04-30
**Scope:** `core/messages.py`, `core/config.py`, `core/bus.py`
**Status:** Approved — ready for implementation planning

---

## 1. Context

quadguide is a multiprocessing flight guidance stack (one OS process per worker).
This spec covers the two foundational modules that every other module depends on:
the config loader/accessors and the shared-memory message bus.

All implementation must conform to ARCHITECTURE.md. Where this spec and
ARCHITECTURE.md conflict, this spec takes precedence (it incorporates post-doc
decisions made during the design session).

---

## 2. `core/messages.py`

### 2.1 Enums

Three enums, all `str` subclasses so they serialise to their string value in logs
and JSON without extra code:

```python
class TrackerHealth(str, Enum):
    NOMINAL   = "nominal"
    UNCERTAIN = "uncertain"
    LOST      = "lost"
    NO_LOCK   = "no_lock"

class ActiveTracker(str, Enum):
    KCF   = "kcf"
    NANO  = "nano"
    FUSED = "fused"

class ProcessState(str, Enum):
    OK       = "ok"
    DEGRADED = "degraded"
    FAILSAFE = "failsafe"
    DEAD     = "dead"
```

Each enum defines two class-level dicts for O(1) wire encoding:
- `_ord: dict[EnumType, int]` — enum value → byte ordinal (position in definition order)
- `_from_ord: dict[int, EnumType]` — byte ordinal → enum value

### 2.2 Message dataclasses

All dataclasses are `frozen=True`. Every field that is embedded in another
message (e.g. `BoundingBox` inside `TrackerEstimate`) is flattened into the
parent's struct format — `BoundingBox` has no standalone `FMT_` or `pack/unpack`.

Each message dataclass exposes:
- `pack(self) → bytes` — instance method, packs to wire bytes
- `unpack(cls, data: bytes) → Self` — classmethod, reconstructs from wire bytes

**Float precision:** Float fields are packed as IEEE 754 single precision (4 bytes, `f` format). Round-trip equality holds to float32 precision, not float64 — dataclasses store Python float64 values but the wire representation is float32. Test assertions on float fields must use `pytest.approx` rather than `==`; integer and enum fields use exact `==`.

#### Wire formats

The struct format string is the source of truth for byte layout and size.
Architecture byte-count comments that disagree with `struct.calcsize(fmt)` are
wrong; the format string wins. Corrections are noted inline.

```
FMT_TRACKER_ESTIMATE = "!QfffffB"
# Q(8) + bbox.x,y,w,h(4×f=16) + confidence(f=4) + health(B=1) = 29 bytes
# Breakdown: 8 + 16 + 4 + 1 = 29. (User said "30" in design session;
# struct.calcsize is the source of truth.)

FMT_TARGET_ESTIMATE = "!QfffffffBB"
# Q(8) + bbox(4×f=16) + centroid_x,y(2×f=8) + confidence(f=4)
#   + tracker_health(B=1) + active_tracker(B=1) = 38 bytes
# Two trailing B bytes: tracker_health ordinal, then active_tracker ordinal.
# Architecture had "!QffffffBB" (6 f's = 34 bytes) — corrected to 7 f's (38 bytes);
# the comment listed confidence as a field but the format string omitted it.

FMT_ATTITUDE_STATE = "!Qffffff"
# Q(8) + roll,pitch,yaw,roll_rate,pitch_rate,yaw_rate (6×f=24) = 32 bytes

FMT_IMU_FRAME = "!Qffffff"
# Q(8) + ax,ay,az,gx,gy,gz (6×f=24) = 32 bytes

FMT_ACCEL_CMD = "!Qff"
# Q(8) + ax,ay (2×f=8) = 16 bytes

FMT_CONTROL_CMD = "!Qffff"
# Q(8) + roll_deg,pitch_deg,yaw_rate_dps,throttle_norm (4×f=16) = 24 bytes

FMT_LOCKON_CMD = "!QHffff"
# Q(8) + seq(H=2) + bbox.x,y,w,h (4×f=16) = 26 bytes
# seq: H = uint16, wraps at 65535. Comparison is always !=, never >.
# Wraparound is safe because the test is identity, not ordering.

FMT_HEALTH_REPORT = "!Q16sB"
# Q(8) + process name (16s=16, zero-padded UTF-8) + state(B=1) = 25 bytes
# detail field is NOT on the wire — logged only.
#
# pack() MUST truncate before packing — struct.pack('16s', name) raises
# struct.error if the encoded name exceeds 16 bytes. Required idiom:
#   name_bytes = self.process.encode('utf-8')[:16].ljust(16, b'\x00')
# unpack() strips null bytes: name_bytes.rstrip(b'\x00').decode('utf-8')
```

#### Dataclass definitions (abbreviated)

```python
@dataclass(frozen=True)
class BoundingBox:
    x: float; y: float; w: float; h: float

@dataclass(frozen=True)
class TrackerEstimate:
    timestamp_ns: int
    bbox: BoundingBox
    confidence: float
    tracker_health: TrackerHealth

@dataclass(frozen=True)
class TargetEstimate:
    timestamp_ns: int
    bbox: BoundingBox
    centroid_norm: tuple[float, float]
    confidence: float
    tracker_health: TrackerHealth
    active_tracker: ActiveTracker

@dataclass(frozen=True)
class AttitudeState:
    timestamp_ns: int
    roll_rad: float; pitch_rad: float; yaw_rad: float
    roll_rate_rps: float; pitch_rate_rps: float; yaw_rate_rps: float

@dataclass(frozen=True)
class IMUFrame:
    timestamp_ns: int
    ax: float; ay: float; az: float
    gx: float; gy: float; gz: float

@dataclass(frozen=True)
class AccelCmd:
    timestamp_ns: int
    ax: float; ay: float

@dataclass(frozen=True)
class ControlCmd:
    timestamp_ns: int
    roll_deg: float; pitch_deg: float
    yaw_rate_dps: float; throttle_norm: float

@dataclass(frozen=True)
class LockOnCmd:
    timestamp_ns: int
    seq: int
    bbox: BoundingBox

@dataclass(frozen=True)
class HealthReport:
    timestamp_ns: int
    process: str       # max 16 bytes UTF-8 on the wire
    state: ProcessState
    detail: str        # not on the wire — log only
```

---

## 3. `core/config.py`

### 3.1 `load_config`

```python
def load_config(path: str, overrides: dict[str, str]) -> dict:
    ...
```

1. Load YAML from `path` using `yaml.safe_load`.
2. Apply each override: split key on `.`, traverse the dict tree, set the leaf
   value (coerced to match the existing value's type). Raise `KeyError`
   immediately if any segment of the path is absent — this is a programming
   error.
3. Validate that the following top-level keys are present:
   `platform`, `airframe`, `tracker`, `guidance`, `watchdog`, `mission`,
   `logging`. Missing key → `KeyError`.
4. Return the raw `dict` (not parsed into dataclasses).

### 3.2 Typed dataclass tree

All dataclasses are `frozen=True`. Construction validates implicitly — a missing
or wrong-type key raises `TypeError` from the dataclass constructor, which
propagates up as a config error at startup.

```
BusConfig           ring_depth: int = 8
LoggingConfig       level, dir, max_bytes, backup_count
HILConfig           target_model, initial_offset_m, target_speed_mps
MissionConfig       mode: str, hil: HILConfig | None = None
WatchdogConfig      target_estimate_ms, fc_attitude_ms, guidance_accel_ms
GuidanceConfig      N, closing_vel_fallback, fov_horizontal_rad
FusionConfig        confidence_gate, iou_divergence_thresh, nano_staleness_ms
NanotrackConfig     exemplar_sz, instance_sz, score_threshold
KCFConfig           detect_thresh, sigma, lambda_
TrackerConfig       kcf: KCFConfig, nanotrack: NanotrackConfig, fusion: FusionConfig
ControlLimitsConfig max_roll_deg, max_pitch_deg, max_roll_rate_dps, max_pitch_rate_dps
AirframeConfig      name, mass_kg, inertia: tuple[float,float,float], control_limits: ControlLimitsConfig
RealtimeConfig      kcf_cpu_core, control_cpu_core, control_sched_fifo, control_fifo_prio
InferenceConfig     device, backbone, head
SerialConfig        port, baud
CameraConfig        backend, pipeline, width, height, fps
PlatformConfig      name, camera, serial, inference, realtime
```

### 3.3 Accessor functions

One function per top-level section; each takes the raw dict and returns the
corresponding typed dataclass:

```python
def cfg_platform(d: dict) -> PlatformConfig: ...
def cfg_airframe(d: dict) -> AirframeConfig: ...
def cfg_tracker(d: dict) -> TrackerConfig: ...
def cfg_guidance(d: dict) -> GuidanceConfig: ...
def cfg_watchdog(d: dict) -> WatchdogConfig: ...
def cfg_mission(d: dict) -> MissionConfig: ...   # hil is None if key absent
def cfg_logging(d: dict) -> LoggingConfig: ...
def cfg_bus(d: dict) -> BusConfig: ...           # defaults ring_depth=8 if bus: absent
```

Workers receive the typed sub-object for their section (e.g. KCF worker gets
`TrackerConfig`), not the raw dict. Accessors are called once in `scripts/run.py`
before spawning.

---

## 4. `core/bus.py`

### 4.1 Topic registry

Module-level constant. All nine bus topics pre-declared. Any call with an
unknown topic name raises `KeyError` — this is a programming error.

```python
TOPICS: dict[str, tuple[type, str]] = {
    "kcf/estimate":    (TrackerEstimate, FMT_TRACKER_ESTIMATE),
    "nano/estimate":   (TrackerEstimate, FMT_TRACKER_ESTIMATE),
    "target/estimate": (TargetEstimate,  FMT_TARGET_ESTIMATE),
    "fc/attitude":     (AttitudeState,   FMT_ATTITUDE_STATE),
    "fc/imu":          (IMUFrame,        FMT_IMU_FRAME),
    "guidance/accel":  (AccelCmd,        FMT_ACCEL_CMD),
    "control/cmd":     (ControlCmd,      FMT_CONTROL_CMD),
    "lockon/cmd":      (LockOnCmd,       FMT_LOCKON_CMD),
    "system/health":   (HealthReport,    FMT_HEALTH_REPORT),
}
```

`frame_buffer` is handled by `core/frame_buffer.py` — it is NOT a bus topic.

### 4.2 Per-topic state

```python
@dataclass
class _TopicState:
    shm:       SharedMemory          # ring_depth × slot_size bytes
    lock:      multiprocessing.Lock  # protects shm write + pipe drain/write atomically
    head:      multiprocessing.Value # 'i', initialised to -1 (empty)
    r_fd:      int                   # os.pipe() read end, set non-blocking via fcntl
    w_fd:      int                   # os.pipe() write end
    slot_size: int                   # struct.calcsize(fmt)
    msg_class: type
    fmt:       str
```

### 4.3 `Bus.__init__(ring_depth: int = 8)`

For each topic in `TOPICS`:
1. Compute `slot_size = struct.calcsize(fmt)`.
2. Create `SharedMemory(create=True, size=ring_depth * slot_size)`.
3. Create `multiprocessing.Lock()` and `multiprocessing.Value('i', -1)`.
4. Call `os.pipe()` → `(r_fd, w_fd)`.
5. Set `r_fd` non-blocking: `fcntl.fcntl(r_fd, fcntl.F_SETFL, os.O_NONBLOCK)`.
6. Store `_TopicState`.

Store `ring_depth` as instance attribute.

### 4.4 `Bus.publish(topic: str, msg) → None`

```
state = _get_state(topic)   # KeyError if unknown
state.lock.acquire()
try:
    data = msg.pack()
    new_head = (state.head.value + 1) % ring_depth
    state.shm.buf[new_head * slot_size : (new_head + 1) * slot_size] = data
    state.head.value = new_head
    # drain stale wakeup byte (non-blocking — r_fd is O_NONBLOCK)
    try: os.read(state.r_fd, 1)
    except BlockingIOError: pass
    os.write(state.w_fd, b'\x00')
finally:
    state.lock.release()
```

The `try/finally` is required: if `os.write` raises during shutdown, the lock
must still be released so other processes do not deadlock.

The drain+write happening inside the lock ensures at most one byte ever lives
in the pipe, regardless of publish rate. This gives edge-triggered semantics:
subscribers always read the latest ring slot, not a queued message per publish.

### 4.5 `Bus.latest(topic: str) → msg | None`

```
state = _get_state(topic)
state.lock.acquire()
try:
    h = state.head.value
    if h == -1:
        return None
    data = bytes(state.shm.buf[h * slot_size : (h + 1) * slot_size])
finally:
    state.lock.release()
return state.msg_class.unpack(data)
```

`try/finally` ensures the lock is always released even if the shm read raises.

### 4.6 `Bus.subscribe_one(topic: str) → msg`

```
state = _get_state(topic)
select.select([state.r_fd], [], [])  # BLOCKS here — no lock held
                                     # select blocks regardless of O_NONBLOCK on fd
os.read(state.r_fd, 1)               # drain the wakeup byte (non-blocking read is fine now)
return self.latest(topic)            # acquires lock briefly for shm read
```

**Critical:** the lock is NOT held across the blocking `select.select`. If it were,
no publisher could acquire the lock to complete the drain+write in `publish`,
causing deadlock. The lock is only held inside `latest()` for the shm read.

`r_fd` is set `O_NONBLOCK` (for the drain in `publish`). A bare `os.read(r_fd, 1)`
on an empty O_NONBLOCK pipe raises `BlockingIOError` immediately — it does NOT
block. `select.select` is used here precisely because it blocks independently of
the `O_NONBLOCK` flag, then the `os.read` safely drains the already-ready byte.

**Constraint:** at most one process may block on a given topic at a time via
`subscribe_one`. The pipe holds at most one wakeup byte; if two processes block
simultaneously only one will be woken per publish. In the current architecture
this is not a problem (fusion worker is the sole blocking subscriber), but
violating this invariant is a programming error.

### 4.7 `Bus.subscribe_any(topics: list[str]) → tuple[str, msg]`

```
states = [_get_state(t) for t in topics]   # KeyError if any unknown
ready, _, _ = select.select([s.r_fd for s in states], [], [])
idx = [s.r_fd for s in states].index(ready[0])
topic_name = topics[idx]
os.read(states[idx].r_fd, 1)   # drain the wakeup byte
return topic_name, self.latest(topic_name)
```

`select.select` blocks with zero CPU until any of the listed topics receives a
publish. Same lock invariant as `subscribe_one` — no lock held during blocking.

Only `ready[0]` is processed per call. If multiple topics fire simultaneously,
`select.select` may return multiple ready fds, but only the first is consumed.
The remaining fds retain their wakeup bytes and will fire immediately on the
next `select.select` call. This is correct behaviour — not a missed message.

### 4.8 `Bus.close() → None`

Calls `shm.close()` + `shm.unlink()` on every topic's SharedMemory, then
`os.close(r_fd)` and `os.close(w_fd)` on every pipe pair.

**Must only be called after all child processes have been joined.** The parent's
`os.close(r_fd)` closes only the parent's fd copy; children retain their copies
until they exit. Calling `unlink()` while children still hold the shm open is
safe on Linux (the mapping persists until the last fd is closed), but it is
still good practice to join first.

### 4.9 `Bus.detach() → None`

Called by each worker process in its SIGTERM handler (before exit), NOT by the
parent. Calls `shm.close()` (NOT `shm.unlink()`) on every topic's SharedMemory
and closes local pipe fd copies. Workers must never call `shm.unlink()` — only
the parent owns the unlink lifecycle. Failing to call `detach()` leaks the
worker's fd references to the shared memory segment.

### 4.9 Process model

`Bus` is created in `scripts/run.py` before any `multiprocessing.Process` is
spawned. Linux `fork` start method means all child processes inherit:
- The `SharedMemory` mappings (shared in virtual memory — writes are visible
  across processes).
- All pipe fds `r_fd` and `w_fd` (harmless — children never use `w_fd` of
  topics they don't publish to; `r_fd` reads only consume one wakeup byte).
- All `multiprocessing.Lock` and `Value` objects (created pre-fork, so they
  use the kernel's futex/semaphore backing and work correctly cross-process).

No reconstruction from names is needed. The Bus object is passed as a
constructor argument to each `Process`.

`ring_depth` is wired from config in `scripts/run.py`:
```python
bus = Bus(ring_depth=cfg_bus(config).ring_depth)
```

---

## 5. Testing approach

### `messages.py`
- Round-trip test for every message type: `assert msg == MsgClass.unpack(msg.pack())`
- Verify `struct.calcsize(FMT_*)` matches expected byte counts.
- Verify enum ordinal encoding round-trips correctly.

### `config.py`
- Load the real `configs/config.yaml` and call all accessor functions — no
  assertion needed beyond "doesn't raise".
- Override application: verify dot-path override changes the correct nested key.
- Missing top-level key → `KeyError`.
- `cfg_mission` with no `hil:` section → `hil=None`.

### `bus.py`
- Single-process: publish then latest returns the message.
- Ring wrap: publish `ring_depth + 1` messages, latest returns the last one.
- `subscribe_one` blocking: spawn a thread that sleeps 50ms then publishes;
  assert the caller waited at least 40ms before returning. This is the critical
  property of the pipe design — it must be tested, not assumed.
- `subscribe_any` with two topics: fires on whichever is published first.
- Unknown topic → `KeyError`.
- No data → `latest` returns `None`.

---

## 6. Files produced

| File | Description |
|------|-------------|
| `src/quadguide/core/messages.py` | Enums, dataclasses, FMT_ strings, pack/unpack; `__all__` exported |
| `src/quadguide/core/config.py` | load_config, typed dataclasses, accessor fns |
| `src/quadguide/core/bus.py` | Bus class, `_TopicState` (private), TOPICS registry; `__all__ = ["Bus", "TOPICS"]` |
| `configs/config.yaml` | Populated with all sections from ARCHITECTURE.md |
| `tests/unit/test_bus.py` | Bus unit tests (including blocking subscribe_one test) |
| `tests/unit/test_messages.py` | Round-trip and format tests (new file) |
| `tests/unit/test_config.py` | Config load and accessor tests (new file) |

# Config + IPC Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `core/messages.py` (wire-format message types), `core/config.py` (YAML config loader + typed accessors), and `core/bus.py` (shared-memory message bus with pipe-backed blocking) — the three foundational modules every other quadguide module depends on.

**Architecture:** Each module is a standalone file in `src/quadguide/core/`. `messages.py` has zero imports from the rest of the codebase. `config.py` is also self-contained. `bus.py` imports only from `messages.py`. All three are created before any other module is touched. Tests live in `tests/unit/`.

**Tech Stack:** Python 3.11+, `multiprocessing.shared_memory`, `os.pipe()`, `select.select`, `struct`, `pyyaml`, `pytest`

**Spec:** `docs/superpowers/specs/2026-04-30-config-ipc-design.md` — read it before starting. Where this plan and the spec conflict, the spec takes precedence.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Create | Package metadata; pytest src layout config |
| `configs/config.yaml` | Create | Full runtime config (all sections) |
| `src/quadguide/__init__.py` | Create | Empty — makes package importable |
| `src/quadguide/core/__init__.py` | Create | Empty |
| `src/quadguide/core/messages.py` | Create | Enums, dataclasses, struct formats, pack/unpack |
| `src/quadguide/core/config.py` | Create | Typed dataclasses, load_config, accessor fns |
| `src/quadguide/core/bus.py` | Create | Bus class, TOPICS registry, shared-memory ring |
| `tests/__init__.py` | Create | Empty |
| `tests/unit/__init__.py` | Create | Empty |
| `tests/unit/test_messages.py` | Create | Round-trip + format size tests |
| `tests/unit/test_config.py` | Create | Load, override, accessor, optional-hil tests |
| `tests/unit/test_bus.py` | Populate | Publish/latest/subscribe/ring-wrap tests |

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/quadguide/__init__.py`
- Create: `src/quadguide/core/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "quadguide"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pyyaml"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create the empty `__init__.py` files**

```bash
touch src/quadguide/__init__.py \
      src/quadguide/core/__init__.py \
      tests/__init__.py \
      tests/unit/__init__.py
```

- [ ] **Step 3: Verify pytest can discover tests**

Run: `pytest tests/ --collect-only`
Expected: "no tests ran" with no import errors.

- [ ] **Step 4: Commit**

```bash
git init
git add pyproject.toml src/quadguide/__init__.py src/quadguide/core/__init__.py \
        tests/__init__.py tests/unit/__init__.py
git commit -m "feat: project scaffolding and pytest configuration"
```

---

## Task 2: `configs/config.yaml`

No TDD here — this is data, not logic. The config tests in Task 4 exercise it.

**Files:**
- Create: `configs/config.yaml`

- [ ] **Step 1: Create `configs/config.yaml`**

```yaml
platform:
  name: orange_pi5           # key into platform/factory.py PLATFORMS dict
  camera:
    backend: gstreamer       # "v4l2" | "gstreamer" | "virtual"
    pipeline: "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=60/1 ! appsink"
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
    confidence_gate: 0.7
    iou_divergence_thresh: 0.3
    nano_staleness_ms: 100

guidance:
  N: 4.0
  closing_vel_fallback: 2.0
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
  max_bytes: 10485760         # 10 MB per file
  backup_count: 3

bus:
  ring_depth: 8
```

- [ ] **Step 2: Commit**

```bash
git add configs/config.yaml
git commit -m "feat: add full configs/config.yaml"
```

---

## Task 3: `core/messages.py` — enums and format sizes

**Files:**
- Create: `tests/unit/test_messages.py` (partial — enums + sizes only)
- Populate: `src/quadguide/core/messages.py` (partial)

- [ ] **Step 1: Write failing tests for enums and struct sizes**

Create `tests/unit/test_messages.py`:

```python
import struct
import pytest
from quadguide.core.messages import (
    TrackerHealth, ActiveTracker, ProcessState,
    FMT_TRACKER_ESTIMATE, FMT_TARGET_ESTIMATE,
    FMT_ATTITUDE_STATE, FMT_IMU_FRAME, FMT_ACCEL_CMD,
    FMT_CONTROL_CMD, FMT_LOCKON_CMD, FMT_HEALTH_REPORT,
)


class TestEnumOrdinals:
    def test_tracker_health_round_trip(self):
        for health in TrackerHealth:
            assert TrackerHealth._from_ord[TrackerHealth._ord[health]] == health

    def test_active_tracker_round_trip(self):
        for tracker in ActiveTracker:
            assert ActiveTracker._from_ord[ActiveTracker._ord[tracker]] == tracker

    def test_process_state_round_trip(self):
        for state in ProcessState:
            assert ProcessState._from_ord[ProcessState._ord[state]] == state

    def test_tracker_health_is_str(self):
        assert TrackerHealth.NOMINAL == "nominal"

    def test_active_tracker_is_str(self):
        assert ActiveTracker.KCF == "kcf"

    def test_process_state_is_str(self):
        assert ProcessState.OK == "ok"


class TestFormatSizes:
    def test_tracker_estimate(self):
        assert struct.calcsize(FMT_TRACKER_ESTIMATE) == 29

    def test_target_estimate(self):
        assert struct.calcsize(FMT_TARGET_ESTIMATE) == 38

    def test_attitude_state(self):
        assert struct.calcsize(FMT_ATTITUDE_STATE) == 32

    def test_imu_frame(self):
        assert struct.calcsize(FMT_IMU_FRAME) == 32

    def test_accel_cmd(self):
        assert struct.calcsize(FMT_ACCEL_CMD) == 16

    def test_control_cmd(self):
        assert struct.calcsize(FMT_CONTROL_CMD) == 24

    def test_lockon_cmd(self):
        assert struct.calcsize(FMT_LOCKON_CMD) == 26

    def test_health_report(self):
        assert struct.calcsize(FMT_HEALTH_REPORT) == 25
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/unit/test_messages.py -v`
Expected: `ModuleNotFoundError: No module named 'quadguide.core.messages'`

- [ ] **Step 3: Implement enums and format strings in `core/messages.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import struct

__all__ = [
    "TrackerHealth", "ActiveTracker", "ProcessState",
    "BoundingBox",
    "TrackerEstimate", "TargetEstimate", "AttitudeState", "IMUFrame",
    "AccelCmd", "ControlCmd", "LockOnCmd", "HealthReport",
    "FMT_TRACKER_ESTIMATE", "FMT_TARGET_ESTIMATE", "FMT_ATTITUDE_STATE",
    "FMT_IMU_FRAME", "FMT_ACCEL_CMD", "FMT_CONTROL_CMD",
    "FMT_LOCKON_CMD", "FMT_HEALTH_REPORT",
]


def _byte_enum(cls):
    """Decorator: adds _ord and _from_ord dicts to an Enum for O(1) wire encoding."""
    members = list(cls)
    cls._ord = {e: i for i, e in enumerate(members)}
    cls._from_ord = {i: e for i, e in enumerate(members)}
    return cls


@_byte_enum
class TrackerHealth(str, Enum):
    NOMINAL   = "nominal"
    UNCERTAIN = "uncertain"
    LOST      = "lost"
    NO_LOCK   = "no_lock"


@_byte_enum
class ActiveTracker(str, Enum):
    KCF   = "kcf"
    NANO  = "nano"
    FUSED = "fused"


@_byte_enum
class ProcessState(str, Enum):
    OK       = "ok"
    DEGRADED = "degraded"
    FAILSAFE = "failsafe"
    DEAD     = "dead"


# Format strings are the source of truth for wire layout.
# Byte-count comments show the arithmetic; if a comment disagrees with
# struct.calcsize(fmt), the format string wins.

FMT_TRACKER_ESTIMATE = "!QfffffB"
# Q(8) + bbox.x,y,w,h(4×f=16) + confidence(f=4) + health(B=1) = 29 bytes

FMT_TARGET_ESTIMATE = "!QfffffffBB"
# Q(8) + bbox.x,y,w,h(4×f=16) + centroid_x,y(2×f=8) + confidence(f=4)
#   + tracker_health(B=1) + active_tracker(B=1) = 38 bytes
# NOTE: architecture.md had "!QffffffBB" (6 f's = 34 bytes) — corrected here.

FMT_ATTITUDE_STATE = "!Qffffff"
# Q(8) + roll,pitch,yaw,roll_rate,pitch_rate,yaw_rate(6×f=24) = 32 bytes

FMT_IMU_FRAME = "!Qffffff"
# Q(8) + ax,ay,az,gx,gy,gz(6×f=24) = 32 bytes

FMT_ACCEL_CMD = "!Qff"
# Q(8) + ax,ay(2×f=8) = 16 bytes

FMT_CONTROL_CMD = "!Qffff"
# Q(8) + roll_deg,pitch_deg,yaw_rate_dps,throttle_norm(4×f=16) = 24 bytes

FMT_LOCKON_CMD = "!QHffff"
# Q(8) + seq(H=2) + bbox.x,y,w,h(4×f=16) = 26 bytes
# seq is uint16 (wraps at 65535). Comparison is always !=, never >; wraparound is safe.

FMT_HEALTH_REPORT = "!Q16sB"
# Q(8) + process(16s=16) + state(B=1) = 25 bytes
# detail is NOT on the wire — logged only.
# pack() truncates process name to 16 UTF-8 bytes before packing.


@dataclass(frozen=True)
class BoundingBox:
    x: float  # top-left x, normalised 0–1
    y: float  # top-left y, normalised 0–1
    w: float  # width, normalised 0–1
    h: float  # height, normalised 0–1
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `pytest tests/unit/test_messages.py -v`
Expected: all 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/core/messages.py tests/unit/test_messages.py
git commit -m "feat: messages enums, BoundingBox, and struct format strings"
```

---

## Task 4: `core/messages.py` — message dataclasses with pack/unpack

**Files:**
- Modify: `tests/unit/test_messages.py` (append round-trip tests)
- Modify: `src/quadguide/core/messages.py` (append dataclasses)

- [ ] **Step 1: Append round-trip tests to `tests/unit/test_messages.py`**

Add after the existing `TestFormatSizes` class:

```python
class TestRoundTrips:
    def test_tracker_estimate(self):
        msg = TrackerEstimate(
            timestamp_ns=1_000_000,
            bbox=BoundingBox(0.1, 0.2, 0.3, 0.4),
            confidence=0.9,
            tracker_health=TrackerHealth.NOMINAL,
        )
        assert TrackerEstimate.unpack(msg.pack()) == msg

    def test_target_estimate(self):
        msg = TargetEstimate(
            timestamp_ns=2_000_000,
            bbox=BoundingBox(0.1, 0.2, 0.3, 0.4),
            centroid_norm=(-0.1, 0.2),
            confidence=0.85,
            tracker_health=TrackerHealth.UNCERTAIN,
            active_tracker=ActiveTracker.FUSED,
        )
        assert TargetEstimate.unpack(msg.pack()) == msg

    def test_attitude_state(self):
        msg = AttitudeState(
            timestamp_ns=3_000_000,
            roll_rad=0.1, pitch_rad=0.2, yaw_rad=0.3,
            roll_rate_rps=0.01, pitch_rate_rps=0.02, yaw_rate_rps=0.03,
        )
        assert AttitudeState.unpack(msg.pack()) == msg

    def test_imu_frame(self):
        msg = IMUFrame(
            timestamp_ns=4_000_000,
            ax=1.0, ay=2.0, az=9.8,
            gx=0.1, gy=0.2, gz=0.3,
        )
        assert IMUFrame.unpack(msg.pack()) == msg

    def test_accel_cmd(self):
        msg = AccelCmd(timestamp_ns=5_000_000, ax=1.5, ay=-0.5)
        assert AccelCmd.unpack(msg.pack()) == msg

    def test_control_cmd(self):
        msg = ControlCmd(
            timestamp_ns=6_000_000,
            roll_deg=10.0, pitch_deg=-5.0,
            yaw_rate_dps=0.0, throttle_norm=0.5,
        )
        assert ControlCmd.unpack(msg.pack()) == msg

    def test_lockon_cmd(self):
        msg = LockOnCmd(
            timestamp_ns=7_000_000, seq=42,
            bbox=BoundingBox(0.3, 0.3, 0.2, 0.2),
        )
        assert LockOnCmd.unpack(msg.pack()) == msg

    def test_lockon_cmd_seq_max(self):
        # seq is uint16; 65535 must survive round-trip without overflow
        msg = LockOnCmd(timestamp_ns=0, seq=65535, bbox=BoundingBox(0, 0, 1, 1))
        assert LockOnCmd.unpack(msg.pack()).seq == 65535

    def test_health_report_round_trip(self):
        msg = HealthReport(
            timestamp_ns=8_000_000,
            process="camera",
            state=ProcessState.OK,
            detail="all good",
        )
        recovered = HealthReport.unpack(msg.pack())
        assert recovered.timestamp_ns == msg.timestamp_ns
        assert recovered.process == "camera"
        assert recovered.state == ProcessState.OK
        assert recovered.detail == ""  # detail is not on the wire

    def test_health_report_truncates_long_name(self):
        # process names > 16 UTF-8 bytes must be silently truncated, not raise
        msg = HealthReport(
            timestamp_ns=0, process="a" * 20,
            state=ProcessState.DEAD, detail="",
        )
        recovered = HealthReport.unpack(msg.pack())
        assert recovered.process == "a" * 16
```

Also add these imports at the top of the file (after the existing imports):

```python
from quadguide.core.messages import (
    TrackerHealth, ActiveTracker, ProcessState,
    BoundingBox, TrackerEstimate, TargetEstimate,
    AttitudeState, IMUFrame, AccelCmd, ControlCmd,
    LockOnCmd, HealthReport,
    FMT_TRACKER_ESTIMATE, FMT_TARGET_ESTIMATE,
    FMT_ATTITUDE_STATE, FMT_IMU_FRAME, FMT_ACCEL_CMD,
    FMT_CONTROL_CMD, FMT_LOCKON_CMD, FMT_HEALTH_REPORT,
)
```

(Replace the existing partial import at the top of the test file.)

- [ ] **Step 2: Run tests — verify new tests fail**

Run: `pytest tests/unit/test_messages.py::TestRoundTrips -v`
Expected: `ImportError` — `TrackerEstimate` not yet defined.

- [ ] **Step 3: Append dataclasses to `src/quadguide/core/messages.py`**

Add after `BoundingBox` (append to end of file). Use precompiled `struct.Struct`
objects (module-level) for performance — one C object per format, reused on every call.

```python
_ST_TRACKER_ESTIMATE = struct.Struct(FMT_TRACKER_ESTIMATE)
_ST_TARGET_ESTIMATE  = struct.Struct(FMT_TARGET_ESTIMATE)
_ST_ATTITUDE_STATE   = struct.Struct(FMT_ATTITUDE_STATE)
_ST_IMU_FRAME        = struct.Struct(FMT_IMU_FRAME)
_ST_ACCEL_CMD        = struct.Struct(FMT_ACCEL_CMD)
_ST_CONTROL_CMD      = struct.Struct(FMT_CONTROL_CMD)
_ST_LOCKON_CMD       = struct.Struct(FMT_LOCKON_CMD)
_ST_HEALTH_REPORT    = struct.Struct(FMT_HEALTH_REPORT)


@dataclass(frozen=True)
class TrackerEstimate:
    timestamp_ns: int
    bbox: BoundingBox
    confidence: float
    tracker_health: TrackerHealth

    def pack(self) -> bytes:
        return _ST_TRACKER_ESTIMATE.pack(
            self.timestamp_ns,
            self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h,
            self.confidence,
            TrackerHealth._ord[self.tracker_health],
        )

    @classmethod
    def unpack(cls, data: bytes) -> TrackerEstimate:
        ts, x, y, w, h, conf, health_b = _ST_TRACKER_ESTIMATE.unpack(data)
        return cls(
            timestamp_ns=ts,
            bbox=BoundingBox(x, y, w, h),
            confidence=conf,
            tracker_health=TrackerHealth._from_ord[health_b],
        )


@dataclass(frozen=True)
class TargetEstimate:
    timestamp_ns: int
    bbox: BoundingBox
    centroid_norm: tuple[float, float]
    confidence: float
    tracker_health: TrackerHealth
    active_tracker: ActiveTracker

    def pack(self) -> bytes:
        return _ST_TARGET_ESTIMATE.pack(
            self.timestamp_ns,
            self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h,
            self.centroid_norm[0], self.centroid_norm[1],
            self.confidence,
            TrackerHealth._ord[self.tracker_health],
            ActiveTracker._ord[self.active_tracker],
        )

    @classmethod
    def unpack(cls, data: bytes) -> TargetEstimate:
        ts, bx, by, bw, bh, cx, cy, conf, health_b, tracker_b = _ST_TARGET_ESTIMATE.unpack(data)
        return cls(
            timestamp_ns=ts,
            bbox=BoundingBox(bx, by, bw, bh),
            centroid_norm=(cx, cy),
            confidence=conf,
            tracker_health=TrackerHealth._from_ord[health_b],
            active_tracker=ActiveTracker._from_ord[tracker_b],
        )


@dataclass(frozen=True)
class AttitudeState:
    timestamp_ns: int
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    roll_rate_rps: float
    pitch_rate_rps: float
    yaw_rate_rps: float

    def pack(self) -> bytes:
        return _ST_ATTITUDE_STATE.pack(
            self.timestamp_ns,
            self.roll_rad, self.pitch_rad, self.yaw_rad,
            self.roll_rate_rps, self.pitch_rate_rps, self.yaw_rate_rps,
        )

    @classmethod
    def unpack(cls, data: bytes) -> AttitudeState:
        ts, roll, pitch, yaw, rr, pr, yr = _ST_ATTITUDE_STATE.unpack(data)
        return cls(ts, roll, pitch, yaw, rr, pr, yr)


@dataclass(frozen=True)
class IMUFrame:
    timestamp_ns: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

    def pack(self) -> bytes:
        return _ST_IMU_FRAME.pack(
            self.timestamp_ns,
            self.ax, self.ay, self.az,
            self.gx, self.gy, self.gz,
        )

    @classmethod
    def unpack(cls, data: bytes) -> IMUFrame:
        ts, ax, ay, az, gx, gy, gz = _ST_IMU_FRAME.unpack(data)
        return cls(ts, ax, ay, az, gx, gy, gz)


@dataclass(frozen=True)
class AccelCmd:
    timestamp_ns: int
    ax: float
    ay: float

    def pack(self) -> bytes:
        return _ST_ACCEL_CMD.pack(self.timestamp_ns, self.ax, self.ay)

    @classmethod
    def unpack(cls, data: bytes) -> AccelCmd:
        ts, ax, ay = _ST_ACCEL_CMD.unpack(data)
        return cls(ts, ax, ay)


@dataclass(frozen=True)
class ControlCmd:
    timestamp_ns: int
    roll_deg: float
    pitch_deg: float
    yaw_rate_dps: float
    throttle_norm: float

    def pack(self) -> bytes:
        return _ST_CONTROL_CMD.pack(
            self.timestamp_ns,
            self.roll_deg, self.pitch_deg,
            self.yaw_rate_dps, self.throttle_norm,
        )

    @classmethod
    def unpack(cls, data: bytes) -> ControlCmd:
        ts, roll, pitch, yaw_rate, throttle = _ST_CONTROL_CMD.unpack(data)
        return cls(ts, roll, pitch, yaw_rate, throttle)


@dataclass(frozen=True)
class LockOnCmd:
    timestamp_ns: int
    seq: int  # uint16; wraps at 65535. Always compare with !=, never >
    bbox: BoundingBox

    def pack(self) -> bytes:
        return _ST_LOCKON_CMD.pack(
            self.timestamp_ns, self.seq,
            self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h,
        )

    @classmethod
    def unpack(cls, data: bytes) -> LockOnCmd:
        ts, seq, x, y, w, h = _ST_LOCKON_CMD.unpack(data)
        return cls(ts, seq, BoundingBox(x, y, w, h))


@dataclass(frozen=True)
class HealthReport:
    timestamp_ns: int
    process: str    # max 16 UTF-8 bytes on the wire; longer names are truncated
    state: ProcessState
    detail: str     # NOT on the wire — logged only; always "" after unpack

    def pack(self) -> bytes:
        name_bytes = self.process.encode("utf-8")[:16].ljust(16, b"\x00")
        return _ST_HEALTH_REPORT.pack(
            self.timestamp_ns,
            name_bytes,
            ProcessState._ord[self.state],
        )

    @classmethod
    def unpack(cls, data: bytes) -> HealthReport:
        ts, name_bytes, state_b = _ST_HEALTH_REPORT.unpack(data)
        return cls(
            timestamp_ns=ts,
            process=name_bytes.rstrip(b"\x00").decode("utf-8"),
            state=ProcessState._from_ord[state_b],
            detail="",
        )
```

- [ ] **Step 4: Run all message tests**

Run: `pytest tests/unit/test_messages.py -v`
Expected: all 24 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/core/messages.py tests/unit/test_messages.py
git commit -m "feat: message dataclasses with struct pack/unpack"
```

---

## Task 5: `core/config.py`

**Files:**
- Create: `tests/unit/test_config.py`
- Create: `src/quadguide/core/config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_config.py`:

```python
import os
import tempfile
import pytest
from quadguide.core.config import (
    load_config,
    cfg_platform, cfg_airframe, cfg_tracker,
    cfg_guidance, cfg_watchdog, cfg_mission,
    cfg_logging, cfg_bus,
    BusConfig,
)

CONFIG_PATH = "configs/config.yaml"


class TestLoadConfig:
    def test_loads_real_config(self):
        config = load_config(CONFIG_PATH, {})
        assert isinstance(config, dict)

    def test_all_required_sections_present(self):
        config = load_config(CONFIG_PATH, {})
        for section in ("platform", "airframe", "tracker", "guidance",
                        "watchdog", "mission", "logging"):
            assert section in config

    def test_missing_top_level_section_raises(self):
        yaml_text = "platform:\n  name: dev_pc\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        try:
            with pytest.raises(KeyError, match="airframe"):
                load_config(path, {})
        finally:
            os.unlink(path)

    def test_override_string_value(self):
        config = load_config(CONFIG_PATH, {"platform.name": "dev_pc"})
        assert config["platform"]["name"] == "dev_pc"

    def test_override_integer_value(self):
        # Width is int in yaml; override must coerce string "320" to int 320
        config = load_config(CONFIG_PATH, {"platform.camera.width": "320"})
        assert config["platform"]["camera"]["width"] == 320

    def test_override_unknown_path_raises(self):
        with pytest.raises(KeyError):
            load_config(CONFIG_PATH, {"nonexistent.deep.key": "value"})


class TestAccessors:
    def setup_method(self):
        self.config = load_config(CONFIG_PATH, {})

    def test_cfg_platform(self):
        p = cfg_platform(self.config)
        assert p.name == "orange_pi5"
        assert p.camera.width == 640
        assert p.camera.fps == 60
        assert p.serial.baud == 115200
        assert p.inference.device == "rknn"
        assert p.realtime.kcf_cpu_core == 1
        assert p.realtime.control_sched_fifo is True

    def test_cfg_airframe(self):
        a = cfg_airframe(self.config)
        assert a.name == "flix_micro"
        assert a.mass_kg == pytest.approx(0.18)
        assert len(a.inertia) == 3
        assert a.control_limits.max_roll_deg == 35

    def test_cfg_tracker(self):
        t = cfg_tracker(self.config)
        assert t.kcf.detect_thresh == pytest.approx(0.5)
        assert t.nanotrack.exemplar_sz == 127
        assert t.fusion.confidence_gate == pytest.approx(0.7)

    def test_cfg_guidance(self):
        g = cfg_guidance(self.config)
        assert g.N == pytest.approx(4.0)
        assert g.closing_vel_fallback == pytest.approx(2.0)

    def test_cfg_watchdog(self):
        w = cfg_watchdog(self.config)
        assert w.target_estimate_ms == 150
        assert w.fc_attitude_ms == 50

    def test_cfg_mission_with_hil(self):
        m = cfg_mission(self.config)
        assert m.mode == "bench_hil"
        assert m.hil is not None
        assert m.hil.target_model == "constant_velocity"
        assert len(m.hil.initial_offset_m) == 3

    def test_cfg_mission_hil_none_when_absent(self):
        config = dict(self.config)
        config["mission"] = {"mode": "flight"}
        m = cfg_mission(config)
        assert m.hil is None

    def test_cfg_logging(self):
        lg = cfg_logging(self.config)
        assert lg.level == "INFO"
        assert lg.max_bytes == 10_485_760

    def test_cfg_bus_from_config(self):
        bus = cfg_bus(self.config)
        assert bus.ring_depth == 8

    def test_cfg_bus_defaults_when_section_absent(self):
        bus = cfg_bus({})
        assert bus == BusConfig(ring_depth=8)
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/unit/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'quadguide.core.config'`

- [ ] **Step 3: Implement `src/quadguide/core/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import yaml


# ── Leaf config dataclasses ──────────────────────────────────────────────────

@dataclass(frozen=True)
class BusConfig:
    ring_depth: int = 8


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    dir: str
    max_bytes: int
    backup_count: int


@dataclass(frozen=True)
class HILConfig:
    target_model: str
    initial_offset_m: tuple[float, float, float]
    target_speed_mps: float


@dataclass(frozen=True)
class MissionConfig:
    mode: str
    hil: HILConfig | None = None


@dataclass(frozen=True)
class WatchdogConfig:
    target_estimate_ms: int
    fc_attitude_ms: int
    guidance_accel_ms: int


@dataclass(frozen=True)
class GuidanceConfig:
    N: float
    closing_vel_fallback: float
    fov_horizontal_rad: float


@dataclass(frozen=True)
class FusionConfig:
    confidence_gate: float
    iou_divergence_thresh: float
    nano_staleness_ms: int


@dataclass(frozen=True)
class NanotrackConfig:
    exemplar_sz: int
    instance_sz: int
    score_threshold: float


@dataclass(frozen=True)
class KCFConfig:
    detect_thresh: float
    sigma: float
    lambda_: float


@dataclass(frozen=True)
class TrackerConfig:
    kcf: KCFConfig
    nanotrack: NanotrackConfig
    fusion: FusionConfig


@dataclass(frozen=True)
class ControlLimitsConfig:
    max_roll_deg: float
    max_pitch_deg: float
    max_roll_rate_dps: float
    max_pitch_rate_dps: float


@dataclass(frozen=True)
class AirframeConfig:
    name: str
    mass_kg: float
    inertia: tuple[float, float, float]
    control_limits: ControlLimitsConfig


@dataclass(frozen=True)
class RealtimeConfig:
    kcf_cpu_core: int
    control_cpu_core: int
    control_sched_fifo: bool
    control_fifo_prio: int


@dataclass(frozen=True)
class InferenceConfig:
    device: str
    backbone: str
    head: str


@dataclass(frozen=True)
class SerialConfig:
    port: str
    baud: int


@dataclass(frozen=True)
class CameraConfig:
    backend: str
    pipeline: str
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class PlatformConfig:
    name: str
    camera: CameraConfig
    serial: SerialConfig
    inference: InferenceConfig
    realtime: RealtimeConfig


# ── Loader ───────────────────────────────────────────────────────────────────

_REQUIRED_SECTIONS = frozenset(
    {"platform", "airframe", "tracker", "guidance", "watchdog", "mission", "logging"}
)


def load_config(path: str, overrides: dict[str, str]) -> dict:
    """Load YAML config, apply dot-notation overrides, validate required sections."""
    with open(path) as f:
        config = yaml.safe_load(f)

    missing = _REQUIRED_SECTIONS - config.keys()
    if missing:
        raise KeyError(f"Required config section(s) missing: {sorted(missing)}")

    for dotpath, str_value in overrides.items():
        parts = dotpath.split(".")
        node = config
        for part in parts[:-1]:
            node = node[part]  # KeyError propagates if path is wrong
        leaf_key = parts[-1]
        existing = node[leaf_key]   # KeyError if leaf doesn't exist
        node[leaf_key] = type(existing)(str_value)

    return config


# ── Typed accessors ──────────────────────────────────────────────────────────

def cfg_platform(d: dict) -> PlatformConfig:
    p = d["platform"]
    cam = p["camera"]
    return PlatformConfig(
        name=p["name"],
        camera=CameraConfig(
            backend=cam["backend"],
            pipeline=cam.get("pipeline", ""),
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"],
        ),
        serial=SerialConfig(port=p["serial"]["port"], baud=p["serial"]["baud"]),
        inference=InferenceConfig(
            device=p["inference"]["device"],
            backbone=p["inference"]["backbone"],
            head=p["inference"]["head"],
        ),
        realtime=RealtimeConfig(
            kcf_cpu_core=p["realtime"]["kcf_cpu_core"],
            control_cpu_core=p["realtime"]["control_cpu_core"],
            control_sched_fifo=p["realtime"]["control_sched_fifo"],
            control_fifo_prio=p["realtime"]["control_fifo_prio"],
        ),
    )


def cfg_airframe(d: dict) -> AirframeConfig:
    a = d["airframe"]
    lim = a["control_limits"]
    return AirframeConfig(
        name=a["name"],
        mass_kg=a["mass_kg"],
        inertia=tuple(a["inertia"]),
        control_limits=ControlLimitsConfig(
            max_roll_deg=lim["max_roll_deg"],
            max_pitch_deg=lim["max_pitch_deg"],
            max_roll_rate_dps=lim["max_roll_rate_dps"],
            max_pitch_rate_dps=lim["max_pitch_rate_dps"],
        ),
    )


def cfg_tracker(d: dict) -> TrackerConfig:
    t = d["tracker"]
    return TrackerConfig(
        kcf=KCFConfig(
            detect_thresh=t["kcf"]["detect_thresh"],
            sigma=t["kcf"]["sigma"],
            lambda_=t["kcf"]["lambda_"],
        ),
        nanotrack=NanotrackConfig(
            exemplar_sz=t["nanotrack"]["exemplar_sz"],
            instance_sz=t["nanotrack"]["instance_sz"],
            score_threshold=t["nanotrack"]["score_threshold"],
        ),
        fusion=FusionConfig(
            confidence_gate=t["fusion"]["confidence_gate"],
            iou_divergence_thresh=t["fusion"]["iou_divergence_thresh"],
            nano_staleness_ms=t["fusion"]["nano_staleness_ms"],
        ),
    )


def cfg_guidance(d: dict) -> GuidanceConfig:
    g = d["guidance"]
    return GuidanceConfig(
        N=g["N"],
        closing_vel_fallback=g["closing_vel_fallback"],
        fov_horizontal_rad=g["fov_horizontal_rad"],
    )


def cfg_watchdog(d: dict) -> WatchdogConfig:
    w = d["watchdog"]
    return WatchdogConfig(
        target_estimate_ms=w["target_estimate_ms"],
        fc_attitude_ms=w["fc_attitude_ms"],
        guidance_accel_ms=w["guidance_accel_ms"],
    )


def cfg_mission(d: dict) -> MissionConfig:
    m = d["mission"]
    hil_raw = m.get("hil")
    hil = None
    if hil_raw is not None:
        hil = HILConfig(
            target_model=hil_raw["target_model"],
            initial_offset_m=tuple(hil_raw["initial_offset_m"]),
            target_speed_mps=hil_raw["target_speed_mps"],
        )
    return MissionConfig(mode=m["mode"], hil=hil)


def cfg_logging(d: dict) -> LoggingConfig:
    lg = d["logging"]
    return LoggingConfig(
        level=lg["level"],
        dir=lg["dir"],
        max_bytes=lg["max_bytes"],
        backup_count=lg["backup_count"],
    )


def cfg_bus(d: dict) -> BusConfig:
    bus_raw = d.get("bus", {})
    return BusConfig(ring_depth=bus_raw.get("ring_depth", 8))
```

- [ ] **Step 4: Run all config tests**

Run: `pytest tests/unit/test_config.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/core/config.py tests/unit/test_config.py
git commit -m "feat: config dataclasses, load_config, and typed accessors"
```

---

## Task 6: `core/bus.py` — init, publish, latest

**Files:**
- Populate: `tests/unit/test_bus.py` (partial — no subscribe tests yet)
- Create: `src/quadguide/core/bus.py` (partial)

- [ ] **Step 1: Write failing tests for init, publish, and latest**

Replace contents of `tests/unit/test_bus.py`:

```python
import pytest
from quadguide.core.bus import Bus, TOPICS
from quadguide.core.messages import AccelCmd, ControlCmd


@pytest.fixture
def bus():
    b = Bus(ring_depth=4)
    yield b
    b.close()


class TestTopicRegistry:
    def test_all_nine_topics_registered(self):
        b = Bus(ring_depth=2)
        try:
            expected = {
                "kcf/estimate", "nano/estimate", "target/estimate",
                "fc/attitude", "fc/imu", "guidance/accel",
                "control/cmd", "lockon/cmd", "system/health",
            }
            assert set(b._topics.keys()) == expected
        finally:
            b.close()

    def test_topics_constant_has_nine_entries(self):
        assert len(TOPICS) == 9


class TestLatest:
    def test_returns_none_when_empty(self, bus):
        assert bus.latest("guidance/accel") is None

    def test_unknown_topic_raises(self, bus):
        with pytest.raises(KeyError):
            bus.latest("nonexistent/topic")


class TestPublish:
    def test_publish_then_latest_returns_message(self, bus):
        msg = AccelCmd(timestamp_ns=1000, ax=1.0, ay=-0.5)
        bus.publish("guidance/accel", msg)
        assert bus.latest("guidance/accel") == msg

    def test_latest_returns_most_recent(self, bus):
        msg1 = AccelCmd(timestamp_ns=1000, ax=1.0, ay=0.0)
        msg2 = AccelCmd(timestamp_ns=2000, ax=2.0, ay=0.0)
        bus.publish("guidance/accel", msg1)
        bus.publish("guidance/accel", msg2)
        assert bus.latest("guidance/accel") == msg2

    def test_ring_wrap_returns_latest(self, bus):
        # ring_depth=4; publishing 5 messages wraps the ring
        msgs = [AccelCmd(timestamp_ns=i * 1000, ax=float(i), ay=0.0) for i in range(5)]
        for msg in msgs:
            bus.publish("guidance/accel", msg)
        assert bus.latest("guidance/accel") == msgs[-1]

    def test_publish_unknown_topic_raises(self, bus):
        msg = AccelCmd(timestamp_ns=0, ax=0.0, ay=0.0)
        with pytest.raises(KeyError):
            bus.publish("nonexistent/topic", msg)

    def test_different_topics_independent(self, bus):
        accel = AccelCmd(timestamp_ns=1, ax=1.0, ay=2.0)
        ctrl = ControlCmd(timestamp_ns=2, roll_deg=5.0, pitch_deg=-2.0,
                          yaw_rate_dps=0.0, throttle_norm=0.5)
        bus.publish("guidance/accel", accel)
        bus.publish("control/cmd", ctrl)
        assert bus.latest("guidance/accel") == accel
        assert bus.latest("control/cmd") == ctrl
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/unit/test_bus.py -v`
Expected: `ModuleNotFoundError: No module named 'quadguide.core.bus'`

- [ ] **Step 3: Implement `src/quadguide/core/bus.py` (init, publish, latest)**

```python
from __future__ import annotations
import fcntl
import multiprocessing
import os
import select
import struct
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

from quadguide.core.messages import (
    TrackerEstimate,  FMT_TRACKER_ESTIMATE,
    TargetEstimate,   FMT_TARGET_ESTIMATE,
    AttitudeState,    FMT_ATTITUDE_STATE,
    IMUFrame,         FMT_IMU_FRAME,
    AccelCmd,         FMT_ACCEL_CMD,
    ControlCmd,       FMT_CONTROL_CMD,
    LockOnCmd,        FMT_LOCKON_CMD,
    HealthReport,     FMT_HEALTH_REPORT,
)

__all__ = ["Bus", "TOPICS"]

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


@dataclass
class _TopicState:
    shm:       SharedMemory
    lock:      multiprocessing.Lock
    head:      multiprocessing.Value
    r_fd:      int
    w_fd:      int
    slot_size: int
    msg_class: type
    fmt:       str


class Bus:
    """Shared-memory message bus. Created once in the parent process before
    forking workers; all state is inherited across fork.

    Topics are pre-declared (see TOPICS). Any call with an unknown topic name
    raises KeyError immediately — that is a programming error.
    """

    def __init__(self, ring_depth: int = 8) -> None:
        self._ring_depth = ring_depth
        self._topics: dict[str, _TopicState] = {}
        for name, (msg_class, fmt) in TOPICS.items():
            slot_size = struct.calcsize(fmt)
            shm = SharedMemory(create=True, size=ring_depth * slot_size)
            lock = multiprocessing.Lock()
            head = multiprocessing.Value("i", -1)
            r_fd, w_fd = os.pipe()
            # r_fd is non-blocking so the drain in publish never blocks.
            fcntl.fcntl(r_fd, fcntl.F_SETFL, os.O_NONBLOCK)
            self._topics[name] = _TopicState(
                shm=shm, lock=lock, head=head,
                r_fd=r_fd, w_fd=w_fd,
                slot_size=slot_size, msg_class=msg_class, fmt=fmt,
            )

    def _get_state(self, topic: str) -> _TopicState:
        try:
            return self._topics[topic]
        except KeyError:
            raise KeyError(
                f"Unknown bus topic: {topic!r}. Valid topics: {sorted(TOPICS)}"
            )

    def publish(self, topic: str, msg) -> None:
        """Write msg to topic's ring and send a wakeup byte to its pipe.

        The drain+write pair is inside the lock so the pipe holds at most one
        byte regardless of publish rate (edge-triggered, not level-triggered).
        try/finally ensures the lock is always released even if os.write raises.
        """
        state = self._get_state(topic)
        state.lock.acquire()
        try:
            data = msg.pack()
            new_head = (state.head.value + 1) % self._ring_depth
            offset = new_head * state.slot_size
            state.shm.buf[offset : offset + state.slot_size] = data
            state.head.value = new_head
            try:
                os.read(state.r_fd, 1)
            except BlockingIOError:
                pass
            os.write(state.w_fd, b"\x00")
        finally:
            state.lock.release()

    def latest(self, topic: str):
        """Return the most recent message on topic, or None if never published."""
        state = self._get_state(topic)
        state.lock.acquire()
        try:
            h = state.head.value
            if h == -1:
                return None
            offset = h * state.slot_size
            data = bytes(state.shm.buf[offset : offset + state.slot_size])
        finally:
            state.lock.release()
        return state.msg_class.unpack(data)

    def close(self) -> None:
        """Unlink all shared memory and close all pipe fds.

        Must only be called by the parent process after all workers have been
        joined. os.close(r_fd) closes only the parent's fd copy; children retain
        theirs until they exit.
        """
        for state in self._topics.values():
            state.shm.close()
            state.shm.unlink()
            os.close(state.r_fd)
            os.close(state.w_fd)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_bus.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/core/bus.py tests/unit/test_bus.py
git commit -m "feat: Bus init, publish, and latest with shared-memory ring"
```

---

## Task 7: `core/bus.py` — subscribe_one, subscribe_any, detach

**Files:**
- Modify: `tests/unit/test_bus.py` (append subscribe + detach tests)
- Modify: `src/quadguide/core/bus.py` (append three methods)

- [ ] **Step 1: Append subscribe and detach tests to `tests/unit/test_bus.py`**

First, add `import threading` and `import time` to the **top** of the file alongside the existing imports.

Then add these classes after the existing `TestPublish` class:

```python
class TestSubscribeOne:
    def test_blocks_until_publish(self, bus):
        msg = AccelCmd(timestamp_ns=999, ax=3.0, ay=1.5)

        def publisher():
            time.sleep(0.05)
            bus.publish("guidance/accel", msg)

        t = threading.Thread(target=publisher)
        t.start()
        start = time.monotonic()
        received = bus.subscribe_one("guidance/accel")
        elapsed = time.monotonic() - start
        t.join()

        assert received == msg
        assert elapsed >= 0.04, (
            f"subscribe_one returned too quickly ({elapsed:.3f}s) — "
            "pipe blocking is broken"
        )

    def test_unknown_topic_raises(self, bus):
        with pytest.raises(KeyError):
            bus.subscribe_one("nonexistent/topic")


class TestSubscribeAny:
    def test_wakes_on_first_publish(self, bus):
        msg = ControlCmd(
            timestamp_ns=5000, roll_deg=5.0, pitch_deg=-2.0,
            yaw_rate_dps=0.0, throttle_norm=0.5,
        )

        def publisher():
            time.sleep(0.02)
            bus.publish("control/cmd", msg)

        t = threading.Thread(target=publisher)
        t.start()
        topic, received = bus.subscribe_any(["guidance/accel", "control/cmd"])
        t.join()

        assert topic == "control/cmd"
        assert received == msg

    def test_unknown_topic_in_list_raises(self, bus):
        with pytest.raises(KeyError):
            bus.subscribe_any(["guidance/accel", "nonexistent/topic"])


class TestDetach:
    def test_detach_does_not_raise(self):
        b = Bus(ring_depth=2)
        b.detach()  # must not raise; no unlink, just close local references
```

- [ ] **Step 2: Run new tests — verify they fail**

Run: `pytest tests/unit/test_bus.py::TestSubscribeOne tests/unit/test_bus.py::TestSubscribeAny tests/unit/test_bus.py::TestDetach -v`
Expected: `AttributeError: 'Bus' object has no attribute 'subscribe_one'`

- [ ] **Step 3: Append subscribe_one, subscribe_any, and detach to `src/quadguide/core/bus.py`**

Add these methods inside the `Bus` class, after `close()`:

```python
    def subscribe_one(self, topic: str):
        """Block until a new message is published on topic, then return it.

        IMPORTANT: the lock is NOT held during the blocking select.select call.
        Holding it would deadlock — publish cannot acquire the lock to complete
        its drain+write. The lock is only acquired briefly inside latest().

        Constraint: at most one process may block on a given topic at a time.
        The pipe holds one wakeup byte; two concurrent subscribers on the same
        topic means one will miss a wakeup signal.
        """
        state = self._get_state(topic)
        select.select([state.r_fd], [], [])  # blocks; O_NONBLOCK on fd doesn't affect select
        os.read(state.r_fd, 1)              # drain the wakeup byte (fd is ready, won't block)
        return self.latest(topic)

    def subscribe_any(self, topics: list[str]) -> tuple[str, object]:
        """Block until any of the listed topics receives a publish.

        Returns (topic_name, msg). Only the first ready fd is consumed per call.
        If multiple topics fire simultaneously, select.select may return multiple
        ready fds, but only ready[0] is processed. The remaining fds retain their
        wakeup bytes and will fire immediately on the next call — this is correct,
        not a missed message.
        """
        states = [self._get_state(t) for t in topics]  # KeyError if any unknown
        r_fds = [s.r_fd for s in states]
        ready, _, _ = select.select(r_fds, [], [])
        idx = r_fds.index(ready[0])
        os.read(states[idx].r_fd, 1)
        topic_name = topics[idx]
        return topic_name, self.latest(topic_name)

    def detach(self) -> None:
        """Close this process's fd references to shared memory and pipes.

        Called by worker processes in their SIGTERM handler — NOT by the parent.
        Uses shm.close() only, never shm.unlink(). Only the parent (Bus.close())
        owns the unlink lifecycle.
        """
        for state in self._topics.values():
            state.shm.close()
            os.close(state.r_fd)
            os.close(state.w_fd)
```

- [ ] **Step 4: Run all bus tests**

Run: `pytest tests/unit/test_bus.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests PASS, no warnings about unknown markers.

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/core/bus.py tests/unit/test_bus.py
git commit -m "feat: Bus subscribe_one, subscribe_any, and detach"
```

---

## Done

At this point the following are fully implemented and tested:
- `src/quadguide/core/messages.py` — all message types with struct pack/unpack
- `src/quadguide/core/config.py` — YAML loader, typed dataclasses, all accessors
- `src/quadguide/core/bus.py` — shared-memory ring bus with pipe-backed blocking

The next modules to implement are `core/clock.py`, `core/health.py`, and `core/logging.py` — each of which depends on `messages.py` and/or `bus.py`, both of which now exist.

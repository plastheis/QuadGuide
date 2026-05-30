# Tracker Worker Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse quadguide's perception layer (CCV + NCV + fusion = 3 processes) into one generic `tracker_worker` process that runs any tracker — `cv2.TrackerKCF` via a built-in adapter, or an externally-imported hybrid tracker library via a structural protocol.

**Architecture:** Single config field `tracker.import` selects the tracker at startup. A new `perception/tracker_worker.py` (~200 lines) replaces ~28 deleted files including the entire `inference/` and `perception/fusion/` packages. The bus `target/estimate` topic now carries `TrackerEstimate` directly; `TargetEstimate` and `ActiveTracker` are deleted. External tracker libraries import nothing from quadguide — the contract is structural (`.bbox.x/y/w/h`, `.confidence`, `.health` string).

**Tech Stack:** Python 3.11+, `cv2`, `numpy`, `multiprocessing` (fork), `struct`, `pytest`, pyyaml.

**Reference spec:** `docs/superpowers/specs/2026-05-29-tracker-refactor-design.md` (read this first; the plan implements that spec verbatim).

---

## Phase A — Additive prep

These tasks add new code with TDD. None of them break the existing system; each commit can land independently. Old workers (CCV/NCV/fusion) continue to run unchanged.

---

### Task 1: Add `bbox_centroid_norm` helper

**Files:**
- Modify: `src/quadguide/core/messages.py`
- Test: `tests/unit/test_messages.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_messages.py` after the existing `TestFormatSizes` class:

```python
from quadguide.core.messages import bbox_centroid_norm


class TestBBoxCentroidNorm:
    def test_centered_bbox_is_origin(self):
        cx, cy = bbox_centroid_norm(BoundingBox(0.45, 0.45, 0.1, 0.1))
        assert cx == pytest.approx(0.0)
        assert cy == pytest.approx(0.0)

    def test_top_left_bbox_is_negative(self):
        cx, cy = bbox_centroid_norm(BoundingBox(0.0, 0.0, 0.1, 0.1))
        assert cx == pytest.approx(-0.9)
        assert cy == pytest.approx(-0.9)

    def test_bottom_right_bbox_is_positive(self):
        cx, cy = bbox_centroid_norm(BoundingBox(0.9, 0.9, 0.1, 0.1))
        assert cx == pytest.approx(0.9)
        assert cy == pytest.approx(0.9)

    def test_x_only_offset(self):
        cx, cy = bbox_centroid_norm(BoundingBox(0.95, 0.45, 0.1, 0.1))
        assert cx == pytest.approx(1.0)
        assert cy == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_messages.py::TestBBoxCentroidNorm -v`
Expected: FAIL with `ImportError: cannot import name 'bbox_centroid_norm'`

- [ ] **Step 3: Implement the helper**

Add to `src/quadguide/core/messages.py` immediately after the `BoundingBox` dataclass (around line 90):

```python
def bbox_centroid_norm(bbox: BoundingBox) -> tuple[float, float]:
    """Image-centre-relative centroid in (-1, 1) range.

    cx = -1 means bbox centre is at the left edge; cx = +1 means right edge.
    Computed from bbox alone; no clock, no quadguide-internal state.
    """
    return (
        (bbox.x + bbox.w * 0.5 - 0.5) * 2.0,
        (bbox.y + bbox.h * 0.5 - 0.5) * 2.0,
    )
```

Add `bbox_centroid_norm` to the `__all__` list at the top of the file.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_messages.py::TestBBoxCentroidNorm -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/core/messages.py tests/unit/test_messages.py
git commit -m "add: bbox_centroid_norm helper"
```

---

### Task 2: Create `tracker_worker.py` skeleton with OpenCV adapter

**Files:**
- Create: `src/quadguide/perception/tracker_worker.py`
- Create: `tests/unit/test_tracker_worker_cv2_adapter.py`

- [ ] **Step 1: Write the failing tests for the cv2 adapter**

Create `tests/unit/test_tracker_worker_cv2_adapter.py`:

```python
import numpy as np
import pytest

from quadguide.core.messages import BoundingBox
from quadguide.perception.tracker_worker import OpenCVTrackerAdapter


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestOpenCVAdapterConstruction:
    def test_kcf_name(self):
        adapter = OpenCVTrackerAdapter("TrackerKCF", {})
        assert adapter.name() == "kcf"

    def test_mosse_name(self):
        adapter = OpenCVTrackerAdapter("TrackerMOSSE", {})
        assert adapter.name() == "mosse"

    def test_unknown_class_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="DoesNotExist"):
            OpenCVTrackerAdapter("DoesNotExist", {})


class TestOpenCVAdapterUpdateBeforeInit:
    def test_health_is_no_lock(self, blank_frame):
        adapter = OpenCVTrackerAdapter("TrackerKCF", {})
        out = adapter.update(blank_frame)
        assert out.health == "no_lock"

    def test_confidence_is_zero(self, blank_frame):
        adapter = OpenCVTrackerAdapter("TrackerKCF", {})
        out = adapter.update(blank_frame)
        assert out.confidence == 0.0

    def test_bbox_is_zero(self, blank_frame):
        adapter = OpenCVTrackerAdapter("TrackerKCF", {})
        out = adapter.update(blank_frame)
        assert out.bbox.x == 0.0
        assert out.bbox.y == 0.0
        assert out.bbox.w == 0.0
        assert out.bbox.h == 0.0


class TestOpenCVAdapterAfterInit:
    def test_init_then_update_returns_nominal(self, blank_frame):
        adapter = OpenCVTrackerAdapter("TrackerKCF", {})
        adapter.init(blank_frame, BoundingBox(0.2, 0.2, 0.3, 0.3))
        out = adapter.update(blank_frame)
        # On a blank frame, KCF may report ok or lost; either is acceptable
        assert out.health in ("nominal", "lost")
        assert out.confidence in (0.0, 1.0)

    def test_reset_clears_initialized(self, blank_frame):
        adapter = OpenCVTrackerAdapter("TrackerKCF", {})
        adapter.init(blank_frame, BoundingBox(0.2, 0.2, 0.3, 0.3))
        adapter.reset()
        out = adapter.update(blank_frame)
        assert out.health == "no_lock"

    def test_close_does_not_raise(self):
        OpenCVTrackerAdapter("TrackerKCF", {}).close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_tracker_worker_cv2_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quadguide.perception.tracker_worker'`

- [ ] **Step 3: Create the file with the adapter and helpers**

Create `src/quadguide/perception/tracker_worker.py`:

```python
"""Generic tracker worker — runs any tracker that satisfies the structural
protocol documented in docs/superpowers/specs/2026-05-29-tracker-refactor-design.md.

The worker treats the tracker as opaque: same calls regardless of whether the
underlying object is a cv2 tracker wrapped by OpenCVTrackerAdapter, or an
externally imported library tracker. Polymorphism happens once at construction.
"""
from __future__ import annotations
import dataclasses
import importlib
import os
import signal
from collections import namedtuple
from typing import Any

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import (
    BoundingBox, HealthReport, LockOnCmd, ProcessState,
    TrackerEstimate, TrackerHealth,
)

__all__ = [
    "OpenCVTrackerAdapter", "TrackerWorker",
    "load_tracker", "run_from_config",
]

_HEALTH_EVERY = 50

# Private structural-output types used by the cv2 adapter. Never exported.
_TrackerOutput = namedtuple("_TrackerOutput", "bbox confidence health")
_BBox          = namedtuple("_BBox",          "x y w h")


# ── OpenCV adapter ──────────────────────────────────────────────────────────

def _resolve_cv2_factory(class_name: str):
    """Locate a cv2 tracker factory by class name on cv2 or cv2.legacy."""
    import cv2
    if hasattr(cv2, class_name):
        return getattr(cv2, class_name).create
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, class_name):
        return getattr(cv2.legacy, class_name).create
    raise AttributeError(
        f"cv2 has no tracker named {class_name!r} on cv2 or cv2.legacy"
    )


class OpenCVTrackerAdapter:
    """Wraps a cv2 tracker (pixel tuples + success bool) in the structural
    output protocol the worker reads.

    `params` is currently unused; cv2 tracker constructors expect typed
    Params objects that don't map cleanly to a YAML dict. Held for a future
    cv2 params bridge.
    """

    def __init__(self, class_name: str, params: dict) -> None:
        self._factory = _resolve_cv2_factory(class_name)
        self._params = params
        self._name = class_name.lower().removeprefix("tracker")
        self._tracker = None
        self._initialized = False

    def name(self) -> str:
        return self._name

    def init(self, frame, bbox) -> None:
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
            return _TrackerOutput(_BBox(x / w, y / h, bw / w, bh / h), 1.0, "nominal")
        return _TrackerOutput(_BBox(0.0, 0.0, 0.0, 0.0), 0.0, "lost")

    def reset(self) -> None:
        self._initialized = False
        self._tracker = None

    def close(self) -> None:
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_tracker_worker_cv2_adapter.py -v`
Expected: 8 passed (or 7 passed + 1 skipped if `cv2.legacy.TrackerMOSSE` isn't installed; both acceptable)

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/perception/tracker_worker.py tests/unit/test_tracker_worker_cv2_adapter.py
git commit -m "add: tracker_worker.py with OpenCV adapter"
```

---

### Task 3: Add `load_tracker` factory

**Files:**
- Modify: `src/quadguide/perception/tracker_worker.py`
- Create: `tests/unit/test_tracker_worker_loader.py`

- [ ] **Step 1: Write the failing tests for the loader**

Create `tests/unit/test_tracker_worker_loader.py`:

```python
import pytest

from quadguide.perception.tracker_worker import (
    OpenCVTrackerAdapter, load_tracker,
)


def _cfg(import_spec: str, params: dict | None = None) -> dict:
    return {"tracker": {"import": import_spec, "params": params or {}}}


class TestLoadTrackerCV2:
    def test_kcf_returns_adapter(self):
        tracker = load_tracker(_cfg("cv2:TrackerKCF"))
        assert isinstance(tracker, OpenCVTrackerAdapter)
        assert tracker.name() == "kcf"

    def test_mosse_returns_adapter(self):
        tracker = load_tracker(_cfg("cv2:TrackerMOSSE"))
        assert isinstance(tracker, OpenCVTrackerAdapter)
        assert tracker.name() == "mosse"

    def test_unknown_cv2_class_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            load_tracker(_cfg("cv2:DoesNotExist"))


class TestLoadTrackerErrors:
    def test_missing_colon_raises_value_error(self):
        with pytest.raises(ValueError, match="module:Class"):
            load_tracker(_cfg("kcf"))

    def test_empty_module_raises_value_error(self):
        with pytest.raises(ValueError, match="module:Class"):
            load_tracker(_cfg(":Whatever"))

    def test_empty_class_raises_value_error(self):
        with pytest.raises(ValueError, match="module:Class"):
            load_tracker(_cfg("somepkg:"))

    def test_missing_module_raises_import_error(self):
        with pytest.raises(ImportError):
            load_tracker(_cfg("nonexistent_pkg_xyz:Whatever"))


class TestLoadTrackerExternal:
    def test_external_class_constructed_with_params(self, tmp_path, monkeypatch):
        # Build a tiny synthetic external tracker module on disk.
        pkg = tmp_path / "stub_tracker_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "tracker.py").write_text(
            "class StubTracker:\n"
            "    def __init__(self, **kw): self.kw = kw\n"
            "    def name(self): return 'stub'\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        cfg = _cfg("stub_tracker_pkg.tracker:StubTracker", {"foo": 1, "bar": "x"})
        tracker = load_tracker(cfg)
        assert tracker.name() == "stub"
        assert tracker.kw == {"foo": 1, "bar": "x"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_tracker_worker_loader.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_tracker'`

- [ ] **Step 3: Implement `load_tracker`**

Append to `src/quadguide/perception/tracker_worker.py` after the `OpenCVTrackerAdapter` class:

```python
# ── Loader ──────────────────────────────────────────────────────────────────

def load_tracker(config: dict):
    """Construct the tracker selected by config['tracker']['import'].

    The cv2: prefix wraps with OpenCVTrackerAdapter. Anything else is imported
    and constructed directly as cls(**params).
    """
    tcfg = config["tracker"]
    spec = tcfg["import"]
    params = tcfg.get("params") or {}

    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise ValueError(
            f"tracker.import must be 'module:Class', got {spec!r}"
        )

    if module_name == "cv2":
        return OpenCVTrackerAdapter(class_name, params)

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**params)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_tracker_worker_loader.py -v`
Expected: 8 passed (or 7 + 1 skipped if MOSSE missing)

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/perception/tracker_worker.py tests/unit/test_tracker_worker_loader.py
git commit -m "add: load_tracker factory for tracker.import"
```

---

### Task 4: Add `TrackerWorker` class

**Files:**
- Modify: `src/quadguide/perception/tracker_worker.py`
- Create: `tests/unit/test_tracker_worker_loop.py`

- [ ] **Step 1: Write the failing tests for the worker loop**

Create `tests/unit/test_tracker_worker_loop.py`. The tests use stub objects for bus, frame_buffer, and tracker — no real IPC or cv2.

```python
import time
import numpy as np
import pytest

from quadguide.core.messages import (
    BoundingBox, LockOnCmd, TrackerEstimate, TrackerHealth,
)
from quadguide.perception.tracker_worker import TrackerWorker


class _StubBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []
        self.latest_map: dict[str, object] = {}

    def publish(self, topic, msg):
        self.published.append((topic, msg))

    def latest(self, topic):
        return self.latest_map.get(topic)

    def detach(self):
        pass


class _StubFrameBuffer:
    def __init__(self, frame=None, ts: int = 0) -> None:
        self.frame = frame
        self.ts = ts

    def read_latest(self):
        return self.frame, self.ts


class _StubBBox:
    def __init__(self, x, y, w, h):
        self.x = x; self.y = y; self.w = w; self.h = h


class _StubTrackerOutput:
    def __init__(self, x, y, w, h, confidence, health):
        self.bbox = _StubBBox(x, y, w, h)
        self.confidence = confidence
        self.health = health


class _StubTracker:
    """Records calls; returns a fixed output."""
    def __init__(self, output: _StubTrackerOutput | None = None) -> None:
        self._output = output or _StubTrackerOutput(0.1, 0.2, 0.3, 0.4, 0.9, "nominal")
        self.init_calls: list[tuple[np.ndarray, object]] = []
        self.update_calls: int = 0
        self.reset_calls: int = 0
        self.close_calls: int = 0

    def name(self): return "stub"
    def init(self, frame, bbox): self.init_calls.append((frame, bbox))
    def update(self, frame):
        self.update_calls += 1
        return self._output
    def reset(self): self.reset_calls += 1
    def close(self): self.close_calls += 1


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _run_one_iteration(worker: TrackerWorker) -> None:
    """Run a single loop iteration by setting _stop after the iteration body."""
    # Patch the stop flag so the loop exits cleanly after first iteration.
    original_publish = worker._bus.publish
    def _publish_then_stop(topic, msg):
        original_publish(topic, msg)
        worker._stop = True
    worker._bus.publish = _publish_then_stop
    worker.run()


class TestTrackerWorkerLockonFlow:
    def test_new_seq_triggers_init(self):
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=_frame(), ts=time.monotonic_ns())
        tracker = _StubTracker()
        bus.latest_map["lockon/cmd"] = LockOnCmd(
            timestamp_ns=time.monotonic_ns(), seq=7,
            bbox=BoundingBox(0.1, 0.1, 0.2, 0.2),
        )
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        worker._stop = True   # ensure single iteration
        worker._check_lockon()
        assert len(tracker.init_calls) == 1
        assert worker._last_seq == 7

    def test_same_seq_does_not_reinit(self):
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=_frame(), ts=time.monotonic_ns())
        tracker = _StubTracker()
        cmd = LockOnCmd(time.monotonic_ns(), 3, BoundingBox(0.1, 0.1, 0.2, 0.2))
        bus.latest_map["lockon/cmd"] = cmd
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        worker._check_lockon()
        worker._check_lockon()  # same seq
        assert len(tracker.init_calls) == 1

    def test_zero_size_bbox_triggers_reset(self):
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=_frame(), ts=time.monotonic_ns())
        tracker = _StubTracker()
        bus.latest_map["lockon/cmd"] = LockOnCmd(
            timestamp_ns=time.monotonic_ns(), seq=1,
            bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
        )
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        worker._check_lockon()
        assert tracker.reset_calls == 1
        assert len(tracker.init_calls) == 0


class TestTrackerWorkerPublish:
    def test_publish_translates_output_to_tracker_estimate(self):
        bus = _StubBus()
        ts = time.monotonic_ns() - 1_000_000   # 1 ms ago
        fb = _StubFrameBuffer(frame=_frame(), ts=ts)
        tracker = _StubTracker(_StubTrackerOutput(0.1, 0.2, 0.3, 0.4, 0.75, "nominal"))
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        _run_one_iteration(worker)
        published = [(t, m) for (t, m) in bus.published if t == "target/estimate"]
        assert len(published) >= 1
        _, est = published[0]
        assert isinstance(est, TrackerEstimate)
        assert est.bbox.x == pytest.approx(0.1)
        assert est.bbox.y == pytest.approx(0.2)
        assert est.bbox.w == pytest.approx(0.3)
        assert est.bbox.h == pytest.approx(0.4)
        assert est.confidence == pytest.approx(0.75)
        assert est.tracker_health == TrackerHealth.NOMINAL
        assert est.latency_ns > 0

    def test_no_frame_skips_publish(self):
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=None, ts=0)
        tracker = _StubTracker()
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        worker._stop = True
        worker._check_lockon()  # also exercises lockon path with no command
        # The frame read happens inside run(); simulate one tick of run-body:
        frame, frame_ts = fb.read_latest()
        assert frame is None
        # No publish call should have been made
        published = [(t, m) for (t, m) in bus.published if t == "target/estimate"]
        assert len(published) == 0


class TestTrackerWorkerLifecycle:
    def test_close_called_after_loop_exits(self):
        bus = _StubBus()
        fb = _StubFrameBuffer(frame=_frame(), ts=time.monotonic_ns())
        tracker = _StubTracker()
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        _run_one_iteration(worker)
        assert tracker.close_calls == 1

    def test_proc_name_uses_tracker_name(self):
        bus = _StubBus()
        fb = _StubFrameBuffer()
        tracker = _StubTracker()
        worker = TrackerWorker(tracker, bus, fb, cpu_core=None, config={})
        assert worker._proc_name == "tracker_stub"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_tracker_worker_loop.py -v`
Expected: FAIL with `ImportError: cannot import name 'TrackerWorker'`

- [ ] **Step 3: Implement `TrackerWorker`**

Append to `src/quadguide/perception/tracker_worker.py` after the `load_tracker` function:

```python
# ── Worker ──────────────────────────────────────────────────────────────────

class TrackerWorker:
    """IPC loop owning lockon/cmd subscription, SHM frame read, and
    target/estimate publish. Tracker is opaque — no per-implementation branching.
    """

    def __init__(
        self,
        tracker,
        bus: Bus,
        frame_buffer: FrameBuffer,
        cpu_core: int | None = None,
        config: dict | None = None,
    ) -> None:
        self._tracker   = tracker
        self._bus       = bus
        self._fb        = frame_buffer
        self._cpu_core  = cpu_core
        self._config    = config or {}
        self._last_seq: int | None = None
        self._stop      = False
        self._proc_name = f"tracker_{tracker.name()}"

    def run(self) -> None:
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

        self._tracker.close()
        self._bus.detach()
        log.info(f"{self._proc_name}: stopped")

    def _check_lockon(self) -> None:
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

    def _handle_sigterm(self, sig, frame) -> None:
        self._stop = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_tracker_worker_loop.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/perception/tracker_worker.py tests/unit/test_tracker_worker_loop.py
git commit -m "add: TrackerWorker IPC loop"
```

---

### Task 5: Add `run_from_config` entry point

**Files:**
- Modify: `src/quadguide/perception/tracker_worker.py`

This task adds the thin wrapper `run.py` will eventually call. There's no new unit test (it's a four-line glue function); coverage comes from the existing tests in Tasks 2–4 and the integration test rewritten in Task 14.

- [ ] **Step 1: Add `run_from_config` to tracker_worker.py**

Append to `src/quadguide/perception/tracker_worker.py` at the bottom:

```python
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

Note: this references `pcfg.realtime.tracker_cpu_core`, which doesn't exist yet — it's added in Task 8. Import will work because attribute access is lazy at call time. Existing tests must still pass.

- [ ] **Step 2: Verify all Phase A tests still pass**

Run: `python -m pytest tests/unit/test_tracker_worker_cv2_adapter.py tests/unit/test_tracker_worker_loader.py tests/unit/test_tracker_worker_loop.py tests/unit/test_messages.py -v`
Expected: all green.

- [ ] **Step 3: Verify the broader unit suite isn't broken**

Run: `python -m pytest tests/unit -v`
Expected: all existing tests pass (Phase A is purely additive).

- [ ] **Step 4: Commit**

```bash
git add src/quadguide/perception/tracker_worker.py
git commit -m "add: run_from_config entry point for tracker worker"
```

---

## Phase B — Hard cutover

These tasks land sequentially; intermediate states may have broken end-to-end behavior. Each task leaves the **build importable** and **unrelated tests passing**. The HIL smoke test at the end (Task 17) is the operational gate.

---

### Task 6: Update guidance to consume `TrackerEstimate`

**Files:**
- Modify: `src/quadguide/guidance/base.py`
- Modify: `src/quadguide/guidance/pronav.py`
- Modify: `src/quadguide/guidance/pure_pursuit.py`
- Modify: `tests/unit/test_pure_pursuit.py`

- [ ] **Step 1: Update `guidance/base.py`**

Replace the imports and the protocol annotation in `src/quadguide/guidance/base.py`:

```python
from __future__ import annotations
from typing import Protocol

from quadguide.core.messages import IMUFrame, LockOnCmd, TrackerEstimate


class GuidanceMethod(Protocol):
    """Strategy interface for guidance algorithms.

    Each method consumes the same inputs but uses what it needs. Returns
    body-frame lateral/longitudinal acceleration commands in m/s² that the
    control worker maps to roll/pitch via the small-angle tilt approximation.
    """

    def compute(
        self,
        est: TrackerEstimate,
        imu: IMUFrame,
        lockon_cmd: LockOnCmd | None,
        now_ns: int,
    ) -> tuple[float, float]: ...

    def name(self) -> str: ...
```

- [ ] **Step 2: Update `guidance/pronav.py`**

In `src/quadguide/guidance/pronav.py`:

- Change the import line to import `TrackerEstimate` and `bbox_centroid_norm`:

  ```python
  from quadguide.core.messages import IMUFrame, LockOnCmd, TrackerEstimate, bbox_centroid_norm
  ```

- Change the `compute` signature `est: TargetEstimate` → `est: TrackerEstimate`.
- Replace the body of `compute` with:

  ```python
  def compute(
      self,
      est: TrackerEstimate,
      imu: IMUFrame,
      lockon_cmd: LockOnCmd | None,
      now_ns: int,
  ) -> tuple[float, float]:
      centroid = bbox_centroid_norm(est.bbox)
      los_r = self._los.update(centroid, imu, lockon_cmd, now_ns)
      v_c = self._cv.update(est.bbox, now_ns, self._cfg)
      return pronav(los_r, v_c, self._cfg.N)
  ```

- [ ] **Step 3: Update `guidance/pure_pursuit.py`**

Apply the same pattern in `src/quadguide/guidance/pure_pursuit.py`:

```python
from __future__ import annotations

from quadguide.core.config import PurePursuitConfig
from quadguide.core.messages import (
    IMUFrame, LockOnCmd, TrackerEstimate, bbox_centroid_norm,
)


class PurePursuitGuidance:
    """Pure pursuit: command acceleration straight toward the target LOS.

        a = K * LOS_angle

    The centroid is computed from bbox via bbox_centroid_norm, converted to a
    physical LOS angle (radians) via the camera FoV, then scaled by K (m/s²
    per radian). No LOS rate, no closing velocity, no body-rate derotation —
    the simplest possible homing law. Maps cleanly into the control
    attitude_cmd downstream via a ≈ g·θ.
    """

    def __init__(self, cfg: PurePursuitConfig, fov_horizontal_rad: float, aspect: float) -> None:
        self._K = cfg.K
        self._scale_x = fov_horizontal_rad * 0.5
        self._scale_y = (fov_horizontal_rad / aspect) * 0.5

    def name(self) -> str:
        return "pure_pursuit"

    def compute(
        self,
        est: TrackerEstimate,
        imu: IMUFrame,
        lockon_cmd: LockOnCmd | None,
        now_ns: int,
    ) -> tuple[float, float]:
        cx, cy = bbox_centroid_norm(est.bbox)
        return self._K * cx * self._scale_x, self._K * cy * self._scale_y
```

- [ ] **Step 4: Update `tests/unit/test_pure_pursuit.py`**

Replace the test file content with:

```python
import math
import time

import pytest

from quadguide.core.config import PurePursuitConfig
from quadguide.core.messages import (
    BoundingBox, IMUFrame, TrackerEstimate, TrackerHealth,
)
from quadguide.guidance.pure_pursuit import PurePursuitGuidance


FOV_H = 1.047  # ~60° horizontal
ASPECT = 640 / 480


def _est_for_centroid(cx: float, cy: float) -> TrackerEstimate:
    """Build an estimate whose computed centroid equals (cx, cy)."""
    w = h = 0.1
    bbox_x = (cx / 2.0 + 0.5) - w * 0.5
    bbox_y = (cy / 2.0 + 0.5) - h * 0.5
    return TrackerEstimate(
        timestamp_ns=time.monotonic_ns(),
        bbox=BoundingBox(bbox_x, bbox_y, w, h),
        confidence=1.0,
        tracker_health=TrackerHealth.NOMINAL,
    )


def _imu() -> IMUFrame:
    return IMUFrame(time.monotonic_ns(), 0, 0, 0, 0, 0, 0)


def test_zero_centroid_gives_zero_accel():
    pp = PurePursuitGuidance(PurePursuitConfig(K=6.0), FOV_H, ASPECT)
    ax, ay = pp.compute(_est_for_centroid(0.0, 0.0), _imu(), None, time.monotonic_ns())
    assert ax == pytest.approx(0.0)
    assert ay == pytest.approx(0.0)


def test_positive_cx_drives_positive_ax():
    pp = PurePursuitGuidance(PurePursuitConfig(K=6.0), FOV_H, ASPECT)
    ax, ay = pp.compute(_est_for_centroid(1.0, 0.0), _imu(), None, time.monotonic_ns())
    assert ax == pytest.approx(6.0 * FOV_H * 0.5)
    assert ay == pytest.approx(0.0)


def test_scales_with_K():
    pp1 = PurePursuitGuidance(PurePursuitConfig(K=1.0), FOV_H, ASPECT)
    pp2 = PurePursuitGuidance(PurePursuitConfig(K=10.0), FOV_H, ASPECT)
    a1 = pp1.compute(_est_for_centroid(0.5, 0.0), _imu(), None, time.monotonic_ns())
    a2 = pp2.compute(_est_for_centroid(0.5, 0.0), _imu(), None, time.monotonic_ns())
    assert math.isclose(a2[0], a1[0] * 10.0)


def test_vertical_scale_uses_vertical_fov():
    pp = PurePursuitGuidance(PurePursuitConfig(K=6.0), FOV_H, ASPECT)
    _, ay = pp.compute(_est_for_centroid(0.0, 1.0), _imu(), None, time.monotonic_ns())
    fov_v = FOV_H / ASPECT
    assert ay == pytest.approx(6.0 * fov_v * 0.5)
```

- [ ] **Step 5: Run guidance unit tests**

Run: `python -m pytest tests/unit/test_pronav.py tests/unit/test_pure_pursuit.py tests/unit/test_los.py -v`
Expected: all pass. `test_pronav.py` is untouched and exercises only the pure `pronav()` function which is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/guidance/base.py src/quadguide/guidance/pronav.py src/quadguide/guidance/pure_pursuit.py tests/unit/test_pure_pursuit.py
git commit -m "update: guidance reads TrackerEstimate, derives centroid from bbox"
```

---

### Task 7: Update ground module

**Files:**
- Modify: `src/quadguide/ground/overlay.py`
- Modify: `src/quadguide/ground/server.py`
- Modify: `tests/unit/test_ground_overlay.py`
- Modify: `tests/unit/test_ground_server.py`

- [ ] **Step 1: Update `ground/overlay.py`**

Replace the import line and the parameter annotation:

```python
from quadguide.core.messages import TrackerEstimate, TrackerHealth
```

And:

```python
def draw_overlay(frame: np.ndarray, estimate: TrackerEstimate | None) -> bytes:
    ...
```

No body changes — `estimate.bbox` and `estimate.tracker_health` exist on `TrackerEstimate` unchanged.

- [ ] **Step 2: Update `ground/server.py`**

In `src/quadguide/ground/server.py`:

1. Delete the two bus reads (around lines 141–142):

   ```python
   ccv      = app.state.bus.latest("ccv_tracker/estimate")
   ncv      = app.state.bus.latest("ncv_tracker/estimate")
   ```

2. Replace the two `ccv_algo` / `ncv_algo` derivation lines (around 150–155) with:

   ```python
   tracker_algo = next(
       (k[8:] for k in app.state.process_health if k.startswith("tracker_")), None
   )
   ```

3. Replace the SSE `data` dict (around lines 158–206) — delete `active_tracker`, `centroid_x`, `centroid_y`, `ccv_algo`, `ccv_health`, `ccv_conf`, `ncv_algo`, `ncv_health`, `ncv_conf`; add `tracker_algo`. The final dict is:

   ```python
   data = {
       # target/estimate
       "tracker_health": estimate.tracker_health.value if estimate else None,
       "confidence":     estimate.confidence           if estimate else None,
       "bbox_x":         estimate.bbox.x               if estimate else None,
       "bbox_y":         estimate.bbox.y               if estimate else None,
       "bbox_w":         estimate.bbox.w               if estimate else None,
       "bbox_h":         estimate.bbox.h               if estimate else None,
       # tracker process name (derived from system/health)
       "tracker_algo":   tracker_algo,
       # fc/attitude
       "roll_deg":       math.degrees(attitude.roll_rad)       if attitude else None,
       "pitch_deg":      math.degrees(attitude.pitch_rad)      if attitude else None,
       "yaw_deg":        math.degrees(attitude.yaw_rad)        if attitude else None,
       "roll_rate_dps":  math.degrees(attitude.roll_rate_rps)  if attitude else None,
       "pitch_rate_dps": math.degrees(attitude.pitch_rate_rps) if attitude else None,
       "yaw_rate_dps":   math.degrees(attitude.yaw_rate_rps)   if attitude else None,
       # fc/imu
       "imu_ax": imu.ax if imu else None,
       "imu_ay": imu.ay if imu else None,
       "imu_az": imu.az if imu else None,
       "imu_gx": imu.gx if imu else None,
       "imu_gy": imu.gy if imu else None,
       "imu_gz": imu.gz if imu else None,
       # guidance/accel
       "accel_ax": accel.ax if accel else None,
       "accel_ay": accel.ay if accel else None,
       # control/cmd
       "ctrl_roll_deg":     control.roll_deg     if control else None,
       "ctrl_pitch_deg":    control.pitch_deg    if control else None,
       "ctrl_yaw_rate_dps": control.yaw_rate_dps if control else None,
       "ctrl_throttle":     control.throttle_norm if control else None,
       # system/health
       "health": dict(app.state.process_health),
       # latency
       "latency_ms":     lat_ms,
       "latency_avg_ms": avg_ms,
       # video stream fps
       "video_fps": app.state.mjpeg_fps if app.state.mjpeg_fps > 0 else None,
   }
   ```

- [ ] **Step 3: Update `tests/unit/test_ground_overlay.py`**

Open `tests/unit/test_ground_overlay.py`. Replace all references to `TargetEstimate` with `TrackerEstimate` and remove the `centroid_norm` and `active_tracker` arguments from any `TargetEstimate(...)` construction calls — they don't exist on `TrackerEstimate`. The overlay only reads `bbox` and `tracker_health`, so the tests should be straightforward to adapt.

Concretely: any `TargetEstimate(timestamp_ns=..., bbox=..., centroid_norm=..., confidence=..., tracker_health=..., active_tracker=...)` becomes `TrackerEstimate(timestamp_ns=..., bbox=..., confidence=..., tracker_health=...)`.

- [ ] **Step 4: Update `tests/unit/test_ground_server.py`**

Open `tests/unit/test_ground_server.py`. Same `TargetEstimate` → `TrackerEstimate` renames; remove any test assertions on the deleted SSE fields (`active_tracker`, `centroid_x`, `centroid_y`, `ccv_*`, `ncv_*`); add assertions on `tracker_algo` where appropriate. Any test that publishes to `ccv_tracker/estimate` or `ncv_tracker/estimate` is now invalid — delete those tests; the integration-test rewrite in Task 14 covers the equivalent flow.

- [ ] **Step 5: (Optional) Update `src/quadguide/ground/static/index.html`**

The HTML HUD currently consumes the deleted SSE fields. Until updated, those widgets render `undefined` — not a blocker for merge per the spec. If you want the HUD clean in the same commit:

- Wherever the HUD reads `data.centroid_x` / `data.centroid_y`, replace with inline JS that computes from `data.bbox_x + data.bbox_w * 0.5 - 0.5` (multiply by 2 for the (-1, 1) range).
- Wherever it shows `data.ccv_algo` / `data.ncv_algo`, replace with `data.tracker_algo`.

Skip this step if HUD-cosmetic work is being tracked separately.

- [ ] **Step 6: Run ground tests**

Run: `python -m pytest tests/unit/test_ground_overlay.py tests/unit/test_ground_server.py -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/quadguide/ground/overlay.py src/quadguide/ground/server.py tests/unit/test_ground_overlay.py tests/unit/test_ground_server.py
# also stage src/quadguide/ground/static/index.html if step 5 was done
git commit -m "update: ground reads TrackerEstimate, single tracker_algo field"
```

---

### Task 8: Replace `TrackerConfig`, rename `kcf_cpu_core`

**Files:**
- Modify: `src/quadguide/core/config.py`
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Update `core/config.py`**

Make the following edits to `src/quadguide/core/config.py`:

1. Delete the following dataclasses: `InferenceConfig`, `FusionConfig`, `KCFConfig`, `NanotrackConfig`, `MOSSEConfig`.

2. Change `PlatformConfig` — delete the `inference: InferenceConfig` field:

   ```python
   @dataclass(frozen=True)
   class PlatformConfig:
       name: str
       camera: CameraConfig
       serial: SerialConfig
       realtime: RealtimeConfig
   ```

3. Change `RealtimeConfig`:

   ```python
   @dataclass(frozen=True)
   class RealtimeConfig:
       tracker_cpu_core: int | None
       control_cpu_core: int
       control_sched_fifo: bool
       control_fifo_prio: int
   ```

4. Replace `TrackerConfig` with:

   ```python
   from typing import Any

   @dataclass(frozen=True)
   class TrackerConfig:
       import_spec: str
       params: dict[str, Any] = field(default_factory=dict)
   ```

5. Update `cfg_platform` — remove the `inference=InferenceConfig(...)` construction and read `realtime.tracker_cpu_core` (tolerating absence):

   ```python
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
           realtime=RealtimeConfig(
               tracker_cpu_core=p["realtime"].get("tracker_cpu_core"),
               control_cpu_core=p["realtime"]["control_cpu_core"],
               control_sched_fifo=p["realtime"]["control_sched_fifo"],
               control_fifo_prio=p["realtime"]["control_fifo_prio"],
           ),
       )
   ```

6. Replace `cfg_tracker`:

   ```python
   def cfg_tracker(d: dict) -> TrackerConfig:
       t = d["tracker"]
       return TrackerConfig(
           import_spec=t["import"],
           params=dict(t.get("params") or {}),
       )
   ```

- [ ] **Step 2: Update `tests/unit/test_config.py`**

Open `tests/unit/test_config.py`. For every test that constructs `KCFConfig`, `NanotrackConfig`, `MOSSEConfig`, `FusionConfig`, or `InferenceConfig`, or any test that loads a yaml fixture with the old `tracker.{ccv, ncv, kcf, nanotrack, mosse, fusion}` keys: delete or rewrite the test against the new schema.

For tests that touch `cfg_tracker`, replace any old-shape config dict with the new shape:

```python
{
    "tracker": {
        "import": "cv2:TrackerKCF",
        "params": {},
    },
}
```

Add a new test class verifying the new `TrackerConfig`:

```python
class TestTrackerConfigNew:
    def test_cfg_tracker_reads_import(self):
        cfg = {
            "tracker": {"import": "cv2:TrackerKCF", "params": {"foo": 1}},
        }
        tcfg = cfg_tracker(cfg)
        assert tcfg.import_spec == "cv2:TrackerKCF"
        assert tcfg.params == {"foo": 1}

    def test_cfg_tracker_missing_import_raises(self):
        with pytest.raises(KeyError):
            cfg_tracker({"tracker": {}})

    def test_cfg_tracker_params_defaults_to_empty(self):
        cfg = {"tracker": {"import": "cv2:TrackerKCF"}}
        assert cfg_tracker(cfg).params == {}
```

For `RealtimeConfig` rename: any test asserting `pcfg.realtime.kcf_cpu_core` becomes `pcfg.realtime.tracker_cpu_core`. If the test doesn't supply that key, the new default is `None`.

- [ ] **Step 3: Run config tests**

Run: `python -m pytest tests/unit/test_config.py -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/quadguide/core/config.py tests/unit/test_config.py
git commit -m "update: replace TrackerConfig with import/params; rename realtime.tracker_cpu_core"
```

---

### Task 9: Update `configs/config.yaml`

**Files:**
- Modify: `configs/config.yaml`

- [ ] **Step 1: Edit `configs/config.yaml`**

In `configs/config.yaml`:

1. Delete the `platform.inference:` subsection (the three lines for `device`, `backbone`, `head`).

2. Rename `platform.realtime.kcf_cpu_core` to `platform.realtime.tracker_cpu_core`.

3. Replace the entire `tracker:` block with:

   ```yaml
   tracker:
     import: cv2:TrackerKCF       # or e.g. "myhybrid.tracker:HybridTracker"
     params: {}                   # passed to tracker constructor as **kwargs
   ```

The resulting `tracker:` block has exactly two keys.

- [ ] **Step 2: Verify the config loads**

Run from a Python REPL or quick script:

```bash
python -c "from quadguide.core.config import load_config, cfg_tracker, cfg_platform; c = load_config('configs/config.yaml', {}); print(cfg_tracker(c)); print(cfg_platform(c).realtime)"
```

Expected: prints the `TrackerConfig(import_spec='cv2:TrackerKCF', params={})` line and a `RealtimeConfig` with `tracker_cpu_core=1`.

- [ ] **Step 3: Commit**

```bash
git add configs/config.yaml
git commit -m "update: configs/config.yaml uses tracker.import schema"
```

---

### Task 10: Update `scripts/run.py`

**Files:**
- Modify: `scripts/run.py`

- [ ] **Step 1: Rewrite `_start_workers` and delete `_ncv_run`**

In `scripts/run.py`:

1. Delete the entire `_ncv_run` function (lines 19–26).

2. Replace `_start_workers` with the single-tracker version:

   ```python
   def _start_workers(config: dict, bus, frame_buffer, *, ground: bool = True) -> list[multiprocessing.Process]:
       from quadguide.perception.camera.worker import run_from_config as camera_run
       from quadguide.perception.tracker_worker import run_from_config as tracker_run
       from quadguide.link.worker import run as link_run
       from quadguide.guidance.worker import run as guidance_run
       from quadguide.control.worker import run as control_run
       from quadguide.ground.worker import run as ground_run

       entries: list[tuple[str, object, tuple]] = [
           ("camera",   camera_run,   (config, bus, frame_buffer)),
           ("tracker",  tracker_run,  (config, bus, frame_buffer)),
           ("link",     link_run,     (config, bus)),
           ("guidance", guidance_run, (config, bus, frame_buffer)),
           ("control",  control_run,  (config, bus, frame_buffer)),
       ]
       if ground:
           entries.append(("ground", ground_run, (config, bus, frame_buffer)))

       procs = []
       for name, target, args in entries:
           p = multiprocessing.Process(target=target, args=args, name=name, daemon=False)
           p.start()
           procs.append(p)
       return procs
   ```

- [ ] **Step 2: Verify import is clean**

Run: `python -c "import scripts.run"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add scripts/run.py
git commit -m "update: run.py spawns one tracker worker, drops CCV/NCV/fusion path"
```

---

### Task 11: Update `core/bus.py` topic table

**Files:**
- Modify: `src/quadguide/core/bus.py`

- [ ] **Step 1: Edit the imports and `TOPICS` dict**

In `src/quadguide/core/bus.py`:

1. Remove `TargetEstimate` and `FMT_TARGET_ESTIMATE` from the import block at the top.

2. Update the `TOPICS` dict — delete the two `*_tracker/estimate` entries and switch `target/estimate` to `TrackerEstimate`:

   ```python
   TOPICS: dict[str, tuple[type, str]] = {
       "target/estimate":      (TrackerEstimate, FMT_TRACKER_ESTIMATE),
       "fc/attitude":          (AttitudeState,   FMT_ATTITUDE_STATE),
       "fc/imu":               (IMUFrame,        FMT_IMU_FRAME),
       "guidance/accel":       (AccelCmd,        FMT_ACCEL_CMD),
       "control/cmd":          (ControlCmd,      FMT_CONTROL_CMD),
       "lockon/cmd":           (LockOnCmd,       FMT_LOCKON_CMD),
       "system/health":        (HealthReport,    FMT_HEALTH_REPORT),
       "arm/cmd":              (ArmCmd,          FMT_ARM_CMD),
   }
   ```

- [ ] **Step 2: Run bus tests**

Run: `python -m pytest tests/unit/test_bus.py -v`
Expected: tests referencing `ccv_tracker/estimate` or `ncv_tracker/estimate` will fail; that's fixed in Task 12. For now, all other bus tests should pass.

If `test_bus.py` cannot import for some other reason (e.g. import of deleted name), make the minimal fix to get it importable — the substantive cleanup happens in Task 12.

- [ ] **Step 3: Commit**

```bash
git add src/quadguide/core/bus.py
git commit -m "update: bus TOPICS drops ccv/ncv tracker, target/estimate carries TrackerEstimate"
```

---

### Task 12: Delete `TargetEstimate` and `ActiveTracker`

**Files:**
- Modify: `src/quadguide/core/messages.py`
- Modify: `tests/unit/test_messages.py`
- Modify: `tests/unit/test_bus.py`

- [ ] **Step 1: Edit `core/messages.py`**

In `src/quadguide/core/messages.py`:

1. Remove `TargetEstimate`, `ActiveTracker`, `FMT_TARGET_ESTIMATE` from the `__all__` list.

2. Delete the `ActiveTracker` enum (the `@_byte_enum class ActiveTracker(str, Enum): ...` block).

3. Delete the `FMT_TARGET_ESTIMATE` format string line and its trailing arithmetic comment.

4. Delete the `_ST_TARGET_ESTIMATE = struct.Struct(FMT_TARGET_ESTIMATE)` line.

5. Delete the `TargetEstimate` dataclass (its `pack` and `unpack` methods come with it).

- [ ] **Step 2: Edit `tests/unit/test_messages.py`**

Remove from the imports: `TargetEstimate`, `ActiveTracker`, `FMT_TARGET_ESTIMATE`. Delete any test method that constructs or round-trips `TargetEstimate` or `ActiveTracker`. Specifically:

- Remove `test_active_tracker_round_trip`, `test_active_tracker_is_str` from `TestEnumOrdinals`.
- Remove `test_target_estimate` from `TestFormatSizes`.
- Remove any other test in the file that names these symbols.

Run the file and remove anything red:

```bash
python -m pytest tests/unit/test_messages.py -v
```

- [ ] **Step 3: Edit `tests/unit/test_bus.py`**

Open `tests/unit/test_bus.py` and remove anything referencing `ccv_tracker/estimate`, `ncv_tracker/estimate`, `TargetEstimate`, or `ActiveTracker`. Tests that exercised `target/estimate` should now use `TrackerEstimate`:

```python
from quadguide.core.messages import (
    BoundingBox, TrackerEstimate, TrackerHealth,
)

# Replace any:
#   bus.publish("target/estimate", TargetEstimate(..., centroid_norm=..., active_tracker=...))
# with:
#   bus.publish("target/estimate", TrackerEstimate(
#       timestamp_ns=..., bbox=BoundingBox(...), confidence=..., tracker_health=TrackerHealth.NOMINAL,
#   ))
```

- [ ] **Step 4: Run all tests under tests/unit**

Run: `python -m pytest tests/unit -v --ignore=tests/unit/test_kcf_tracker.py --ignore=tests/unit/test_mosse_tracker.py --ignore=tests/unit/test_nanotrack_tracker.py --ignore=tests/unit/test_nanotrack_preprocess.py --ignore=tests/unit/test_nanotrack_postprocess.py --ignore=tests/unit/test_fusion.py --ignore=tests/unit/test_inference.py`

Expected: all green. (The ignored tests cover the modules deleted in Task 13.)

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/core/messages.py tests/unit/test_messages.py tests/unit/test_bus.py
git commit -m "delete: TargetEstimate, ActiveTracker, FMT_TARGET_ESTIMATE"
```

---

### Task 13: Delete old packages and their tests

**Files:**
- Delete: `src/quadguide/inference/` (whole package)
- Delete: `src/quadguide/perception/fusion/` (whole package)
- Delete: `src/quadguide/perception/nanotrack/` (whole package)
- Delete: `src/quadguide/perception/kcf/` (whole package)
- Delete: `src/quadguide/perception/mosse/` (whole package)
- Delete: `src/quadguide/perception/ccv_tracker_worker.py`
- Delete: `src/quadguide/perception/ncv_tracker_worker.py`
- Delete: `src/quadguide/perception/tracker_factories.py`
- Delete: `tests/unit/test_kcf_tracker.py`
- Delete: `tests/unit/test_mosse_tracker.py`
- Delete: `tests/unit/test_nanotrack_tracker.py`
- Delete: `tests/unit/test_nanotrack_preprocess.py`
- Delete: `tests/unit/test_nanotrack_postprocess.py`
- Delete: `tests/unit/test_fusion.py`
- Delete: `tests/unit/test_inference.py`

- [ ] **Step 1: Remove source packages**

Run from the repo root (use whichever shell syntax fits — both PowerShell and bash listed):

PowerShell:
```powershell
Remove-Item -Recurse -Force src/quadguide/inference
Remove-Item -Recurse -Force src/quadguide/perception/fusion
Remove-Item -Recurse -Force src/quadguide/perception/nanotrack
Remove-Item -Recurse -Force src/quadguide/perception/kcf
Remove-Item -Recurse -Force src/quadguide/perception/mosse
Remove-Item src/quadguide/perception/ccv_tracker_worker.py
Remove-Item src/quadguide/perception/ncv_tracker_worker.py
Remove-Item src/quadguide/perception/tracker_factories.py
```

Bash:
```bash
rm -r src/quadguide/inference src/quadguide/perception/fusion \
      src/quadguide/perception/nanotrack src/quadguide/perception/kcf \
      src/quadguide/perception/mosse
rm src/quadguide/perception/ccv_tracker_worker.py \
   src/quadguide/perception/ncv_tracker_worker.py \
   src/quadguide/perception/tracker_factories.py
```

- [ ] **Step 2: Remove the corresponding unit test files**

PowerShell:
```powershell
Remove-Item tests/unit/test_kcf_tracker.py
Remove-Item tests/unit/test_mosse_tracker.py
Remove-Item tests/unit/test_nanotrack_tracker.py
Remove-Item tests/unit/test_nanotrack_preprocess.py
Remove-Item tests/unit/test_nanotrack_postprocess.py
Remove-Item tests/unit/test_fusion.py
Remove-Item tests/unit/test_inference.py
```

Bash:
```bash
rm tests/unit/test_kcf_tracker.py tests/unit/test_mosse_tracker.py \
   tests/unit/test_nanotrack_tracker.py tests/unit/test_nanotrack_preprocess.py \
   tests/unit/test_nanotrack_postprocess.py tests/unit/test_fusion.py \
   tests/unit/test_inference.py
```

- [ ] **Step 3: Verify no lingering imports of deleted modules**

Run: `python -m pytest tests/unit -v`
Expected: all pass — no ImportError, no test failures.

If a test fails due to a stray import, grep for the deleted module name (e.g. `grep -r "from quadguide.inference" tests src scripts`) and fix the caller.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "delete: inference, fusion, nanotrack, kcf, mosse packages and their tests"
```

---

### Task 14: Rewrite the perception integration test

**Files:**
- Modify: `tests/integration/test_perception_pipeline.py`
- Delete: `tests/integration/test_preception_pipeline.py` (typo'd duplicate — verify it's truly redundant before deleting)

- [ ] **Step 1: Decide whether the duplicate file is redundant**

```bash
git log --oneline -- tests/integration/test_preception_pipeline.py
diff tests/integration/test_perception_pipeline.py tests/integration/test_preception_pipeline.py
```

If the typo'd file is a stale copy (likely), delete it:

PowerShell: `Remove-Item tests/integration/test_preception_pipeline.py`
Bash: `rm tests/integration/test_preception_pipeline.py`

If the duplicate carries unique tests, copy any unique content into the canonical file before deleting.

- [ ] **Step 2: Replace `tests/integration/test_perception_pipeline.py`**

Write the new test that exercises camera → tracker_worker → bus end-to-end with a real `cv2.TrackerKCF`. Replace the whole file:

```python
"""Integration test: camera + single tracker worker over the bus.

Uses 'fork' (Linux default) so workers inherit shared-memory handles and pipe
fds. The synthetic camera feeds blank frames; the tracker is the real cv2
TrackerKCF wrapped by OpenCVTrackerAdapter.
"""
from __future__ import annotations
import multiprocessing
import os
import pathlib
import signal
import time

import numpy as np
import pytest

from quadguide.core.bus import Bus
from quadguide.core.config import load_config
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.messages import BoundingBox, LockOnCmd, TrackerEstimate
from quadguide.perception.camera.sources import CameraSource
from quadguide.perception.camera.worker import run as run_camera
from quadguide.perception.tracker_worker import (
    OpenCVTrackerAdapter, TrackerWorker,
)

CONFIG_PATH = str(pathlib.Path(__file__).parents[2] / "configs" / "config.yaml")


class _SyntheticCamera(CameraSource):
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._width, self._height = width, height
        self._i = 0

    def open(self) -> None: pass
    def close(self) -> None: pass

    def read(self) -> tuple[np.ndarray, int]:
        # Draw a stable bright rectangle so KCF has something to track.
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame[100:300, 200:400] = 200
        self._i += 1
        return frame, time.monotonic_ns()


def _run_camera(source, fb, bus):
    run_camera(source, fb, bus)


def _run_tracker(tracker, fb, bus):
    TrackerWorker(tracker, bus, fb, cpu_core=None, config={}).run()


@pytest.fixture
def bus_and_fb():
    bus = Bus(ring_depth=8)
    fb = FrameBuffer(640, 480)
    yield bus, fb
    bus.close()
    fb.unlink()


def test_tracker_worker_publishes_target_estimate(bus_and_fb):
    bus, fb = bus_and_fb
    ctx = multiprocessing.get_context("fork")
    camera_proc = ctx.Process(
        target=_run_camera, args=(_SyntheticCamera(), fb, bus),
        name="camera", daemon=False,
    )
    tracker = OpenCVTrackerAdapter("TrackerKCF", {})
    tracker_proc = ctx.Process(
        target=_run_tracker, args=(tracker, fb, bus),
        name="tracker", daemon=False,
    )

    camera_proc.start()
    tracker_proc.start()
    try:
        # Let the camera fill a few frames, then issue a lockon command.
        time.sleep(0.2)
        bus.publish("lockon/cmd", LockOnCmd(
            timestamp_ns=time.monotonic_ns(), seq=1,
            bbox=BoundingBox(200/640, 100/480, 200/640, 200/480),
        ))

        # Poll target/estimate for up to 2 seconds.
        deadline = time.monotonic() + 2.0
        est = None
        while time.monotonic() < deadline:
            est = bus.latest("target/estimate")
            if est is not None:
                break
            time.sleep(0.05)

        assert isinstance(est, TrackerEstimate), "no estimate published"
        assert est.bbox.w > 0 and est.bbox.h > 0
    finally:
        for p in (camera_proc, tracker_proc):
            if p.is_alive():
                os.kill(p.pid, signal.SIGTERM)
            p.join(timeout=2.0)
            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)
                p.join()


@pytest.mark.skipif(os.name != "posix", reason="fork is Linux-only")
def test_config_path_loads_cleanly():
    """Sanity: the shipped config.yaml is loadable and produces TrackerConfig."""
    from quadguide.core.config import cfg_tracker, cfg_platform
    cfg = load_config(CONFIG_PATH, {})
    tcfg = cfg_tracker(cfg)
    assert tcfg.import_spec
    pcfg = cfg_platform(cfg)
    assert pcfg.realtime.control_cpu_core == 3
```

- [ ] **Step 3: Run the integration test**

Run: `python -m pytest tests/integration/test_perception_pipeline.py -v`
Expected: 2 passed (Linux) or 1 passed + 1 skipped (other OS).

If the fork-based test cannot run on the development machine (Windows), it's acceptable to skip the worker-fork test with a platform check and rely on the HIL smoke test (Task 17) to validate end-to-end.

- [ ] **Step 4: Commit**

Stage the whole `tests/integration/` directory so that both the rewrite and the deletion of the typo'd duplicate (if it was removed in step 1) are picked up:

```bash
git add tests/integration/
git commit -m "update: rewrite perception integration test for single tracker worker"
```

---

### Task 15: Update dev scripts

**Files:**
- Modify: `scripts/bench_tracker.py`
- Modify: `scripts/dev_ground_perception.py`

These are dev-only tooling. The exact rewrites depend on what each script currently does; the goal is for both to drive the new single-tracker path using `load_tracker(config)` and `TrackerWorker` (or, for `bench_tracker.py`, `tracker.update(frame)` in a loop without the worker harness).

- [ ] **Step 1: Inspect each script and design the rewrite**

```bash
git log --oneline -- scripts/bench_tracker.py scripts/dev_ground_perception.py
```

Then read each file and decide:

- `bench_tracker.py`: walk the input video frame-by-frame, call `tracker.update(frame)` from a `load_tracker(config)` instance, log CSV rows containing `out.bbox.{x,y,w,h}`, `out.confidence`, `out.health`, plus per-frame latency. Drop all CCV/NCV/fusion-specific logic.
- `dev_ground_perception.py`: spawn camera + tracker_worker + ground worker (same shape as `_start_workers` in `scripts/run.py` minus link/control/guidance).

- [ ] **Step 2: Rewrite each script**

The rewrite of `bench_tracker.py` is roughly:

```python
from quadguide.core.config import load_config
from quadguide.perception.tracker_worker import load_tracker

config = load_config(args.config, {})
tracker = load_tracker(config)

cap = cv2.VideoCapture(args.video)
ok, frame = cap.read()
if not ok:
    raise SystemExit("video has no frames")
# Initial bbox: middle 25% of the frame, or user-supplied
h, w = frame.shape[:2]
bbox = BoundingBox(0.375, 0.375, 0.25, 0.25)
tracker.init(frame, bbox)

with open(args.csv, "w") as f:
    f.write("frame,x,y,w,h,confidence,health,latency_ns\n")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        t0 = time.monotonic_ns()
        out = tracker.update(frame)
        t1 = time.monotonic_ns()
        f.write(f"{i},{out.bbox.x},{out.bbox.y},{out.bbox.w},{out.bbox.h},"
                f"{out.confidence},{out.health},{t1-t0}\n")
        i += 1

tracker.close()
```

Adapt to whatever CLI surface the original `bench_tracker.py` exposes. Same pattern for `dev_ground_perception.py`.

- [ ] **Step 3: Smoke-test each script**

For `dev_ground_perception.py`:

```bash
python scripts/dev_ground_perception.py --config configs/config.yaml
# Open http://localhost:8080 in a browser, confirm video stream renders.
# Ctrl-C to stop.
```

For `bench_tracker.py`: run it on a small sample video and confirm the CSV is produced. If no sample video is at hand, deferring the smoke-test to manual validation post-merge is acceptable for this dev-only script.

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_tracker.py scripts/dev_ground_perception.py
git commit -m "update: dev scripts use load_tracker + single tracker worker"
```

---

### Task 16: Update systemd unit files

**Files:**
- Modify: `systemd/` (specifics depend on current unit files)

- [ ] **Step 1: Inspect the current unit files**

```bash
ls systemd/
```

Identify `qg-kcf.service`, `qg-nano.service`, `qg-fusion.service` (or whatever subset is present).

- [ ] **Step 2: Delete the obsolete services and add `qg-tracker.service`**

PowerShell:
```powershell
Remove-Item systemd/qg-kcf.service -ErrorAction SilentlyContinue
Remove-Item systemd/qg-nano.service -ErrorAction SilentlyContinue
Remove-Item systemd/qg-fusion.service -ErrorAction SilentlyContinue
```

Then create `systemd/qg-tracker.service` modeled on the existing services. Example structure (adapt `User`, `WorkingDirectory`, and `ExecStart` to match the existing pattern in `systemd/qg-camera.service`):

```ini
[Unit]
Description=quadguide tracker worker
After=qg-camera.service
Requires=qg-camera.service

[Service]
Type=simple
User=plas
WorkingDirectory=/opt/quadguide
ExecStart=/usr/bin/python -m quadguide.perception.tracker_worker
Restart=on-failure
RestartSec=2
TimeoutStopSec=2

[Install]
WantedBy=multi-user.target
```

(Note: if `tracker_worker.py` is not yet runnable as `python -m`, the existing pattern likely has each service call into `scripts/run.py` with a worker-specific entry. Mirror whatever the existing pattern is — the goal is to replace 3 services with 1.)

- [ ] **Step 3: Update any `qg-guidance.service` / `qg-control.service` ordering**

Anywhere a service has `After=qg-kcf.service` or `After=qg-fusion.service`, replace with `After=qg-tracker.service`.

- [ ] **Step 4: Commit**

```bash
git add -A systemd/
git commit -m "update: systemd replaces qg-kcf/qg-nano/qg-fusion with qg-tracker"
```

---

### Task 17: HIL smoke test

**Files:** none modified; this is an operational gate.

- [ ] **Step 1: Run SWIL with cv2:TrackerKCF**

In `configs/config.yaml` confirm `tracker.import: cv2:TrackerKCF`, `mission.mode: swil`.

Run: `python scripts/run.py --config configs/config.yaml`

In a second terminal, open `http://localhost:8080`. Expectations:

- Video stream renders the synthetic target.
- Health grid shows `camera`, `tracker_kcf`, `link`, `guidance`, `control`, `ground`.
- Click-and-drag a lockon. The `tracker_health` field flips to `nominal`.
- Within the arm dwell window after issuing arm command, `control` exits failsafe.

- [ ] **Step 2: Run bench_hil with cv2:TrackerKCF**

Edit `configs/config.yaml` to `mission.mode: bench_hil`. Repeat the test against the real bench rig (camera + bench-FC).

- [ ] **Step 3: If both pass, you're done with operational validation**

If either fails, debug per normal — the spec lists no operational behavior change versus today's setup besides the producer process change.

- [ ] **Step 4: No commit needed; this is verification only**

---

### Task 18: Update `architecture.md`

**Files:**
- Modify: `architecture.md`

- [ ] **Step 1: Update the process count**

In `architecture.md` §1 and §2.1, change references to "8 processes" / "3 perception processes" to reflect 6 processes / 1 perception worker. The owned-resource table loses the `ncv worker`, `ccv worker (kcf or mosse)`, and `fusion worker` rows and gains a single `tracker worker` row.

- [ ] **Step 2: Replace §2.4 example**

Anywhere §2.4 references `inference/factory.py` or `tracker_factories.py`, remove. Adding a new SBC no longer touches an inference factory; adding a new tracker is just a new `tracker.import` value in config.

- [ ] **Step 3: Update §4 data-flow diagram**

Replace the camera → CCV/NCV → fusion diagram with camera → tracker.

- [ ] **Step 4: Update §6 (file-by-file)**

Remove §6.3 (inference/), the perception/ccv_tracker_worker, ncv_tracker_worker, kcf, mosse, nanotrack, fusion subsections. Add a §6.4 entry for `perception/tracker_worker.py` describing the loader, adapter, worker, and entry point.

- [ ] **Step 5: Update §7 IPC summary**

Remove the two `*_tracker/estimate` rows and update the `target/estimate` row to show producer = tracker worker, payload = `TrackerEstimate`.

- [ ] **Step 6: Update §11 Known Constraints**

Add the five limitations from the spec ("Known limitations" section): loss of intra-perception fault isolation, NPU handle leak on SIGKILL now library-owned, cv2 tracker params unconfigurable, no multi-tracker redundancy, bench_tracker rewritten.

- [ ] **Step 7: Commit**

```bash
git add architecture.md
git commit -m "docs: architecture.md reflects single tracker worker refactor"
```

---

## Final verification

- [ ] **Step 1: Full test suite green**

Run: `python -m pytest tests -v`
Expected: all green. Any test under `tests/hil/` that requires actual hardware can be xfail'd or skipped as previously.

- [ ] **Step 2: Repo is clean of dead references**

```bash
git grep "TargetEstimate\|ActiveTracker\|FMT_TARGET_ESTIMATE\|ccv_tracker/estimate\|ncv_tracker/estimate\|kcf_cpu_core\|InferenceConfig\|FusionConfig\|NanotrackConfig\|KCFConfig\|MOSSEConfig\|get_runtime\|get_ccv_tracker\|get_ncv_tracker\|CCV_TRACKERS\|NCV_TRACKERS\|CCVTrackerWorker\|NCVTrackerWorker" -- src scripts tests configs
```

Expected: no matches (or only matches in deprecated docs/comments — clean those up too).

- [ ] **Step 3: Spec and plan files match the merged state**

A reviewer reading the spec should find every claim it makes reflected in the code. Re-skim `docs/superpowers/specs/2026-05-29-tracker-refactor-design.md` after Task 18 lands.

---

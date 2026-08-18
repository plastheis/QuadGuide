# edgecv Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `edgecv` package foundation — every dependency the trackers rely on (core types, runtime/IPC, backends, fusion/predictor abstractions, model manifest, packaging) — with no concrete trackers, fully runnable and tested on x86 with no NPU.

**Architecture:** Flat `edgecv/` package (matches `ARCHITECTURE.md` §13). Bottom-up build: core dataclasses → backend HAL + registry → shared-memory primitives (seqlock, frame ring, payload channel) → orchestrator/worker/placement → fusion & predictor ABCs → CF transferable-filter contract → model manifest. Backends `mock`/`onnx` are fully implemented; `rknn` is a lazy adapter that only errors when used.

**Tech Stack:** Python 3.10+, numpy, PyYAML, hatchling (build), onnxruntime (optional extra), pytest + ruff + mypy (dev), `multiprocessing.shared_memory` + ctypes for IPC.

**Authoritative reference:** `ARCHITECTURE.md`. **Spec:** `docs/superpowers/specs/2026-05-31-edgecv-foundation-design.md`. Where this plan and `ARCHITECTURE.md` disagree, the architecture doc wins.

---

## File Structure

```
pyproject.toml                     # hatchling, deps, extras, entry points, tool config
.gitignore
README.md
.github/workflows/ci.yml           # x86 CI: ruff + mypy + pytest (mock/onnx only)
edgecv/
├── __init__.py                    # version, top-level re-exports
├── core/
│   ├── __init__.py
│   ├── bbox.py                    # BoundingBox (0–1), PixelBox, conversions
│   ├── result.py                  # TrackStatus, TrackResult
│   └── tracker.py                 # Tracker ABC
├── backends/
│   ├── __init__.py
│   ├── base.py                    # TensorSpec, IOSpec, Handle, Model, InferenceBackend
│   ├── registry.py                # lazy entry-point registry
│   ├── mock/__init__.py           # MockBackend, MockModel (full)
│   ├── onnx/__init__.py           # OnnxBackend, OnnxModel (full, lazy import)
│   └── rknn/__init__.py           # RknnBackend (lazy adapter, errors on use)
├── runtime/
│   ├── __init__.py
│   ├── shm/
│   │   ├── __init__.py
│   │   ├── structs.py             # MAGIC, ABI_VERSION, dtype codes, header helpers
│   │   ├── seqlock.py             # SeqLock (wait-free reads)
│   │   ├── frame_ring.py          # zero-copy frame ring, latest-only
│   │   └── payload.py             # variable-shape numpy payload channel
│   ├── placement.py               # BoardProfile, YAML loader, affinity/sched
│   ├── worker.py                  # child entrypoint; backend init in-child
│   └── orchestrator.py            # spawn workers, own SHM lifecycle, heartbeat
├── fusion/
│   ├── __init__.py
│   ├── policy.py                  # FusionPolicy ABC, DetectorOutput, FusionDecision
│   └── predict.py                 # MotionPredictor ABC
├── trackers/
│   ├── __init__.py
│   ├── cf/
│   │   ├── __init__.py
│   │   ├── base.py                # CorrelationFilterTracker ABC, FilterState, EvalResult
│   │   └── ops/__init__.py        # skeleton only
│   ├── nn/__init__.py             # skeleton only
│   └── hybrid/__init__.py         # skeleton only
├── models/
│   ├── __init__.py
│   ├── manifest.py                # ModelManifest schema + YAML loader
│   └── profiles/rk3588.yaml       # shipped board profile (package data)
└── tools/
    └── README.md                  # host-only; not a runtime dependency
tests/
├── __init__.py
├── test_bbox.py
├── test_result.py
├── test_tracker.py
├── test_registry.py
├── test_mock_backend.py
├── test_onnx_backend.py
├── test_seqlock.py
├── test_frame_ring.py
├── test_payload.py
├── test_placement.py
├── test_manifest.py
├── test_fusion_abcs.py
└── test_cf_base.py
```

---

## Task 1: Project scaffold and packaging

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`
- Create: all `__init__.py` files listed in the File Structure (empty except `edgecv/__init__.py`)
- Create: `edgecv/models/profiles/` (dir), `tests/__init__.py`

- [ ] **Step 1: Create the package directory tree and empty `__init__.py` files**

```bash
cd /home/plas/edgecv
mkdir -p edgecv/core edgecv/backends/mock edgecv/backends/onnx edgecv/backends/rknn \
         edgecv/runtime/shm edgecv/fusion edgecv/trackers/cf/ops edgecv/trackers/nn \
         edgecv/trackers/hybrid edgecv/models/profiles edgecv/tools tests
for d in edgecv edgecv/core edgecv/backends edgecv/backends/mock edgecv/backends/onnx \
         edgecv/backends/rknn edgecv/runtime edgecv/runtime/shm edgecv/fusion \
         edgecv/trackers edgecv/trackers/cf edgecv/trackers/cf/ops edgecv/trackers/nn \
         edgecv/trackers/hybrid edgecv/models tests; do
  touch "$d/__init__.py"
done
```

- [ ] **Step 2: Write `edgecv/__init__.py`**

```python
"""edgecv — single-object visual trackers for real-time edge deployment."""

__version__ = "0.0.1"

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.core.tracker import Tracker

__all__ = [
    "BoundingBox",
    "PixelBox",
    "TrackResult",
    "TrackStatus",
    "Tracker",
    "__version__",
]
```

> Note: this import block depends on Tasks 2–4. CI in Task 1 only checks the build metadata; the package imports become valid after Task 4. That is expected for a bottom-up TDD build.

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "edgecv"
version = "0.0.1"
description = "Single-object visual trackers for real-time edge deployment"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
authors = [{ name = "plastheis" }]
dependencies = [
    "numpy>=1.23",
    "PyYAML>=6.0",
]

[project.optional-dependencies]
onnx = ["onnxruntime>=1.16"]
# rknn-toolkit-lite2 is NOT on PyPI; install it manually on-device.
# This extra exists only to signal intent — see README.
rknn = []
dev = []
test = ["pytest>=7.0", "ruff>=0.4", "mypy>=1.8"]

[project.entry-points."edgecv.backends"]
mock = "edgecv.backends.mock:MockBackend"
onnx = "edgecv.backends.onnx:OnnxBackend"
rknn = "edgecv.backends.rknn:RknnBackend"

[tool.hatch.build.targets.wheel]
packages = ["edgecv"]

[tool.hatch.build.targets.wheel.force-include]
"edgecv/models/profiles" = "edgecv/models/profiles"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
warn_unused_ignores = false
```

- [ ] **Step 4: Write `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
build/
dist/
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.rknn
```

- [ ] **Step 5: Write `README.md`**

```markdown
# edgecv

Single-object visual trackers scoped to real-time deployment on edge hardware.
See `ARCHITECTURE.md` for the design.

## Install

```bash
pip install edgecv             # core: numpy CF runtime, fusion abstractions, mock backend
pip install edgecv[onnx]       # ONNXRuntime CPU/dev backend
pip install edgecv[rknn]       # registers the RKNN backend (see device note below)
pip install edgecv[test]       # test + lint tooling
```

### RKNN on-device note

`rknn-toolkit-lite2` is **not on PyPI** and is **installed manually on the device**
(Rockchip release archive). The `[rknn]` extra only registers the backend adapter;
it does not and cannot pull the runtime.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .[onnx,test]
pytest -q
```

## Status

Foundation only — no concrete trackers yet. See
`docs/superpowers/specs/2026-05-31-edgecv-foundation-design.md`.
```

- [ ] **Step 6: Create remaining placeholder package files and tools README**

```bash
cd /home/plas/edgecv
touch edgecv/backends/onnx/__init__.py edgecv/backends/rknn/__init__.py
cat > edgecv/tools/README.md <<'EOF'
# tools — host-only

Conversion (ONNX → RKNN via rknn-toolkit2) and training (pytracking) helpers.
**Host-only. Not a runtime dependency.** Nothing here is imported by the `edgecv`
package at runtime. Empty for now; see `ARCHITECTURE.md` §11.
EOF
```

- [ ] **Step 7: Create a virtualenv and install the package editable**

Run:
```bash
cd /home/plas/edgecv
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[onnx,test]
```
Expected: install succeeds (it will fail to import `edgecv` until Task 4 — that is fine; the build metadata installs).

- [ ] **Step 8: Verify the build metadata and entry points resolve**

Run:
```bash
. .venv/bin/activate
python -c "from importlib.metadata import entry_points; print(sorted(e.name for e in entry_points(group='edgecv.backends')))"
```
Expected: `['mock', 'onnx', 'rknn']`

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore: scaffold edgecv package, packaging, and entry points"
```

---

## Task 2: core/bbox — BoundingBox and PixelBox

**Files:**
- Create: `edgecv/core/bbox.py`
- Test: `tests/test_bbox.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bbox.py
import math

import pytest

from edgecv.core.bbox import BoundingBox, PixelBox


def test_to_pixels_scales_by_dimensions():
    bb = BoundingBox(x=0.1, y=0.2, w=0.5, h=0.25)
    px = bb.to_pixels(width=200, height=100)
    assert px == PixelBox(x=20.0, y=20.0, w=100.0, h=25.0)


def test_from_pixels_is_inverse_of_to_pixels():
    bb = BoundingBox(x=0.3, y=0.4, w=0.2, h=0.1)
    px = bb.to_pixels(640, 480)
    back = BoundingBox.from_pixels(px, 640, 480)
    assert math.isclose(back.x, bb.x) and math.isclose(back.y, bb.y)
    assert math.isclose(back.w, bb.w) and math.isclose(back.h, bb.h)


def test_negative_dimension_rejected():
    with pytest.raises(ValueError):
        BoundingBox(x=0.0, y=0.0, w=-0.1, h=0.5)


def test_clamp_keeps_box_inside_unit_square():
    bb = BoundingBox(x=0.8, y=0.8, w=0.5, h=0.5).clamp()
    assert bb.x + bb.w <= 1.0 + 1e-9
    assert bb.y + bb.h <= 1.0 + 1e-9


def test_pixelbox_center():
    px = PixelBox(x=10.0, y=20.0, w=4.0, h=6.0)
    assert px.center == (12.0, 23.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bbox.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.core.bbox'`

- [ ] **Step 3: Write `edgecv/core/bbox.py`**

```python
"""Bounding-box types. BoundingBox is always normalised 0–1; PixelBox is the
explicit pixel-space helper used only at the pixel boundary. Never let a raw
pixel tuple masquerade as a BoundingBox (see ARCHITECTURE.md §5.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PixelBox:
    """Axis-aligned box in pixel coordinates (sub-pixel allowed)."""

    x: float  # top-left x, pixels
    y: float  # top-left y, pixels
    w: float  # width, pixels
    h: float  # height, pixels

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box, normalised to 0–1 in both dimensions."""

    x: float  # top-left x, normalised 0–1
    y: float  # top-left y, normalised 0–1
    w: float  # width,  normalised 0–1
    h: float  # height, normalised 0–1

    def __post_init__(self) -> None:
        if self.w < 0.0 or self.h < 0.0:
            raise ValueError(f"BoundingBox dimensions must be non-negative: {self!r}")

    def to_pixels(self, width: int, height: int) -> PixelBox:
        return PixelBox(
            x=self.x * width,
            y=self.y * height,
            w=self.w * width,
            h=self.h * height,
        )

    @classmethod
    def from_pixels(cls, box: PixelBox, width: int, height: int) -> BoundingBox:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        return cls(
            x=box.x / width,
            y=box.y / height,
            w=box.w / width,
            h=box.h / height,
        )

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def clamp(self) -> BoundingBox:
        """Return a copy fully contained in the unit square."""
        x = min(max(self.x, 0.0), 1.0)
        y = min(max(self.y, 0.0), 1.0)
        w = min(self.w, 1.0 - x)
        h = min(self.h, 1.0 - y)
        return BoundingBox(x=x, y=y, w=w, h=h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bbox.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add edgecv/core/bbox.py tests/test_bbox.py
git commit -m "feat(core): BoundingBox (normalised) and PixelBox with conversions"
```

---

## Task 3: core/result — TrackStatus and TrackResult

**Files:**
- Create: `edgecv/core/result.py`
- Test: `tests/test_result.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_result.py
from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackResult, TrackStatus


def test_status_values_match_contract():
    assert TrackStatus.INITIALIZING == 0
    assert TrackStatus.LOCKED == 1
    assert TrackStatus.COASTING == 2
    assert TrackStatus.LOST == 3


def test_track_result_allows_none_bbox_and_confidence():
    r = TrackResult(bbox=None, confidence=None, status=TrackStatus.LOST,
                    timestamp=1.5, seq=7)
    assert r.bbox is None
    assert r.confidence is None
    assert r.seq == 7


def test_track_result_carries_bbox_and_score():
    bb = BoundingBox(0.1, 0.1, 0.2, 0.2)
    r = TrackResult(bbox=bb, confidence=12.3, status=TrackStatus.LOCKED,
                    timestamp=2.0, seq=42)
    assert r.bbox is bb
    assert r.confidence == 12.3
    assert r.status is TrackStatus.LOCKED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_result.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.core.result'`

- [ ] **Step 3: Write `edgecv/core/result.py`**

```python
"""Tracker output types (ARCHITECTURE.md §5.2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from edgecv.core.bbox import BoundingBox


class TrackStatus(IntEnum):
    INITIALIZING = 0  # workers warming up / no lock yet
    LOCKED = 1        # confident track
    COASTING = 2      # low confidence, extrapolating / awaiting correction
    LOST = 3          # track lost


@dataclass
class TrackResult:
    bbox: BoundingBox | None       # None when no estimate is available
    confidence: float | None       # None when the tracker has no meaningful score
    status: TrackStatus
    timestamp: float               # monotonic seconds, source-frame time
    seq: int                       # frame sequence number this result corresponds to
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_result.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add edgecv/core/result.py tests/test_result.py
git commit -m "feat(core): TrackStatus and TrackResult"
```

---

## Task 4: core/tracker — Tracker ABC

**Files:**
- Create: `edgecv/core/tracker.py`
- Test: `tests/test_tracker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tracker.py
import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.core.tracker import Tracker


class _Dummy(Tracker):
    def __init__(self):
        self._status = TrackStatus.INITIALIZING
        self.closed = False

    def init(self, frame, bbox):
        self._status = TrackStatus.LOCKED

    def update(self, frame):
        return TrackResult(bbox=BoundingBox(0, 0, 0.1, 0.1), confidence=1.0,
                           status=self._status, timestamp=0.0, seq=0)

    @property
    def status(self):
        return self._status

    def name(self):
        return "Dummy"

    def close(self):
        self.closed = True


def test_tracker_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Tracker()


def test_context_manager_calls_close():
    with _Dummy() as t:
        t.init(np.zeros((4, 4), np.uint8), BoundingBox(0, 0, 0.5, 0.5))
        assert t.status is TrackStatus.LOCKED
    assert t.closed is True


def test_default_close_is_noop_for_inline_tracker():
    class _NoClose(_Dummy):
        pass
    # Should not raise even though we did not override close beyond _Dummy.
    _NoClose().__exit__(None, None, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tracker.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.core.tracker'`

- [ ] **Step 3: Write `edgecv/core/tracker.py`**

```python
"""Tracker abstract base class (ARCHITECTURE.md §5.3)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackResult, TrackStatus


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
        """Human-readable tracker name, e.g. "MOSSE", "SiamFC", "MAFiD"."""

    def close(self) -> None:
        """Tear down any owned process group / shared memory. No-op for inline trackers."""

    def __enter__(self) -> Tracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tracker.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Verify top-level package imports now resolve**

Run: `python -c "import edgecv; print(edgecv.__version__, edgecv.Tracker)"`
Expected: prints `0.0.1 <class 'edgecv.core.tracker.Tracker'>`

- [ ] **Step 6: Commit**

```bash
git add edgecv/core/tracker.py tests/test_tracker.py
git commit -m "feat(core): Tracker ABC with context-manager lifecycle"
```

---

## Task 5: backends/base + registry

**Files:**
- Create: `edgecv/backends/base.py`, `edgecv/backends/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
import pytest

from edgecv.backends.registry import (
    BackendNotFoundError,
    available_backends,
    get_backend,
    list_backends,
)


def test_list_backends_includes_builtins():
    names = list_backends()
    assert {"mock", "onnx", "rknn"} <= set(names)


def test_get_backend_returns_singleton_like_instance():
    b1 = get_backend("mock")
    b2 = get_backend("mock")
    assert b1.name == "mock"
    assert b1 is b2  # cached


def test_unknown_backend_raises():
    with pytest.raises(BackendNotFoundError):
        get_backend("does-not-exist")


def test_available_backends_subset_of_all():
    assert set(available_backends()) <= set(list_backends())
    assert "mock" in available_backends()  # mock is always available
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.backends.registry'`

- [ ] **Step 3: Write `edgecv/backends/base.py`**

```python
"""Hardware-abstraction interfaces (ARCHITECTURE.md §10). Trackers depend on
these, never on a vendor runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from edgecv.models.manifest import ModelManifest


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]          # -1 for dynamic dims
    dtype: str                      # numpy dtype name, e.g. "float32", "int8"
    layout: str = "NCHW"            # informational
    quant: dict | None = None       # {"scale": ..., "zero_point": ...} for INT8, else None


@dataclass(frozen=True)
class IOSpec:
    inputs: tuple[TensorSpec, ...] = field(default_factory=tuple)
    outputs: tuple[TensorSpec, ...] = field(default_factory=tuple)


class Handle(ABC):
    """Async inference handle (NPUs pipeline). `wait` blocks for the result."""

    @abstractmethod
    def wait(self) -> dict[str, np.ndarray]: ...


class Model(ABC):
    @property
    @abstractmethod
    def io_spec(self) -> IOSpec: ...

    @abstractmethod
    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

    def infer_async(self, inputs: dict[str, np.ndarray]) -> Handle:
        """Optional. Default raises; backends that pipeline override this."""
        raise NotImplementedError(f"{type(self).__name__} does not support infer_async")

    @abstractmethod
    def close(self) -> None: ...


class InferenceBackend(ABC):
    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """True if this backend's runtime can actually load and run a model here."""

    @abstractmethod
    def load(self, manifest: ModelManifest) -> Model: ...
```

- [ ] **Step 4: Write `edgecv/backends/registry.py`**

```python
"""Lazy, entry-point-driven backend registry (ARCHITECTURE.md §10).

Backends register under the `edgecv.backends` entry-point group and are imported
lazily, so a missing vendor runtime only errors when that backend is used. A
built-in fallback map covers the case where the package metadata is unavailable.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from edgecv.backends.base import InferenceBackend

_BUILTIN: dict[str, str] = {
    "mock": "edgecv.backends.mock:MockBackend",
    "onnx": "edgecv.backends.onnx:OnnxBackend",
    "rknn": "edgecv.backends.rknn:RknnBackend",
}

_instances: dict[str, InferenceBackend] = {}


class BackendNotFoundError(KeyError):
    pass


def _entry_point_targets() -> dict[str, str]:
    targets: dict[str, str] = dict(_BUILTIN)
    try:
        for ep in entry_points(group="edgecv.backends"):
            targets[ep.name] = ep.value
    except Exception:
        # Metadata unavailable (e.g. source tree without install) — builtins suffice.
        pass
    return targets


def list_backends() -> list[str]:
    return sorted(_entry_point_targets())


def _load_class(target: str) -> type[InferenceBackend]:
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, attr)
    if not issubclass(cls, InferenceBackend):
        raise TypeError(f"{target} is not an InferenceBackend")
    return cls


def get_backend(name: str) -> InferenceBackend:
    if name in _instances:
        return _instances[name]
    targets = _entry_point_targets()
    if name not in targets:
        raise BackendNotFoundError(name)
    instance = _load_class(targets[name])()
    _instances[name] = instance
    return instance


def available_backends() -> list[str]:
    """Names whose runtime is importable and usable on this machine."""
    out: list[str] = []
    for name in list_backends():
        try:
            if get_backend(name).is_available():
                out.append(name)
        except Exception:
            continue
    return out
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_registry.py -q`
Expected: PASS (4 passed) — depends on Tasks 6 & 7 providing the backend classes; if run before those, `list_backends` still passes but `get_backend("mock")` import fails. Implement Tasks 6–7 before running the full file, or run only `test_list_backends_includes_builtins` here.

- [ ] **Step 6: Commit**

```bash
git add edgecv/backends/base.py edgecv/backends/registry.py tests/test_registry.py
git commit -m "feat(backends): HAL interfaces and lazy entry-point registry"
```

---

## Task 6: backends/mock — full

**Files:**
- Create: `edgecv/backends/mock/__init__.py`
- Test: `tests/test_mock_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mock_backend.py
import numpy as np

from edgecv.backends.mock import MockBackend
from edgecv.models.manifest import ModelManifest


def _manifest():
    return ModelManifest(
        name="t",
        task="sot_template_matching",
        preprocessing={},
        inputs=[{"name": "x", "shape": [1, 3, 8, 8], "dtype": "float32"}],
        outputs=[{"name": "y", "shape": [1, 1, 4, 4], "dtype": "float32"}],
        artifacts={"mock": {}},
    )


def test_mock_backend_is_always_available():
    assert MockBackend().is_available() is True


def test_mock_model_io_spec_from_manifest():
    model = MockBackend().load(_manifest())
    spec = model.io_spec
    assert spec.inputs[0].name == "x"
    assert spec.outputs[0].shape == (1, 1, 4, 4)


def test_mock_infer_returns_zeros_of_declared_shape():
    model = MockBackend().load(_manifest())
    out = model.infer({"x": np.zeros((1, 3, 8, 8), np.float32)})
    assert set(out) == {"y"}
    assert out["y"].shape == (1, 1, 4, 4)
    assert out["y"].dtype == np.float32


def test_mock_async_handle_matches_sync():
    model = MockBackend().load(_manifest())
    out = model.infer_async({"x": np.zeros((1, 3, 8, 8), np.float32)}).wait()
    assert out["y"].shape == (1, 1, 4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mock_backend.py -q`
Expected: FAIL — `ImportError: cannot import name 'MockBackend'` (module is empty)

- [ ] **Step 3: Write `edgecv/backends/mock/__init__.py`**

```python
"""Mock backend: canned, manifest-shaped outputs. Lets the full runtime/IPC/
fusion stack run with no model and no accelerator (ARCHITECTURE.md §10)."""

from __future__ import annotations

import numpy as np

from edgecv.backends.base import Handle, IOSpec, Model, TensorSpec
from edgecv.models.manifest import ModelManifest


def _specs(entries: list[dict]) -> tuple[TensorSpec, ...]:
    return tuple(
        TensorSpec(
            name=e["name"],
            shape=tuple(e["shape"]),
            dtype=e.get("dtype", "float32"),
            layout=e.get("layout", "NCHW"),
            quant=e.get("quant"),
        )
        for e in entries
    )


def _concrete_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    # Replace dynamic (-1) dims with 1 so a concrete array can be produced.
    return tuple(d if d > 0 else 1 for d in shape)


class _ImmediateHandle(Handle):
    def __init__(self, result: dict[str, np.ndarray]):
        self._result = result

    def wait(self) -> dict[str, np.ndarray]:
        return self._result


class MockModel(Model):
    def __init__(self, io_spec: IOSpec):
        self._io_spec = io_spec

    @property
    def io_spec(self) -> IOSpec:
        return self._io_spec

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            spec.name: np.zeros(_concrete_shape(spec.shape), dtype=np.dtype(spec.dtype))
            for spec in self._io_spec.outputs
        }

    def infer_async(self, inputs: dict[str, np.ndarray]) -> Handle:
        return _ImmediateHandle(self.infer(inputs))

    def close(self) -> None:  # nothing to release
        pass


class MockBackend:
    name = "mock"

    def is_available(self) -> bool:
        return True

    def load(self, manifest: ModelManifest) -> Model:
        io_spec = IOSpec(inputs=_specs(manifest.inputs), outputs=_specs(manifest.outputs))
        return MockModel(io_spec)
```

> Note: `MockBackend` structurally satisfies `InferenceBackend`. The registry checks
> `issubclass(cls, InferenceBackend)`, so register it as a subclass — make the class
> declaration `class MockBackend(InferenceBackend):` and import `InferenceBackend` from
> `edgecv.backends.base`. Apply the same to `OnnxBackend` and `RknnBackend`.

- [ ] **Step 4: Fix the base-class declaration**

Edit the import line to `from edgecv.backends.base import Handle, InferenceBackend, IOSpec, Model, TensorSpec`
and change `class MockBackend:` to `class MockBackend(InferenceBackend):`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_mock_backend.py -q`
Expected: PASS (4 passed) — requires Task 15 `ModelManifest`. If implementing in order, defer running this file until after Task 15, or implement Task 15 first. (See Self-Review note on ordering.)

- [ ] **Step 6: Commit**

```bash
git add edgecv/backends/mock/__init__.py tests/test_mock_backend.py
git commit -m "feat(backends): full mock backend with manifest-shaped output"
```

---

## Task 7: backends/onnx (full) and backends/rknn (lazy adapter)

**Files:**
- Create: `edgecv/backends/onnx/__init__.py`, `edgecv/backends/rknn/__init__.py`
- Test: `tests/test_onnx_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onnx_backend.py
import numpy as np
import pytest

from edgecv.backends.rknn import RknnBackend

ort = pytest.importorskip("onnxruntime")


def _make_identity_onnx(path):
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    import onnx
    onnx.save(model, str(path))


def test_onnx_backend_loads_and_infers(tmp_path):
    pytest.importorskip("onnx")
    from edgecv.backends.onnx import OnnxBackend
    from edgecv.models.manifest import ModelManifest

    onnx_path = tmp_path / "id.onnx"
    _make_identity_onnx(onnx_path)
    man = ModelManifest(
        name="id", task="test", preprocessing={},
        inputs=[{"name": "x", "shape": [1, 3], "dtype": "float32"}],
        outputs=[{"name": "y", "shape": [1, 3], "dtype": "float32"}],
        artifacts={"onnx": {"path": str(onnx_path)}},
    )
    model = OnnxBackend().load(man)
    out = model.infer({"x": np.array([[1.0, 2.0, 3.0]], np.float32)})
    np.testing.assert_allclose(out["y"], [[1.0, 2.0, 3.0]])
    assert model.io_spec.inputs[0].name == "x"
    model.close()


def test_rknn_backend_reports_unavailable_without_runtime():
    # On x86/CI there is no rknnlite; the adapter must say so, not crash.
    assert RknnBackend().is_available() is False


def test_rknn_load_raises_clear_error_without_runtime():
    from edgecv.models.manifest import ModelManifest

    man = ModelManifest(name="m", task="t", preprocessing={},
                        inputs=[], outputs=[], artifacts={"rknn": {"path": "x.rknn"}})
    with pytest.raises(RuntimeError, match="rknn-toolkit-lite2"):
        RknnBackend().load(man)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_onnx_backend.py -q`
Expected: FAIL — `ImportError: cannot import name 'RknnBackend'`

- [ ] **Step 3: Write `edgecv/backends/onnx/__init__.py`**

```python
"""ONNXRuntime CPU backend (ARCHITECTURE.md §10). Lazy import: onnxruntime is an
optional extra, imported only when this backend is actually loaded."""

from __future__ import annotations

import numpy as np

from edgecv.backends.base import InferenceBackend, IOSpec, Model, TensorSpec
from edgecv.models.manifest import ModelManifest

# ONNX tensor element type -> numpy dtype name (the common subset).
_ORT_TO_NP = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(float16)": "float16",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(int8)": "int8",
    "tensor(uint8)": "uint8",
}


def _dims(shape: list) -> tuple[int, ...]:
    return tuple(d if isinstance(d, int) and d > 0 else -1 for d in shape)


class OnnxModel(Model):
    def __init__(self, session):
        self._session = session
        inputs = tuple(
            TensorSpec(name=i.name, shape=_dims(i.shape),
                       dtype=_ORT_TO_NP.get(i.type, "float32"))
            for i in session.get_inputs()
        )
        outputs = tuple(
            TensorSpec(name=o.name, shape=_dims(o.shape),
                       dtype=_ORT_TO_NP.get(o.type, "float32"))
            for o in session.get_outputs()
        )
        self._io_spec = IOSpec(inputs=inputs, outputs=outputs)

    @property
    def io_spec(self) -> IOSpec:
        return self._io_spec

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        names = [o.name for o in self._io_spec.outputs]
        results = self._session.run(names, inputs)
        return dict(zip(names, results))

    def close(self) -> None:
        self._session = None


class OnnxBackend(InferenceBackend):
    name = "onnx"

    def is_available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, manifest: ModelManifest) -> Model:
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - covered by is_available path
            raise RuntimeError(
                "onnxruntime is not installed; install with `pip install edgecv[onnx]`"
            ) from e
        artifact = manifest.artifacts.get("onnx")
        if not artifact or "path" not in artifact:
            raise ValueError(f"manifest {manifest.name!r} has no onnx artifact path")
        session = ort.InferenceSession(
            artifact["path"], providers=["CPUExecutionProvider"]
        )
        return OnnxModel(session)
```

- [ ] **Step 4: Write `edgecv/backends/rknn/__init__.py`**

```python
"""RKNN backend adapter (ARCHITECTURE.md §10). Lazy: rknn-toolkit-lite2 is NOT on
PyPI and is installed manually on-device. This adapter reports unavailability
cleanly off-device and raises an actionable error if used without the runtime.
It must be initialised inside the worker process, never the parent."""

from __future__ import annotations

from edgecv.backends.base import InferenceBackend, Model
from edgecv.models.manifest import ModelManifest

_INSTALL_HINT = (
    "rknn-toolkit-lite2 is not available. It is not on PyPI; install it manually "
    "on the Rockchip device (see README RKNN note). The [rknn] extra only registers "
    "this adapter."
)


def _import_rknnlite():
    from rknnlite.api import RKNNLite  # type: ignore

    return RKNNLite


class RknnBackend(InferenceBackend):
    name = "rknn"

    def is_available(self) -> bool:
        try:
            _import_rknnlite()
        except Exception:
            return False
        return True

    def load(self, manifest: ModelManifest) -> Model:
        try:
            _import_rknnlite()
        except Exception as e:
            raise RuntimeError(_INSTALL_HINT) from e
        # Concrete RKNNLite model wiring lands with the first NN tracker; the
        # adapter is intentionally minimal in the foundation build.
        raise NotImplementedError(
            "RKNN model loading is implemented alongside the first NN tracker."
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_onnx_backend.py -q`
Expected: PASS (onnx tests pass if `onnx`+`onnxruntime` present, else skipped; rknn tests pass). Requires Task 15.

- [ ] **Step 6: Commit**

```bash
git add edgecv/backends/onnx/__init__.py edgecv/backends/rknn/__init__.py tests/test_onnx_backend.py
git commit -m "feat(backends): full onnx CPU backend and lazy rknn adapter"
```

---

## Task 8: runtime/shm/structs and seqlock

**Files:**
- Create: `edgecv/runtime/shm/structs.py`, `edgecv/runtime/shm/seqlock.py`
- Test: `tests/test_seqlock.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seqlock.py
import threading
from multiprocessing import shared_memory

import numpy as np

from edgecv.runtime.shm.seqlock import SeqLock
from edgecv.runtime.shm.structs import ABI_VERSION, MAGIC, dtype_to_code, code_to_dtype


def test_magic_and_abi_present():
    assert isinstance(MAGIC, int)
    assert ABI_VERSION >= 1


def test_dtype_code_roundtrip():
    for name in ("float32", "uint8", "int8", "complex64", "float64"):
        assert code_to_dtype(dtype_to_code(np.dtype(name))) == np.dtype(name)


def test_seqlock_torn_read_is_retried():
    shm = shared_memory.SharedMemory(create=True, size=64)
    try:
        lock = SeqLock(shm.buf, offset=0)
        payload = np.ndarray((4,), dtype=np.int64, buffer=shm.buf, offset=8)
        payload[:] = [0, 0, 0, 0]

        stop = threading.Event()
        torn = {"seen": False}

        def writer():
            v = 0
            while not stop.is_set():
                v += 1
                lock.write_begin()
                for i in range(4):
                    payload[i] = v       # multi-field write, not atomic
                lock.write_end()

        def reader():
            for _ in range(20000):
                def read():
                    return [int(payload[i]) for i in range(4)]
                vals = lock.read(read)
                if len(set(vals)) != 1:
                    torn["seen"] = True
                    break

        w = threading.Thread(target=writer)
        w.start()
        reader()
        stop.set()
        w.join()

        assert torn["seen"] is False  # seqlock retried away every torn read
    finally:
        shm.close()
        shm.unlink()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_seqlock.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.runtime.shm.seqlock'`

- [ ] **Step 3: Write `edgecv/runtime/shm/structs.py`**

```python
"""Single source of truth for shared-memory layouts (ARCHITECTURE.md §7.5).

Every shared header begins with MAGIC + ABI_VERSION, validated on attach. Any
change to a shared layout MUST bump ABI_VERSION and update both producer and
consumer.
"""

from __future__ import annotations

import ctypes

import numpy as np

MAGIC = 0xED6EC711          # "edgecv" tag; arbitrary but fixed
ABI_VERSION = 1

# numpy dtype <-> stable integer code. Append-only; never renumber.
_CODE_TO_NAME: dict[int, str] = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "int32",
    5: "int64",
    6: "float16",
    7: "float32",
    8: "float64",
    9: "complex64",
    10: "complex128",
    11: "bool",
}
_NAME_TO_CODE: dict[str, int] = {v: k for k, v in _CODE_TO_NAME.items()}


def dtype_to_code(dtype: np.dtype) -> int:
    name = np.dtype(dtype).name
    try:
        return _NAME_TO_CODE[name]
    except KeyError as e:
        raise ValueError(f"unsupported dtype for IPC: {name}") from e


def code_to_dtype(code: int) -> np.dtype:
    try:
        return np.dtype(_CODE_TO_NAME[code])
    except KeyError as e:
        raise ValueError(f"unknown dtype code: {code}") from e


class FrameControl(ctypes.Structure):
    """Control word published by the frame-ring producer (one writer)."""

    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("seq", ctypes.c_uint64),
        ("seqlock", ctypes.c_uint64),     # seqlock word (odd while writing)
        ("timestamp", ctypes.c_double),
        ("slot", ctypes.c_uint32),
        ("h", ctypes.c_uint32),
        ("w", ctypes.c_uint32),
        ("c", ctypes.c_uint32),
        ("dtype_code", ctypes.c_uint32),
    ]


def validate_header(magic: int, abi_version: int) -> None:
    if magic != MAGIC:
        raise ValueError(f"bad shared-memory magic: {magic:#x} != {MAGIC:#x}")
    if abi_version != ABI_VERSION:
        raise ValueError(
            f"ABI mismatch: segment v{abi_version}, library v{ABI_VERSION}"
        )
```

- [ ] **Step 4: Write `edgecv/runtime/shm/seqlock.py`**

```python
"""Seqlock for wait-free cross-process reads (ARCHITECTURE.md §7.3).

The writer bumps the seq word odd, writes the payload, then bumps it even. A
reader retries while the seq is odd or changed across the read. Reads never block
the writer.

Honest caveat: pure Python has no explicit memory barriers, so this is "correct
in practice for aligned word-size stores on ARM64/x86" rather than provably
correct. If stronger guarantees are needed, back the control word with a tiny C
extension or a microsecond-held lock on ONLY the control word.
"""

from __future__ import annotations

import ctypes
from typing import Callable, TypeVar

T = TypeVar("T")


class SeqLock:
    """Wraps a uint64 seq word living at `offset` inside a shared buffer."""

    def __init__(self, buf, offset: int = 0):
        self._word = ctypes.c_uint64.from_buffer(buf, offset)

    def write_begin(self) -> None:
        self._word.value += 1          # -> odd: a write is in progress

    def write_end(self) -> None:
        self._word.value += 1          # -> even: write complete

    def read(self, fn: Callable[[], T], max_retries: int = 10_000) -> T:
        """Run `fn` (which reads the guarded payload) under seqlock retry."""
        for _ in range(max_retries):
            before = self._word.value
            if before & 1:             # writer mid-update
                continue
            value = fn()
            after = self._word.value
            if before == after:
                return value
        raise RuntimeError("seqlock read exceeded max_retries (writer starving reader?)")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_seqlock.py -q`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add edgecv/runtime/shm/structs.py edgecv/runtime/shm/seqlock.py tests/test_seqlock.py
git commit -m "feat(runtime): shared-struct ABI source of truth and seqlock"
```

---

## Task 9: runtime/shm/frame_ring

**Files:**
- Create: `edgecv/runtime/shm/frame_ring.py`
- Test: `tests/test_frame_ring.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frame_ring.py
import numpy as np

from edgecv.runtime.shm.frame_ring import FrameRing


def test_publish_then_read_latest_roundtrips():
    ring = FrameRing.create(slots=4, max_h=8, max_w=8, max_c=3, dtype="uint8")
    try:
        frame = (np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
        ring.publish(frame, seq=1, timestamp=10.0)
        got = ring.read_latest()
        assert got is not None
        view, seq, ts = got
        assert seq == 1 and ts == 10.0
        np.testing.assert_array_equal(view, frame)
    finally:
        ring.close(unlink=True)


def test_latest_only_skips_to_newest():
    ring = FrameRing.create(slots=4, max_h=4, max_w=4, max_c=1, dtype="uint8")
    try:
        for s in range(1, 6):  # more than slots, forces wraparound
            ring.publish(np.full((4, 4, 1), s, np.uint8), seq=s, timestamp=float(s))
        view, seq, ts = ring.read_latest()
        assert seq == 5
        assert int(view[0, 0, 0]) == 5
    finally:
        ring.close(unlink=True)


def test_read_before_any_publish_returns_none():
    ring = FrameRing.create(slots=2, max_h=4, max_w=4, max_c=1, dtype="uint8")
    try:
        assert ring.read_latest() is None
    finally:
        ring.close(unlink=True)


def test_attach_reads_producer_frames():
    producer = FrameRing.create(slots=3, max_h=4, max_w=4, max_c=1, dtype="uint8")
    try:
        producer.publish(np.full((4, 4, 1), 7, np.uint8), seq=99, timestamp=1.0)
        consumer = FrameRing.attach(producer.name, slots=3, max_h=4, max_w=4,
                                    max_c=1, dtype="uint8")
        try:
            view, seq, ts = consumer.read_latest()
            assert seq == 99 and int(view[0, 0, 0]) == 7
        finally:
            consumer.close(unlink=False)
    finally:
        producer.close(unlink=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_ring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.runtime.shm.frame_ring'`

- [ ] **Step 3: Write `edgecv/runtime/shm/frame_ring.py`**

```python
"""Zero-copy, latest-only frame ring (ARCHITECTURE.md §7.1).

N fixed-size slots sized for the max supported resolution. The single producer
writes the next slot then publishes the control word under a seqlock. Consumers
read a zero-copy numpy view of the newest slot; a consumer that fell behind jumps
to the newest seq rather than draining (latest-only). Triple-or-more buffering
plus latest-only reads handle slot recycling without refcounts.
"""

from __future__ import annotations

import ctypes
from multiprocessing import shared_memory

import numpy as np

from edgecv.runtime.shm.seqlock import SeqLock
from edgecv.runtime.shm.structs import (
    ABI_VERSION,
    MAGIC,
    FrameControl,
    code_to_dtype,
    dtype_to_code,
    validate_header,
)

_CONTROL_SIZE = ctypes.sizeof(FrameControl)
# seqlock word for the control struct is the FrameControl.seqlock field; locate it.
_SEQLOCK_OFFSET = FrameControl.seqlock.offset


class FrameRing:
    def __init__(self, shm: shared_memory.SharedMemory, slots: int, max_h: int,
                 max_w: int, max_c: int, dtype: str, owner: bool):
        self._shm = shm
        self._slots = slots
        self._max_h = max_h
        self._max_w = max_w
        self._max_c = max_c
        self._dtype = np.dtype(dtype)
        self._owner = owner
        self._slot_bytes = max_h * max_w * max_c * self._dtype.itemsize
        self._data_offset = _CONTROL_SIZE
        self._control = FrameControl.from_buffer(shm.buf, 0)
        self._seqlock = SeqLock(shm.buf, _SEQLOCK_OFFSET)
        self._write_count = 0
        if owner:
            self._control.magic = MAGIC
            self._control.abi_version = ABI_VERSION
            self._control.seq = 0
            self._control.seqlock = 0
            self._control.slot = 0
        else:
            validate_header(self._control.magic, self._control.abi_version)

    @property
    def name(self) -> str:
        return self._shm.name

    @staticmethod
    def _size(slots: int, max_h: int, max_w: int, max_c: int, dtype: str) -> int:
        slot_bytes = max_h * max_w * max_c * np.dtype(dtype).itemsize
        return _CONTROL_SIZE + slots * slot_bytes

    @classmethod
    def create(cls, slots: int, max_h: int, max_w: int, max_c: int,
               dtype: str = "uint8", name: str | None = None) -> FrameRing:
        size = cls._size(slots, max_h, max_w, max_c, dtype)
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        return cls(shm, slots, max_h, max_w, max_c, dtype, owner=True)

    @classmethod
    def attach(cls, name: str, slots: int, max_h: int, max_w: int, max_c: int,
               dtype: str = "uint8") -> FrameRing:
        shm = shared_memory.SharedMemory(name=name)
        return cls(shm, slots, max_h, max_w, max_c, dtype, owner=False)

    def _slot_view(self, slot: int, h: int, w: int, c: int) -> np.ndarray:
        offset = self._data_offset + slot * self._slot_bytes
        return np.ndarray((h, w, c), dtype=self._dtype,
                          buffer=self._shm.buf, offset=offset)

    def publish(self, frame: np.ndarray, seq: int, timestamp: float) -> None:
        if frame.dtype != self._dtype:
            raise ValueError(f"frame dtype {frame.dtype} != ring dtype {self._dtype}")
        h, w = frame.shape[0], frame.shape[1]
        c = frame.shape[2] if frame.ndim == 3 else 1
        if h > self._max_h or w > self._max_w or c > self._max_c:
            raise ValueError(f"frame {frame.shape} exceeds ring capacity "
                             f"({self._max_h},{self._max_w},{self._max_c})")
        slot = self._write_count % self._slots
        dst = self._slot_view(slot, h, w, c)
        dst[...] = frame.reshape(h, w, c)
        self._seqlock.write_begin()
        self._control.slot = slot
        self._control.seq = seq
        self._control.timestamp = timestamp
        self._control.h = h
        self._control.w = w
        self._control.c = c
        self._control.dtype_code = dtype_to_code(self._dtype)
        self._seqlock.write_end()
        self._write_count += 1

    def read_latest(self) -> tuple[np.ndarray, int, float] | None:
        def snapshot():
            return (int(self._control.seq), int(self._control.slot),
                    float(self._control.timestamp), int(self._control.h),
                    int(self._control.w), int(self._control.c))
        seq, slot, ts, h, w, c = self._seqlock.read(snapshot)
        if seq == 0:
            return None
        view = self._slot_view(slot, h, w, c).copy()  # decouple from later writers
        return view, seq, ts

    def close(self, unlink: bool) -> None:
        # Drop ctypes views into the buffer before closing it.
        del self._control
        self._shm.close()
        if unlink and self._owner:
            self._shm.unlink()
```

> Note on `read_latest` returning a `.copy()`: the architecture calls for zero-copy
> *views*, but a copy at the read boundary keeps the test and the inline consumer safe
> from a producer overwriting the slot mid-use. The zero-copy fast path (returning the
> raw view) lands with the hybrid hot loop, where preallocation and latest-only timing
> are managed deliberately (ARCHITECTURE.md §14.9). Keep the copy for the foundation.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_ring.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add edgecv/runtime/shm/frame_ring.py tests/test_frame_ring.py
git commit -m "feat(runtime): zero-copy latest-only frame ring"
```

---

## Task 10: runtime/shm/payload

**Files:**
- Create: `edgecv/runtime/shm/payload.py`
- Test: `tests/test_payload.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_payload.py
import numpy as np
import pytest

from edgecv.runtime.shm.payload import PayloadChannel


def test_try_read_before_publish_is_none():
    ch = PayloadChannel.create(capacity_bytes=64 * 1024, max_arrays=8)
    try:
        assert ch.try_read() is None
    finally:
        ch.close(unlink=True)


def test_variable_shape_roundtrip():
    ch = PayloadChannel.create(capacity_bytes=256 * 1024, max_arrays=8)
    try:
        arrays = {
            "boxes": np.array([[0.1, 0.2, 0.3, 0.4]], np.float32),
            "scores": np.array([0.9], np.float32),
            "H": (np.random.rand(13, 21) + 1j * np.random.rand(13, 21)).astype(np.complex64),
        }
        ch.publish(arrays, seq=5)
        out = ch.try_read()
        assert out is not None
        seq, got = out
        assert seq == 5
        assert set(got) == set(arrays)
        for k in arrays:
            np.testing.assert_array_equal(got[k], arrays[k])
    finally:
        ch.close(unlink=True)


def test_latest_publish_wins():
    ch = PayloadChannel.create(capacity_bytes=64 * 1024, max_arrays=4)
    try:
        ch.publish({"a": np.array([1], np.int32)}, seq=1)
        ch.publish({"a": np.array([2], np.int32)}, seq=2)
        seq, got = ch.try_read()
        assert seq == 2 and int(got["a"][0]) == 2
    finally:
        ch.close(unlink=True)


def test_capacity_overflow_raises():
    ch = PayloadChannel.create(capacity_bytes=128, max_arrays=2)
    try:
        with pytest.raises(ValueError):
            ch.publish({"big": np.zeros(10_000, np.float64)}, seq=1)
    finally:
        ch.close(unlink=True)


def test_attach_reads_other_handle():
    a = PayloadChannel.create(capacity_bytes=64 * 1024, max_arrays=4)
    try:
        a.publish({"x": np.array([3.0], np.float32)}, seq=7)
        b = PayloadChannel.attach(a.name, capacity_bytes=64 * 1024, max_arrays=4)
        try:
            seq, got = b.try_read()
            assert seq == 7 and got["x"][0] == 3.0
        finally:
            b.close(unlink=False)
    finally:
        a.close(unlink=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_payload.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.runtime.shm.payload'`

- [ ] **Step 3: Write `edgecv/runtime/shm/payload.py`**

```python
"""Variable-shape numpy payload channel (ARCHITECTURE.md §7.2).

Carries detector boxes+scores AND the candidate FilterState (complex arrays whose
shape depends on ROI size). Layout: a fixed header (magic, abi, seqlock, seq,
n_arrays) + a fixed-size array-descriptor table + a max-size data region. All
reads go through the seqlock; a fixed ctypes struct alone is insufficient because
the filter is variable-shape.
"""

from __future__ import annotations

import ctypes
from multiprocessing import shared_memory

import numpy as np

from edgecv.runtime.shm.seqlock import SeqLock
from edgecv.runtime.shm.structs import (
    ABI_VERSION,
    MAGIC,
    code_to_dtype,
    dtype_to_code,
    validate_header,
)

_MAX_NAME = 24
_MAX_NDIM = 6


class _Header(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("seqlock", ctypes.c_uint64),
        ("seq", ctypes.c_uint64),
        ("n_arrays", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
    ]


class _ArrayDesc(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * _MAX_NAME),
        ("dtype_code", ctypes.c_uint32),
        ("ndim", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * _MAX_NDIM),
        ("offset", ctypes.c_uint64),   # bytes from start of data region
        ("nbytes", ctypes.c_uint64),
    ]


_HEADER_SIZE = ctypes.sizeof(_Header)
_DESC_SIZE = ctypes.sizeof(_ArrayDesc)
_SEQLOCK_OFFSET = _Header.seqlock.offset


class PayloadChannel:
    def __init__(self, shm, capacity_bytes: int, max_arrays: int, owner: bool):
        self._shm = shm
        self._capacity = capacity_bytes
        self._max_arrays = max_arrays
        self._owner = owner
        self._header = _Header.from_buffer(shm.buf, 0)
        self._desc_offset = _HEADER_SIZE
        self._data_offset = _HEADER_SIZE + max_arrays * _DESC_SIZE
        self._seqlock = SeqLock(shm.buf, _SEQLOCK_OFFSET)
        if owner:
            self._header.magic = MAGIC
            self._header.abi_version = ABI_VERSION
            self._header.seqlock = 0
            self._header.seq = 0
            self._header.n_arrays = 0
        else:
            validate_header(self._header.magic, self._header.abi_version)

    @property
    def name(self) -> str:
        return self._shm.name

    @staticmethod
    def _size(capacity_bytes: int, max_arrays: int) -> int:
        return _HEADER_SIZE + max_arrays * _DESC_SIZE + capacity_bytes

    @classmethod
    def create(cls, capacity_bytes: int, max_arrays: int = 8,
               name: str | None = None) -> PayloadChannel:
        shm = shared_memory.SharedMemory(
            create=True, size=cls._size(capacity_bytes, max_arrays), name=name)
        return cls(shm, capacity_bytes, max_arrays, owner=True)

    @classmethod
    def attach(cls, name: str, capacity_bytes: int, max_arrays: int = 8) -> PayloadChannel:
        shm = shared_memory.SharedMemory(name=name)
        return cls(shm, capacity_bytes, max_arrays, owner=False)

    def _desc(self, i: int) -> _ArrayDesc:
        return _ArrayDesc.from_buffer(self._shm.buf, self._desc_offset + i * _DESC_SIZE)

    def publish(self, arrays: dict[str, np.ndarray], seq: int) -> None:
        if len(arrays) > self._max_arrays:
            raise ValueError(f"{len(arrays)} arrays exceeds max_arrays={self._max_arrays}")
        # Pre-plan layout and check capacity before touching the seqlock.
        plan = []
        cursor = 0
        for name, arr in arrays.items():
            if len(name.encode()) >= _MAX_NAME:
                raise ValueError(f"array name too long: {name!r}")
            if arr.ndim > _MAX_NDIM:
                raise ValueError(f"array {name!r} ndim {arr.ndim} > {_MAX_NDIM}")
            arr = np.ascontiguousarray(arr)
            nbytes = arr.nbytes
            if cursor + nbytes > self._capacity:
                raise ValueError("payload exceeds channel capacity")
            plan.append((name, arr, cursor, nbytes))
            cursor += nbytes

        self._seqlock.write_begin()
        self._header.n_arrays = len(plan)
        for i, (name, arr, off, nbytes) in enumerate(plan):
            desc = self._desc(i)
            desc.name = name.encode()
            desc.dtype_code = dtype_to_code(arr.dtype)
            desc.ndim = arr.ndim
            for d in range(_MAX_NDIM):
                desc.shape[d] = arr.shape[d] if d < arr.ndim else 0
            desc.offset = off
            desc.nbytes = nbytes
            dst = np.ndarray((nbytes,), dtype=np.uint8, buffer=self._shm.buf,
                             offset=self._data_offset + off)
            dst[...] = arr.view(np.uint8).reshape(-1)
        self._header.seq = seq
        self._seqlock.write_end()

    def try_read(self) -> tuple[int, dict[str, np.ndarray]] | None:
        def snapshot():
            seq = int(self._header.seq)
            n = int(self._header.n_arrays)
            out: dict[str, np.ndarray] = {}
            for i in range(n):
                desc = self._desc(i)
                name = bytes(desc.name).rstrip(b"\x00").decode()
                dtype = code_to_dtype(desc.dtype_code)
                shape = tuple(int(desc.shape[d]) for d in range(int(desc.ndim)))
                raw = np.ndarray((int(desc.nbytes),), dtype=np.uint8,
                                 buffer=self._shm.buf,
                                 offset=self._data_offset + int(desc.offset))
                out[name] = raw.view(dtype).reshape(shape).copy()
            return seq, out
        seq, out = self._seqlock.read(snapshot)
        if seq == 0:
            return None
        return seq, out

    def close(self, unlink: bool) -> None:
        del self._header
        self._shm.close()
        if unlink and self._owner:
            self._shm.unlink()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_payload.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add edgecv/runtime/shm/payload.py tests/test_payload.py
git commit -m "feat(runtime): variable-shape numpy payload channel under seqlock"
```

---

## Task 11: runtime/placement

**Files:**
- Create: `edgecv/runtime/placement.py`
- Test: `tests/test_placement.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_placement.py
import textwrap

from edgecv.runtime.placement import BoardProfile, ProcessPlacement, load_profile, default_profile


def test_load_profile_parses_processes(tmp_path):
    p = tmp_path / "board.yaml"
    p.write_text(textwrap.dedent("""
        board: rk3588
        processes:
          caller:
            cpu_affinity: [4, 5, 6, 7]
            sched: {policy: FIFO, priority: 80}
          detector:
            cpu_affinity: [0, 1]
            npu_core: 0
            backend: rknn
    """))
    prof = load_profile(p)
    assert prof.board == "rk3588"
    assert prof.processes["caller"].cpu_affinity == [4, 5, 6, 7]
    assert prof.processes["caller"].sched == {"policy": "FIFO", "priority": 80}
    assert prof.processes["detector"].npu_core == 0
    assert prof.processes["detector"].backend == "rknn"


def test_default_profile_is_rk3588_and_has_caller_and_detector():
    prof = default_profile()
    assert prof.board == "rk3588"
    assert "caller" in prof.processes
    assert "detector" in prof.processes


def test_apply_affinity_is_safe_noop_when_unsupported(monkeypatch):
    # If os.sched_setaffinity is missing/raises, apply() must not crash the process.
    placement = ProcessPlacement(cpu_affinity=[0], sched=None, npu_core=None, backend=None)
    import edgecv.runtime.placement as mod
    monkeypatch.setattr(mod.os, "sched_setaffinity",
                        lambda pid, mask: (_ for _ in ()).throw(OSError("nope")),
                        raising=False)
    placement.apply()  # should swallow the error and return
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_placement.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.runtime.placement'`

- [ ] **Step 3: Write `edgecv/runtime/placement.py`**

```python
"""Declarative process->hardware placement (ARCHITECTURE.md §7.6).

A board profile maps each process to CPU affinity, optional SCHED_FIFO, an NPU
core, and a backend. Shipped defaults for rk3588; fully user-overridable. No
placement is ever hardcoded in a tracker. Applying affinity/sched is best-effort:
it needs privileges (CAP_SYS_NICE) that may be absent in CI, so failures are
swallowed with a warning rather than raised.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger("edgecv.placement")

# Shipped default board profile (also available as packaged YAML in models/profiles).
_DEFAULT_RK3588 = {
    "board": "rk3588",
    "processes": {
        "caller": {
            "cpu_affinity": [4, 5, 6, 7],
            "sched": {"policy": "FIFO", "priority": 80},
        },
        "detector": {
            "cpu_affinity": [0, 1],
            "npu_core": 0,
            "backend": "rknn",
        },
    },
}


@dataclass
class ProcessPlacement:
    cpu_affinity: list[int] | None = None
    sched: dict | None = None
    npu_core: int | None = None
    backend: str | None = None

    def apply(self) -> None:
        """Best-effort: pin affinity and (optionally) set SCHED_FIFO for this process."""
        if self.cpu_affinity:
            try:
                os.sched_setaffinity(0, set(self.cpu_affinity))
            except Exception as e:  # missing on non-Linux, or EPERM in CI
                log.warning("could not set CPU affinity %s: %s", self.cpu_affinity, e)
        if self.sched and self.sched.get("policy") == "FIFO":
            try:
                param = os.sched_param(int(self.sched.get("priority", 1)))
                os.sched_setscheduler(0, os.SCHED_FIFO, param)
            except Exception as e:
                log.warning("could not set SCHED_FIFO: %s", e)


@dataclass
class BoardProfile:
    board: str
    processes: dict[str, ProcessPlacement] = field(default_factory=dict)


def _from_dict(data: dict) -> BoardProfile:
    procs = {
        name: ProcessPlacement(
            cpu_affinity=spec.get("cpu_affinity"),
            sched=spec.get("sched"),
            npu_core=spec.get("npu_core"),
            backend=spec.get("backend"),
        )
        for name, spec in (data.get("processes") or {}).items()
    }
    return BoardProfile(board=data.get("board", "unknown"), processes=procs)


def load_profile(path: str | os.PathLike) -> BoardProfile:
    data = yaml.safe_load(Path(path).read_text())
    return _from_dict(data)


def default_profile() -> BoardProfile:
    return _from_dict(_DEFAULT_RK3588)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_placement.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add edgecv/runtime/placement.py tests/test_placement.py
git commit -m "feat(runtime): board profile placement (affinity/sched), rk3588 default"
```

---

## Task 12: runtime/worker and orchestrator

**Files:**
- Create: `edgecv/runtime/worker.py`, `edgecv/runtime/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator.py
import time

from edgecv.runtime.orchestrator import Orchestrator, WorkerSpec


def _echo_worker(ctx, started, stop):
    """Module-level so it is picklable under spawn. Sets `started`, waits for stop."""
    started.set()
    while not stop.is_set():
        time.sleep(0.01)


def test_orchestrator_spawns_and_stops_worker():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    started = ctx.Event()
    stop = ctx.Event()
    orch = Orchestrator(mp_context="spawn")
    orch.add_worker(WorkerSpec(name="echo", target=_echo_worker,
                               args=(None, started, stop)))
    with orch:
        orch.start()
        assert started.wait(timeout=10.0), "worker never signalled start"
        assert orch.is_alive("echo")
        stop.set()
    # After context exit, the worker is reaped.
    assert not orch.is_alive("echo")


def test_close_is_idempotent():
    orch = Orchestrator(mp_context="spawn")
    orch.close()
    orch.close()  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.runtime.orchestrator'`

- [ ] **Step 3: Write `edgecv/runtime/worker.py`**

```python
"""Worker child entrypoint helpers (ARCHITECTURE.md §7.4).

Children attach to shared memory (never unlink), detach the resource_tracker for
attached segments to avoid multiprocessing double-unlink warnings, request death
with the parent via PR_SET_PDEATHSIG (Linux), and initialise their backend
in-process (NPU/RKNN contexts do not survive fork and must be created here).
"""

from __future__ import annotations

import ctypes
import logging
import platform

log = logging.getLogger("edgecv.worker")

_PR_SET_PDEATHSIG = 1  # from <sys/prctl.h>


def request_death_with_parent() -> None:
    """Ask the kernel to send SIGTERM to this process when the parent dies."""
    if platform.system() != "Linux":
        return
    try:
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception as e:  # pragma: no cover - platform dependent
        log.warning("PR_SET_PDEATHSIG failed: %s", e)


def detach_resource_tracker(shm_name: str) -> None:
    """Stop this process's resource_tracker from trying to unlink an attached segment.

    Only the orchestrator (owner) unlinks. Children attach only (§7.4 / §14.8)."""
    try:
        from multiprocessing import resource_tracker

        resource_tracker.unregister(f"/{shm_name}", "shared_memory")
    except Exception as e:  # pragma: no cover - internal API drift
        log.debug("resource_tracker.unregister(%s) failed: %s", shm_name, e)


def child_main(target, args: tuple) -> None:
    """Generic child bootstrap: install death signal, then run the worker target."""
    request_death_with_parent()
    target(*args)
```

- [ ] **Step 4: Write `edgecv/runtime/orchestrator.py`**

```python
"""Process-group orchestrator (ARCHITECTURE.md §7.4).

Spawns workers with the 'spawn' (or 'forkserver') start method — never 'fork',
because NPU runtime contexts do not survive fork. Owns shared-memory lifecycle:
the orchestrator creates and unlinks all segments; children only attach. Provides
a heartbeat reaper and deterministic teardown via close()/context manager.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Callable

from edgecv.runtime.worker import child_main

log = logging.getLogger("edgecv.orchestrator")


@dataclass
class WorkerSpec:
    name: str
    target: Callable
    args: tuple = field(default_factory=tuple)


class Orchestrator:
    def __init__(self, mp_context: str = "spawn"):
        if mp_context == "fork":
            raise ValueError("fork is forbidden: NPU contexts do not survive fork (§7.4)")
        self._ctx = mp.get_context(mp_context)
        self._specs: dict[str, WorkerSpec] = {}
        self._procs: dict[str, mp.process.BaseProcess] = {}
        self._closed = False

    def add_worker(self, spec: WorkerSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate worker name: {spec.name}")
        self._specs[spec.name] = spec

    def start(self) -> None:
        for name, spec in self._specs.items():
            if name in self._procs and self._procs[name].is_alive():
                continue
            proc = self._ctx.Process(
                target=child_main, args=(spec.target, spec.args), name=name, daemon=True
            )
            proc.start()
            self._procs[name] = proc
            log.info("started worker %s (pid=%s)", name, proc.pid)

    def is_alive(self, name: str) -> bool:
        proc = self._procs.get(name)
        return bool(proc and proc.is_alive())

    def reap(self, restart: bool = False) -> None:
        """Join finished workers; optionally restart any that died."""
        for name, proc in list(self._procs.items()):
            if not proc.is_alive():
                proc.join(timeout=0)
                log.info("worker %s exited (code=%s)", name, proc.exitcode)
                if restart:
                    self._procs.pop(name)
        if restart:
            self.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name, proc in self._procs.items():
            if proc.is_alive():
                proc.terminate()
        for name, proc in self._procs.items():
            proc.join(timeout=5.0)
            if proc.is_alive():  # pragma: no cover - escalation path
                proc.kill()
                proc.join(timeout=5.0)
            log.info("worker %s reaped", name)
        self._procs.clear()

    def __enter__(self) -> Orchestrator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_orchestrator.py -q`
Expected: PASS (2 passed). May take a few seconds for spawn startup.

- [ ] **Step 6: Commit**

```bash
git add edgecv/runtime/worker.py edgecv/runtime/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(runtime): spawn-based orchestrator and worker bootstrap"
```

---

## Task 13: fusion/policy and predict (ABCs only)

**Files:**
- Create: `edgecv/fusion/policy.py`, `edgecv/fusion/predict.py`
- Test: `tests/test_fusion_abcs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fusion_abcs.py
import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.fusion.policy import DetectorOutput, FusionDecision, FusionPolicy
from edgecv.fusion.predict import MotionPredictor
from edgecv.trackers.cf.base import EvalResult


def test_fusion_policy_is_abstract():
    with pytest.raises(TypeError):
        FusionPolicy()


def test_motion_predictor_is_abstract():
    with pytest.raises(TypeError):
        MotionPredictor()


def test_detector_output_and_decision_construct():
    do = DetectorOutput(boxes=np.zeros((1, 4), np.float32),
                        scores=np.array([0.5], np.float32))
    assert do.boxes.shape == (1, 4)
    dec = FusionDecision(take_candidate=True)
    assert dec.take_candidate is True


def test_concrete_policy_can_be_implemented():
    class KeepIncumbent(FusionPolicy):
        def fuse(self, incumbent, candidate, detector_out):
            return FusionDecision(take_candidate=False)

    er = EvalResult(bbox=BoundingBox(0, 0, 0.1, 0.1),
                    response_map=np.zeros((2, 2)), psr=3.0)
    assert KeepIncumbent().fuse(er, None, None).take_candidate is False


def test_concrete_predictor_can_be_implemented():
    class Hold(MotionPredictor):
        def predict(self, history, dt):
            return history[-1][1]

    bb = BoundingBox(0.2, 0.2, 0.1, 0.1)
    assert Hold().predict([(0.0, bb)], dt=0.03) is bb
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fusion_abcs.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.fusion.policy'`

- [ ] **Step 3: Write `edgecv/fusion/policy.py`**

```python
"""Fusion abstractions (ARCHITECTURE.md §8). The library ships the abstractions
hybrids need, not specific hybrid trackers. The reference PSR-gate policy lands in
a later build."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from edgecv.trackers.cf.base import EvalResult


@dataclass
class DetectorOutput:
    boxes: np.ndarray       # (N, 4), normalised
    scores: np.ndarray      # (N,)
    meta: dict | None = None


@dataclass
class FusionDecision:
    take_candidate: bool


class FusionPolicy(ABC):
    @abstractmethod
    def fuse(self,
             incumbent: EvalResult,
             candidate: EvalResult | None,
             detector_out: DetectorOutput | None) -> FusionDecision: ...
```

- [ ] **Step 4: Write `edgecv/fusion/predict.py`**

```python
"""Motion predictor abstraction (ARCHITECTURE.md §9). Supplies the search window
for set_filter, bridging detection latency at edge frame rates. The
constant-velocity default lands in a later build."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from edgecv.core.bbox import BoundingBox


class MotionPredictor(ABC):
    @abstractmethod
    def predict(self,
                history: Sequence[tuple[float, BoundingBox]],
                dt: float) -> BoundingBox:
        """Predict the box dt seconds after the last history sample."""
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_fusion_abcs.py -q`
Expected: PASS (5 passed). Depends on Task 14 for `EvalResult`; implement Task 14 first or together.

- [ ] **Step 6: Commit**

```bash
git add edgecv/fusion/policy.py edgecv/fusion/predict.py tests/test_fusion_abcs.py
git commit -m "feat(fusion): FusionPolicy and MotionPredictor abstractions"
```

---

## Task 14: trackers/cf/base — transferable-filter contract

**Files:**
- Create: `edgecv/trackers/cf/base.py`
- Test: `tests/test_cf_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cf_base.py
import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.trackers.cf.base import CorrelationFilterTracker, EvalResult, FilterState


def test_filter_state_and_eval_result_construct():
    fs = FilterState(arrays={"H": np.zeros((3, 3), np.complex64)},
                     bbox=BoundingBox(0, 0, 0.1, 0.1), meta={"feature": "raw"})
    assert "H" in fs.arrays
    er = EvalResult(bbox=fs.bbox, response_map=np.zeros((3, 3)), psr=5.0)
    assert er.psr == 5.0


def test_cf_tracker_is_abstract():
    with pytest.raises(TypeError):
        CorrelationFilterTracker()


def test_contract_requires_pure_ops_and_state_access():
    # A subclass missing build_filter must not be instantiable.
    class Incomplete(CorrelationFilterTracker):
        def init(self, frame, bbox): ...
        def update(self, frame): ...
        @property
        def status(self): ...
        def name(self): return "x"
        # missing build_filter/evaluate/get_filter/set_filter/response_map/psr
    with pytest.raises(TypeError):
        Incomplete()


def test_fully_implemented_subclass_instantiates():
    class Ok(CorrelationFilterTracker):
        def init(self, frame, bbox): self._fs = FilterState({}, bbox, {})
        def update(self, frame): ...
        @property
        def status(self): return None
        def name(self): return "Ok"
        def build_filter(self, frame, bbox): return FilterState({}, bbox, {})
        def evaluate(self, frame, state):
            return EvalResult(state.bbox, np.zeros((2, 2)), 1.0)
        def get_filter(self): return self._fs
        def set_filter(self, state, search_box=None): self._fs = state
        @property
        def response_map(self): return np.zeros((2, 2))
        @property
        def psr(self): return 1.0
    t = Ok()
    t.init(np.zeros((4, 4), np.uint8), BoundingBox(0, 0, 0.5, 0.5))
    assert t.name() == "Ok"
    assert t.get_filter().bbox.w == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cf_base.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.trackers.cf.base'`

- [ ] **Step 3: Write `edgecv/trackers/cf/base.py`**

```python
"""Correlation-filter base contract (ARCHITECTURE.md §6.1, §14.5).

Every CF tracker subclasses CorrelationFilterTracker and implements both the
online (mutating) loop AND the pure ops. build_filter/evaluate MUST NOT mutate
self: a worker builds a FilterState in one process and the caller evaluates
incumbent vs candidate on the current frame in another. This purity is what makes
build-elsewhere / evaluate-here / swap safe across processes."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.core.tracker import Tracker


@dataclass
class FilterState:
    """Transferable CF model state (ARCHITECTURE.md §5/§6.1)."""

    arrays: dict[str, np.ndarray]   # e.g. {"H": ..., "A": ..., "B": ...}, arbitrary shapes
    bbox: BoundingBox               # ROI the filter was built for
    meta: dict                      # feature type, window params, scale/aspect, abi tag


@dataclass
class EvalResult:
    bbox: BoundingBox
    response_map: np.ndarray
    psr: float


class CorrelationFilterTracker(Tracker):
    # --- pure ops: MUST NOT mutate self ---
    @abstractmethod
    def build_filter(self, frame: np.ndarray, bbox: BoundingBox) -> FilterState: ...

    @abstractmethod
    def evaluate(self, frame: np.ndarray, state: FilterState) -> EvalResult: ...

    # --- state access ---
    @abstractmethod
    def get_filter(self) -> FilterState: ...

    @abstractmethod
    def set_filter(self, state: FilterState,
                   search_box: BoundingBox | None = None) -> None: ...

    @property
    @abstractmethod
    def response_map(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def psr(self) -> float: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cf_base.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/cf/base.py tests/test_cf_base.py
git commit -m "feat(trackers): mandatory CF transferable-filter contract"
```

---

## Task 15: models/manifest and shipped board profile

**Files:**
- Create: `edgecv/models/manifest.py`, `edgecv/models/profiles/rk3588.yaml`
- Test: `tests/test_manifest.py`

> **Ordering note:** `ModelManifest` is imported by the mock/onnx backends (Tasks 6–7).
> If executing strictly in order, implement this task's `manifest.py` **before** running
> the Task 6/7 test files. The plan lists it here to keep the models section together;
> a subagent executing Task 6 should create `manifest.py` first (it is small and shown
> in full below).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
import textwrap

import pytest

from edgecv.models.manifest import ModelManifest, load_manifest


def test_load_manifest_parses_yaml(tmp_path):
    p = tmp_path / "siamfc.yaml"
    p.write_text(textwrap.dedent("""
        name: siamfc_generic
        task: sot_template_matching
        preprocessing: {color: gray, exemplar: 127, search: 255}
        io:
          inputs:
            - {name: exemplar, shape: [1, 1, 127, 127], dtype: float32}
            - {name: search, shape: [1, 1, 255, 255], dtype: float32}
          outputs:
            - {name: score_map, shape: [1, 1, 17, 17], dtype: float32}
        artifacts:
          onnx: {path: siamfc_generic.onnx}
          rknn: {path: siamfc_generic.rk3588.rknn, quant: int8}
    """))
    man = load_manifest(p)
    assert man.name == "siamfc_generic"
    assert man.task == "sot_template_matching"
    assert man.inputs[0]["name"] == "exemplar"
    assert man.outputs[0]["shape"] == [1, 1, 17, 17]
    assert man.artifacts["rknn"]["quant"] == "int8"
    assert man.preprocessing["color"] == "gray"


def test_manifest_requires_name_and_task(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("io: {inputs: [], outputs: []}\nartifacts: {}\n")
    with pytest.raises(ValueError):
        load_manifest(p)


def test_artifact_for_backend_helper():
    man = ModelManifest(name="m", task="t", preprocessing={},
                        inputs=[], outputs=[],
                        artifacts={"onnx": {"path": "m.onnx"}})
    assert man.artifact_for("onnx") == {"path": "m.onnx"}
    assert man.artifact_for("rknn") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_manifest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.models.manifest'`

- [ ] **Step 3: Write `edgecv/models/manifest.py`**

```python
"""Model manifest schema + loader (ARCHITECTURE.md §10.1).

A manifest maps one logical model to per-backend artifacts plus preprocessing and
an I/O spec. Trackers depend on the manifest, never on a vendor artifact file.
I/O entries are kept as plain dicts here; backends translate them into TensorSpec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelManifest:
    name: str
    task: str
    preprocessing: dict = field(default_factory=dict)
    inputs: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    artifacts: dict[str, dict] = field(default_factory=dict)

    def artifact_for(self, backend: str) -> dict | None:
        return self.artifacts.get(backend)


def load_manifest(path: str | Path) -> ModelManifest:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} is not a mapping")
    name = data.get("name")
    task = data.get("task")
    if not name or not task:
        raise ValueError(f"manifest {path} must define 'name' and 'task'")
    io = data.get("io") or {}
    return ModelManifest(
        name=name,
        task=task,
        preprocessing=data.get("preprocessing") or {},
        inputs=io.get("inputs") or [],
        outputs=io.get("outputs") or [],
        artifacts=data.get("artifacts") or {},
    )
```

- [ ] **Step 4: Write `edgecv/models/profiles/rk3588.yaml`**

```yaml
# Shipped default board profile for RK3588 (ARCHITECTURE.md §7.6).
# User-overridable: pass your own YAML to edgecv.runtime.placement.load_profile.
board: rk3588
processes:
  caller:                       # CF + fusion run inline here; sets output rate
    cpu_affinity: [4, 5, 6, 7]            # A76 big cores
    sched: {policy: FIFO, priority: 80}   # optional; needs CAP_SYS_NICE
  detector:                     # the async NPU worker
    cpu_affinity: [0, 1]                  # A55 little cores for pre/post-processing
    npu_core: 0                           # one of the RK3588's three NPU cores
    backend: rknn
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_manifest.py -q`
Expected: PASS (3 passed)

- [ ] **Step 6: Run the previously backend-dependent tests now that the manifest exists**

Run: `pytest tests/test_mock_backend.py tests/test_onnx_backend.py tests/test_registry.py -q`
Expected: PASS (onnx tests skip if onnx/onnxruntime missing)

- [ ] **Step 7: Commit**

```bash
git add edgecv/models/manifest.py edgecv/models/profiles/rk3588.yaml tests/test_manifest.py
git commit -m "feat(models): manifest schema/loader and shipped rk3588 profile"
```

---

## Task 16: CI workflow and full-suite green

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Run the entire test suite locally**

Run:
```bash
. .venv/bin/activate
pytest -q
```
Expected: all tests pass (onnx tests skip only if onnxruntime absent; install it with `pip install -e .[onnx,test]` so they run).

- [ ] **Step 2: Run lint and type checks**

Run:
```bash
. .venv/bin/activate
ruff check edgecv tests
mypy edgecv
```
Expected: ruff reports no errors; mypy reports no errors (ignore_missing_imports handles optional deps).

- [ ] **Step 3: Write `.github/workflows/ci.yml`**

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e .[onnx,test]
      - name: Lint
        run: ruff check edgecv tests
      - name: Type-check
        run: mypy edgecv
      - name: Test
        run: pytest -q
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: x86 GitHub Actions running ruff, mypy, and pytest"
```

- [ ] **Step 5: Final verification**

Run:
```bash
. .venv/bin/activate
pytest -q && ruff check edgecv tests && mypy edgecv && echo "FOUNDATION GREEN"
```
Expected: prints `FOUNDATION GREEN`

---

## Self-Review

**Spec coverage** (against `2026-05-31-edgecv-foundation-design.md`):
- §3.1 Packaging → Task 1 (pyproject, extras, entry points, .gitignore, README); CI → Task 16. LICENSE: pyproject declares Apache-2.0; no `LICENSE` file task — **flagged for user**, see Open items.
- §3.2 core/ → Tasks 2–4 (bbox, result, tracker). ✓
- §3.3 backends/ → Task 5 (base+registry), Task 6 (mock), Task 7 (onnx+rknn). ✓
- §3.4 runtime/ → Task 8 (structs+seqlock), 9 (frame_ring), 10 (payload), 11 (placement), 12 (orchestrator+worker). ✓
- §3.5 fusion ABCs → Task 13. ✓ (no reference impl — matches "abstractions only")
- §3.6 trackers contract+skeleton → Task 14 (cf/base) + skeleton dirs from Task 1. ✓ (no concrete trackers, no ops)
- §3.7 models/ → Task 15 (manifest + rk3588 profile). ✓
- §3.8 tools/ README → Task 1 Step 6. ✓
- §3.9 tests → one test task per module. ✓
- §4 invariants → seqlock (Task 8), latest-only (Task 9), ABI magic/version (Task 8 structs, validated on attach in 9/10), pure CF ops contract (Task 14), spawn-not-fork + backend-in-child (Task 12/worker), centralised SHM ownership (owner flag in 9/10, orchestrator in 12), seq/timestamp travel with frame/payload (9/10). ✓

**Placeholder scan:** No "TBD"/"implement later" in code steps. The rknn `load` raising `NotImplementedError` is an intentional, spec-described lazy adapter behaviour (§3.3), not a plan placeholder. The frame-ring `.copy()` and the rknn concrete wiring are explicitly deferred with rationale tied to architecture sections.

**Type consistency:** `ModelManifest(name, task, preprocessing, inputs, outputs, artifacts)` is constructed identically in Tasks 6, 7, 15. `TensorSpec(name, shape, dtype, layout, quant)` defined in Task 5, consumed in 6/7. `EvalResult(bbox, response_map, psr)` and `FilterState(arrays, bbox, meta)` defined in Task 14, consumed in Task 13. `FusionDecision(take_candidate)` consistent in 13. `SeqLock(buf, offset)` / `.write_begin()/.write_end()/.read(fn)` consistent across 8/9/10. `FrameRing.create/attach/publish/read_latest/close(unlink=)` consistent in Task 9. `PayloadChannel.create/attach/publish/try_read/close(unlink=)` consistent in Task 10.

**Cross-task ordering caveat (important for the executor):** `manifest.py` (Task 15) is a dependency of Tasks 6–7. A subagent doing Task 6 must create `edgecv/models/manifest.py` (full code in Task 15 Step 3) first, or the executor should run Task 15 before Tasks 6–7. Similarly Task 13 depends on Task 14's `EvalResult`. These are noted inline in the affected steps.

**Open items (non-blocking, from spec §6):** add a real `LICENSE` file (user choice — pyproject currently declares Apache-2.0); model-artifact packaging deferred to first NN tracker; seqlock C-extension revisit only if a correctness issue appears.

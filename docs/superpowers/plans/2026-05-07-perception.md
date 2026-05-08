# Perception Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full perception pipeline — camera, KCF, MOSSE, NanoTrack trackers, inference runtimes, modular slot workers — and verify end-to-end IPC message flow with a multiprocess integration test.

**Architecture:** Two bus topic slots (`ccv_tracker/estimate` for classical CV, `ncv_tracker/estimate` for neural CV) are populated by generic slot workers (`CCVTrackerWorker`, `NCVTrackerWorker`) that accept any conforming tracker object. Tracker algorithm files have zero bus/IPC imports; workers own all IPC. A factory dict per slot maps config strings to tracker classes — adding a new tracker means one class + one dict entry.

**Tech Stack:** Python 3.11+, numpy, opencv-contrib-python (KCF + MOSSE + camera capture), Pillow (NanoTrack resize), onnxruntime (CPU/CUDA inference), rknnlite/rknn (on-device inference), pytest + multiprocessing for integration tests.

---

## File Map

```
# Modified
src/quadguide/core/bus.py                           rename two TOPICS entries
src/quadguide/core/config.py                        add ccv/ncv fields + MOSSEConfig
configs/config.yaml                                 add tracker.ccv, tracker.ncv, tracker.mosse
tests/unit/test_bus.py                              update expected topic names
tests/unit/test_config.py                           add ccv/ncv assertions

# New — inference/
src/quadguide/inference/__init__.py
src/quadguide/inference/base.py                     NPURuntime Protocol
src/quadguide/inference/onnx_cpu.py                 OnnxCPURuntime
src/quadguide/inference/onnx_cuda.py                OnnxCUDARuntime
src/quadguide/inference/rknn.py                     RKNNRuntime
src/quadguide/inference/factory.py                  get_runtime()

# New — perception/
src/quadguide/perception/__init__.py
src/quadguide/perception/ccv_tracker_worker.py      CCVTrackerWorker (IPC loop)
src/quadguide/perception/ncv_tracker_worker.py      NCVTrackerWorker (IPC loop)
src/quadguide/perception/tracker_factories.py       CCV_TRACKERS, NCV_TRACKERS dicts

# New — camera/
src/quadguide/perception/camera/__init__.py
src/quadguide/perception/camera/sources.py          CameraSource ABC + implementations
src/quadguide/perception/camera/worker.py           run() entry point

# New — kcf/
src/quadguide/perception/kcf/__init__.py
src/quadguide/perception/kcf/tracker.py             KCFTracker
src/quadguide/perception/kcf/worker.py              run() entry point

# New — mosse/
src/quadguide/perception/mosse/__init__.py
src/quadguide/perception/mosse/tracker.py           MOSSETracker
src/quadguide/perception/mosse/worker.py            run() entry point

# New — nanotrack/
src/quadguide/perception/nanotrack/__init__.py
src/quadguide/perception/nanotrack/preprocess.py    crop + normalise functions
src/quadguide/perception/nanotrack/postprocess.py   decode_response()
src/quadguide/perception/nanotrack/tracker.py       NanoTracker
src/quadguide/perception/nanotrack/worker.py        run() entry point

# New — tests
tests/unit/test_inference.py
tests/unit/test_kcf_tracker.py
tests/unit/test_mosse_tracker.py
tests/unit/test_nanotrack_preprocess.py
tests/unit/test_nanotrack_postprocess.py
tests/unit/test_nanotrack_tracker.py
tests/integration/test_perception_pipeline.py
```

---

## Task 1: Bus topic rename

**Files:**
- Modify: `src/quadguide/core/bus.py`
- Modify: `tests/unit/test_bus.py`

- [ ] **Step 1: Update the failing assertion first — change the expected set in test_bus.py**

In `tests/unit/test_bus.py`, find `TestTopicRegistry.test_all_nine_topics_registered` and update the expected set:

```python
def test_all_nine_topics_registered(self):
    b = Bus(ring_depth=2)
    try:
        expected = {
            "ccv_tracker/estimate", "ncv_tracker/estimate", "target/estimate",
            "fc/attitude", "fc/imu", "guidance/accel",
            "control/cmd", "lockon/cmd", "system/health",
        }
        assert set(b._topics.keys()) == expected
    finally:
        b.close()
```

- [ ] **Step 2: Run the test to confirm it fails with the old names**

```bash
cd /home/plas/quadguide && .venv/bin/python -m pytest tests/unit/test_bus.py::TestTopicRegistry -v
```

Expected: FAIL — `assert {'kcf/estimate', 'nano/estimate', ...} == {'ccv_tracker/estimate', ...}`

- [ ] **Step 3: Rename the two entries in bus.py TOPICS**

In `src/quadguide/core/bus.py`, replace the TOPICS dict entirely:

```python
TOPICS: dict[str, tuple[type, str]] = {
    "ccv_tracker/estimate": (TrackerEstimate, FMT_TRACKER_ESTIMATE),
    "ncv_tracker/estimate": (TrackerEstimate, FMT_TRACKER_ESTIMATE),
    "target/estimate":      (TargetEstimate,  FMT_TARGET_ESTIMATE),
    "fc/attitude":          (AttitudeState,   FMT_ATTITUDE_STATE),
    "fc/imu":               (IMUFrame,        FMT_IMU_FRAME),
    "guidance/accel":       (AccelCmd,        FMT_ACCEL_CMD),
    "control/cmd":          (ControlCmd,      FMT_CONTROL_CMD),
    "lockon/cmd":           (LockOnCmd,       FMT_LOCKON_CMD),
    "system/health":        (HealthReport,    FMT_HEALTH_REPORT),
}
```

- [ ] **Step 4: Run the full bus test suite**

```bash
.venv/bin/python -m pytest tests/unit/test_bus.py -v
```

Expected: all 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/core/bus.py tests/unit/test_bus.py
git commit -m "feat: rename bus topics to ccv_tracker/estimate and ncv_tracker/estimate"
```

---

## Task 2: Config extension (ccv/ncv + MOSSE)

**Files:**
- Modify: `src/quadguide/core/config.py`
- Modify: `configs/config.yaml`
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Add MOSSEConfig and extend TrackerConfig in config.py**

In `src/quadguide/core/config.py`, add after the existing `KCFConfig` dataclass:

```python
@dataclass(frozen=True)
class MOSSEConfig:
    pass  # OpenCV MOSSE exposes no tunable parameters
```

Replace the existing `TrackerConfig` dataclass with:

```python
@dataclass(frozen=True)
class TrackerConfig:
    ccv: str          # "kcf" | "mosse"
    ncv: str          # "nanotrack"
    kcf: KCFConfig
    nanotrack: NanotrackConfig
    fusion: FusionConfig
    mosse: MOSSEConfig = field(default_factory=MOSSEConfig)
```

Add `from dataclasses import dataclass, field` (replace the existing `from dataclasses import dataclass` import).

- [ ] **Step 2: Update cfg_tracker() to read ccv and ncv fields**

Replace the existing `cfg_tracker` function body:

```python
def cfg_tracker(d: dict) -> TrackerConfig:
    t = d["tracker"]
    return TrackerConfig(
        ccv=t["ccv"],
        ncv=t["ncv"],
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
        mosse=MOSSEConfig(),
    )
```

- [ ] **Step 3: Add ccv, ncv, and mosse to configs/config.yaml under tracker:**

```yaml
tracker:
  ccv: kcf            # "kcf" | "mosse"
  ncv: nanotrack      # "nanotrack"
  mosse: {}           # no tunable params
  kcf:
    detect_thresh: 0.5
    sigma: 0.2
    lambda_: 0.0001
  # ... rest unchanged
```

The existing kcf/nanotrack/fusion keys are unchanged; just add the three new lines at the top of the tracker block.

- [ ] **Step 4: Write the failing tests**

In `tests/unit/test_config.py`, add inside `class TestAccessors`:

```python
def test_cfg_tracker_ccv_field(self):
    config = load_config(CONFIG_PATH, {})
    tracker = cfg_tracker(config)
    assert tracker.ccv == "kcf"

def test_cfg_tracker_ncv_field(self):
    config = load_config(CONFIG_PATH, {})
    tracker = cfg_tracker(config)
    assert tracker.ncv == "nanotrack"

def test_cfg_tracker_mosse_is_mosse_config(self):
    from quadguide.core.config import MOSSEConfig
    config = load_config(CONFIG_PATH, {})
    tracker = cfg_tracker(config)
    assert isinstance(tracker.mosse, MOSSEConfig)
```

- [ ] **Step 5: Run the full config test suite**

```bash
.venv/bin/python -m pytest tests/unit/test_config.py -v
```

Expected: all tests PASS including the three new ones.

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/core/config.py configs/config.yaml tests/unit/test_config.py
git commit -m "feat: add ccv/ncv tracker slot fields and MOSSEConfig to TrackerConfig"
```

---

## Task 3: Inference protocol + OnnxCPURuntime + factory

**Files:**
- Create: `src/quadguide/inference/__init__.py`
- Create: `src/quadguide/inference/base.py`
- Create: `src/quadguide/inference/onnx_cpu.py`
- Create: `src/quadguide/inference/factory.py`
- Create: `tests/unit/test_inference.py`

- [ ] **Step 1: Create __init__.py files**

Create empty files:
- `src/quadguide/inference/__init__.py` (empty)

- [ ] **Step 2: Write the tests first**

Create `tests/unit/test_inference.py`:

```python
import pathlib
import pytest
import numpy as np
from quadguide.core.config import load_config
from quadguide.inference.base import NPURuntime
from quadguide.inference.onnx_cpu import OnnxCPURuntime
from quadguide.inference.factory import get_runtime, RUNTIMES

CONFIG_PATH = str(pathlib.Path(__file__).parents[2] / "configs" / "config.yaml")


class TestNPURuntimeProtocol:
    def test_onnx_cpu_has_load(self):
        assert hasattr(OnnxCPURuntime(), "load")

    def test_onnx_cpu_has_infer(self):
        assert hasattr(OnnxCPURuntime(), "infer")

    def test_onnx_cpu_has_close(self):
        assert hasattr(OnnxCPURuntime(), "close")


class TestInferenceFactory:
    def test_runtimes_dict_has_expected_keys(self):
        assert "cpu" in RUNTIMES
        assert "cuda" in RUNTIMES
        assert "rknn" in RUNTIMES

    def test_get_runtime_cpu_returns_onnx_cpu(self):
        config = load_config(CONFIG_PATH, {"platform.inference.device": "cpu"})
        runtime = get_runtime(config)
        assert isinstance(runtime, OnnxCPURuntime)

    def test_get_runtime_unknown_raises(self):
        config = load_config(CONFIG_PATH, {"platform.inference.device": "unknown"})
        with pytest.raises(KeyError):
            get_runtime(config)

    def test_close_is_no_op_for_cpu(self):
        runtime = OnnxCPURuntime()
        runtime.close()  # must not raise
```

- [ ] **Step 3: Run to confirm they fail (ImportError)**

```bash
.venv/bin/python -m pytest tests/unit/test_inference.py -v
```

Expected: ImportError / ModuleNotFoundError on quadguide.inference.*

- [ ] **Step 4: Write inference/base.py**

```python
from __future__ import annotations
from typing import Any, Protocol
import numpy as np

__all__ = ["NPURuntime"]


class NPURuntime(Protocol):
    """Structural protocol for all inference runtimes.

    All NanoTrack inference calls go through this interface. No tracker file
    imports onnxruntime, rknn, or any backend directly.
    """

    def load(self, path: str) -> Any:
        """Load a model file and return a model handle."""
        ...

    def infer(
        self, model: Any, inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Run inference. Returns output tensors keyed by name (ONNX) or
        positional index as string ("0", "1", ...) for RKNN."""
        ...

    def close(self) -> None:
        """Release any hardware resources held by this runtime."""
        ...
```

- [ ] **Step 5: Write inference/onnx_cpu.py**

```python
from __future__ import annotations
from typing import Any
import numpy as np

__all__ = ["OnnxCPURuntime"]


class OnnxCPURuntime:
    """ONNX Runtime with CPU execution. Universal fallback on any platform."""

    def load(self, path: str) -> Any:
        import onnxruntime as ort
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def infer(self, model: Any, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        output_names = [o.name for o in model.get_outputs()]
        results = model.run(output_names, inputs)
        return dict(zip(output_names, results))

    def close(self) -> None:
        pass  # onnxruntime sessions are garbage-collected
```

- [ ] **Step 6: Write inference/factory.py**

```python
from __future__ import annotations
from quadguide.inference.onnx_cpu import OnnxCPURuntime
from quadguide.inference.onnx_cuda import OnnxCUDARuntime
from quadguide.inference.rknn import RKNNRuntime

__all__ = ["RUNTIMES", "get_runtime"]

RUNTIMES = {
    "cpu":  OnnxCPURuntime,
    "cuda": OnnxCUDARuntime,
    "rknn": RKNNRuntime,
}


def get_runtime(config: dict):
    """Return a runtime instance selected by config["platform"]["inference"]["device"].

    Raises KeyError for unknown device strings — that is a configuration error.
    """
    device = config["platform"]["inference"]["device"]
    try:
        cls = RUNTIMES[device]
    except KeyError:
        raise KeyError(
            f"Unknown inference device {device!r}. Valid options: {sorted(RUNTIMES)}"
        )
    return cls()
```

- [ ] **Step 7: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_inference.py -v
```

Expected: all tests PASS (onnx_cpu + factory tests; CUDA and RKNN classes are lazy-imported so no import errors at this stage).

- [ ] **Step 8: Commit**

```bash
git add src/quadguide/inference/ tests/unit/test_inference.py
git commit -m "feat: inference NPURuntime protocol, OnnxCPURuntime, and factory"
```

---

## Task 4: OnnxCUDARuntime + RKNNRuntime

**Files:**
- Create: `src/quadguide/inference/onnx_cuda.py`
- Create: `src/quadguide/inference/rknn.py`

These runtimes use optional hardware. Their backend imports are deferred to `load()` so the module can be imported on any machine without the hardware-specific library installed.

- [ ] **Step 1: Write inference/onnx_cuda.py**

```python
from __future__ import annotations
from typing import Any
import numpy as np

__all__ = ["OnnxCUDARuntime"]


class OnnxCUDARuntime:
    """ONNX Runtime with CUDA execution. Requires onnxruntime-gpu and a CUDA GPU."""

    def load(self, path: str) -> Any:
        import onnxruntime as ort
        return ort.InferenceSession(
            path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def infer(self, model: Any, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        output_names = [o.name for o in model.get_outputs()]
        results = model.run(output_names, inputs)
        return dict(zip(output_names, results))

    def close(self) -> None:
        pass
```

- [ ] **Step 2: Write inference/rknn.py**

```python
from __future__ import annotations
from typing import Any
import numpy as np

__all__ = ["RKNNRuntime"]


class RKNNRuntime:
    """RKNN Lite runtime for RK3576/RK3588 NPU.

    On device: uses rknnlite (lightweight, no model conversion capability).
    On x86 sim: falls back to rknn-toolkit2's RKNN class for simulation.
    RKNN outputs are positional; keys in the returned dict are "0", "1", ...
    NanoTrack tracker accesses outputs via list(outputs.values()) in model order.
    """

    def load(self, path: str) -> Any:
        try:
            from rknnlite.api import RKNNLite
            rknn = RKNNLite()
        except ImportError:
            from rknn.api import RKNN
            rknn = RKNN()
        ret = rknn.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f"RKNNRuntime: load_rknn({path!r}) failed with code {ret}")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNNRuntime: init_runtime() failed with code {ret}")
        return rknn

    def infer(self, model: Any, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        input_list = list(inputs.values())
        outputs = model.inference(inputs=input_list)
        return {str(i): out for i, out in enumerate(outputs)}

    def close(self) -> None:
        pass  # RKNN handle released by GC; explicit release via model.release() if needed
```

- [ ] **Step 3: Verify factory imports cleanly (no hardware required)**

```bash
.venv/bin/python -c "from quadguide.inference.factory import RUNTIMES; print(list(RUNTIMES))"
```

Expected: `['cpu', 'cuda', 'rknn']`

- [ ] **Step 4: Commit**

```bash
git add src/quadguide/inference/onnx_cuda.py src/quadguide/inference/rknn.py
git commit -m "feat: OnnxCUDARuntime and RKNNRuntime (lazy hardware imports)"
```

---

## Task 5: Camera sources

**Files:**
- Create: `src/quadguide/perception/__init__.py`
- Create: `src/quadguide/perception/camera/__init__.py`
- Create: `src/quadguide/perception/camera/sources.py`
- Create: `tests/unit/test_camera_sources.py`

- [ ] **Step 1: Create __init__.py files**

Create these two empty files:
- `src/quadguide/perception/__init__.py`
- `src/quadguide/perception/camera/__init__.py`

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_camera_sources.py`:

```python
import numpy as np
import pytest
from quadguide.perception.camera.sources import CameraSource, USBCamera, CSICamera, VirtualCamera


class TestCameraSourceABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            CameraSource()

    def test_usb_camera_is_camera_source(self):
        assert issubclass(USBCamera, CameraSource)

    def test_csi_camera_is_camera_source(self):
        assert issubclass(CSICamera, CameraSource)

    def test_virtual_camera_is_camera_source(self):
        assert issubclass(VirtualCamera, CameraSource)


class TestVirtualCameraStub:
    def test_open_raises_not_implemented(self):
        cam = VirtualCamera()
        with pytest.raises(NotImplementedError):
            cam.open()

    def test_read_raises_not_implemented(self):
        cam = VirtualCamera()
        with pytest.raises(NotImplementedError):
            cam.read()
```

- [ ] **Step 3: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_camera_sources.py -v
```

Expected: ImportError on `quadguide.perception.camera.sources`

- [ ] **Step 4: Write perception/camera/sources.py**

```python
from __future__ import annotations
import abc
import time
import numpy as np

__all__ = ["CameraSource", "USBCamera", "CSICamera", "VirtualCamera"]


class CameraSource(abc.ABC):
    """Abstract base for all camera input sources."""

    @abc.abstractmethod
    def open(self) -> None:
        """Open the camera device or pipeline."""

    @abc.abstractmethod
    def read(self) -> tuple[np.ndarray, int]:
        """Return (frame_bgr, timestamp_ns). Blocks until a frame is available."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the camera device or pipeline."""

    def __iter__(self):
        while True:
            yield self.read()


class USBCamera(CameraSource):
    """V4L2 USB camera via cv2.VideoCapture."""

    def __init__(self, config) -> None:
        # config is a CameraConfig dataclass or dict-like with width/height/fps
        self._width  = getattr(config, "width",  640)
        self._height = getattr(config, "height", 480)
        self._fps    = getattr(config, "fps",     30)
        self._cap    = None

    def open(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        if not self._cap.isOpened():
            raise RuntimeError("USBCamera: failed to open /dev/video0")

    def read(self) -> tuple[np.ndarray, int]:
        ts = time.monotonic_ns()
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("USBCamera: frame capture failed")
        return frame, ts

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CSICamera(CameraSource):
    """CSI camera via a GStreamer pipeline string, read through cv2.VideoCapture."""

    def __init__(self, config) -> None:
        self._pipeline = getattr(config, "pipeline", "")
        self._cap      = None

    def open(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(f"CSICamera: failed to open pipeline: {self._pipeline!r}")

    def read(self) -> tuple[np.ndarray, int]:
        ts = time.monotonic_ns()
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("CSICamera: frame capture failed")
        return frame, ts

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VirtualCamera(CameraSource):
    """Stub camera for HIL mode.

    # STUB: HIL not yet implemented — see hil/virtual_source.py for the full
    # implementation that renders synthetic frames from the dynamics simulation.
    # Replace this class body when building the hil/ module.
    """

    def open(self) -> None:
        raise NotImplementedError("VirtualCamera: HIL not yet implemented")

    def read(self) -> tuple[np.ndarray, int]:
        raise NotImplementedError("VirtualCamera: HIL not yet implemented")

    def close(self) -> None:
        pass
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_camera_sources.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/perception/__init__.py src/quadguide/perception/camera/ tests/unit/test_camera_sources.py
git commit -m "feat: CameraSource ABC, USBCamera, CSICamera, VirtualCamera stub"
```

---

## Task 6: Camera worker

**Files:**
- Modify: `src/quadguide/perception/camera/worker.py`

The camera worker's `run()` accepts a constructed `CameraSource` so it can be tested with any source without touching config.

- [ ] **Step 1: Write perception/camera/worker.py**

```python
from __future__ import annotations
import signal
import time

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, ProcessState
from quadguide.perception.camera.sources import CameraSource, USBCamera, CSICamera, VirtualCamera

__all__ = ["run", "run_from_config"]

_SOURCES = {"v4l2": USBCamera, "gstreamer": CSICamera, "virtual": VirtualCamera}
_HEALTH_EVERY = 60  # publish health every N frames


def run(source: CameraSource, frame_buffer: FrameBuffer, bus: Bus,
        config: dict | None = None) -> None:
    """Camera worker process entry point.

    Opens source, writes frames into frame_buffer, publishes system/health.
    Runs until SIGTERM sets the stop flag.
    """
    log = setup_logging("camera", config or {})
    stop = False

    def _on_sigterm(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    source.open()
    log.info("camera: source opened")
    i = 0
    try:
        while not stop:
            frame, ts = source.read()
            frame_buffer.write_frame(frame, ts)
            i += 1
            if i % _HEALTH_EVERY == 0:
                bus.publish(
                    "system/health",
                    HealthReport(monotonic_ns(), "camera", ProcessState.OK, ""),
                )
    except Exception as exc:
        log.error(f"camera: fatal error: {exc}")
    finally:
        source.close()
        bus.detach()
        log.info("camera: stopped")


def run_from_config(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    """Construct a CameraSource from config and call run()."""
    from quadguide.core.config import cfg_platform
    pcfg = cfg_platform(config)
    source_cls = _SOURCES[pcfg.camera.backend]
    run(source_cls(pcfg.camera), frame_buffer, bus, config)
```

- [ ] **Step 2: Smoke-import check**

```bash
.venv/bin/python -c "from quadguide.perception.camera.worker import run, run_from_config; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/quadguide/perception/camera/worker.py
git commit -m "feat: camera worker run() and run_from_config()"
```

---

## Task 7: KCF tracker

**Files:**
- Create: `src/quadguide/perception/kcf/__init__.py`
- Modify: `src/quadguide/perception/kcf/tracker.py`
- Create: `tests/unit/test_kcf_tracker.py`

KCF wraps `cv2.TrackerKCF`. `update()` returns health `NO_LOCK` until `init()` is called. After `init()`, OpenCV returns `(success: bool, bbox_px: tuple)`. Confidence is 1.0 on success, 0.0 on failure. Health: `NOMINAL` on success, `LOST` on failure.

`detect_thresh` and `sigma` from `KCFConfig` are applied to `cv2.TrackerKCF.Params` when available (requires OpenCV 4.5+); the tracker falls back to defaults silently if the `Params` class is absent.

- [ ] **Step 1: Create __init__.py**

Create empty `src/quadguide/perception/kcf/__init__.py`.

- [ ] **Step 2: Write failing tests**

Create `tests/unit/test_kcf_tracker.py`:

```python
import numpy as np
import pytest
from quadguide.core.config import KCFConfig
from quadguide.core.messages import TrackerHealth, BoundingBox
from quadguide.perception.kcf.tracker import KCFTracker


@pytest.fixture
def cfg():
    return KCFConfig(detect_thresh=0.5, sigma=0.2, lambda_=0.0001)


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestKCFTrackerNoLock:
    def test_update_before_init_returns_no_lock(self, cfg, blank_frame):
        tracker = KCFTracker(cfg)
        est = tracker.update(blank_frame)
        assert est.tracker_health == TrackerHealth.NO_LOCK

    def test_update_before_init_confidence_is_zero(self, cfg, blank_frame):
        tracker = KCFTracker(cfg)
        est = tracker.update(blank_frame)
        assert est.confidence == 0.0


class TestKCFTrackerInit:
    def test_init_sets_initialized_flag(self, cfg, blank_frame):
        tracker = KCFTracker(cfg)
        bbox = BoundingBox(0.2, 0.2, 0.3, 0.3)
        tracker.init(blank_frame, bbox)
        assert tracker._initialized is True

    def test_update_after_init_returns_tracker_estimate(self, cfg, blank_frame):
        from quadguide.core.messages import TrackerEstimate
        tracker = KCFTracker(cfg)
        bbox = BoundingBox(0.2, 0.2, 0.3, 0.3)
        tracker.init(blank_frame, bbox)
        est = tracker.update(blank_frame)
        assert isinstance(est, TrackerEstimate)

    def test_update_after_init_health_is_not_no_lock(self, cfg, blank_frame):
        tracker = KCFTracker(cfg)
        bbox = BoundingBox(0.2, 0.2, 0.3, 0.3)
        tracker.init(blank_frame, bbox)
        est = tracker.update(blank_frame)
        assert est.tracker_health != TrackerHealth.NO_LOCK

    def test_close_does_not_raise(self, cfg):
        KCFTracker(cfg).close()
```

- [ ] **Step 3: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_kcf_tracker.py -v
```

Expected: ImportError or AttributeError on `quadguide.perception.kcf.tracker`

- [ ] **Step 4: Write perception/kcf/tracker.py**

```python
from __future__ import annotations
import time

import numpy as np

from quadguide.core.config import KCFConfig
from quadguide.core.messages import BoundingBox, TrackerEstimate, TrackerHealth

__all__ = ["KCFTracker"]


class KCFTracker:
    """KCF tracker wrapping cv2.TrackerKCF. No bus/IPC imports."""

    def __init__(self, config: KCFConfig) -> None:
        self._config      = config
        self._tracker     = None
        self._initialized = False
        self._frame_shape: tuple[int, int] | None = None  # (height, width)

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        """Initialise (or re-initialise) KCF on the given frame and bbox."""
        import cv2
        h, w = frame.shape[:2]
        self._frame_shape = (h, w)
        self._tracker     = self._build_cv_tracker()
        bbox_px = (
            int(bbox.x * w),
            int(bbox.y * h),
            max(1, int(bbox.w * w)),
            max(1, int(bbox.h * h)),
        )
        self._tracker.init(frame, bbox_px)
        self._initialized = True

    def update(self, frame: np.ndarray) -> TrackerEstimate:
        if not self._initialized:
            return TrackerEstimate(
                timestamp_ns=time.monotonic_ns(),
                bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
                confidence=0.0,
                tracker_health=TrackerHealth.NO_LOCK,
            )
        h, w = frame.shape[:2]
        success, bbox_px = self._tracker.update(frame)
        if success:
            x, y, bw, bh = bbox_px
            bbox_norm = BoundingBox(
                x=float(x) / w, y=float(y) / h,
                w=float(bw) / w, h=float(bh) / h,
            )
            return TrackerEstimate(
                timestamp_ns=time.monotonic_ns(),
                bbox=bbox_norm,
                confidence=1.0,
                tracker_health=TrackerHealth.NOMINAL,
            )
        return TrackerEstimate(
            timestamp_ns=time.monotonic_ns(),
            bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
            confidence=0.0,
            tracker_health=TrackerHealth.LOST,
        )

    def close(self) -> None:
        pass

    def _build_cv_tracker(self):
        import cv2
        try:
            p = cv2.TrackerKCF.Params()
            p.sigma         = self._config.sigma
            p.lambda_       = self._config.lambda_
            p.detect_thresh = self._config.detect_thresh
            return cv2.TrackerKCF.create(p)
        except AttributeError:
            return cv2.TrackerKCF.create()
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_kcf_tracker.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/perception/kcf/__init__.py src/quadguide/perception/kcf/tracker.py tests/unit/test_kcf_tracker.py
git commit -m "feat: KCFTracker wrapping cv2.TrackerKCF"
```

---

## Task 8: MOSSE tracker

**Files:**
- Create: `src/quadguide/perception/mosse/__init__.py`
- Modify: `src/quadguide/perception/mosse/tracker.py`
- Create: `tests/unit/test_mosse_tracker.py`

MOSSE wraps `cv2.legacy.TrackerMOSSE`. Requires `opencv-contrib-python`. Same interface as KCFTracker. Confidence is always 1.0 on success (MOSSE exposes no score), 0.0 on failure.

- [ ] **Step 1: Create __init__.py**

Create empty `src/quadguide/perception/mosse/__init__.py`.

- [ ] **Step 2: Write failing tests**

Create `tests/unit/test_mosse_tracker.py`:

```python
import numpy as np
import pytest
from quadguide.core.messages import TrackerHealth, BoundingBox
from quadguide.perception.mosse.tracker import MOSSETracker


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestMOSSETrackerNoLock:
    def test_update_before_init_returns_no_lock(self, blank_frame):
        tracker = MOSSETracker()
        est = tracker.update(blank_frame)
        assert est.tracker_health == TrackerHealth.NO_LOCK

    def test_update_before_init_confidence_zero(self, blank_frame):
        tracker = MOSSETracker()
        est = tracker.update(blank_frame)
        assert est.confidence == 0.0


class TestMOSSETrackerInit:
    def test_init_sets_initialized(self, blank_frame):
        tracker = MOSSETracker()
        tracker.init(blank_frame, BoundingBox(0.2, 0.2, 0.3, 0.3))
        assert tracker._initialized is True

    def test_update_after_init_not_no_lock(self, blank_frame):
        tracker = MOSSETracker()
        tracker.init(blank_frame, BoundingBox(0.2, 0.2, 0.3, 0.3))
        est = tracker.update(blank_frame)
        assert est.tracker_health != TrackerHealth.NO_LOCK

    def test_close_does_not_raise(self):
        MOSSETracker().close()
```

- [ ] **Step 3: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_mosse_tracker.py -v
```

Expected: ImportError on `quadguide.perception.mosse.tracker`

- [ ] **Step 4: Write perception/mosse/tracker.py**

```python
from __future__ import annotations
import time

import numpy as np

from quadguide.core.messages import BoundingBox, TrackerEstimate, TrackerHealth

__all__ = ["MOSSETracker"]


class MOSSETracker:
    """MOSSE tracker wrapping cv2.legacy.TrackerMOSSE.

    Requires opencv-contrib-python. Exposes no tunable parameters —
    OpenCV's MOSSE implementation has no public Params class.
    Confidence is always binary: 1.0 on success, 0.0 on failure.
    Health is therefore always NOMINAL or LOST, never UNCERTAIN.
    """

    def __init__(self) -> None:
        self._tracker     = None
        self._initialized = False

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        import cv2
        h, w = frame.shape[:2]
        self._tracker = cv2.legacy.TrackerMOSSE.create()
        bbox_px = (
            int(bbox.x * w),
            int(bbox.y * h),
            max(1, int(bbox.w * w)),
            max(1, int(bbox.h * h)),
        )
        self._tracker.init(frame, bbox_px)
        self._initialized = True

    def update(self, frame: np.ndarray) -> TrackerEstimate:
        if not self._initialized:
            return TrackerEstimate(
                timestamp_ns=time.monotonic_ns(),
                bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
                confidence=0.0,
                tracker_health=TrackerHealth.NO_LOCK,
            )
        h, w = frame.shape[:2]
        success, bbox_px = self._tracker.update(frame)
        if success:
            x, y, bw, bh = bbox_px
            return TrackerEstimate(
                timestamp_ns=time.monotonic_ns(),
                bbox=BoundingBox(
                    x=float(x) / w, y=float(y) / h,
                    w=float(bw) / w, h=float(bh) / h,
                ),
                confidence=1.0,
                tracker_health=TrackerHealth.NOMINAL,
            )
        return TrackerEstimate(
            timestamp_ns=time.monotonic_ns(),
            bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
            confidence=0.0,
            tracker_health=TrackerHealth.LOST,
        )

    def close(self) -> None:
        pass
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_mosse_tracker.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/perception/mosse/__init__.py src/quadguide/perception/mosse/tracker.py tests/unit/test_mosse_tracker.py
git commit -m "feat: MOSSETracker wrapping cv2.legacy.TrackerMOSSE"
```

---

## Task 9: NanoTrack preprocessing

**Files:**
- Create: `src/quadguide/perception/nanotrack/__init__.py`
- Modify: `src/quadguide/perception/nanotrack/preprocess.py`
- Create: `tests/unit/test_nanotrack_preprocess.py`

No OpenCV. Uses numpy for all array ops, Pillow for bilinear resize.

Context padding follows the SiamTrack convention: `p = (w + h) / 2`, `s = sqrt((w + p) * (h + p))`. This produces a square crop that includes enough background context for the backbone to build a meaningful template.

- [ ] **Step 1: Create __init__.py**

Create empty `src/quadguide/perception/nanotrack/__init__.py`.

- [ ] **Step 2: Write failing tests**

Create `tests/unit/test_nanotrack_preprocess.py`:

```python
import numpy as np
import pytest
from quadguide.core.messages import BoundingBox
from quadguide.perception.nanotrack.preprocess import (
    get_exemplar_crop, get_search_crop, normalise,
)


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def bbox():
    return BoundingBox(x=0.3, y=0.3, w=0.2, h=0.2)


class TestGetExemplarCrop:
    def test_output_shape_is_exemplar_sz(self, frame, bbox):
        crop = get_exemplar_crop(frame, bbox, exemplar_sz=127)
        assert crop.shape == (127, 127, 3)

    def test_output_dtype_uint8(self, frame, bbox):
        crop = get_exemplar_crop(frame, bbox, exemplar_sz=127)
        assert crop.dtype == np.uint8


class TestGetSearchCrop:
    def test_output_shape_is_instance_sz(self, frame, bbox):
        crop = get_search_crop(frame, bbox, scale=2.0, instance_sz=255)
        assert crop.shape == (255, 255, 3)

    def test_output_dtype_uint8(self, frame, bbox):
        crop = get_search_crop(frame, bbox, scale=2.0, instance_sz=255)
        assert crop.dtype == np.uint8


class TestNormalise:
    def test_output_shape_is_1_3_H_W(self, frame, bbox):
        crop = get_exemplar_crop(frame, bbox, exemplar_sz=127)
        norm = normalise(crop)
        assert norm.shape == (1, 3, 127, 127)

    def test_output_dtype_float32(self, frame, bbox):
        crop = get_exemplar_crop(frame, bbox, exemplar_sz=127)
        norm = normalise(crop)
        assert norm.dtype == np.float32

    def test_mean_approximately_zero(self, frame, bbox):
        # After ImageNet normalisation the mean of a random image ≈ 0
        crop = get_exemplar_crop(frame, bbox, exemplar_sz=127)
        norm = normalise(crop)
        assert abs(float(norm.mean())) < 1.5
```

- [ ] **Step 3: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_nanotrack_preprocess.py -v
```

Expected: ImportError on `quadguide.perception.nanotrack.preprocess`

- [ ] **Step 4: Write perception/nanotrack/preprocess.py**

```python
from __future__ import annotations
import math

import numpy as np

from quadguide.core.messages import BoundingBox

__all__ = ["get_exemplar_crop", "get_search_crop", "normalise"]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_exemplar_crop(
    frame: np.ndarray, bbox: BoundingBox, exemplar_sz: int
) -> np.ndarray:
    """Return an exemplar_sz × exemplar_sz BGR uint8 crop centred on bbox.

    Uses SiamTrack context padding: p = (w + h) / 2, s = sqrt((w+p)*(h+p)).
    The context area gives the backbone enough background to build a reliable template.
    """
    h, w = frame.shape[:2]
    bw_px = bbox.w * w
    bh_px = bbox.h * h
    p     = (bw_px + bh_px) / 2
    s     = math.sqrt((bw_px + p) * (bh_px + p))
    cx    = (bbox.x + bbox.w / 2) * w
    cy    = (bbox.y + bbox.h / 2) * h
    return _crop_and_resize(frame, cx, cy, s, exemplar_sz)


def get_search_crop(
    frame: np.ndarray, bbox: BoundingBox, scale: float, instance_sz: int
) -> np.ndarray:
    """Return an instance_sz × instance_sz BGR uint8 search crop.

    The search region is scale × the exemplar crop size, centred on bbox.
    Caller computes scale = instance_sz / exemplar_sz at init time.
    """
    h, w = frame.shape[:2]
    bw_px = bbox.w * w
    bh_px = bbox.h * h
    p     = (bw_px + bh_px) / 2
    s_z   = math.sqrt((bw_px + p) * (bh_px + p))
    s_x   = s_z * scale
    cx    = (bbox.x + bbox.w / 2) * w
    cy    = (bbox.y + bbox.h / 2) * h
    return _crop_and_resize(frame, cx, cy, s_x, instance_sz)


def normalise(crop: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 crop to float32 (1, 3, H, W) with ImageNet normalisation."""
    rgb = crop[:, :, ::-1].astype(np.float32) / 255.0  # BGR → RGB, scale to [0,1]
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD        # ImageNet normalise
    return rgb.transpose(2, 0, 1)[np.newaxis]           # (H,W,C) → (1,C,H,W)


def _crop_and_resize(
    frame: np.ndarray, cx: float, cy: float, size: float, out_sz: int
) -> np.ndarray:
    """Crop a square of `size` pixels centred at (cx, cy) and resize to out_sz × out_sz.

    Out-of-bound regions are filled with the per-channel mean of the full frame.
    """
    from PIL import Image

    h, w = frame.shape[:2]
    half  = size / 2
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = int(round(cx + half)), int(round(cy + half))

    pad_l = max(0, -x1);  pad_r = max(0, x2 - w)
    pad_t = max(0, -y1);  pad_b = max(0, y2 - h)

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    crop = frame[y1c:y2c, x1c:x2c]

    if pad_l or pad_r or pad_t or pad_b:
        mean_color = tuple(int(v) for v in frame.mean(axis=(0, 1)))  # (B, G, R)
        canvas_h = crop.shape[0] + pad_t + pad_b
        canvas_w = crop.shape[1] + pad_l + pad_r
        canvas   = np.full((canvas_h, canvas_w, 3), mean_color, dtype=np.uint8)
        canvas[pad_t:pad_t + crop.shape[0], pad_l:pad_l + crop.shape[1]] = crop
        crop = canvas

    pil  = Image.fromarray(crop[:, :, ::-1])      # BGR → RGB for PIL
    pil  = pil.resize((out_sz, out_sz), Image.BILINEAR)
    return np.array(pil)[:, :, ::-1]              # RGB → BGR, uint8
```

- [ ] **Step 5: Install Pillow if not present, then run tests**

```bash
.venv/bin/pip install Pillow
.venv/bin/python -m pytest tests/unit/test_nanotrack_preprocess.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/perception/nanotrack/__init__.py src/quadguide/perception/nanotrack/preprocess.py tests/unit/test_nanotrack_preprocess.py
git commit -m "feat: NanoTrack preprocessing — exemplar crop, search crop, ImageNet normalise"
```

---

## Task 10: NanoTrack postprocessing

**Files:**
- Modify: `src/quadguide/perception/nanotrack/postprocess.py`
- Create: `tests/unit/test_nanotrack_postprocess.py`

`decode_response` finds the peak in the score map, reads the ltrb regression at that location, and returns the target bbox in search-crop-relative normalised coordinates plus a sigmoid confidence.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_nanotrack_postprocess.py`:

```python
import numpy as np
import pytest
from quadguide.perception.nanotrack.postprocess import decode_response


class TestDecodeResponse:
    def test_returns_four_coord_tuple_and_float(self):
        score_map = np.zeros((1, 1, 25, 25), dtype=np.float32)
        bbox_map  = np.zeros((1, 4, 25, 25), dtype=np.float32)
        coords, conf = decode_response(score_map, bbox_map, stride=8, instance_sz=255)
        assert len(coords) == 4
        assert isinstance(conf, float)

    def test_confidence_in_zero_one(self):
        score_map = np.zeros((1, 1, 25, 25), dtype=np.float32)
        bbox_map  = np.zeros((1, 4, 25, 25), dtype=np.float32)
        _, conf = decode_response(score_map, bbox_map, stride=8, instance_sz=255)
        assert 0.0 <= conf <= 1.0

    def test_peak_at_known_location_recovers_roughly_correct_center(self):
        score_map = np.zeros((1, 1, 25, 25), dtype=np.float32)
        score_map[0, 0, 12, 12] = 10.0   # strong peak at centre cell (12,12)
        bbox_map  = np.zeros((1, 4, 25, 25), dtype=np.float32)
        # ltrb all zero → decoded bbox is a degenerate point at the peak cell centre
        (cx_n, cy_n, w_n, h_n), conf = decode_response(score_map, bbox_map, stride=8, instance_sz=255)
        # Peak at cell (12,12): px centre = (12+0.5)*8 = 100.
        # Normalised: 100/255 ≈ 0.392
        assert abs(cx_n - 100 / 255) < 0.02
        assert abs(cy_n - 100 / 255) < 0.02

    def test_high_score_gives_high_confidence(self):
        score_map = np.full((1, 1, 25, 25), 10.0, dtype=np.float32)
        bbox_map  = np.zeros((1, 4, 25, 25), dtype=np.float32)
        _, conf = decode_response(score_map, bbox_map, stride=8, instance_sz=255)
        assert conf > 0.9

    def test_zero_score_gives_moderate_confidence(self):
        score_map = np.zeros((1, 1, 25, 25), dtype=np.float32)
        bbox_map  = np.zeros((1, 4, 25, 25), dtype=np.float32)
        _, conf = decode_response(score_map, bbox_map, stride=8, instance_sz=255)
        # sigmoid(0) = 0.5
        assert abs(conf - 0.5) < 0.01
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_nanotrack_postprocess.py -v
```

Expected: ImportError on `quadguide.perception.nanotrack.postprocess`

- [ ] **Step 3: Write perception/nanotrack/postprocess.py**

```python
from __future__ import annotations
import numpy as np

__all__ = ["decode_response"]


def decode_response(
    score_map: np.ndarray,  # (1, 1, H, W) raw logits
    bbox_map: np.ndarray,   # (1, 4, H, W) ltrb offsets in feature-map units
    stride: int,
    instance_sz: int,
) -> tuple[tuple[float, float, float, float], float]:
    """Decode NanoTrack head outputs into a target location and confidence.

    Returns:
        coords: (cx_norm, cy_norm, w_norm, h_norm) — all normalised to [0, 1]
                relative to the search crop of size instance_sz × instance_sz.
        conf:   sigmoid of the peak score value, in [0, 1].

    Coordinate convention:
        Peak cell index (cy_idx, cx_idx) maps to pixel centre
        (cx_px, cy_px) = ((cx_idx + 0.5) * stride, (cy_idx + 0.5) * stride).
        LTRB offsets l, t, r, b (in feature-map units, scaled by stride) give:
            x1 = cx_px - l,  y1 = cy_px - t
            x2 = cx_px + r,  y2 = cy_px + b
        Normalised centre and size relative to instance_sz:
            cx_norm = (x1 + x2) / 2 / instance_sz
            cy_norm = (y1 + y2) / 2 / instance_sz
            w_norm  = (x2 - x1) / instance_sz
            h_norm  = (y2 - y1) / instance_sz
    """
    score = score_map[0, 0]  # (H, W)
    h_map, w_map = score.shape
    flat_idx = int(np.argmax(score))
    cy_idx   = flat_idx // w_map
    cx_idx   = flat_idx % w_map

    peak_val = float(score[cy_idx, cx_idx])
    conf     = float(1.0 / (1.0 + np.exp(-peak_val)))  # sigmoid

    cx_px = (cx_idx + 0.5) * stride
    cy_px = (cy_idx + 0.5) * stride

    l = float(bbox_map[0, 0, cy_idx, cx_idx]) * stride
    t = float(bbox_map[0, 1, cy_idx, cx_idx]) * stride
    r = float(bbox_map[0, 2, cy_idx, cx_idx]) * stride
    b = float(bbox_map[0, 3, cy_idx, cx_idx]) * stride

    x1, y1 = cx_px - l, cy_px - t
    x2, y2 = cx_px + r, cy_px + b

    cx_norm = (x1 + x2) / 2 / instance_sz
    cy_norm = (y1 + y2) / 2 / instance_sz
    w_norm  = max(0.0, (x2 - x1)) / instance_sz
    h_norm  = max(0.0, (y2 - y1)) / instance_sz

    return (cx_norm, cy_norm, w_norm, h_norm), conf
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_nanotrack_postprocess.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/perception/nanotrack/postprocess.py tests/unit/test_nanotrack_postprocess.py
git commit -m "feat: NanoTrack postprocessing — decode_response with sigmoid confidence"
```

---

## Task 11: NanoTracker

**Files:**
- Modify: `src/quadguide/perception/nanotrack/tracker.py`
- Create: `tests/unit/test_nanotrack_tracker.py`

`NanoTracker` coordinates inference: backbone encodes exemplar + search crops; head produces score and regression maps; `decode_response` converts to a bbox; tracker maps the bbox from search-crop space back to full-frame normalised coordinates.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_nanotrack_tracker.py`:

```python
import numpy as np
import pytest
from quadguide.core.config import NanotrackConfig
from quadguide.core.messages import TrackerHealth, BoundingBox, TrackerEstimate
from quadguide.perception.nanotrack.tracker import NanoTracker


class _MockRuntime:
    def load(self, path: str):
        return {"path": path}

    def infer(self, model, inputs: dict) -> dict:
        if "input" in inputs:
            return {"features": np.zeros((1, 256, 6, 6), dtype=np.float32)}
        # head call: inputs has "z" and "x"
        return {
            "score": np.zeros((1, 1, 25, 25), dtype=np.float32),
            "bbox":  np.zeros((1, 4, 25, 25), dtype=np.float32),
        }

    def close(self) -> None:
        pass


@pytest.fixture
def cfg():
    return NanotrackConfig(exemplar_sz=127, instance_sz=255, score_threshold=0.7)


@pytest.fixture
def tracker(cfg):
    runtime = _MockRuntime()
    return NanoTracker(runtime, runtime.load("backbone.onnx"), runtime.load("head.onnx"), cfg)


@pytest.fixture
def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestNanoTrackerNoLock:
    def test_update_before_init_returns_no_lock(self, tracker, frame):
        est = tracker.update(frame)
        assert est.tracker_health == TrackerHealth.NO_LOCK

    def test_update_before_init_confidence_zero(self, tracker, frame):
        est = tracker.update(frame)
        assert est.confidence == 0.0


class TestNanoTrackerAfterInit:
    def test_update_after_init_returns_tracker_estimate(self, tracker, frame):
        tracker.init(frame, BoundingBox(0.3, 0.3, 0.2, 0.2))
        est = tracker.update(frame)
        assert isinstance(est, TrackerEstimate)

    def test_update_after_init_not_no_lock(self, tracker, frame):
        tracker.init(frame, BoundingBox(0.3, 0.3, 0.2, 0.2))
        est = tracker.update(frame)
        assert est.tracker_health != TrackerHealth.NO_LOCK

    def test_reinit_works_after_first_init(self, tracker, frame):
        tracker.init(frame, BoundingBox(0.1, 0.1, 0.2, 0.2))
        tracker.init(frame, BoundingBox(0.5, 0.5, 0.2, 0.2))
        est = tracker.update(frame)
        assert est.tracker_health != TrackerHealth.NO_LOCK

    def test_close_does_not_raise(self, tracker):
        tracker.close()
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_nanotrack_tracker.py -v
```

Expected: ImportError on `quadguide.perception.nanotrack.tracker`

- [ ] **Step 3: Write perception/nanotrack/tracker.py**

```python
from __future__ import annotations
import math
import time

import numpy as np

from quadguide.core.config import NanotrackConfig
from quadguide.core.messages import BoundingBox, TrackerEstimate, TrackerHealth
from quadguide.perception.nanotrack.postprocess import decode_response
from quadguide.perception.nanotrack.preprocess import (
    get_exemplar_crop, get_search_crop, normalise,
)

__all__ = ["NanoTracker"]

_STRIDE = 8  # NanoTrack feature stride (backbone downsampling factor)


class NanoTracker:
    """NanoTrack inference pipeline. No OpenCV, no bus/IPC imports.

    Backbone encodes both the exemplar template (stored in self._z_feat) and
    search crops on each update. Head scores and regresses bboxes. Search-crop
    coordinates are mapped back to full-frame normalised coords using the stored
    scale factor computed at init time.
    """

    def __init__(self, runtime, backbone_model, head_model, config: NanotrackConfig) -> None:
        self._runtime  = runtime
        self._backbone = backbone_model
        self._head     = head_model
        self._cfg      = config
        self._z_feat:    dict[str, np.ndarray] | None = None
        self._last_bbox: BoundingBox | None            = None
        self._scale:     float | None                  = None  # s_x / exemplar_sz
        self._initialized = False

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        """Extract exemplar features and store search-region scale."""
        h, w = frame.shape[:2]
        bw_px = bbox.w * w
        bh_px = bbox.h * h
        p     = (bw_px + bh_px) / 2
        s_z   = math.sqrt((bw_px + p) * (bh_px + p))  # exemplar crop size in frame px
        self._scale     = (self._cfg.instance_sz / self._cfg.exemplar_sz)
        self._s_z       = s_z
        self._last_bbox = bbox

        crop         = get_exemplar_crop(frame, bbox, self._cfg.exemplar_sz)
        self._z_feat = self._runtime.infer(self._backbone, {"input": normalise(crop)})
        self._initialized = True

    def update(self, frame: np.ndarray) -> TrackerEstimate:
        if not self._initialized:
            return TrackerEstimate(
                timestamp_ns=time.monotonic_ns(),
                bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
                confidence=0.0,
                tracker_health=TrackerHealth.NO_LOCK,
            )

        h, w = frame.shape[:2]

        # Encode search region
        s_crop = get_search_crop(
            frame, self._last_bbox, self._scale, self._cfg.instance_sz
        )
        x_feat = self._runtime.infer(self._backbone, {"input": normalise(s_crop)})

        # Head: takes exemplar and search features
        z_arr  = list(self._z_feat.values())[0]
        x_arr  = list(x_feat.values())[0]
        out    = self._runtime.infer(self._head, {"z": z_arr, "x": x_arr})
        out_vals = list(out.values())
        score_map = out_vals[0]  # (1, 1, H, W)
        bbox_map  = out_vals[1]  # (1, 4, H, W) ltrb

        (cx_n, cy_n, w_n, h_n), conf = decode_response(
            score_map, bbox_map, stride=_STRIDE, instance_sz=self._cfg.instance_sz
        )

        # Map search-crop coords back to full-frame normalised coords.
        # Search crop is centred at last_bbox centre; its pixel extent = s_z * scale.
        s_x = self._s_z * self._scale  # search region size in frame pixels
        search_cx_norm = self._last_bbox.x + self._last_bbox.w / 2
        search_cy_norm = self._last_bbox.y + self._last_bbox.h / 2

        target_cx_px = (cx_n - 0.5) * s_x + search_cx_norm * w
        target_cy_px = (cy_n - 0.5) * s_x + search_cy_norm * h
        target_w_px  = w_n * s_x
        target_h_px  = h_n * s_x

        bbox_out = BoundingBox(
            x=max(0.0, (target_cx_px - target_w_px / 2) / w),
            y=max(0.0, (target_cy_px - target_h_px / 2) / h),
            w=min(1.0, target_w_px / w),
            h=min(1.0, target_h_px / h),
        )
        self._last_bbox = bbox_out

        health = (
            TrackerHealth.NOMINAL
            if conf >= self._cfg.score_threshold
            else TrackerHealth.UNCERTAIN
        )
        return TrackerEstimate(
            timestamp_ns=time.monotonic_ns(),
            bbox=bbox_out,
            confidence=conf,
            tracker_health=health,
        )

    def close(self) -> None:
        self._runtime.close()
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_nanotrack_tracker.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/perception/nanotrack/tracker.py tests/unit/test_nanotrack_tracker.py
git commit -m "feat: NanoTracker — backbone+head inference, coord transform, TrackerEstimate output"
```

---

## Task 12: CCVTrackerWorker + NCVTrackerWorker

**Files:**
- Create: `src/quadguide/perception/ccv_tracker_worker.py`
- Create: `src/quadguide/perception/ncv_tracker_worker.py`

Both workers own the IPC loop. Tracker objects (KCFTracker, MOSSETracker, NanoTracker) are passed in pre-constructed — workers have no factory knowledge. SIGTERM sets a flag; the loop exits cleanly after the current iteration.

- [ ] **Step 1: Write perception/ccv_tracker_worker.py**

```python
from __future__ import annotations
import os
import signal
import time

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, LockOnCmd, ProcessState

__all__ = ["CCVTrackerWorker"]

_HEALTH_EVERY = 50   # publish HealthReport every N iterations


class CCVTrackerWorker:
    """IPC loop for the classical CV tracker slot.

    Publishes to ccv_tracker/estimate. Sets CPU affinity if requested.
    Tracker object must implement init(frame, bbox) and update(frame).
    """

    def __init__(self, tracker, bus: Bus, frame_buffer: FrameBuffer,
                 cpu_core: int = 1, config: dict | None = None) -> None:
        self._tracker      = tracker
        self._bus          = bus
        self._fb           = frame_buffer
        self._cpu_core     = cpu_core
        self._config       = config or {}
        self._last_seq: int | None = None
        self._stop         = False

    def run(self) -> None:
        log = setup_logging("ccv_tracker", self._config)

        signal.signal(signal.SIGTERM, self._handle_sigterm)

        try:
            os.sched_setaffinity(0, {self._cpu_core})
        except (AttributeError, OSError):
            pass  # dev machine or permission denied — continue without affinity

        log.info("ccv_tracker: started")
        i = 0
        while not self._stop:
            self._check_lockon()
            frame, _ = self._fb.read_latest()
            if frame is not None:
                est = self._tracker.update(frame)
                self._bus.publish("ccv_tracker/estimate", est)
            i += 1
            if i % _HEALTH_EVERY == 0:
                self._bus.publish(
                    "system/health",
                    HealthReport(monotonic_ns(), "ccv_tracker", ProcessState.OK, ""),
                )

        self._bus.detach()
        log.info("ccv_tracker: stopped")

    def _check_lockon(self) -> None:
        cmd: LockOnCmd | None = self._bus.latest("lockon/cmd")
        if cmd is None:
            return
        if cmd.seq != self._last_seq:
            self._last_seq = cmd.seq
            frame, _ = self._fb.read_latest()
            if frame is not None:
                self._tracker.init(frame, cmd.bbox)

    def _handle_sigterm(self, sig, frame) -> None:
        self._stop = True
```

- [ ] **Step 2: Write perception/ncv_tracker_worker.py**

```python
from __future__ import annotations
import signal
import time

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, LockOnCmd, ProcessState

__all__ = ["NCVTrackerWorker"]

_HEALTH_EVERY = 10   # NCV runs slower (~30 Hz); publish health more frequently


class NCVTrackerWorker:
    """IPC loop for the neural CV tracker slot.

    Publishes to ncv_tracker/estimate. Calls tracker.close() on SIGTERM
    to release the NPU handle before exit — critical for RKNN.
    """

    def __init__(self, tracker, bus: Bus, frame_buffer: FrameBuffer,
                 config: dict | None = None) -> None:
        self._tracker      = tracker
        self._bus          = bus
        self._fb           = frame_buffer
        self._config       = config or {}
        self._last_seq: int | None = None
        self._stop         = False

    def run(self) -> None:
        log = setup_logging("ncv_tracker", self._config)
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        log.info("ncv_tracker: started")
        i = 0
        while not self._stop:
            self._check_lockon()
            frame, _ = self._fb.read_latest()
            if frame is not None:
                est = self._tracker.update(frame)
                self._bus.publish("ncv_tracker/estimate", est)
            i += 1
            if i % _HEALTH_EVERY == 0:
                self._bus.publish(
                    "system/health",
                    HealthReport(monotonic_ns(), "ncv_tracker", ProcessState.OK, ""),
                )

        self._tracker.close()  # release NPU handle before exit
        self._bus.detach()
        log.info("ncv_tracker: stopped")

    def _check_lockon(self) -> None:
        cmd: LockOnCmd | None = self._bus.latest("lockon/cmd")
        if cmd is None:
            return
        if cmd.seq != self._last_seq:
            self._last_seq = cmd.seq
            frame, _ = self._fb.read_latest()
            if frame is not None:
                self._tracker.init(frame, cmd.bbox)

    def _handle_sigterm(self, sig, frame) -> None:
        self._stop = True
```

- [ ] **Step 3: Smoke-import check**

```bash
.venv/bin/python -c "
from quadguide.perception.ccv_tracker_worker import CCVTrackerWorker
from quadguide.perception.ncv_tracker_worker import NCVTrackerWorker
print('slot workers OK')
"
```

Expected: `slot workers OK`

- [ ] **Step 4: Commit**

```bash
git add src/quadguide/perception/ccv_tracker_worker.py src/quadguide/perception/ncv_tracker_worker.py
git commit -m "feat: CCVTrackerWorker and NCVTrackerWorker IPC loop classes"
```

---

## Task 13: Tracker factories + thin worker entry points

**Files:**
- Create: `src/quadguide/perception/tracker_factories.py`
- Modify: `src/quadguide/perception/kcf/worker.py`
- Modify: `src/quadguide/perception/mosse/worker.py`
- Modify: `src/quadguide/perception/nanotrack/worker.py`
- Create: `src/quadguide/perception/mosse/__init__.py`

The factory dicts are the extension points. Each thin worker.py imports a tracker class and its slot worker, constructs them from config, and calls `.run()`.

- [ ] **Step 1: Write perception/tracker_factories.py**

```python
from __future__ import annotations

__all__ = ["CCV_TRACKERS", "NCV_TRACKERS", "get_ccv_tracker", "get_ncv_tracker"]

# To add a new CCV tracker: implement the class, add one entry here.
CCV_TRACKERS: dict[str, type] = {}
NCV_TRACKERS: dict[str, type] = {}


def _register_ccv() -> None:
    from quadguide.perception.kcf.tracker import KCFTracker
    from quadguide.perception.mosse.tracker import MOSSETracker
    CCV_TRACKERS["kcf"]   = KCFTracker
    CCV_TRACKERS["mosse"] = MOSSETracker


def _register_ncv() -> None:
    from quadguide.perception.nanotrack.tracker import NanoTracker
    NCV_TRACKERS["nanotrack"] = NanoTracker


_register_ccv()
_register_ncv()


def get_ccv_tracker(config: dict):
    """Return a constructed CCV tracker instance selected by config.tracker.ccv."""
    from quadguide.core.config import cfg_tracker
    tcfg = cfg_tracker(config)
    name = tcfg.ccv
    try:
        cls = CCV_TRACKERS[name]
    except KeyError:
        raise KeyError(f"Unknown ccv tracker {name!r}. Valid: {sorted(CCV_TRACKERS)}")
    if name == "kcf":
        return cls(tcfg.kcf)
    if name == "mosse":
        return cls()
    return cls()


def get_ncv_tracker(config: dict, runtime):
    """Return a constructed NCV tracker instance with a loaded runtime."""
    from quadguide.core.config import cfg_tracker, cfg_platform
    tcfg = cfg_tracker(config)
    pcfg = cfg_platform(config)
    name = tcfg.ncv
    try:
        cls = NCV_TRACKERS[name]
    except KeyError:
        raise KeyError(f"Unknown ncv tracker {name!r}. Valid: {sorted(NCV_TRACKERS)}")
    if name == "nanotrack":
        backbone = runtime.load(pcfg.inference.backbone)
        head     = runtime.load(pcfg.inference.head)
        return cls(runtime, backbone, head, tcfg.nanotrack)
    return cls()
```

- [ ] **Step 2: Write perception/kcf/worker.py**

```python
from quadguide.core.bus import Bus
from quadguide.core.config import cfg_tracker, cfg_platform
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.perception.ccv_tracker_worker import CCVTrackerWorker
from quadguide.perception.kcf.tracker import KCFTracker


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    tcfg    = cfg_tracker(config)
    pcfg    = cfg_platform(config)
    tracker = KCFTracker(tcfg.kcf)
    worker  = CCVTrackerWorker(
        tracker, bus, frame_buffer,
        cpu_core=pcfg.realtime.kcf_cpu_core,
        config=config,
    )
    worker.run()
```

- [ ] **Step 3: Write perception/mosse/worker.py**

```python
from quadguide.core.bus import Bus
from quadguide.core.config import cfg_platform
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.perception.ccv_tracker_worker import CCVTrackerWorker
from quadguide.perception.mosse.tracker import MOSSETracker


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    pcfg    = cfg_platform(config)
    tracker = MOSSETracker()
    worker  = CCVTrackerWorker(
        tracker, bus, frame_buffer,
        cpu_core=pcfg.realtime.kcf_cpu_core,
        config=config,
    )
    worker.run()
```

- [ ] **Step 4: Write perception/nanotrack/worker.py**

```python
from quadguide.core.bus import Bus
from quadguide.core.config import cfg_tracker, cfg_platform
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.inference.factory import get_runtime
from quadguide.perception.ncv_tracker_worker import NCVTrackerWorker
from quadguide.perception.nanotrack.tracker import NanoTracker


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    tcfg    = cfg_tracker(config)
    pcfg    = cfg_platform(config)
    runtime = get_runtime(config)
    backbone = runtime.load(pcfg.inference.backbone)
    head     = runtime.load(pcfg.inference.head)
    tracker  = NanoTracker(runtime, backbone, head, tcfg.nanotrack)
    worker   = NCVTrackerWorker(tracker, bus, frame_buffer, config=config)
    worker.run()
```

- [ ] **Step 5: Smoke-import all new modules**

```bash
.venv/bin/python -c "
from quadguide.perception.tracker_factories import CCV_TRACKERS, NCV_TRACKERS, get_ccv_tracker, get_ncv_tracker
print('ccv:', list(CCV_TRACKERS))
print('ncv:', list(NCV_TRACKERS))
"
```

Expected:
```
ccv: ['kcf', 'mosse']
ncv: ['nanotrack']
```

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/perception/tracker_factories.py \
        src/quadguide/perception/kcf/worker.py \
        src/quadguide/perception/mosse/__init__.py \
        src/quadguide/perception/mosse/worker.py \
        src/quadguide/perception/nanotrack/worker.py
git commit -m "feat: tracker factories and thin worker entry points for kcf, mosse, nanotrack"
```

---

## Task 14: IPC integration test

**Files:**
- Modify: `tests/integration/test_perception_pipeline.py`
- Create: `tests/integration/__init__.py`

Spawns real `multiprocessing.Process` workers using the 'fork' context (Linux). Camera uses a test-only `_SyntheticCamera` that generates solid-colour frames. NanoTrack uses `_MockRuntime`. Asserts that valid `TrackerEstimate` messages appear on both tracker topics within 2 seconds.

- [ ] **Step 1: Create tests/integration/__init__.py**

Create empty `tests/integration/__init__.py`.

- [ ] **Step 2: Write the test**

```python
"""Integration test: camera + CCV + NCV tracker workers communicate over the bus.

Uses 'fork' explicitly (Linux default) so workers inherit shared-memory handles
and pipe fds without pickling. Non-picklable objects (_SyntheticCamera,
_MockRuntime) survive the fork because they are already in the parent's memory.
"""
from __future__ import annotations
import multiprocessing
import os
import signal
import time

import numpy as np
import pytest

from quadguide.core.bus import Bus
from quadguide.core.config import load_config
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.messages import TrackerEstimate, TrackerHealth
from quadguide.perception.camera.sources import CameraSource
from quadguide.perception.camera.worker import run as run_camera
from quadguide.perception.ccv_tracker_worker import CCVTrackerWorker
from quadguide.perception.kcf.tracker import KCFTracker
from quadguide.perception.ncv_tracker_worker import NCVTrackerWorker
from quadguide.perception.nanotrack.tracker import NanoTracker

import pathlib
CONFIG_PATH = str(pathlib.Path(__file__).parents[2] / "configs" / "config.yaml")


# ── test-only fakes ──────────────────────────────────────────────────────────

class _SyntheticCamera(CameraSource):
    """Generates 640×480 frames at max speed — no hardware required."""
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._width, self._height = width, height
        self._i = 0

    def open(self) -> None:
        pass

    def read(self) -> tuple[np.ndarray, int]:
        frame = np.full((self._height, self._width, 3), self._i % 255, dtype=np.uint8)
        self._i += 1
        return frame, time.monotonic_ns()

    def close(self) -> None:
        pass


class _MockRuntime:
    """Returns correctly-shaped zero arrays so postprocess never crashes."""

    def load(self, path: str):
        return {"path": path}

    def infer(self, model, inputs: dict) -> dict:
        if "input" in inputs:
            return {"features": np.zeros((1, 256, 6, 6), dtype=np.float32)}
        # head call
        return {
            "score": np.zeros((1, 1, 25, 25), dtype=np.float32),
            "bbox":  np.zeros((1, 4, 25, 25), dtype=np.float32),
        }

    def close(self) -> None:
        pass


# ── worker entry points (top-level for fork) ─────────────────────────────────

def _run_camera(source, fb, bus):
    run_camera(source, fb, bus)


def _run_ccv(tracker, fb, bus):
    CCVTrackerWorker(tracker, bus, fb).run()


def _run_ncv(tracker, fb, bus):
    NCVTrackerWorker(tracker, bus, fb).run()


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def bus_and_fb():
    config = load_config(CONFIG_PATH, {"platform.inference.device": "cpu"})
    bus = Bus(ring_depth=8)
    fb  = FrameBuffer(width=640, height=480)
    yield bus, fb, config
    bus.close()
    fb.unlink()


# ── test ─────────────────────────────────────────────────────────────────────

def test_ccv_and_ncv_publish_tracker_estimates(bus_and_fb):
    """All three workers run concurrently; both tracker topics receive valid messages."""
    bus, fb, config = bus_and_fb
    ctx = multiprocessing.get_context("fork")

    camera  = _SyntheticCamera()
    runtime = _MockRuntime()
    from quadguide.core.config import cfg_tracker
    tcfg    = cfg_tracker(config)

    kcf_tracker  = KCFTracker(tcfg.kcf)
    nano_tracker = NanoTracker(
        runtime, runtime.load("bb.onnx"), runtime.load("hd.onnx"), tcfg.nanotrack
    )

    procs = [
        ctx.Process(target=_run_camera, args=(camera, fb, bus)),
        ctx.Process(target=_run_ccv,    args=(kcf_tracker, fb, bus)),
        ctx.Process(target=_run_ncv,    args=(nano_tracker, fb, bus)),
    ]
    for p in procs:
        p.start()

    deadline = time.monotonic() + 2.0
    ccv_ok = ncv_ok = False
    while time.monotonic() < deadline and not (ccv_ok and ncv_ok):
        time.sleep(0.05)
        ccv_ok = bus.latest("ccv_tracker/estimate") is not None
        ncv_ok = bus.latest("ncv_tracker/estimate") is not None

    for p in procs:
        os.kill(p.pid, signal.SIGTERM)
    for p in procs:
        p.join(timeout=3.0)

    # Assertions
    ccv_msg = bus.latest("ccv_tracker/estimate")
    ncv_msg = bus.latest("ncv_tracker/estimate")

    assert ccv_msg is not None, "ccv_tracker/estimate: no message within 2 s"
    assert ncv_msg is not None, "ncv_tracker/estimate: no message within 2 s"

    assert isinstance(ccv_msg, TrackerEstimate), f"expected TrackerEstimate, got {type(ccv_msg)}"
    assert isinstance(ncv_msg, TrackerEstimate), f"expected TrackerEstimate, got {type(ncv_msg)}"

    assert ccv_msg.tracker_health in list(TrackerHealth), \
        f"ccv health {ccv_msg.tracker_health!r} is not a valid TrackerHealth"
    assert ncv_msg.tracker_health in list(TrackerHealth), \
        f"ncv health {ncv_msg.tracker_health!r} is not a valid TrackerHealth"

    # No lock-on sent, so both should be NO_LOCK
    assert ccv_msg.tracker_health == TrackerHealth.NO_LOCK
    assert ncv_msg.tracker_health == TrackerHealth.NO_LOCK
```

- [ ] **Step 3: Run the integration test**

```bash
.venv/bin/python -m pytest tests/integration/test_perception_pipeline.py -v -s
```

Expected:
```
PASSED tests/integration/test_perception_pipeline.py::test_ccv_and_ncv_publish_tracker_estimates
```

- [ ] **Step 4: Run the full test suite to check for regressions**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests PASS, no regressions in unit tests.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_perception_pipeline.py
git commit -m "test: IPC integration test — camera, CCV, NCV workers communicate over bus"
```

---

## Self-Review Notes

**Spec coverage check:**

| Spec section | Task |
|---|---|
| Bus topic rename (§1) | Task 1 |
| Config ccv/ncv/mosse (§8) | Task 2 |
| Inference protocol + runtimes (§6) | Tasks 3–4 |
| Camera sources ABC (§4) | Task 5 |
| Camera worker (§4) | Task 6 |
| KCF tracker (§5) | Task 7 |
| MOSSE tracker (§5) | Task 8 |
| NanoTrack preprocessing (§7) | Task 9 |
| NanoTrack postprocessing (§7) | Task 10 |
| NanoTrack tracker (§7) | Task 11 |
| CCVTrackerWorker / NCVTrackerWorker (§3) | Task 12 |
| Tracker factories + entry points (§3.2) | Task 13 |
| IPC integration test (§9) | Task 14 |

All spec sections have a corresponding task. ✓

**Type consistency:** `KCFTracker` receives `KCFConfig` throughout (Tasks 7, 13, 14). `NanoTracker` receives `(runtime, backbone, head, NanotrackConfig)` throughout (Tasks 11, 13, 14). `CCVTrackerWorker` signature `(tracker, bus, frame_buffer, cpu_core, config)` is consistent across Tasks 12 and 13. ✓

**Placeholder scan:** No TBDs. All code steps contain real code. ✓

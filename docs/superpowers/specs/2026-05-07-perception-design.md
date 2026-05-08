# Perception Module Design

**Date:** 2026-05-07  
**Status:** Approved  
**Scope:** `perception/` (camera, kcf, mosse, nanotrack), `inference/`, bus topic rename, config extension, IPC integration test. Fusion is excluded.

---

## 1. Bus Topic Rename

Two bus topics are renamed to reflect tracker slot roles rather than specific algorithms:

| Old name | New name |
|---|---|
| `kcf/estimate` | `ccv_tracker/estimate` |
| `nano/estimate` | `ncv_tracker/estimate` |

`ccv` = classical computer vision. `ncv` = neural computer vision.

Wire format (`FMT_TRACKER_ESTIMATE = "!QfffffB"`) is unchanged. `bus.py` TOPICS table and all dependent tests are updated. The fusion worker (not yet built) will subscribe to `["ccv_tracker/estimate", "ncv_tracker/estimate"]`.

---

## 2. Architecture Overview

```
camera/worker.py
  └─ CameraSource (USBCamera | CSICamera | VirtualCamera[stub])
       │ writes frames → FrameBuffer (shared memory)

perception/ccv_tracker_worker.py   (CCVTrackerWorker)
  └─ get_ccv_tracker(config) → KCFTracker | MOSSETracker
       │ publishes → ccv_tracker/estimate

perception/ncv_tracker_worker.py   (NCVTrackerWorker)
  └─ get_ncv_tracker(config, runtime) → NanoTracker
       │ publishes → ncv_tracker/estimate

inference/factory.py
  └─ get_runtime(config) → OnnxCPURuntime | OnnxCUDARuntime | RKNNRuntime
```

**Key constraint:** All tracker algorithm code (kcf/, mosse/, nanotrack/) has zero bus/IPC imports. Workers own IPC; trackers own math.

---

## 3. Tracker Modularity

### 3.1 Slot workers

**`perception/ccv_tracker_worker.py`** — `CCVTrackerWorker`:
- Sets CPU affinity via `platform.set_realtime(core=config.realtime.kcf_cpu_core, prio=0)`
- Loop (no rate limit — runs at CPU speed):
  1. Check `lockon/cmd` latest; reinit tracker if `seq != last_seq`
  2. `frame, ts = frame_buffer.read_latest()`
  3. `est = tracker.update(frame)`
  4. `bus.publish("ccv_tracker/estimate", est)`
- Publishes `HealthReport("ccv_tracker", ...)` every 50 iterations
- SIGTERM: stop flag → exit 0

**`perception/ncv_tracker_worker.py`** — `NCVTrackerWorker`:
- Same structure; no CPU affinity
- Publishes to `ncv_tracker/estimate`
- SIGTERM: calls `tracker.close()` to release NPU handle before exit

### 3.2 Tracker factories (`perception/tracker_factories.py`)

```python
CCV_TRACKERS = {
    "kcf":   KCFTracker,
    "mosse": MOSSETracker,
}
NCV_TRACKERS = {
    "nanotrack": NanoTracker,
}

def get_ccv_tracker(config) -> CCV tracker instance
def get_ncv_tracker(config, runtime) -> NCV tracker instance
```

Adding a new tracker: implement the class, add one dict entry.

### 3.3 Tracker interface (implicit duck typing)

Both slot workers expect their tracker to implement:

```python
def init(frame: np.ndarray, bbox: BoundingBox) -> None: ...
def update(frame: np.ndarray) -> TrackerEstimate: ...
def close() -> None: ...   # no-op for classical trackers
```

No formal ABC — structural typing. The factory is the contract enforcer.

---

## 4. Camera Module (`perception/camera/`)

### `camera/sources.py`

- `CameraSource` ABC: `open()`, `read() → (np.ndarray, int)`, `close()`, `__iter__`
- `USBCamera(config: CameraConfig)` — `cv2.VideoCapture(index)`, V4L2
- `CSICamera(config: CameraConfig)` — `cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)`
- `VirtualCamera` — **stub**; `open()` and `read()` raise `NotImplementedError` with comment: `# HIL not yet implemented — see hil/virtual_source.py`

Source selection in `worker.py`:
```python
SOURCES = {"v4l2": USBCamera, "gstreamer": CSICamera, "virtual": VirtualCamera}
```

### `camera/worker.py`

```
source = SOURCES[config.platform.camera.backend](config.platform.camera)
source.open()
loop:
    frame, ts = source.read()
    frame_buffer.write_frame(frame, ts)
    bus.publish("system/health", HealthReport("camera", ProcessState.OK, "", monotonic_ns()))
SIGTERM → source.close(); bus.detach(); exit 0
```

---

## 5. KCF and MOSSE (`perception/kcf/`, `perception/mosse/`)

### `kcf/tracker.py` — `KCFTracker(config: KCFConfig)`

- Wraps `cv2.TrackerKCF_create()` with params from config
- Pre-lock state: `update()` returns `TrackerEstimate` with `tracker_health=NO_LOCK`
- Post-lock: confidence from `cv2` success flag; health mapping:
  - `success and conf ≥ detect_thresh → NOMINAL`
  - `success and conf < detect_thresh → UNCERTAIN`
  - `not success → LOST`

### `mosse/tracker.py` — `MOSSETracker(config)`

- Wraps `cv2.legacy.TrackerMOSSE_create()` — same OpenCV interface
- Same `init` / `update` / `close` signature as KCFTracker
- No MOSSE-specific config params (OpenCV exposes none)
- Health mapping: same logic as KCF (success/fail from OpenCV return value)

**Note:** KCF provides a confidence float; MOSSE does not. MOSSE confidence is 1.0 on success, 0.0 on failure. This means MOSSE estimates always have health `NOMINAL` or `LOST`, never `UNCERTAIN`. Fusion must account for this.

---

## 6. Inference Module (`inference/`)

### `inference/base.py` — `NPURuntime` Protocol

```python
class NPURuntime(Protocol):
    def load(self, path: str) -> Any: ...
    def infer(self, model: Any, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...
    def close(self) -> None: ...
```

All NanoTrack inference code calls `runtime.infer(model, inputs)`. No tracker file imports RKNN, ONNX, or CUDA directly.

### Implementations

| File | Class | Backend |
|---|---|---|
| `onnx_cpu.py` | `OnnxCPURuntime` | `onnxruntime` + `CPUExecutionProvider` |
| `onnx_cuda.py` | `OnnxCUDARuntime` | `onnxruntime` + `CUDAExecutionProvider` |
| `rknn.py` | `RKNNRuntime` | `rknnlite` (device) or `rknn` (x86 sim) |

### `inference/factory.py`

```python
RUNTIMES = {"cpu": OnnxCPURuntime, "cuda": OnnxCUDARuntime, "rknn": RKNNRuntime}

def get_runtime(config) -> NPURuntime
```

---

## 7. NanoTrack (`perception/nanotrack/`)

**No OpenCV dependency.** All processing is done through the `NPURuntime` interface.

### `nanotrack/tracker.py` — `NanoTracker(runtime, backbone_model, head_model, config: NanotrackConfig)`

- `init(frame, bbox)`:
  1. `crop = get_exemplar_crop(frame, bbox, config.exemplar_sz)`
  2. `norm = normalise(crop)`
  3. `self._z_feat = runtime.infer(backbone_model, {"input": norm})["output"]`
- `update(frame) → TrackerEstimate`:
  1. `crop = get_search_crop(frame, self._last_bbox, config.instance_sz)`
  2. `x_feat = runtime.infer(backbone_model, {"input": normalise(crop)})["output"]`
  3. `score_map, bbox_map = runtime.infer(head_model, {"z": self._z_feat, "x": x_feat}).values()`
  4. `bbox, conf = decode_response(score_map, bbox_map, stride=8)`
  5. Health: `conf ≥ config.score_threshold → NOMINAL`, else `UNCERTAIN`
- `close()`: `runtime.close()`

### `nanotrack/preprocess.py`

- `get_exemplar_crop(frame, bbox, exemplar_sz) → np.ndarray` — centre-crop with context padding
- `get_search_crop(frame, bbox, instance_sz) → np.ndarray` — larger crop centred on last bbox
- `normalise(crop) → np.ndarray` — float32, ImageNet mean/std, shape `(1, 3, H, W)`

### `nanotrack/postprocess.py`

- `decode_response(score_map, bbox_map, stride) → (BoundingBox, float)` — argmax of score map, regression at that location, map back to normalised image coords

---

## 8. Config Changes

### `configs/config.yaml` additions under `tracker:`

```yaml
tracker:
  ccv: kcf          # "kcf" | "mosse"
  ncv: nanotrack    # "nanotrack" (extensible)
  mosse: {}         # no tunable params exposed
  kcf: ...          # unchanged
  nanotrack: ...    # unchanged
  fusion: ...       # unchanged
```

### `core/config.py` additions

`TrackerConfig` gains two fields: `ccv: str` and `ncv: str`.  
`cfg_tracker()` reads them from the YAML.

---

## 9. IPC Integration Test (`tests/integration/test_perception_pipeline.py`)

Spawns real `multiprocessing.Process` workers using a `Bus` and `FrameBuffer` created in the test process. Camera uses a `TestVirtualCamera` (produces solid-colour frames in-process, not the HIL stub). NanoTrack uses a `MockRuntime` that returns zero arrays with the expected output shapes.

**Assertions (within 2-second timeout):**
1. `bus.latest("ccv_tracker/estimate")` is not None
2. `bus.latest("ncv_tracker/estimate")` is not None
3. Both are `TrackerEstimate` instances
4. `tracker_health` field is a valid `TrackerHealth` member
5. `bbox` fields are all floats in range `[0, 1]`

The `MockRuntime` implements the `NPURuntime` Protocol. `infer()` returns zero `float32` arrays with the shapes the NanoTrack postprocessor expects (`score_map`: `(1,1,H,W)`, `bbox_map`: `(1,4,H,W)` where `H=W=25` for instance_sz=255 with stride=8). This keeps `decode_response` from crashing on bad shapes.

Test tears down all processes via SIGTERM and waits for clean exit.

---

## 10. File Checklist

```
# Modified
src/quadguide/core/bus.py              — topic rename
src/quadguide/core/config.py           — TrackerConfig.ccv/ncv
configs/config.yaml                    — tracker.ccv, tracker.ncv, tracker.mosse
tests/unit/test_bus.py                 — updated topic names

# New
src/quadguide/perception/ccv_tracker_worker.py
src/quadguide/perception/ncv_tracker_worker.py
src/quadguide/perception/tracker_factories.py
src/quadguide/perception/camera/sources.py
src/quadguide/perception/camera/worker.py
src/quadguide/perception/kcf/tracker.py
src/quadguide/perception/kcf/worker.py        — thin entry point, delegates to CCVTrackerWorker
src/quadguide/perception/mosse/tracker.py
src/quadguide/perception/mosse/worker.py      — thin entry point, delegates to CCVTrackerWorker
src/quadguide/perception/nanotrack/tracker.py
src/quadguide/perception/nanotrack/preprocess.py
src/quadguide/perception/nanotrack/postprocess.py
src/quadguide/perception/nanotrack/worker.py  — thin entry point, delegates to NCVTrackerWorker
src/quadguide/inference/base.py
src/quadguide/inference/onnx_cpu.py
src/quadguide/inference/onnx_cuda.py
src/quadguide/inference/rknn.py
src/quadguide/inference/factory.py
tests/integration/test_perception_pipeline.py
```

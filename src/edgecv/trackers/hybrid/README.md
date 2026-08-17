# MAFiD-Style Hybrid Trackers

Mutual-Assistance tracker of Feature Filters and Detectors. Runs a
correlation-filter (CF) tracker and a neural-network (NN) detector in parallel
across two OS processes, with the detector opportunistically injecting improved
CF filters into the live tracking loop.

```
CF tracker (inline, full frame rate)  +  NN detector (async worker, lower FPS)
          │                                        │
          └────────── mutual assistance ───────────┘
                     (MAFiD method)
```

## Purpose

Single-object visual tracking on edge hardware has a speed-vs-accuracy tradeoff:

| Approach | Speed | Accuracy | Can recover from drift? |
|---|---|---|---|
| CF tracker alone (MOSSE) | ~600+ FPS | Low — drifts over time | No |
| NN detector alone (YOLO) | ~70 FPS | High | Frame-by-frame |
| **MAFiD hybrid** | CF rate (600 FPS output) | NN quality (IoU ~0.84) | Yes — opportunistic injection |

The MAFiD method resolves this by running both in parallel. The CF tracker
produces every output frame at high speed. The slower NN detector crops around
the current ROI, detects the target, builds a **new CF filter** from the
detection, and ships it to the CF tracker. The tracker evaluates both filters on
the *current* frame and keeps whichever has higher confidence. The result: CF
speed with NN accuracy.

**Reference:** Matsuo & Yamakawa, "High-Speed Tracking with Mutual Assistance of
Feature Filters and Detectors," *Sensors* 2023.

## Architecture

```
┌── CALLER PROCESS (your code) ──────────────────────────────────────────┐
│                                                                         │
│   tracker = MAFiDMosseYOLO()                                            │
│   tracker.init(frame, bbox)     ← spawns worker, creates shared memory │
│                                                                         │
│   for each frame:                                                       │
│     result = tracker.update(frame)   ← 1-2 ms, non-blocking            │
│                                                                         │
│   tracker.close()                ← clean worker shutdown               │
│                                                                         │
│   Owns: FrameRing (writes frames)                                       │
│         SearchROI channel (writes crop region)                          │
│         PayloadChannel (reads candidate filters)                        │
│         Orchestrator (manages worker lifecycle)                         │
└─────────────────────────────────────────────────────────────────────────┘
          │ FrameRing          │ SearchROI ch.       ▲ PayloadChannel
          │ (latest-only)      │ (caller→worker)     │ (worker→caller)
          ▼                    ▼                     │
┌── DETECTOR WORKER PROCESS ─────────────────────────────────────────────┐
│                                                                         │
│   loop:                                                                 │
│     1. Read latest frame + search ROI (non-blocking seqlock)            │
│     2. Crop → NN inference (YOLO via ONNX/RKNN) → detect object        │
│     3. NN confidence gate → build CF filter → publish candidate         │
│                                                                         │
│   Owns: nothing (attaches SHM, never unlinks)                           │
│   Loads: Backend model inside child (spawn, never fork)                 │
└─────────────────────────────────────────────────────────────────────────┘
```

**Parallelism:** The caller and worker are independent OS processes on different
CPU cores (or the worker runs on an NPU). No process ever blocks on another —
all IPC uses wait-free seqlocks over shared memory.

## Codebase map

```
edgecv/
├── trackers/hybrid/
│   ├── base.py                  ← MAFiDHybrid (building-block constructor)
│   ├── mafid_mosse_yolo.py      ← MAFiDMosseYOLO (concrete, flat kwargs)
│   ├── worker.py                ← _detector_main() entrypoint
│   ├── detector_adapter.py      ← NNDetectorAdapter ABC + YoloDetectorAdapter
│   └── serialise.py             ← FilterState ↔ payload dict
├── fusion/
│   ├── calibrator.py            ← ScoreCalibrator, LinearCalibrator, SigmoidCalibrator
│   ├── psr_gate.py              ← PSRGateParams, PSRGatePolicy
│   └── weighted.py              ← WeightedFusionParams, WeightedFusionPolicy
├── runtime/shm/
│   ├── search_roi.py            ← SearchROI channel (caller→worker)
│   ├── frame_ring.py            ← FrameRing (zero-copy frame buffer)
│   ├── payload.py               ← PayloadChannel (variable-shape arrays)
│   └── seqlock.py               ← Wait-free synchronisation primitive
└── trackers/cf/mosse.py         ← MOSSE CF tracker (+ default_calibrator)
    trackers/nn/yolo.py          ← YOLO detector (+ default_calibrator)
```

## Quick start

```python
from edgecv.trackers.hybrid import MAFiDMosseYOLO

# Zero-config — sensible defaults for MOSSE + YOLO
tracker = MAFiDMosseYOLO()

# Acquire initial bounding box (your application logic)
tracker.init(first_frame, bbox)

# Main loop — returns plain CF results until first NN injection
while True:
    result = tracker.update(frame)
    # result.bbox      → tracking output
    # result.confidence → PSR (CF) or fused confidence
    # result.status    → LOCKED / COASTING / LOST

tracker.close()  # or use `with MAFiDMosseYOLO() as tracker:`
```

### Deployment

```python
# Dev machine (ONNX CPU)
tracker = MAFiDMosseYOLO(placement="dev")

# Rockchip RK3588 (RKNN NPU)
tracker = MAFiDMosseYOLO(placement="rk3588")

# Custom board profile
tracker = MAFiDMosseYOLO(placement="/path/to/my_board.yaml")
```

## Tuning guide

All parameters are flat `__init__` kwargs. Tune only what you need — the rest
use sensible defaults calibrated for MOSSE + YOLO.

### CF tracker params (`mosse_*` prefix)

Passed through to the MOSSE constructor. See `Mosse` docstring for details.

| Parameter | Default | Effect |
|---|---|---|
| `mosse_padding` | 1.0 | Template size relative to target |
| `mosse_sigma` | 2.0 | Gaussian label spread |
| `mosse_eta` | 0.125 | Online update learning rate |
| `mosse_lmbda` | 1e-3 | Regularisation |
| `mosse_psr_lock` | 7.0 | PSR above this → LOCKED |
| `mosse_psr_lost` | 5.0 | PSR below this → LOST |

### NN detector params (`yolo_*` prefix)

| Parameter | Default | Effect |
|---|---|---|
| `yolo_input_size` | 640 | NN input resolution |
| `yolo_conf_thresh` | 0.25 | Minimum raw YOLO confidence |
| `yolo_iou_thresh` | 0.45 | NMS overlap threshold |
| `yolo_backend` | `"auto"` | `"auto"` / `"onnx"` / `"rknn"` / `"mock"` |
| `yolo_manifest` | `None` | Path to custom YAML model manifest |

### Search area

| Parameter | Default | Effect |
|---|---|---|
| `search_padding` | 2.0 | Crop region = target bbox × factor |

Higher values let the detector find the target if it moves fast, at the cost of
more pixels to process (slower detection). Lower values are faster but risk
missing the target if it's far from the last known position.

### Fusion — PSR gate

Controls when the NN-detected filter replaces the incumbent CF filter.

| Parameter | Default | Effect |
|---|---|---|
| `psr_margin` | 0.5 | Candidate *calibrated* confidence must exceed incumbent by this much |
| `candidate_floor` | 0.3 | Candidate below this → rejected regardless |
| `incumbent_floor` | 0.1 | Incumbent below this → emergency mode (margin halved) |
| `use_hysteresis` | True | Once candidate is taken, margin to switch BACK is doubled |

**Tuning workflow:**
1. Set `psr_margin` higher (0.7–1.0) if the tracker oscillates between filters
2. Set `candidate_floor` higher (0.4–0.5) if bad detections keep injecting junk filters
3. Set `incumbent_floor` higher (0.2–0.3) to accept candidates more eagerly when the track is failing
4. Set `use_hysteresis=False` if the tracker switches too conservatively

### Score calibration

CF PSR (~2–50) and NN confidence (0–1) live on different scales. Calibrators
map both to a common 0–1 range before the fusion policy compares them.

#### CF calibration (PSR → 0–1)

| Parameter | Default | Effect |
|---|---|---|
| `psr_low` | 3.0 | PSR at this value → calibrated confidence 0 |
| `psr_high` | 15.0 | PSR at this value → calibrated confidence 1 |

Set `psr_low` near your tracker's "lost" PSR threshold and `psr_high` near the
typical "locked" PSR for your scene.

#### NN calibration (YOLO score → 0–1)

| Parameter | Default | Effect |
|---|---|---|
| `nn_centre` | 0.4 | Raw NN score where calibrated confidence = 0.5 |
| `nn_steepness` | 12.0 | Sharper → more binary decision; gentler → smoother |
| `nn_confidence_floor` | 0.3 | Worker-side gate: NN confidence below this → skip filter build |

**Important:** `nn_confidence_floor` is on the *calibrated* 0–1 scale, not the
raw YOLO score. If you change `nn_centre` or `nn_steepness`, the meaning of 0.3
changes. Tune calibration first, then set the floor based on observed values.

### Tuning example

```python
tracker = MAFiDMosseYOLO(
    # Faster online adaptation for fast-moving targets
    mosse_eta=0.2,
    mosse_psr_lock=8.0,

    # Lower YOLO threshold — accept more detections
    yolo_conf_thresh=0.15,

    # Wider search area for fast motion
    search_padding=3.0,

    # Gate: require strong evidence before switching filters
    psr_margin=0.7,
    candidate_floor=0.4,

    # Calibration: wider PSR dynamic range
    psr_low=2.0,
    psr_high=25.0,

    # Deployment
    placement="rk3588",
)
```

### Default calibrator values per tracker

Each tracker class exposes its recommended calibration range:

| Tracker | Calibrator | Values |
|---|---|---|
| `Mosse` | `LinearCalibrator` | `low=3.0, high=15.0` |
| `YoloDetector` | `SigmoidCalibrator` | `centre=0.4, steepness=12.0` |

These are used when no explicit calibrator is passed. The hybrid auto-resolves
them from the tracker class.

## Adding a new hybrid

Every hybrid tracker follows the same two-layer pattern:

### Layer 1 — `MAFiDHybrid` (building blocks)

Accepts pre-built components for advanced users and custom hybrids:

```python
from edgecv.trackers.hybrid import MAFiDHybrid

tracker = MAFiDHybrid(
    cf_tracker=Mosse(eta=0.2),
    detector_config={"manifest": "my_model.yaml", "backend": "onnx"},
    detector_factory=my_custom_factory,
    fusion_policy=WeightedFusionPolicy(cf_weight=0.7, nn_weight=0.3),
)
```

### Layer 2 — Concrete named hybrid (what users import)

Subclass `MAFiDHybrid` with a pre-wired CF+NN pairing and flat kwargs:

```python
class MAFiDKCFYOLO(MAFiDHybrid):
    def __init__(self, *, kcf_sigma=0.5, yolo_conf=0.25, ...):
        cf = KCF(sigma=kcf_sigma)
        detector_config = {"conf_thresh": yolo_conf, ...}
        super().__init__(cf, detector_config, detector_factory=_make_yolo_adapter, ...)
        self._cf_kwargs = {"sigma": kcf_sigma}  # for worker
```

Requirements for a new CF tracker to work as a hybrid component:
1. Subclass `CorrelationFilterTracker`
2. Implement `build_filter()` (pure, no mutation) — used by the worker
3. Implement `evaluate()` (pure) — used by the caller for comparison
4. Implement `get_filter()` / `set_filter()` — for filter swap
5. Provide a `default_calibrator` class attribute

See `edgecv/trackers/cf/mosse.py` for the reference implementation.

Requirements for a new NN detector to work as a hybrid component:
1. Subclass `NNDetectorAdapter`
2. Implement `detect(frame, search_roi) → DetectorOutput`
3. Provide a module-level factory function (picklable for `spawn`)
4. The factory must construct the backend model **inside** the child process

See `edgecv/trackers/hybrid/detector_adapter.py` for the reference implementation.

## How the MAFiD method works (one frame)

1. **Caller publishes** the current frame and search ROI to shared memory
2. **Caller runs CF tracking** inline on the frame (the rate-limiting step)
3. **Worker reads** the latest frame + ROI, crops around the ROI, runs NN detection
4. **Worker builds a CF filter** from the detection box (pure `build_filter`, no mutation)
5. **Worker publishes** the candidate filter + detection metadata to the payload channel
6. **Caller polls** the payload channel (non-blocking — may be empty)
7. **If candidate ready:** caller evaluates *both* filters on the current frame
   using the *same* CF engine, compares PSR, and swaps if the candidate wins
8. **Return** the chosen filter's bounding box

The key insight: the worker ships a **filter** (appearance template), not a
**box** (position). The filter is evaluated on the *current* frame by the *same*
CF engine that's doing the live tracking. This means the detection's position is
only used to build the filter; the tracking position always comes from the CF
engine evaluating on the most recent frame. Detection latency is bridged without
a motion predictor.

## Testing

```bash
# Run all hybrid-related tests
python3 -m pytest tests/test_mafid_hybrid.py tests/test_calibrator.py \
    tests/test_psr_gate.py tests/test_weighted_fusion.py \
    tests/test_search_roi.py -v

# Run the full test suite
python3 -m pytest tests/ -v
```

## References

- **MAFiD paper:** Matsuo & Yamakawa, *Sensors* 2023, 23, 7082
- **Full design spec:** `docs/specs/mafid_hybrid_v1.md`
- **Project architecture:** `ARCHITECTURE.md`
- **MOSSE tracker:** Bolme et al., CVPR 2010

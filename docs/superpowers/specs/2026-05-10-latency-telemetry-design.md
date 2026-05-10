# Latency Telemetry — Design Spec

**Date:** 2026-05-10
**Scope:** Add tracker pipeline latency (latest and moving average) to the ground station web UI.

---

## Goal

Display the total tracker pipeline latency — from camera frame capture to fusion output — in the ground station HUD. Show the most recent measurement and a 20-sample moving average. The measurement must not include ground station SSE polling delay.

---

## Latency Definition

`latency_ns = tracker_update_time - frame_capture_time`

Where:
- `frame_capture_time` = `timestamp_ns` written by the camera worker into `FrameBuffer` (defaults to `monotonic_ns()` at write time)
- `tracker_update_time` = `monotonic_ns()` immediately after `tracker.update(frame)` returns in the tracker worker

This captures: time the frame spent waiting in the ring buffer + tracker inference time.  
This deliberately excludes: fusion processing time (sub-ms, negligible) and ground station SSE polling delay.

Fusion forwards the `latency_ns` of the active tracker's estimate into `TargetEstimate`. The ground server reads it directly from there — no independent time measurement.

---

## Changes

### 1. `core/messages.py`

Add `latency_ns: int` to `TrackerEstimate` and `TargetEstimate`. Sentinel value `0` means no frame was available (pre-lock-on).

Wire format updates:
- `FMT_TRACKER_ESTIMATE`: `"!QfffffBI"` (29 → 33 bytes, `I` = uint32)
- `FMT_TARGET_ESTIMATE`: `"!QfffffffBBI"` (38 → 42 bytes, `I` = uint32)

`uint32` covers up to ~4.3 seconds — sufficient for any real latency budget.

### 2. `perception/ccv_tracker_worker.py` and `perception/ncv_tracker_worker.py`

Read the frame timestamp instead of discarding it:

```python
frame, frame_ts = frame_buffer.read_latest()
```

After `tracker.update(frame)` returns, compute latency and attach it to the estimate:

```python
latency_ns = (monotonic_ns() - frame_ts) if frame_ts > 0 else 0
est = dataclasses.replace(est, latency_ns=latency_ns)
```

No changes to the tracker algorithm contract (`init`, `update`, `close`, `name`). The worker owns the latency computation, not the tracker.

### 3. `perception/fusion/fusion.py`

`fuse()` copies `latency_ns` from the active tracker's estimate into `TargetEstimate`:

- `ActiveTracker.NCV` → `ncv.latency_ns`
- `ActiveTracker.CCV` or `ActiveTracker.FUSED` → `ccv.latency_ns`
- Passthrough (only one tracker available) → use that tracker's `latency_ns`
- Neither available → `0`

No changes to `fusion/worker.py`.

### 4. `ground/server.py`

Add a `deque(maxlen=20)` on `app.state` to accumulate latency samples.

In `_sse()`, after reading `estimate`:

```python
lat_ms = estimate.latency_ns / 1e6 if estimate and estimate.latency_ns > 0 else None
if lat_ms is not None:
    app.state.latency_window.append(estimate.latency_ns)
avg_ms = (sum(app.state.latency_window) / len(app.state.latency_window) / 1e6
          if app.state.latency_window else None)
```

Add to the SSE payload:
```json
"latency_ms": <float|null>,
"latency_avg_ms": <float|null>
```

### 5. `ground/static/index.html`

Add a **LATENCY** section to the HUD grid (1-column width, alongside the existing CROSSHAIR section):

```
LATENCY
latest   12.4 ms
avg (20) 14.1 ms
```

Colour coding on the `avg` value:
- Normal (green): avg ≤ 50 ms
- Warn (yellow): 50 ms < avg ≤ 100 ms
- Danger (red): avg > 100 ms

Thresholds chosen relative to NanoTrack's ~33ms cycle time. Values are `null`-safe (show `—` before first lock-on).

---

## What Is Not Changed

- Tracker algorithm files (`kcf/tracker.py`, `mosse/tracker.py`, `nanotrack/tracker.py`) — latency is computed in the worker, not the algorithm
- `fusion/worker.py` — no loop changes needed
- `ground/overlay.py` — latency not drawn on the video frame
- Config — no new config keys; window size (20) is a constant

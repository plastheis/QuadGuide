# Latency Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tracker pipeline latency (latest + 20-sample moving average) to the ground station HUD, measuring from camera frame capture through to fusion output.

**Architecture:** `latency_ns` is computed in each tracker worker as `monotonic_ns() - frame_ts` after `tracker.update()` returns, then attached to `TrackerEstimate` via `dataclasses.replace`. Fusion copies the active tracker's `latency_ns` into `TargetEstimate`. The ground server reads it directly from the estimate — no independent time measurement — so SSE polling delay is excluded.

**Tech Stack:** Python dataclasses + struct (wire format), FastAPI SSE, vanilla JS

---

### Task 1: Add `latency_ns` to wire messages

**Files:**
- Modify: `src/quadguide/core/messages.py`
- Modify: `tests/unit/test_messages.py`

- [ ] **Step 1: Update size tests to expect new wire sizes (they will fail)**

In `tests/unit/test_messages.py`, update `TestFormatSizes`:

```python
def test_tracker_estimate(self):
    assert struct.calcsize(FMT_TRACKER_ESTIMATE) == 33  # was 29

def test_target_estimate(self):
    assert struct.calcsize(FMT_TARGET_ESTIMATE) == 42  # was 38
```

- [ ] **Step 2: Run size tests to confirm they fail**

```bash
pytest tests/unit/test_messages.py::TestFormatSizes::test_tracker_estimate \
       tests/unit/test_messages.py::TestFormatSizes::test_target_estimate -v
```

Expected: FAIL — `assert 29 == 33` and `assert 38 == 42`

- [ ] **Step 3: Update `FMT_TRACKER_ESTIMATE` and `TrackerEstimate` in `messages.py`**

Replace lines 52–53 (format constant):

```python
FMT_TRACKER_ESTIMATE = "!QfffffBI"
# Q(8) + bbox.x,y,w,h(4×f=16) + confidence(f=4) + health(B=1) + latency_ns(I=4) = 33 bytes
```

Replace lines 100–123 (`TrackerEstimate` dataclass):

```python
@dataclass(frozen=True)
class TrackerEstimate:
    timestamp_ns: int
    bbox: BoundingBox
    confidence: float
    tracker_health: TrackerHealth
    latency_ns: int = 0  # set by tracker worker via dataclasses.replace; 0 = no frame yet

    def pack(self) -> bytes:
        return _ST_TRACKER_ESTIMATE.pack(
            self.timestamp_ns,
            self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h,
            self.confidence,
            TrackerHealth._ord[self.tracker_health],
            self.latency_ns,
        )

    @classmethod
    def unpack(cls, data: bytes) -> TrackerEstimate:
        ts, x, y, w, h, conf, health_b, latency = _ST_TRACKER_ESTIMATE.unpack(data)
        return cls(
            timestamp_ns=ts,
            bbox=BoundingBox(x, y, w, h),
            confidence=conf,
            tracker_health=TrackerHealth._from_ord[health_b],
            latency_ns=latency,
        )
```

- [ ] **Step 4: Update `FMT_TARGET_ESTIMATE` and `TargetEstimate` in `messages.py`**

Replace lines 55–57 (format constant):

```python
FMT_TARGET_ESTIMATE = "!QfffffffBBI"
# Q(8) + bbox.x,y,w,h(4×f=16) + centroid_x,y(2×f=8) + confidence(f=4)
#   + tracker_health(B=1) + active_tracker(B=1) + latency_ns(I=4) = 42 bytes
```

Replace lines 126–155 (`TargetEstimate` dataclass):

```python
@dataclass(frozen=True)
class TargetEstimate:
    timestamp_ns: int
    bbox: BoundingBox
    centroid_norm: tuple[float, float]
    confidence: float
    tracker_health: TrackerHealth
    active_tracker: ActiveTracker
    latency_ns: int = 0  # set by fusion from the active tracker's latency_ns

    def pack(self) -> bytes:
        return _ST_TARGET_ESTIMATE.pack(
            self.timestamp_ns,
            self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h,
            self.centroid_norm[0], self.centroid_norm[1],
            self.confidence,
            TrackerHealth._ord[self.tracker_health],
            ActiveTracker._ord[self.active_tracker],
            self.latency_ns,
        )

    @classmethod
    def unpack(cls, data: bytes) -> TargetEstimate:
        ts, bx, by, bw, bh, cx, cy, conf, health_b, tracker_b, latency = \
            _ST_TARGET_ESTIMATE.unpack(data)
        return cls(
            timestamp_ns=ts,
            bbox=BoundingBox(bx, by, bw, bh),
            centroid_norm=(cx, cy),
            confidence=conf,
            tracker_health=TrackerHealth._from_ord[health_b],
            active_tracker=ActiveTracker._from_ord[tracker_b],
            latency_ns=latency,
        )
```

- [ ] **Step 5: Update round-trip tests to cover `latency_ns`**

In `tests/unit/test_messages.py`, update `TestRoundTrips.test_tracker_estimate`:

```python
def test_tracker_estimate(self):
    msg = TrackerEstimate(
        timestamp_ns=1_000_000,
        bbox=BoundingBox(0.1, 0.2, 0.3, 0.4),
        confidence=0.9,
        tracker_health=TrackerHealth.NOMINAL,
        latency_ns=12_345_678,
    )
    r = TrackerEstimate.unpack(msg.pack())
    assert r.timestamp_ns == msg.timestamp_ns
    assert r.bbox.x == pytest.approx(msg.bbox.x, rel=1e-6)
    assert r.bbox.y == pytest.approx(msg.bbox.y, rel=1e-6)
    assert r.bbox.w == pytest.approx(msg.bbox.w, rel=1e-6)
    assert r.bbox.h == pytest.approx(msg.bbox.h, rel=1e-6)
    assert r.confidence == pytest.approx(msg.confidence, rel=1e-6)
    assert r.tracker_health == msg.tracker_health
    assert r.latency_ns == msg.latency_ns
```

Update `TestRoundTrips.test_target_estimate`:

```python
def test_target_estimate(self):
    msg = TargetEstimate(
        timestamp_ns=2_000_000,
        bbox=BoundingBox(0.1, 0.2, 0.3, 0.4),
        centroid_norm=(-0.1, 0.2),
        confidence=0.85,
        tracker_health=TrackerHealth.UNCERTAIN,
        active_tracker=ActiveTracker.FUSED,
        latency_ns=8_000_000,
    )
    r = TargetEstimate.unpack(msg.pack())
    assert r.timestamp_ns == msg.timestamp_ns
    assert r.bbox.x == pytest.approx(msg.bbox.x, rel=1e-6)
    assert r.bbox.y == pytest.approx(msg.bbox.y, rel=1e-6)
    assert r.bbox.w == pytest.approx(msg.bbox.w, rel=1e-6)
    assert r.bbox.h == pytest.approx(msg.bbox.h, rel=1e-6)
    assert r.centroid_norm[0] == pytest.approx(msg.centroid_norm[0], rel=1e-6)
    assert r.centroid_norm[1] == pytest.approx(msg.centroid_norm[1], rel=1e-6)
    assert r.confidence == pytest.approx(msg.confidence, rel=1e-6)
    assert r.tracker_health == msg.tracker_health
    assert r.active_tracker == msg.active_tracker
    assert r.latency_ns == msg.latency_ns
```

- [ ] **Step 6: Run all message tests**

```bash
pytest tests/unit/test_messages.py -v
```

Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/quadguide/core/messages.py tests/unit/test_messages.py
git commit -m "feat(messages): add latency_ns field to TrackerEstimate and TargetEstimate"
```

---

### Task 2: Compute latency in tracker workers

**Files:**
- Modify: `src/quadguide/perception/ccv_tracker_worker.py`
- Modify: `src/quadguide/perception/ncv_tracker_worker.py`

No new unit tests: the run-loop is infinite and integration-tested via `tests/integration/test_perception_pipeline.py`. The latency value itself is verified in Task 3's fusion tests.

- [ ] **Step 1: Update `ccv_tracker_worker.py`**

Add `import dataclasses` after the existing `import os` on line 2 (the file currently has `import os`, `import signal`, `import time`):

```python
import dataclasses
import os
import signal
import time
```

In `CCVTrackerWorker.run()`, replace lines 49–52:

```python
# before
frame, _ = self._fb.read_latest()
if frame is not None:
    est = self._tracker.update(frame)
    self._bus.publish("ccv_tracker/estimate", est)
```

with:

```python
frame, frame_ts = self._fb.read_latest()
if frame is not None:
    est = self._tracker.update(frame)
    latency_ns = monotonic_ns() - frame_ts if frame_ts > 0 else 0
    est = dataclasses.replace(est, latency_ns=latency_ns)
    self._bus.publish("ccv_tracker/estimate", est)
```

- [ ] **Step 2: Update `ncv_tracker_worker.py`**

Add `import dataclasses` after `from __future__ import annotations` and before `import signal`:

```python
from __future__ import annotations
import dataclasses
import signal
```

In `NCVTrackerWorker.run()`, replace lines 39–42:

```python
# before
frame, _ = self._fb.read_latest()
if frame is not None:
    est = self._tracker.update(frame)
    self._bus.publish("ncv_tracker/estimate", est)
```

with:

```python
frame, frame_ts = self._fb.read_latest()
if frame is not None:
    est = self._tracker.update(frame)
    latency_ns = monotonic_ns() - frame_ts if frame_ts > 0 else 0
    est = dataclasses.replace(est, latency_ns=latency_ns)
    self._bus.publish("ncv_tracker/estimate", est)
```

- [ ] **Step 3: Run full unit test suite to check nothing broke**

```bash
pytest tests/unit/ -v
```

Expected: all PASS (tracker algorithm tests create `TrackerEstimate` without `latency_ns`; the default of `0` keeps them working)

- [ ] **Step 4: Commit**

```bash
git add src/quadguide/perception/ccv_tracker_worker.py \
        src/quadguide/perception/ncv_tracker_worker.py
git commit -m "feat(perception): attach latency_ns to tracker estimates in worker loops"
```

---

### Task 3: Forward latency through fusion

**Files:**
- Modify: `src/quadguide/perception/fusion/fusion.py`
- Modify: `tests/unit/test_fusion.py`

- [ ] **Step 1: Write failing fusion latency tests**

Replace the entire content of `tests/unit/test_fusion.py` with:

```python
import types
import pytest

from quadguide.core.messages import (
    ActiveTracker, BoundingBox, TrackerEstimate, TrackerHealth,
)
from quadguide.perception.fusion.fusion import fuse

_CFG = types.SimpleNamespace(
    ncv_staleness_ms=100,
    confidence_gate=0.7,
    iou_divergence_thresh=0.3,
)

_BBOX = BoundingBox(0.2, 0.2, 0.3, 0.3)


def _ccv(health=TrackerHealth.NOMINAL, conf=0.5, latency_ns=1_000_000):
    return TrackerEstimate(
        timestamp_ns=1_000_000_000,
        bbox=_BBOX,
        confidence=conf,
        tracker_health=health,
        latency_ns=latency_ns,
    )


def _ncv(health=TrackerHealth.NOMINAL, conf=0.8, latency_ns=3_000_000):
    return TrackerEstimate(
        timestamp_ns=1_000_000_000,
        bbox=_BBOX,
        confidence=conf,
        tracker_health=health,
        latency_ns=latency_ns,
    )


class TestFuseLatencyPassthrough:
    def test_ccv_only_preserves_latency(self):
        result = fuse(_ccv(latency_ns=5_000_000), None, _CFG)
        assert result is not None
        assert result.latency_ns == 5_000_000

    def test_ncv_only_preserves_latency(self):
        result = fuse(None, _ncv(latency_ns=8_000_000), _CFG)
        assert result is not None
        assert result.latency_ns == 8_000_000

    def test_both_no_lock_uses_ccv_latency(self):
        ccv = _ccv(health=TrackerHealth.NO_LOCK, latency_ns=500_000)
        ncv = _ncv(health=TrackerHealth.NO_LOCK, latency_ns=1_000_000)
        result = fuse(ccv, ncv, _CFG)
        assert result is not None
        assert result.latency_ns == 500_000


class TestFuseLatencyFull:
    def test_ncv_active_uses_ncv_latency(self):
        # ncv.confidence (0.9) > gate (0.7) → NCV active
        ccv = _ccv(conf=0.5, latency_ns=1_000_000)
        ncv = _ncv(conf=0.9, latency_ns=3_000_000)
        result = fuse(ccv, ncv, _CFG)
        assert result is not None
        assert result.active_tracker == ActiveTracker.NCV
        assert result.latency_ns == 3_000_000

    def test_fused_active_uses_ccv_latency(self):
        # both confidences below gate (0.7) → FUSED
        ccv = _ccv(conf=0.5, latency_ns=1_000_000)
        ncv = _ncv(conf=0.5, latency_ns=3_000_000)
        result = fuse(ccv, ncv, _CFG)
        assert result is not None
        assert result.active_tracker == ActiveTracker.FUSED
        assert result.latency_ns == 1_000_000

    def test_stale_ncv_falls_back_to_ccv_latency(self):
        # ncv.timestamp_ns = 0, so age_ms will be huge → ncv dropped → passthrough ccv
        stale_ncv = TrackerEstimate(
            timestamp_ns=0,
            bbox=_BBOX,
            confidence=0.9,
            tracker_health=TrackerHealth.NOMINAL,
            latency_ns=9_000_000,
        )
        ccv = _ccv(latency_ns=1_000_000)
        result = fuse(ccv, stale_ncv, _CFG)
        assert result is not None
        assert result.active_tracker == ActiveTracker.CCV
        assert result.latency_ns == 1_000_000
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_fusion.py -v
```

Expected: FAIL — `assert 0 == 5_000_000` (fusion doesn't forward latency_ns yet)

- [ ] **Step 3: Update `_passthrough` in `fusion.py` to forward `latency_ns`**

Replace lines 25–33 (`_passthrough` function):

```python
def _passthrough(est: TrackerEstimate, label: ActiveTracker) -> TargetEstimate:
    return TargetEstimate(
        timestamp_ns=est.timestamp_ns,
        bbox=est.bbox,
        centroid_norm=_centroid(est.bbox),
        confidence=est.confidence,
        tracker_health=est.tracker_health,
        active_tracker=label,
        latency_ns=est.latency_ns,
    )
```

- [ ] **Step 4: Update `fuse()` to select and propagate `latency_ns`**

In `fuse()`, replace lines 78–96 (the confidence gate and weighted average block) with:

```python
    # confidence gate: prefer nano when it is confident enough
    if ncv.confidence > cfg.confidence_gate:
        fused_bbox = ncv.bbox
        fused_conf = ncv.confidence
        active = ActiveTracker.NCV
        latency_ns = ncv.latency_ns
    else:
        total = ccv.confidence + ncv.confidence
        if total == 0.0:
            w_ccv, w_ncv = 0.5, 0.5
        else:
            w_ccv = ccv.confidence / total
            w_ncv = ncv.confidence / total
        fused_bbox = BoundingBox(
            x=ccv.bbox.x * w_ccv + ncv.bbox.x * w_ncv,
            y=ccv.bbox.y * w_ccv + ncv.bbox.y * w_ncv,
            w=ccv.bbox.w * w_ccv + ncv.bbox.w * w_ncv,
            h=ccv.bbox.h * w_ccv + ncv.bbox.h * w_ncv,
        )
        fused_conf = max(ccv.confidence, ncv.confidence)
        active = ActiveTracker.FUSED
        latency_ns = ccv.latency_ns
```

Then update the final `TargetEstimate(...)` constructor at lines 104–111 to add `latency_ns`:

```python
    return TargetEstimate(
        timestamp_ns=now_ns,
        bbox=fused_bbox,
        centroid_norm=_centroid(fused_bbox),
        confidence=fused_conf,
        tracker_health=health,
        active_tracker=active,
        latency_ns=latency_ns,
    )
```

- [ ] **Step 5: Run fusion tests**

```bash
pytest tests/unit/test_fusion.py -v
```

Expected: all PASS

- [ ] **Step 6: Run full unit test suite**

```bash
pytest tests/unit/ -v
```

Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/quadguide/perception/fusion/fusion.py tests/unit/test_fusion.py
git commit -m "feat(fusion): propagate latency_ns from active tracker into TargetEstimate"
```

---

### Task 4: Add latency to ground server SSE

**Files:**
- Modify: `src/quadguide/ground/server.py`
- Modify: `tests/unit/test_ground_server.py`

- [ ] **Step 1: Write failing tests for SSE latency fields**

Add the following to `tests/unit/test_ground_server.py` (after the existing imports, add the new imports; add the fixtures and tests at the end of the file):

New imports to add at the top of the file:

```python
import json
from quadguide.core.messages import (
    ActiveTracker, BoundingBox, LockOnCmd, TargetEstimate, TrackerHealth,
)
```

New fixture and tests to append at the end of the file:

```python
@pytest.fixture
def client_with_estimate():
    class _EstimateBus(_MockBus):
        def latest(self, topic: str):
            if topic == "target/estimate":
                return TargetEstimate(
                    timestamp_ns=1_000_000_000,
                    bbox=BoundingBox(0.2, 0.2, 0.3, 0.3),
                    centroid_norm=(0.0, 0.0),
                    confidence=0.9,
                    tracker_health=TrackerHealth.NOMINAL,
                    active_tracker=ActiveTracker.CCV,
                    latency_ns=5_000_000,  # 5 ms
                )
            return None
    app = create_app(_EstimateBus(), _MockFrameBuffer())
    with TestClient(app) as c:
        yield c


def _read_one_sse_event(client, path: str) -> dict:
    with client.stream("GET", path) as resp:
        for line in resp.iter_lines():
            if line.startswith("data:"):
                return json.loads(line[len("data:"):].strip())
    raise AssertionError("no SSE event received")


def test_telemetry_includes_latency_keys(client):
    data = _read_one_sse_event(client, "/telemetry")
    assert "latency_ms" in data
    assert "latency_avg_ms" in data


def test_telemetry_latency_null_when_no_estimate(client):
    data = _read_one_sse_event(client, "/telemetry")
    assert data["latency_ms"] is None
    assert data["latency_avg_ms"] is None


def test_telemetry_latency_ms_matches_estimate(client_with_estimate):
    data = _read_one_sse_event(client_with_estimate, "/telemetry")
    assert data["latency_ms"] == pytest.approx(5.0, rel=0.01)


def test_telemetry_latency_avg_ms_after_one_sample(client_with_estimate):
    data = _read_one_sse_event(client_with_estimate, "/telemetry")
    assert data["latency_avg_ms"] == pytest.approx(5.0, rel=0.01)
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
pytest tests/unit/test_ground_server.py::test_telemetry_includes_latency_keys \
       tests/unit/test_ground_server.py::test_telemetry_latency_null_when_no_estimate \
       tests/unit/test_ground_server.py::test_telemetry_latency_ms_matches_estimate \
       tests/unit/test_ground_server.py::test_telemetry_latency_avg_ms_after_one_sample -v
```

Expected: FAIL — `KeyError: 'latency_ms'`

- [ ] **Step 3: Update `server.py`**

Add `from collections import deque` to the imports at the top (after `import math`):

```python
from collections import deque
```

In the `_lifespan` context manager (around line 37), add `latency_window` initialisation after `app.state.process_health`:

```python
app.state.process_health: dict[str, str] = {}
app.state.latency_window: deque = deque(maxlen=20)
```

In `_sse()`, add latency computation after reading `estimate` (around line 100, after `estimate = app.state.bus.latest("target/estimate")`):

```python
lat_ms = estimate.latency_ns / 1e6 if estimate and estimate.latency_ns > 0 else None
if lat_ms is not None:
    app.state.latency_window.append(estimate.latency_ns)
avg_ms = (
    sum(app.state.latency_window) / len(app.state.latency_window) / 1e6
    if app.state.latency_window else None
)
```

In the `data = { ... }` dict, add the two new keys (anywhere in the dict — add them after `"health"`):

```python
"latency_ms":     lat_ms,
"latency_avg_ms": avg_ms,
```

- [ ] **Step 4: Run all ground server tests**

```bash
pytest tests/unit/test_ground_server.py -v
```

Expected: all PASS

- [ ] **Step 5: Run full unit test suite**

```bash
pytest tests/unit/ -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/ground/server.py tests/unit/test_ground_server.py
git commit -m "feat(ground): add latency_ms and latency_avg_ms to SSE telemetry stream"
```

---

### Task 5: Add LATENCY section to web UI

**Files:**
- Modify: `src/quadguide/ground/static/index.html`

No automated tests. Verify visually: open the UI in a browser and confirm the new section appears and updates.

- [ ] **Step 1: Expand CROSSHAIR section to `wide2` and add LATENCY section**

In `index.html`, find the CROSSHAIR section (around line 119):

```html
  <!-- CROSSHAIR SIZE -->
  <div class="section">
```

Change it to `wide2`:

```html
  <!-- CROSSHAIR SIZE -->
  <div class="section wide2">
```

After the closing `</div>` of the CROSSHAIR section and before the SYSTEM HEALTH section, add:

```html
  <!-- LATENCY -->
  <div class="section wide2">
    <div class="sec-title">LATENCY</div>
    <div class="row"><span class="lbl">latest</span><span class="val dim" id="h-lat-latest">—</span></div>
    <div class="row"><span class="lbl">avg (20)</span><span class="val dim" id="h-lat-avg">—</span></div>
  </div>
```

- [ ] **Step 2: Add latency display logic to the SSE handler in the `<script>` block**

In the `sse.onmessage` handler, after the `// system/health` block and before the closing `};`, add:

```javascript
  // latency
  const avgMs = d.latency_avg_ms;
  const latClass = avgMs == null ? '' : avgMs > 100 ? 'danger' : avgMs > 50 ? 'warn' : '';
  set('h-lat-latest', f(d.latency_ms,     1, ' ms'));
  set('h-lat-avg',    f(d.latency_avg_ms, 1, ' ms'), latClass);
```

- [ ] **Step 3: Commit**

```bash
git add src/quadguide/ground/static/index.html
git commit -m "feat(ground/ui): add LATENCY HUD section showing latest and avg pipeline latency"
```

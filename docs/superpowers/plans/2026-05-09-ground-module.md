# Ground Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the quadguide ground station — MJPEG live stream with green centered crosshair for lock-on, orange tracking bbox overlay, SSE telemetry panel, and `+`/`-`/Enter keyboard controls — plus a dev launcher so the module can be tested standalone on the RPi with just a USB camera.

**Architecture:** FastAPI serves a single-page UI. The MJPEG generator reads `frame_buffer` and calls `overlay.draw_overlay()` at 15 Hz; an SSE endpoint polls the bus at 10 Hz for telemetry. The crosshair is drawn client-side on a `<canvas>` stacked over the stream `<img>`; the tracking bbox is drawn server-side in `overlay.py`. All bus access uses `bus.latest()` only — no blocking calls. The ground worker is a separate OS process; interference with the critical pipeline is isolated at the process boundary.

**Tech Stack:** Python 3.13, FastAPI, uvicorn, cv2 (JPEG encoding), vanilla JS (EventSource, fetch, canvas 2D API), multiprocessing shared memory bus.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/quadguide/ground/overlay.py` | CREATE | `draw_overlay(frame, estimate) → bytes` |
| `src/quadguide/ground/server.py` | CREATE | `create_app(bus, frame_buffer) → FastAPI` |
| `src/quadguide/ground/worker.py` | CREATE | `run(config, bus, frame_buffer)` — uvicorn entry point |
| `src/quadguide/ground/static/index.html` | CREATE | Single-page UI |
| `tests/unit/test_ground_overlay.py` | CREATE | Overlay unit tests |
| `tests/unit/test_ground_server.py` | CREATE | Server endpoint tests |
| `pyproject.toml` | MODIFY | Add optional `ground` dep group |
| `scripts/dev_ground.py` | CREATE | Dev launcher: camera + ground worker, no full stack |

---

## Task 1: Install FastAPI + uvicorn

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Install into the venv**

```bash
pip install fastapi "uvicorn[standard]"
```

Expected output includes lines like:
```
Successfully installed fastapi-... uvicorn-... anyio-... httptools-... ...
```

- [ ] **Step 2: Add optional dependency group to pyproject.toml**

Open `pyproject.toml`. After the `[project]` block, add:

```toml
[project.optional-dependencies]
ground = ["fastapi", "uvicorn[standard]"]
```

- [ ] **Step 3: Verify import**

```bash
python -c "import fastapi, uvicorn; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add fastapi+uvicorn optional dep group for ground module"
```

---

## Task 2: overlay.py — tests then implementation

**Files:**
- Create: `src/quadguide/ground/overlay.py`
- Create: `tests/unit/test_ground_overlay.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ground_overlay.py`:

```python
import cv2
import numpy as np
import pytest

from quadguide.core.messages import (
    ActiveTracker, BoundingBox, TargetEstimate, TrackerHealth,
)
from quadguide.ground.overlay import draw_overlay


def _black_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _estimate(health: TrackerHealth) -> TargetEstimate:
    return TargetEstimate(
        timestamp_ns=0,
        bbox=BoundingBox(0.25, 0.25, 0.5, 0.5),
        centroid_norm=(0.0, 0.0),
        confidence=0.9,
        tracker_health=health,
        active_tracker=ActiveTracker.KCF,
    )


def _plain_jpeg(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def _is_jpeg(data: bytes) -> bool:
    return data[:2] == b"\xff\xd8"


def test_none_estimate_returns_jpeg():
    assert _is_jpeg(draw_overlay(_black_frame(), None))


def test_no_lock_returns_jpeg():
    assert _is_jpeg(draw_overlay(_black_frame(), _estimate(TrackerHealth.NO_LOCK)))


def test_lost_returns_jpeg():
    assert _is_jpeg(draw_overlay(_black_frame(), _estimate(TrackerHealth.LOST)))


def test_nominal_returns_jpeg():
    assert _is_jpeg(draw_overlay(_black_frame(), _estimate(TrackerHealth.NOMINAL)))


def test_uncertain_returns_jpeg():
    assert _is_jpeg(draw_overlay(_black_frame(), _estimate(TrackerHealth.UNCERTAIN)))


def test_no_lock_matches_plain_encode():
    frame = _black_frame()
    assert draw_overlay(frame, _estimate(TrackerHealth.NO_LOCK)) == _plain_jpeg(frame)


def test_lost_matches_plain_encode():
    frame = _black_frame()
    assert draw_overlay(frame, _estimate(TrackerHealth.LOST)) == _plain_jpeg(frame)


def test_none_matches_plain_encode():
    frame = _black_frame()
    assert draw_overlay(frame, None) == _plain_jpeg(frame)


def test_nominal_modifies_frame():
    frame = _black_frame()
    assert draw_overlay(frame, _estimate(TrackerHealth.NOMINAL)) != _plain_jpeg(frame)


def test_uncertain_modifies_frame():
    frame = _black_frame()
    assert draw_overlay(frame, _estimate(TrackerHealth.UNCERTAIN)) != _plain_jpeg(frame)


def test_does_not_mutate_input_frame():
    frame = _black_frame()
    original = frame.copy()
    draw_overlay(frame, _estimate(TrackerHealth.NOMINAL))
    assert np.array_equal(frame, original)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/unit/test_ground_overlay.py -v
```

Expected: all fail with `ModuleNotFoundError` or `ImportError` — `overlay` does not exist yet.

- [ ] **Step 3: Implement overlay.py**

Create `src/quadguide/ground/overlay.py`:

```python
from __future__ import annotations

import cv2
import numpy as np

from quadguide.core.messages import TargetEstimate, TrackerHealth

_JPEG_PARAMS     = [cv2.IMWRITE_JPEG_QUALITY, 80]
_COLOR_NOMINAL   = (0, 165, 255)   # orange BGR
_COLOR_UNCERTAIN = (0, 255, 255)   # yellow BGR


def draw_overlay(frame: np.ndarray, estimate: TargetEstimate | None) -> bytes:
    """Return frame encoded as JPEG, with tracking bbox drawn if tracker is active.

    Does not mutate the input frame. Returns a plain encode when estimate is
    None, NO_LOCK, or LOST — nothing is drawn in those states.
    """
    if estimate is None or estimate.tracker_health in (
        TrackerHealth.NO_LOCK, TrackerHealth.LOST
    ):
        return _encode(frame)

    h, w = frame.shape[:2]
    b = estimate.bbox
    x1 = int(b.x * w)
    y1 = int(b.y * h)
    x2 = int((b.x + b.w) * w)
    y2 = int((b.y + b.h) * h)
    color = (
        _COLOR_NOMINAL
        if estimate.tracker_health == TrackerHealth.NOMINAL
        else _COLOR_UNCERTAIN
    )
    out = frame.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return _encode(out)


def _encode(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, _JPEG_PARAMS)
    return buf.tobytes()
```

- [ ] **Step 4: Run tests — all should pass**

```bash
pytest tests/unit/test_ground_overlay.py -v
```

Expected: 11 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/ground/overlay.py tests/unit/test_ground_overlay.py
git commit -m "feat: ground overlay — bbox draw on NOMINAL/UNCERTAIN, JPEG encode"
```

---

## Task 3: server.py — tests then implementation

**Files:**
- Create: `src/quadguide/ground/server.py`
- Create: `tests/unit/test_ground_server.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ground_server.py`:

```python
import pytest
from starlette.testclient import TestClient

from quadguide.core.messages import BoundingBox, LockOnCmd
from quadguide.ground.server import create_app


class _MockBus:
    def __init__(self):
        self.published: list[tuple[str, object]] = []

    def latest(self, topic: str):
        return None

    def publish(self, topic: str, msg) -> None:
        self.published.append((topic, msg))

    def detach(self) -> None:
        pass


class _MockFrameBuffer:
    def read_latest(self):
        return None, 0


@pytest.fixture
def client():
    app = create_app(_MockBus(), _MockFrameBuffer())
    with TestClient(app) as c:
        yield c


@pytest.fixture
def bus_client():
    bus = _MockBus()
    app = create_app(bus, _MockFrameBuffer())
    with TestClient(app) as c:
        yield bus, c


def test_lockon_returns_ok(client):
    resp = client.post("/lockon", json={"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_lockon_publishes_to_lockon_cmd(bus_client):
    bus, client = bus_client
    client.post("/lockon", json={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4})
    lockon = [(t, m) for t, m in bus.published if t == "lockon/cmd"]
    assert len(lockon) == 1
    cmd = lockon[0][1]
    assert isinstance(cmd, LockOnCmd)
    assert cmd.bbox.x == pytest.approx(0.1)
    assert cmd.bbox.y == pytest.approx(0.2)
    assert cmd.bbox.w == pytest.approx(0.3)
    assert cmd.bbox.h == pytest.approx(0.4)


def test_lockon_seq_starts_at_1(bus_client):
    bus, client = bus_client
    client.post("/lockon", json={"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2})
    cmd = [m for t, m in bus.published if t == "lockon/cmd"][0]
    assert cmd.seq == 1


def test_lockon_seq_increments(bus_client):
    bus, client = bus_client
    client.post("/lockon", json={"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2})
    client.post("/lockon", json={"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2})
    cmds = [m for t, m in bus.published if t == "lockon/cmd"]
    assert cmds[0].seq == 1
    assert cmds[1].seq == 2


def test_stream_content_type(client):
    with client.stream("GET", "/stream") as r:
        assert r.status_code == 200
        assert "multipart/x-mixed-replace" in r.headers["content-type"]
        assert "boundary=frame" in r.headers["content-type"]


def test_telemetry_content_type(client):
    with client.stream("GET", "/telemetry") as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/unit/test_ground_server.py -v
```

Expected: all fail — `server.py` does not exist yet.

- [ ] **Step 3: Implement server.py**

Create `src/quadguide/ground/server.py`:

```python
from __future__ import annotations
import asyncio
import json
import math
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from quadguide.core.clock import monotonic_ns
from quadguide.core.messages import BoundingBox, HealthReport, LockOnCmd, ProcessState
from quadguide.ground import overlay

_STATIC      = Path(__file__).parent / "static"
_MJPEG_RATE  = 1 / 15   # 15 Hz
_SSE_RATE    = 0.1       # 10 Hz
_HEALTH_RATE = 0.2       # 5 Hz


def create_app(bus, frame_buffer) -> FastAPI:

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        app.state.bus            = bus
        app.state.frame_buffer   = frame_buffer
        app.state.lockon_seq     = 0
        app.state.process_health: dict[str, str] = {}
        task = asyncio.create_task(_health_task(app))
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(lifespan=_lifespan)

    @app.get("/")
    async def index():
        return FileResponse(_STATIC / "index.html")

    @app.get("/stream")
    async def stream(request: Request):
        return StreamingResponse(
            _mjpeg(request.app),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/telemetry")
    async def telemetry(request: Request):
        return StreamingResponse(
            _sse(request.app),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/lockon")
    async def lockon(body: _LockOnBody, request: Request):
        request.app.state.lockon_seq += 1
        cmd = LockOnCmd(
            timestamp_ns=monotonic_ns(),
            seq=request.app.state.lockon_seq,
            bbox=BoundingBox(body.x, body.y, body.w, body.h),
        )
        request.app.state.bus.publish("lockon/cmd", cmd)
        return {"ok": True}

    return app


class _LockOnBody(BaseModel):
    x: float
    y: float
    w: float
    h: float


async def _mjpeg(app: FastAPI):
    while True:
        await asyncio.sleep(_MJPEG_RATE)
        frame, _ = app.state.frame_buffer.read_latest()
        if frame is None:
            continue
        estimate = app.state.bus.latest("target/estimate")
        jpeg = overlay.draw_overlay(frame, estimate)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


async def _sse(app: FastAPI):
    while True:
        await asyncio.sleep(_SSE_RATE)
        estimate = app.state.bus.latest("target/estimate")
        attitude = app.state.bus.latest("fc/attitude")
        report   = app.state.bus.latest("system/health")
        if report is not None:
            app.state.process_health[report.process] = report.state.value
        data = {
            "tracker_health": estimate.tracker_health.value if estimate else None,
            "confidence":     estimate.confidence            if estimate else None,
            "roll_deg":       math.degrees(attitude.roll_rad)  if attitude else None,
            "pitch_deg":      math.degrees(attitude.pitch_rad) if attitude else None,
            "health":         dict(app.state.process_health),
        }
        yield f"data: {json.dumps(data)}\n\n"


async def _health_task(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(_HEALTH_RATE)
        app.state.bus.publish(
            "system/health",
            HealthReport(monotonic_ns(), "ground", ProcessState.OK, ""),
        )
```

- [ ] **Step 4: Run tests — all should pass**

```bash
pytest tests/unit/test_ground_server.py -v
```

Expected: 6 tests PASSED. (`/` route is verified by the browser smoke test in Task 6 — `index.html` doesn't exist yet at this step.)

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/ground/server.py tests/unit/test_ground_server.py
git commit -m "feat: ground server — MJPEG stream, SSE telemetry, POST /lockon"
```

---

## Task 4: worker.py

**Files:**
- Create: `src/quadguide/ground/worker.py`

No unit test — the worker is a thin uvicorn launcher. Covered by dev-launch smoke test in Task 6.

- [ ] **Step 1: Implement worker.py**

Create `src/quadguide/ground/worker.py`:

```python
from __future__ import annotations
import uvicorn

from quadguide.core.bus import Bus
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.ground.server import create_app


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    """Ground worker entry point. Blocks until uvicorn exits (SIGTERM)."""
    port = config.get("ground", {}).get("port", 8080)
    app  = create_app(bus, frame_buffer)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    bus.detach()
```

- [ ] **Step 2: Verify import**

```bash
python -c "from quadguide.ground.worker import run; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/quadguide/ground/worker.py
git commit -m "feat: ground worker — uvicorn entry point wrapping FastAPI app"
```

---

## Task 5: index.html

**Files:**
- Create: `src/quadguide/ground/static/index.html`

- [ ] **Step 1: Write index.html**

Create `src/quadguide/ground/static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>quadguide</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #111;
      color: #eee;
      font-family: monospace;
      font-size: 13px;
      padding: 10px;
    }
    h1 { font-size: 14px; letter-spacing: 3px; color: #888; margin-bottom: 8px; }

    #viewport {
      position: relative;
      display: inline-block;
      border: 1px solid #333;
    }
    #stream { display: block; max-width: 100%; }
    #overlay {
      position: absolute;
      top: 0; left: 0;
      pointer-events: none;
    }

    #controls {
      margin-top: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    button {
      background: #1a1a1a;
      color: #eee;
      border: 1px solid #444;
      padding: 3px 10px;
      font-family: monospace;
      font-size: 13px;
      cursor: pointer;
    }
    button:hover { background: #2a2a2a; }
    #btn-lock {
      border-color: #00cc00;
      color: #00cc00;
    }
    #btn-lock.flash { background: #002200; }

    #status {
      margin-top: 8px;
      line-height: 2;
      color: #aaa;
    }
    #status span.ok       { color: #eee; }
    #status span.degraded { color: #ffaa00; }
    #status span.failsafe,
    #status span.dead     { color: #ff4444; }
    #status span.unknown  { color: #555; }
  </style>
</head>
<body>
  <h1>QUADGUIDE</h1>

  <div id="viewport">
    <img id="stream" src="/stream" alt="camera stream">
    <canvas id="overlay"></canvas>
  </div>

  <div id="controls">
    <span>size: <span id="size-val">120</span>px</span>
    <button id="btn-plus">+</button>
    <button id="btn-minus">−</button>
    <button id="btn-lock">LOCK ON</button>
  </div>

  <div id="status">
    <div id="line-health">health: <span class="unknown">—</span></div>
    <div id="line-tracker">tracker: <span class="unknown">—</span>&nbsp;&nbsp;conf: <span class="unknown">—</span></div>
    <div id="line-attitude">roll: <span class="unknown">—</span>&nbsp;&nbsp;pitch: <span class="unknown">—</span></div>
  </div>

  <script>
    'use strict';

    const img     = document.getElementById('stream');
    const canvas  = document.getElementById('overlay');
    const ctx     = canvas.getContext('2d');
    const sizeVal = document.getElementById('size-val');
    const btnLock = document.getElementById('btn-lock');

    const STEP     = 10;
    const MIN_SIZE = 40;
    let size = 120;

    // ── Canvas sizing ─────────────────────────────────────────────────────────

    function syncCanvas() {
      canvas.width  = img.offsetWidth;
      canvas.height = img.offsetHeight;
      drawCrosshair();
    }

    new ResizeObserver(syncCanvas).observe(img);
    img.addEventListener('load', syncCanvas);

    // ── Crosshair drawing ─────────────────────────────────────────────────────

    function drawCrosshair() {
      const W = canvas.width;
      const H = canvas.height;
      if (W === 0 || H === 0) return;

      ctx.clearRect(0, 0, W, H);

      const cx   = W / 2;
      const cy   = H / 2;
      const half = size / 2;
      const clen = size / 5;

      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth   = 2;
      ctx.beginPath();

      // Centre lines (center → edge of square)
      ctx.moveTo(cx,        cy);        ctx.lineTo(cx,        cy - half); // up
      ctx.moveTo(cx,        cy);        ctx.lineTo(cx,        cy + half); // down
      ctx.moveTo(cx,        cy);        ctx.lineTo(cx - half, cy);        // left
      ctx.moveTo(cx,        cy);        ctx.lineTo(cx + half, cy);        // right

      // Corner L-shapes
      // top-left
      ctx.moveTo(cx - half,        cy - half + clen);
      ctx.lineTo(cx - half,        cy - half);
      ctx.lineTo(cx - half + clen, cy - half);
      // top-right
      ctx.moveTo(cx + half - clen, cy - half);
      ctx.lineTo(cx + half,        cy - half);
      ctx.lineTo(cx + half,        cy - half + clen);
      // bottom-left
      ctx.moveTo(cx - half,        cy + half - clen);
      ctx.lineTo(cx - half,        cy + half);
      ctx.lineTo(cx - half + clen, cy + half);
      // bottom-right
      ctx.moveTo(cx + half - clen, cy + half);
      ctx.lineTo(cx + half,        cy + half);
      ctx.lineTo(cx + half,        cy + half - clen);

      ctx.stroke();
    }

    // ── Size control ──────────────────────────────────────────────────────────

    function maxSize() {
      return Math.min(canvas.width, canvas.height) - 20;
    }

    function changeSize(delta) {
      size = Math.max(MIN_SIZE, Math.min(size + delta, maxSize()));
      sizeVal.textContent = size;
      drawCrosshair();
    }

    // ── Lock-on ───────────────────────────────────────────────────────────────

    function doLockon() {
      const W = canvas.width;
      const H = canvas.height;
      if (W === 0 || H === 0) return;

      const x = (W / 2 - size / 2) / W;
      const y = (H / 2 - size / 2) / H;
      const w = size / W;
      const h = size / H;

      fetch('/lockon', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x, y, w, h }),
      });

      btnLock.classList.add('flash');
      setTimeout(() => btnLock.classList.remove('flash'), 300);
    }

    // ── Button + keyboard handlers ────────────────────────────────────────────

    document.getElementById('btn-plus').addEventListener('click', () => changeSize(STEP));
    document.getElementById('btn-minus').addEventListener('click', () => changeSize(-STEP));
    btnLock.addEventListener('click', doLockon);

    document.addEventListener('keydown', e => {
      if (e.key === '+' || e.key === '=') { e.preventDefault(); changeSize(STEP); }
      else if (e.key === '-')             { e.preventDefault(); changeSize(-STEP); }
      else if (e.key === 'Enter')         { e.preventDefault(); doLockon(); }
    });

    // ── SSE telemetry ─────────────────────────────────────────────────────────

    const lineHealth  = document.getElementById('line-health');
    const lineTracker = document.getElementById('line-tracker');
    const lineAtt     = document.getElementById('line-attitude');

    function cls(state) {
      if (!state)                              return 'unknown';
      if (state === 'ok')                      return 'ok';
      if (state === 'degraded')                return 'degraded';
      return 'dead';  // failsafe | dead
    }

    function fmt(val, decimals, suffix) {
      return val != null ? val.toFixed(decimals) + suffix : '—';
    }

    const es = new EventSource('/telemetry');
    es.onmessage = ev => {
      const d = JSON.parse(ev.data);

      // Health row
      if (d.health) {
        const parts = Object.entries(d.health)
          .map(([k, v]) => `<span class="${cls(v)}">${k}:${v}</span>`)
          .join('  ');
        lineHealth.innerHTML = 'health: ' + (parts || '<span class="unknown">—</span>');
      }

      // Tracker row
      const th   = d.tracker_health ?? '—';
      const conf = fmt(d.confidence, 2, '');
      lineTracker.innerHTML =
        `tracker: <span class="${cls(d.tracker_health === 'ok' ? 'ok' : d.tracker_health)}">${th}</span>` +
        `&nbsp;&nbsp;conf: ${conf}`;

      // Attitude row
      lineAtt.textContent =
        `roll: ${fmt(d.roll_deg, 1, '°')}   pitch: ${fmt(d.pitch_deg, 1, '°')}`;
    };

    es.onerror = () => {
      lineHealth.innerHTML = 'health: <span class="dead">connection lost</span>';
    };
  </script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add src/quadguide/ground/static/index.html
git commit -m "feat: ground station single-page UI — MJPEG stream, crosshair, telemetry"
```

---

## Task 6: dev launcher — camera + ground worker

**Files:**
- Create: `scripts/dev_ground.py`

This script lets you test the ground module on the RPi with just a USB camera — no trackers, no flight controller needed.

- [ ] **Step 1: Create dev_ground.py**

Create `scripts/dev_ground.py`:

```python
#!/usr/bin/env python3
"""Dev launcher: USB camera + ground worker only. No trackers or FC needed.

Usage:
    python scripts/dev_ground.py --config configs/config.yaml
Then open http://<rpi-ip>:8080 in a browser.
"""
from __future__ import annotations
import argparse
import multiprocessing
import signal
import sys
import time

import cv2

from quadguide.core.bus import Bus
from quadguide.core.config import cfg_platform, load_config
from quadguide.core.frame_buffer import FrameBuffer
import quadguide.ground.worker as ground_worker


def _camera_loop(frame_buffer: FrameBuffer, width: int, height: int) -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("ERROR: could not open /dev/video0", flush=True)
        return
    while True:
        ok, frame = cap.read()
        if ok:
            frame_buffer.write_frame(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config, {})
    pcfg   = cfg_platform(config)

    bus = Bus()
    fb  = FrameBuffer(pcfg.camera.width, pcfg.camera.height)

    cam_proc = multiprocessing.Process(
        target=_camera_loop,
        args=(fb, pcfg.camera.width, pcfg.camera.height),
        daemon=True,
    )
    cam_proc.start()
    print(f"camera  PID {cam_proc.pid}", flush=True)

    gnd_proc = multiprocessing.Process(
        target=ground_worker.run,
        args=(config, bus, fb),
        daemon=True,
    )
    gnd_proc.start()
    print(f"ground  PID {gnd_proc.pid}", flush=True)
    print(f"open    http://0.0.0.0:8080  (or your RPi's LAN IP)", flush=True)

    def _shutdown(sig, _frame):
        print("\nshutting down...", flush=True)
        cam_proc.terminate()
        gnd_proc.terminate()
        cam_proc.join(timeout=3)
        gnd_proc.join(timeout=3)
        bus.close()
        fb.close()
        fb.unlink()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test startup**

From the repo root with the USB camera plugged in:

```bash
python scripts/dev_ground.py --config configs/config.yaml
```

Expected output:
```
camera  PID <n>
ground  PID <n>
open    http://0.0.0.0:8080  (or your RPi's LAN IP)
```

No tracebacks. Process stays alive.

- [ ] **Step 3: Browser smoke test**

Open `http://<rpi-lan-ip>:8080` in a browser on your laptop.

Verify:
- Camera stream is visible
- Green crosshair is centered on the stream
- `+` and `-` keys (and buttons) change the crosshair size
- Pressing `Enter` (or LOCK ON button) flashes the button green briefly
- The health row shows `ground:ok` within ~2 seconds
- `tracker`, `conf`, `roll`, `pitch` show `—` (no trackers running — expected)

- [ ] **Step 4: Commit**

```bash
git add scripts/dev_ground.py
git commit -m "feat: dev_ground launcher — USB camera + ground worker for UI testing"
```

---

## Final test run

- [ ] **Run the full unit test suite to confirm nothing regressed**

```bash
pytest tests/unit/ -v
```

Expected: all existing tests pass plus the 18 new overlay + server tests.

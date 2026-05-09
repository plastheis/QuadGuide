# Ground Module Design

**Date:** 2026-05-09
**Status:** approved

---

## 1. Scope

Implement the four ground module files:

- `src/quadguide/ground/worker.py`
- `src/quadguide/ground/server.py`
- `src/quadguide/ground/overlay.py`
- `src/quadguide/ground/static/index.html`

The ground module serves a web UI for operator situational awareness and target lock-on. It runs as a separate OS process and must not interfere with the critical tracking/guidance/control pipeline.

---

## 2. Architecture

### 2.1 Process isolation

The ground worker is a `multiprocessing.Process`. It inherits `bus` and `frame_buffer` handles from the parent at fork time. All other workers (kcf, nanotrack, fusion, guidance, control) continue running independently. The ground worker's failure or restart has no effect on them.

### 2.2 Non-interference rules

- Only `bus.latest()` (non-blocking) is ever called. `subscribe_one` and `subscribe_any` are never used — no blocking waits on the bus from the ground worker.
- MJPEG stream is rate-capped at **15 Hz** via `asyncio.sleep(1/15)`.
- SSE telemetry is rate-capped at **10 Hz** via `asyncio.sleep(0.1)`.
- `frame_buffer.read_latest()` is lock-free (atomic head pointer); safe to call at any rate.

### 2.3 Data flow

```
frame_buffer ──► /stream generator ──► overlay.py ──► MJPEG ──► browser <img>
                                                                   ↑
                                                        <canvas> draws crosshair on top

bus topics ──► /telemetry SSE ──► browser EventSource ──► status panel

browser Enter ──► POST /lockon (normalised bbox) ──► bus.publish("lockon/cmd")
```

---

## 3. ground/worker.py

Entry point: `run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None`

- Builds the FastAPI app from `server.py`, sets `app.state.bus`, `app.state.frame_buffer`, `app.state.lockon_seq = 0`
- Publishes `HealthReport("ground", ProcessState.OK, "")` to `system/health` at 5 Hz from a background `asyncio.Task` started in the FastAPI lifespan
- Runs `uvicorn.run(app, host="0.0.0.0", port=8080)` — this call blocks for the lifetime of the process
- On SIGTERM: uvicorn exits, worker calls `bus.detach()` and exits cleanly

---

## 4. ground/server.py

FastAPI app with three endpoints.

### GET /stream

```
Content-Type: multipart/x-mixed-replace; boundary=frame
```

Async generator:
1. `asyncio.sleep(1/15)` — rate cap
2. `frame, _ = app.state.frame_buffer.read_latest()` — skip if None
3. `estimate = app.state.bus.latest("target/estimate")`
4. `jpeg = overlay.draw_overlay(frame, estimate)`
5. Yield multipart chunk

### GET /telemetry

```
Content-Type: text/event-stream
```

Async generator, 10 Hz:
1. `asyncio.sleep(0.1)`
2. Read `bus.latest("target/estimate")`, `bus.latest("fc/attitude")`
3. Collect latest HealthReport per process name from `bus.latest("system/health")`
   (ground worker maintains an in-memory `dict[str, HealthReport]` updated by the health-publish task)
4. Emit SSE event:

```json
{
  "tracker_health": "nominal",
  "confidence": 0.87,
  "roll_deg": 2.1,
  "pitch_deg": -1.4,
  "health": {
    "camera": "ok",
    "ccv_tracker": "ok",
    "ncv_tracker": "ok",
    "fusion": "ok",
    "link": "ok",
    "ground": "ok"
  }
}
```

Fields are `null` when the corresponding bus topic has never published.

### POST /lockon

Body: `{"x": float, "y": float, "w": float, "h": float}` — normalised bbox (0–1).

- Increments `app.state.lockon_seq`
- Publishes `LockOnCmd(monotonic_ns(), seq, BoundingBox(x, y, w, h))` to `lockon/cmd`
- Returns `{"ok": true}`

---

## 5. ground/overlay.py

```python
def draw_overlay(frame: np.ndarray, estimate: TargetEstimate | None) -> bytes
```

- If `estimate` is `None` or `tracker_health` is `NO_LOCK` or `LOST`: encode frame as-is.
- Otherwise draw a 2px rectangle at the tracker bbox:
  - `NOMINAL` → orange `(0, 165, 255)` in BGR
  - `UNCERTAIN` → yellow `(0, 255, 255)` in BGR
- Convert normalised bbox to pixel coords using `frame.shape`.
- Encode with `cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])`.
- Returns `bytes`.

The overlay does not draw the crosshair, text, attitude HUD, or confidence bar. Those are the browser's responsibility.

---

## 6. ground/static/index.html

Single-file, vanilla JS, no framework, no external resources.

### 6.1 Layout

```
┌─────────────────────────────────────────┐
│  quadguide                              │
├─────────────────────────────────────────┤
│   ┌──────────────────────────────────┐  │
│   │  <img id="stream" src="/stream"> │  │
│   │  <canvas> (absolute, on top)     │  │
│   └──────────────────────────────────┘  │
│                                         │
│  Size: 120px  [+]  [-]  [LOCK ON]       │
│                                         │
│  camera: ok   ccv: ok   ncv: ok         │
│  fusion: ok   link: ok  ground: ok      │
│  tracker: nominal   conf: 0.87          │
│  roll: 2.1°   pitch: -1.4°             │
└─────────────────────────────────────────┘
```

Dark background (`#111`), white/coloured monospace text. No decorative elements.

### 6.2 Stream + canvas stacking

```html
<div id="viewport" style="position:relative; display:inline-block">
  <img id="stream" src="/stream">
  <canvas id="overlay" style="position:absolute; top:0; left:0; pointer-events:none">
</div>
```

`pointer-events: none` on the canvas so the img tag receives any future mouse events.

A `ResizeObserver` on `#stream` keeps the canvas sized to match the rendered image. On resize, crosshair redraws immediately.

### 6.3 Crosshair drawing

Green (`#00ff00`), 2px line width. Drawn on every `requestAnimationFrame`.

```
center = (canvas.width/2, canvas.height/2)
half   = size/2
cornerLen = size/5

four corner L-shapes:
  top-left:     horizontal: (cx-half, cy-half) → (cx-half+cornerLen, cy-half)
                vertical:   (cx-half, cy-half) → (cx-half, cy-half+cornerLen)
  (mirror for other three corners)

four centre lines from cx,cy to inner edge of square:
  up:    (cx, cy) → (cx, cy-half)
  down:  (cx, cy) → (cx, cy+half)
  left:  (cx, cy) → (cx-half, cy)
  right: (cx, cy) → (cx+half, cy)
```

Default `size = 120px`. Min = 40px. Max = `Math.min(canvas.width, canvas.height) - 20`.

### 6.4 Keyboard handlers

Listeners on `document`:

| Key      | Action                  |
|----------|-------------------------|
| `+` / `=` | `size = Math.min(size + 10, maxSize)` |
| `-`      | `size = Math.max(size - 10, minSize)` |
| `Enter`  | compute bbox, POST `/lockon`, flash viewport border green for 300ms |

### 6.5 Lock-on bbox calculation

The crosshair square is always centered. Normalised bbox:

```
x = (canvas.width/2  - size/2) / canvas.width
y = (canvas.height/2 - size/2) / canvas.height
w = size / canvas.width
h = size / canvas.height
```

### 6.6 SSE telemetry handler

`EventSource("/telemetry")` updates text fields in-place on each message. No DOM rebuild.

Health field colours:
- `ok` → white
- `degraded` → `#ffaa00` (amber)
- `failsafe` / `dead` → `#ff4444` (red)
- missing (null) → `#888` (grey)

---

## 7. Dependencies

FastAPI and uvicorn must be added to the venv:

```bash
pip install fastapi "uvicorn[standard]"
```

`pyproject.toml` optional dependency group:

```toml
[project.optional-dependencies]
ground = ["fastapi", "uvicorn[standard]"]
```

---

## 8. Health publishing detail

Two independent responsibilities:

**Publishing ground's own health:** A 5 Hz `asyncio.Task` (started in the FastAPI lifespan) publishes `HealthReport("ground", ProcessState.OK, "")` to `system/health`.

**Collecting other workers' health:** `app.state.process_health: dict[str, str]` maps process name → latest state string. The SSE handler reads `bus.latest("system/health")` on each 10 Hz tick and upserts into this dict. Because `system/health` is a single shared topic with multiple publishers, `bus.latest()` returns only the most recently published report; reports from two processes published close together may be collapsed into one poll tick. This is acceptable for a status display — per-process state converges within one publish period (~200 ms at 5 Hz).

---

## 9. Out of scope

- Authentication
- CORS configuration
- HTTPS
- Physical button input (future hardware; keyboard is the current input method)
- Re-acquisition logic (operator presses Enter again with crosshair repositioned)

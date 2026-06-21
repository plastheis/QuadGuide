# Minimal Ground UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal 800×480 kiosk ground UI selectable via `ground.ui_mode` config, reusing all existing backend endpoints, with zero impact on the guidance/control pipeline.

**Architecture:** One new static file (`static/minimal.html`) rendered entirely client-side; one route change in `server.py` to choose which HTML `GET /` serves; a config key and a dev-launcher flag to toggle it. No new endpoints, no second video stream, no message/bus/pipeline changes. The PIP magnifier, crosshair, and status text are computed in the kiosk browser from the existing `/stream` (640×480 MJPEG, bbox burned in) and `/telemetry` (SSE).

**Tech Stack:** FastAPI + Starlette `FileResponse`, vanilla JS + `<canvas>` (no framework), pytest + Starlette `TestClient`.

**Spec:** `docs/superpowers/specs/2026-06-21-minimal-ground-ui-design.md`

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/quadguide/ground/static/minimal.html` | The kiosk UI (layout + crosshair + PIP + status + key handlers) | **Create** |
| `src/quadguide/ground/server.py` | `GET /` serves `minimal.html` when `ground.ui_mode == "minimal"` | Modify (`create_app`) |
| `tests/unit/test_ground_server.py` | Route-selection tests | Modify (append 2 tests) |
| `configs/config.yaml` | Document/enable the `ground` section + `ui_mode` | Modify (append section) |
| `scripts/dev_ground_perception.py` | Stop clobbering `ground` config; add `--minimal` flag | Modify (arg + config build) |

`minimal.html` is static (no JS test harness exists in this repo, matching `index.html`); it is verified manually in Task 4. The only automated coverage is the route selection (Task 2).

---

### Task 1: Create the minimal kiosk UI

**Files:**
- Create: `src/quadguide/ground/static/minimal.html`

This file is static markup + client JS. It depends only on telemetry fields `server.py` actually emits (`tracker_health`, `bbox_x/y/w/h`, `latency_ms`, `latency_avg_ms`, `health`). The route that serves it is added in Task 2; visual verification is Task 4.

- [ ] **Step 1: Create `src/quadguide/ground/static/minimal.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>quadguide kiosk</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 800px; height: 480px; overflow: hidden;
    background: #000; color: #e0e0e0; font-family: monospace;
  }
  #app { display: flex; width: 800px; height: 480px; }

  /* Left strip: 5 health boxes (fill height) + latency block */
  #left { width: 160px; height: 480px; display: flex; flex-direction: column;
          gap: 4px; padding: 4px; }
  #health { display: flex; flex-direction: column; gap: 4px; flex: 1; }
  .mod { flex: 1; border-radius: 3px; display: flex; align-items: center;
         justify-content: center; font-size: 14px; letter-spacing: 2px;
         text-transform: uppercase; color: #000; font-weight: bold;
         background: #444; transition: background-color 120ms linear;
         user-select: none; }
  .mod.uninit   { background: #444; color: #888; }
  .mod.ok       { background: #2ecc40; color: #000; }
  .mod.degraded { background: #ffdc00; color: #000; }
  .mod.failsafe,
  .mod.dead     { background: #ff4136; color: #000; }

  #latency { flex-shrink: 0; background: #181818; border: 1px solid #2a2a2a;
             padding: 6px; text-align: center; }
  #latency .lbl { color: #888; font-size: 10px; letter-spacing: 1px; }
  #lat-avg { font-size: 24px; color: #0f0; line-height: 1.1; }
  #lat-avg.warn   { color: #fa0; }
  #lat-avg.danger { color: #f44; }
  #lat-latest { font-size: 11px; color: #888; }

  /* Main feed: native 640x480, flush right */
  #feed { position: relative; width: 640px; height: 480px; }
  #stream  { position: absolute; top: 0; left: 0; width: 640px; height: 480px; }
  #overlay { position: absolute; top: 0; left: 0; width: 640px; height: 480px;
             pointer-events: none; }
  #pip { position: absolute; top: 0; right: 0; width: 160px; height: 160px;
         border: 1px solid #2a2a2a; background: #000; }

  /* On-screen status text */
  .status { position: absolute; left: 0; width: 640px; text-align: center;
            font-weight: bold; letter-spacing: 3px; pointer-events: none;
            text-shadow: 0 0 4px #000, 0 0 4px #000; }
  #armed { top: 10px; font-size: 26px; color: #ff4136; display: none; }
  #lock  { bottom: 10px; font-size: 24px; display: none; }
</style>
</head>
<body>
<div id="app">
  <div id="left">
    <div id="health">
      <div class="mod uninit" data-mod="camera">CAM</div>
      <div class="mod uninit" data-mod="tracker">TRK</div>
      <div class="mod uninit" data-mod="link">LINK</div>
      <div class="mod uninit" data-mod="guidance">GUID</div>
      <div class="mod uninit" data-mod="control">CTRL</div>
    </div>
    <div id="latency">
      <div class="lbl">LATENCY ms</div>
      <div id="lat-avg">&mdash;</div>
      <div id="lat-latest">&mdash;</div>
    </div>
  </div>

  <div id="feed">
    <img id="stream" src="/stream" alt="stream">
    <canvas id="overlay" width="640" height="480"></canvas>
    <canvas id="pip" width="160" height="160"></canvas>
    <div id="armed" class="status">ARMED</div>
    <div id="lock" class="status"></div>
  </div>
</div>

<script>
'use strict';

const W = 640, H = 480;
const streamImg = document.getElementById('stream');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const pip = document.getElementById('pip');
const pctx = pip.getContext('2d');
pctx.imageSmoothingEnabled = false;

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ── Crosshair (same model as the verbose UI) ───────────────────────────────
let crosshairSize = 160;
const STEP = 20, MIN_SIZE = 40, MAX_SIZE = Math.min(W, H) - 20;

function drawCrosshair() {
  octx.clearRect(0, 0, W, H);
  const cx = W / 2, cy = H / 2, half = crosshairSize / 2;
  octx.strokeStyle = '#00ff00';
  octx.lineWidth = 2;
  octx.beginPath();
  octx.moveTo(0, cy);          octx.lineTo(cx - half, cy);
  octx.moveTo(cx + half, cy);  octx.lineTo(W, cy);
  octx.moveTo(cx, 0);          octx.lineTo(cx, cy - half);
  octx.moveTo(cx, cy + half);  octx.lineTo(cx, H);
  octx.stroke();
  const arm = crosshairSize * 0.25;
  const corners = [
    [cx - half, cy - half,  arm, 0, 0,  arm],
    [cx + half, cy - half, -arm, 0, 0,  arm],
    [cx - half, cy + half,  arm, 0, 0, -arm],
    [cx + half, cy + half, -arm, 0, 0, -arm],
  ];
  octx.beginPath();
  for (const [ox, oy, dx1, dy1, dx2, dy2] of corners) {
    octx.moveTo(ox + dx1, oy + dy1);
    octx.lineTo(ox, oy);
    octx.lineTo(ox + dx2, oy + dy2);
  }
  octx.stroke();
}
drawCrosshair();

// ── PIP magnifier (client-side crop of the streamed frame) ─────────────────
const PAD = 0.20, MIN_PIP = 48, MAX_PIP = 480;
let trackState = { health: null, bx: 0, by: 0, bw: 0, bh: 0 };

function pipCropRect() {
  const h = trackState.health;
  const locked = (h === 'nominal' || h === 'uncertain') && trackState.bh > 0;
  if (locked) {
    const cx = (trackState.bx + trackState.bw / 2) * W;
    const cy = (trackState.by + trackState.bh / 2) * H;
    const side = clamp(trackState.bh * H * (1 + 2 * PAD), MIN_PIP, MAX_PIP);
    const sx = clamp(cx - side / 2, 0, W - side);
    const sy = clamp(cy - side / 2, 0, H - side);
    return [sx, sy, side];
  }
  const side = crosshairSize;
  return [(W - side) / 2, (H - side) / 2, side];
}

function drawPip() {
  if (!streamImg.complete || streamImg.naturalWidth === 0) return;
  const [sx, sy, side] = pipCropRect();
  pctx.clearRect(0, 0, 160, 160);
  try {
    pctx.drawImage(streamImg, sx, sy, side, side, 0, 0, 160, 160);
  } catch (e) { /* frame not ready this tick */ }
}
// Cap PIP redraw at the MJPEG rate (~15 Hz); also redrawn on crosshair/SSE change.
setInterval(drawPip, 1000 / 15);

// ── On-screen status text ──────────────────────────────────────────────────
let armState = false;
const elArmed = document.getElementById('armed');
const elLock  = document.getElementById('lock');

function updateArmedText() { elArmed.style.display = armState ? 'block' : 'none'; }

function updateLockText(health) {
  let txt = '', color = '';
  if (health === 'nominal' || health === 'uncertain') { txt = 'LOCK'; color = '#2ecc40'; }
  else if (health === 'lost')      { txt = 'LOST'; color = '#ffdc00'; }
  else if (health === 'acquiring') { txt = 'ACQ';  color = '#00ffff'; }
  if (txt) { elLock.textContent = txt; elLock.style.color = color; elLock.style.display = 'block'; }
  else     { elLock.style.display = 'none'; }
}

// ── Health column ──────────────────────────────────────────────────────────
const MODULE_KEYS = ['camera', 'tracker', 'link', 'guidance', 'control'];
function modStateClass(state) {
  if (state === 'ok')       return 'ok';
  if (state === 'degraded') return 'degraded';
  if (state === 'failsafe' || state === 'dead') return 'failsafe';
  return 'uninit';
}
function updateModuleHealth(health) {
  for (const key of MODULE_KEYS) {
    let state = null;
    if (health) {
      if (key === 'tracker') {
        const k = Object.keys(health).find(k => k === 'tracker' || k.startsWith('tracker_'));
        if (k) state = health[k];
      } else if (key in health) {
        state = health[key];
      }
    }
    const el = document.querySelector(`.mod[data-mod="${key}"]`);
    if (el) el.className = 'mod ' + modStateClass(state);
  }
}

// ── Latency block ──────────────────────────────────────────────────────────
const elLatAvg    = document.getElementById('lat-avg');
const elLatLatest = document.getElementById('lat-latest');
function updateLatency(latest, avg) {
  elLatAvg.textContent = avg != null ? avg.toFixed(1) : '—';
  elLatAvg.className = avg == null ? '' : avg > 100 ? 'danger' : avg > 50 ? 'warn' : '';
  elLatLatest.textContent = latest != null ? 'now ' + latest.toFixed(1) : '—';
}

// ── SSE telemetry ──────────────────────────────────────────────────────────
let lastHealth = null;
const sse = new EventSource('/telemetry');
sse.onmessage = (ev) => {
  let d;
  try { d = JSON.parse(ev.data); } catch { return; }
  lastHealth = d.tracker_health;
  trackState = {
    health: d.tracker_health,
    bx: d.bbox_x != null ? d.bbox_x : 0,
    by: d.bbox_y != null ? d.bbox_y : 0,
    bw: d.bbox_w != null ? d.bbox_w : 0,
    bh: d.bbox_h != null ? d.bbox_h : 0,
  };
  updateLockText(d.tracker_health);
  updateModuleHealth(d.health);
  updateLatency(d.latency_ms, d.latency_avg_ms);
  drawPip();
};
sse.onerror = () => { updateModuleHealth(null); };

// ── Commands ───────────────────────────────────────────────────────────────
function postJson(url, body) {
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  }).catch(() => {});
}
function sendLockon() {
  const half = crosshairSize / 2;
  postJson('/lockon', {
    x: (W / 2 - half) / W,
    y: (H / 2 - half) / H,
    w: crosshairSize / W,
    h: crosshairSize / H,
  });
}
function toggleLock() {
  // Direction derived from live tracker health so it self-heals across reload.
  const active = lastHealth === 'nominal' || lastHealth === 'uncertain' || lastHealth === 'lost';
  if (active) postJson('/reset_lockon', null);
  else        sendLockon();
}
function sendArm(armed) {
  armState = armed;
  updateArmedText();
  postJson('/arm', { armed });
}

// ── Keys (physical buttons / arm switch mapped to keys on the kiosk Pi) ─────
document.addEventListener('keydown', (e) => {
  if (e.key === '+' || e.key === '=') {
    crosshairSize = clamp(crosshairSize + STEP, MIN_SIZE, MAX_SIZE);
    drawCrosshair(); drawPip();
  } else if (e.key === '-' || e.key === '_') {
    crosshairSize = clamp(crosshairSize - STEP, MIN_SIZE, MAX_SIZE);
    drawCrosshair(); drawPip();
  } else if (e.key === 'Enter') {
    toggleLock();
  } else if (e.key === 'a') {
    sendArm(true);
  } else if (e.key === 'd') {
    sendArm(false);
  }
});
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add src/quadguide/ground/static/minimal.html
git commit -m "feat(ground): add minimal kiosk UI static page"
```

---

### Task 2: Serve minimal.html via ground.ui_mode (TDD)

**Files:**
- Modify: `src/quadguide/ground/server.py` (inside `create_app`, around line 31 and the `index` route at lines 52-54)
- Test: `tests/unit/test_ground_server.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_ground_server.py`:

```python
def test_index_serves_verbose_by_default():
    app = create_app(_MockBus(), _MockFrameBuffer())
    with TestClient(app) as c:
        body = c.get("/").text
    assert 'id="right-col"' in body          # verbose-only marker


def test_index_serves_minimal_when_configured():
    app = create_app(_MockBus(), _MockFrameBuffer(), {"ground": {"ui_mode": "minimal"}})
    with TestClient(app) as c:
        body = c.get("/").text
    assert 'id="pip"' in body                # minimal-only marker
```

- [ ] **Step 2: Run the tests to verify the minimal one fails**

Run: `python -m pytest tests/unit/test_ground_server.py -k "serves" -v`
Expected: `test_index_serves_verbose_by_default` PASSES (default already serves `index.html`); `test_index_serves_minimal_when_configured` FAILS — `assert 'id="pip"' in body` is False because `create_app` ignores `ui_mode` and serves `index.html`.

- [ ] **Step 3: Implement ui_mode resolution**

In `src/quadguide/ground/server.py`, inside `create_app`, just after the existing
`acquire_crop = overlay.acquire_crop_from_config(config)` line, add:

```python
    ui_mode = (config or {}).get("ground", {}).get("ui_mode", "verbose")
    index_file = "minimal.html" if ui_mode == "minimal" else "index.html"
```

Then change the index route from:

```python
    @app.get("/")
    async def index():
        return FileResponse(_STATIC / "index.html")
```

to:

```python
    @app.get("/")
    async def index():
        return FileResponse(_STATIC / index_file)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_ground_server.py -k "serves" -v`
Expected: both PASS.

- [ ] **Step 5: Run the full ground server test file (no regressions)**

Run: `python -m pytest tests/unit/test_ground_server.py -v`
Expected: all tests PASS (the existing lockon/arm/telemetry tests are unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/ground/server.py tests/unit/test_ground_server.py
git commit -m "feat(ground): serve minimal UI when ground.ui_mode=minimal"
```

---

### Task 3: Config key + dev launcher toggle

**Files:**
- Modify: `configs/config.yaml` (append `ground` section after the `bus:` block, lines 89-90)
- Modify: `scripts/dev_ground_perception.py` (arg parser ~line 30; config build lines 34-36)

- [ ] **Step 1: Add the `ground` section to `configs/config.yaml`**

Append at the end of the file (after `bus:\n  ring_depth: 8`):

```yaml

ground:
  port: 8080
  ui_mode: verbose      # "verbose" (full HUD) | "minimal" (800x480 kiosk)
```

- [ ] **Step 2: Stop the dev launcher clobbering `ground`, add `--minimal`**

In `scripts/dev_ground_perception.py`, add the flag in the arg parser (after the
`--log` argument, before `args = parser.parse_args()`):

```python
    parser.add_argument("--minimal", action="store_true",
                        help="Serve the minimal kiosk UI (ground.ui_mode=minimal)")
```

Then replace the config build line that overwrites `ground`:

```python
    config["ground"] = {"port": args.port}
```

with (preserves `ui_mode` from config.yaml, applies the flag override):

```python
    config.setdefault("ground", {})
    config["ground"]["port"] = args.port
    if args.minimal:
        config["ground"]["ui_mode"] = "minimal"
```

- [ ] **Step 3: Verify config loads and the launcher parses the flag**

Run: `python -c "from quadguide.core.config import load_config; c = load_config('configs/config.yaml', {}); print(c['ground'])"`
Expected: prints `{'port': 8080, 'ui_mode': 'verbose'}`

Run: `python scripts/dev_ground_perception.py --help`
Expected: help text lists `--minimal` with its description; exits 0.

- [ ] **Step 4: Confirm the full unit suite is green**

Run: `python -m pytest tests/unit -q`
Expected: all pass (no collection or import errors from the edits).

- [ ] **Step 5: Commit**

```bash
git add configs/config.yaml scripts/dev_ground_perception.py
git commit -m "feat(ground): config ground.ui_mode + dev --minimal flag"
```

---

### Task 4: Manual end-to-end verification (no commit)

**Files:** none (verification only)

This task confirms the client-side UI renders and behaves correctly. Requires a
camera at `/dev/video0` (or set `platform.camera.backend: virtual` in
`configs/config.yaml` for a no-hardware check). Run on/through a browser sized to
800×480 (kiosk, or a desktop browser window/devtools device set to 800×480).

- [ ] **Step 1: Launch the ground + perception stack with the minimal UI**

Run: `python scripts/dev_ground_perception.py --minimal`
Expected: console prints `ground station → http://0.0.0.0:8080`; processes start.

- [ ] **Step 2: Open the kiosk UI and verify layout**

Open `http://localhost:8080/` in a browser at 800×480.
Verify: no scrollbars; left 160px column shows 5 health boxes (CAM/TRK/LINK/GUID/CTRL)
filling the height with a latency block at the bottom; the 640×480 feed is flush
right; the 160×160 PIP sits in the feed's top-right corner.

- [ ] **Step 3: Verify crosshair + PIP zoom (unlocked)**

Press `+` and `-`.
Expected: the green crosshair grows/shrinks; the PIP shows a center crop that zooms
in as the crosshair shrinks and out as it grows.

- [ ] **Step 4: Verify lock toggle + locked PIP + status text**

Press `Enter` to lock onto the crosshair box.
Expected: `LOCK` (green) appears bottom-center once the tracker reports nominal/
uncertain; the bbox is drawn on the feed (from `/stream`); the PIP switches to a
square crop centered on the bbox. Press `Enter` again.
Expected: `/reset_lockon` fires, `LOCK` clears, PIP reverts to the crosshair crop.

- [ ] **Step 5: Verify arm text + latency**

Press `a` then `d`.
Expected: `ARMED` (red) appears top-center on `a` and disappears on `d`. With a lock
active, the latency block shows a glass→control number (avg large, `now` small).

- [ ] **Step 6: Confirm the verbose UI is unaffected**

Stop the stack, restart without `--minimal`: `python scripts/dev_ground_perception.py`
Open `http://localhost:8080/`.
Expected: the original verbose HUD loads (right-hand telemetry columns present).

---

## Self-Review

**Spec coverage:**
- §3 config `ground.ui_mode` → Task 3 Step 1.
- §4 server route selection → Task 2.
- §5.1 layout (160px left strip, 640×480 feed flush right, top-right PIP) → Task 1 CSS + Task 4 Step 2.
- §5.2 health column (5 modules, fill height, prefix-match tracker) → Task 1 `updateModuleHealth` + `#health` CSS.
- §5.3 latency (glass→control avg primary, latest dimmed, color thresholds) → Task 1 `updateLatency` + `#latency` CSS.
- §5.4 crosshair (same model, 160 default, STEP 20, 40..460) → Task 1 `drawCrosshair`.
- §5.5 PIP (client crop; crosshair-scale unlocked; bbox-centered locked, `bbox_h·(1+2·0.20)`, clamp 48..480, edge-clamped) → Task 1 `pipCropRect`/`drawPip`.
- §5.6 status text (ARMED top-center; LOCK/LOST/ACQ bottom-center) → Task 1 `updateArmedText`/`updateLockText` + `.status` CSS.
- §5.7 key map (`+`/`-`/`Enter`/`a`/`d`) → Task 1 `keydown` handler.
- §5.8 lock toggle (direction from tracker_health) → Task 1 `toggleLock`.
- §5.9 arm (latched level, two keys, idempotent; no backend change) → Task 1 `sendArm` (uses existing `/arm`).
- §6 performance (no new endpoints/stream; PIP capped at 15 Hz) → Task 1 `setInterval(drawPip, 1000/15)`; no backend additions beyond route selection.
- §7 verification (route test + manual) → Task 2 + Task 4.
- Dev-launcher `ground` clobber (would block testing the toggle) → Task 3 Step 2.

**Placeholder scan:** No TBD/TODO; the one `catch (e) { /* ... */ }` is an intentional no-op for a not-yet-ready frame, not a placeholder.

**Type/name consistency:** `crosshairSize`, `trackState`, `lastHealth`, `pipCropRect`, `drawPip`, `drawCrosshair`, `updateModuleHealth`, `updateLatency`, `updateLockText`, `updateArmedText`, `sendArm`, `toggleLock`, `sendLockon`, `postJson` are each defined once and referenced consistently. Route-test markers (`id="right-col"` verbose, `id="pip"` minimal) match the actual files. SSE field names (`tracker_health`, `bbox_x/y/w/h`, `latency_ms`, `latency_avg_ms`, `health`) match `server.py:_sse`.

---

## Out of scope

- The "fire" control and any fire/commit command (deferred).
- Camera / `/stream` resolution changes (stays 640×480).
- Auth / HTTPS / CORS.
- JS unit-test harness (none exists; `minimal.html` is verified manually like `index.html`).

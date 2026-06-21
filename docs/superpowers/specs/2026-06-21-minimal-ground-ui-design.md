# Minimal Ground UI Design

**Date:** 2026-06-21
**Status:** draft

---

## 1. Scope

Add a second, minimal operator GUI alongside the existing verbose `index.html`,
selectable via config. The minimal UI targets a Raspberry Pi web kiosk on a 7"
**800×480** display, streamed from the SBC over WiFi HaLow. Operator input comes
from physical controls wired to the kiosk Pi's GPIO and mapped to keyboard keys:
**3 momentary buttons** (zoom out, zoom in, lock-on toggle) and **1 maintained
toggle switch** (arm).

The verbose UI stays the default and is untouched. The minimal UI reuses every
existing backend endpoint unchanged.

### Non-goals / constraints

- **Zero impact on the guidance/control pipeline.** The ground worker is already
  a separate process that only calls `bus.latest()`; this design adds no new bus
  access and no new worker.
- **Minimal added SBC load and HaLow bandwidth.** No second video stream, no
  extra server-side crop/encode. All minimal-UI-specific rendering (PIP
  magnifier, crosshair, status text) runs **client-side in the kiosk browser**.
- The "fire" control is **deferred** (no command exists yet); the layout and key
  map leave room for it.

---

## 2. Architecture

### 2.1 What changes

| Area | Change |
|------|--------|
| `configs/config.yaml` | New `ground.ui_mode: verbose \| minimal` (default `verbose`). |
| `ground/server.py` | `GET /` serves `minimal.html` when `ui_mode == "minimal"`, else `index.html`. |
| `ground/static/minimal.html` | **New file** — the kiosk UI (single-file vanilla JS). |

**Unchanged:** `/stream`, `/telemetry`, `/lockon`, `/reset_lockon`, `/arm`,
`overlay.py`, `worker.py`, every `core/` / message / bus / pipeline module, and
`index.html`.

### 2.2 Non-interference

The minimal UI consumes the **same** `/stream` (15 Hz MJPEG, 640×480, bbox
already burned in by `overlay.draw_overlay`) and `/telemetry` (10 Hz SSE) the
verbose UI uses. The SBC does exactly the same work whether the kiosk runs the
verbose or minimal UI. The PIP magnifier and on-screen text are computed in the
browser from frames and telemetry the SBC already sends — so the airframe-side
cost is unchanged, and the kiosk Pi (not the SBC) renders the extras.

### 2.3 Data flow

```
/stream (640×480 MJPEG, bbox burned in) ─► <img id="stream">
                                              ├─► <canvas id="overlay"> crosshair (client)
                                              └─► <canvas id="pip"> drawImage(crop) (client)

/telemetry (SSE) ─► EventSource ─► health boxes + latency + status text + lock/PIP state

keydown a/d   ─► POST /arm {armed}          (latched level; link reads latest @50Hz)
keydown Enter ─► POST /lockon | /reset_lockon  (toggle; direction from tracker_health)
keydown +/-   ─► local crosshairSize (drives crosshair + unlocked PIP zoom)
```

---

## 3. Config change

`configs/config.yaml` — add a `ground` section (the port was already read from
`config["ground"]["port"]` in `worker.py`, defaulting to 8080):

```yaml
ground:
  port: 8080
  ui_mode: minimal      # "verbose" (default — full HUD) | "minimal" (kiosk)
```

`ui_mode` absent → `verbose` (no behavior change for existing configs).

---

## 4. server.py change

`create_app(bus, frame_buffer, config)` already receives the full config dict.
Resolve the index file once at app construction:

```python
ui_mode = (config or {}).get("ground", {}).get("ui_mode", "verbose")
_index_file = "minimal.html" if ui_mode == "minimal" else "index.html"

@app.get("/")
async def index():
    return FileResponse(_STATIC / _index_file)
```

No other route changes. This is the only Python edit.

---

## 5. ground/static/minimal.html

Single-file, vanilla JS, no framework, no external resources — same conventions
as `index.html`. Body is locked to 800×480 with `overflow:hidden` (kiosk, no
scroll).

### 5.1 Layout (800×480)

```
┌────────┬─────────────────────────────────────────────────┐
│ CAMERA │                                          ┌─────┐ │
│ TRACKER│                  ARMED                   │ PIP │ │ 160×160
│ LINK   │                                          └─────┘ │
│GUIDANCE│            MAIN FEED 640 × 480                    │
│CONTROL │         crosshair (client) + bbox (stream)        │
│────────│                  LOCK / LOST / ACQ                │
│ LAT ms │                                                   │
└────────┴─────────────────────────────────────────────────┘
  160px                     640px (native, full height)
```

- **Left strip — 160 px wide, full height.** 5 health boxes (flex column,
  `flex:1` each so they stretch to fill the 480 height) above a latency block
  pinned at the bottom.
- **Main feed — 640×480, flush right.** `<img id="stream" src="/stream">` at
  native size; `<canvas id="overlay" 640×480>` absolutely on top
  (`pointer-events:none`) for the crosshair. The tracking bbox arrives already
  drawn in the MJPEG (server `overlay.py`), same as the verbose UI.
- **PIP — `<canvas id="pip" 160×160>`** absolutely positioned at the feed's
  **top-right corner**.
- **Status text** — absolutely positioned over the feed: `ARMED` top-center,
  `LOCK`/`LOST`/`ACQ` bottom-center (only one of these shows at a time).

### 5.2 Health column

Reuse the verbose UI's module model verbatim: keys
`['camera','tracker','link','guidance','control']`, classes
`uninit / ok / degraded / failsafe / dead`, and the same `updateModuleHealth`
logic (tracker matched by `tracker` or `tracker_` prefix), driven by SSE
`d.health`. Boxes are restyled to a 160 px column that fills the 480 height.

### 5.3 Latency block (bottom-left)

The end-to-end **glass→control** latency — the only true end-to-end figure the
stack exposes. SSE already provides `latency_avg_ms` (smoothed over 20) and
`latency_ms` (latest). Show the smoothed average as the primary figure with the
latest below it dimmed. Color thresholds match the verbose UI: `>100 ms` danger
(red), `>50 ms` warn (amber), else normal. `—` until a lock exists
(`latency_*` is null pre-lock).

### 5.4 Crosshair (same as now)

Identical to `index.html`: green `#00ff00`, 2 px, scalable, centered on image
center. `crosshairSize` defaults to 160, `STEP=20`, `MIN_SIZE=40`,
`MAX_SIZE=min(640,480)-20=460`. `+`/`=` grow, `-`/`_` shrink. Redrawn on the
overlay canvas event-driven (key change / resize), not per frame. The acquire
crop guide is already burned into `/stream` by the server for AcquireTrack
trackers, so the client does not redraw it.

### 5.5 PIP magnifier (client-side crop)

A `setInterval` at ~15 Hz (the MJPEG rate) plus a redraw on each crosshair change
and SSE tick calls:

```js
pipCtx.drawImage(streamImg, sx, sy, side, side, 0, 0, 160, 160);
```

The `<img>` natural size is 640×480, so canvas px == image px (1:1). Crop rect:

- **Unlocked / lost / acquiring / no-lock** — crop = the crosshair square,
  centered on image center: `side = crosshairSize`,
  `sx = 320 - side/2`, `sy = 240 - side/2`. Smaller crosshair → higher zoom; it
  zooms live with `+`/`-`.
- **Locked** (`tracker_health ∈ {nominal, uncertain}` with a bbox present) — crop
  centered on the bbox centroid:
  - `cx = (bbox_x + bbox_w/2)·640`, `cy = (bbox_y + bbox_h/2)·480`
  - `side = clamp(bbox_h·480·(1 + 2·PAD), MIN_PIP, MAX_PIP)` with `PAD = 0.20`
    (20% of bbox height each side), `MIN_PIP = 48`, `MAX_PIP = 480` — so a tiny
    bbox doesn't zoom to mush and a large one can't exceed the frame.
  - `sx = clamp(cx - side/2, 0, 640 - side)`, `sy = clamp(cy - side/2, 0,
    480 - side)` so the crop stays inside the frame near edges.

The bbox is already drawn in the stream, so it appears (zoomed) inside the PIP
automatically — no separate PIP bbox draw needed.

### 5.6 On-screen status text

Plain absolutely-positioned HTML/CSS, updated each SSE tick (and on `/arm`):

| Text | Condition | Color |
|------|-----------|-------|
| `ARMED` (top-center) | local `armState === true` | red |
| `LOCK` (bottom-center) | `tracker_health ∈ {nominal, uncertain}` | green |
| `LOST` (bottom-center) | `tracker_health === "lost"` | amber |
| `ACQ` (bottom-center) | `tracker_health === "acquiring"` | cyan |

Bottom-center shows at most one of LOCK/LOST/ACQ; `no_lock`/null → hidden.

### 5.7 Controls / key map

| Physical control | Key(s) | Action |
|------------------|--------|--------|
| Button — zoom out | `-` / `_` | `crosshairSize -= STEP` (PIP zooms out) |
| Button — zoom in  | `+` / `=` | `crosshairSize += STEP` (PIP zooms in) |
| Button — lock     | `Enter`   | **toggle** lock ⇄ release (§5.8) |
| Toggle switch — arm | `a` / `d` | latched arm / disarm (§5.9) |

(`fire` deferred — a future key/button + command slots in here.)

### 5.8 Lock-on toggle

`Enter` toggles, with direction derived from the latest SSE `tracker_health`
(stored in a `lastHealth` var) so it self-heals across a kiosk reload:

```
active = lastHealth ∈ {nominal, uncertain, lost}
active ? POST /reset_lockon : POST /lockon {crosshair box}
```

`acquiring` / `no_lock` / null are treated as "not active" → `Enter` commits a
lock. Reuses both existing endpoints; no backend change. The crosshair-box
payload is computed exactly as in `index.html` (`sendLockon`).

### 5.9 Arm (maintained toggle switch)

`arm/cmd` is left exactly as-is — a **latched level** (`ArmCmd(armed: bool)`)
that the link worker reads via `bus.latest("arm/cmd")` every 20 ms. A maintained
switch *is* a level, so it maps cleanly:

- `keydown 'a'` → `POST /arm {armed:true}`; set `armState=true`; show `ARMED`.
- `keydown 'd'` → `POST /arm {armed:false}`; set `armState=false`; hide `ARMED`.

Re-sends are idempotent (link just keeps reading `latest()`). Two distinct keys
(not a JS toggle) keep the **physical switch position authoritative** — a JS
toggle flag could desync from the switch after a browser reload, which is a
safety hazard for arm.

**GPIO-side guidance (firmware, not quadguide code):** emit one keystroke per
switch *edge* (not a held key, which would auto-repeat POSTs). Recommended polish:
re-emit the current switch position every ~1 s so arm state self-heals if the
kiosk browser reloads (a reload otherwise loses the UI's knowledge of switch
position; the latched bus value is unaffected, so the system stays in its commanded
state — only the on-screen `ARMED` indicator would be stale until the next edge).

---

## 6. Performance notes

- No new endpoints, no second stream, no extra server-side encode → SBC ground
  worker load and HaLow bandwidth are identical to today.
- PIP is a single 160×160 `drawImage` from an already-decoded `<img>`, capped at
  the ~15 Hz stream rate, running on the **kiosk Pi**. Redrawing faster than the
  stream only re-crops the same frame, so it's capped deliberately.
- Crosshair and status text are event-driven (key/SSE), not per-frame.
- The guidance/control processes are never touched: the ground worker still only
  calls `bus.latest()`, and this design adds zero bus access.

---

## 7. Testing / verification

- **Route selection (integration test):** with `ground.ui_mode: minimal`,
  `GET /` returns the `minimal.html` body; with `verbose`/absent it returns
  `index.html`. This is the only added Python logic.
- **Manual (browser at 800×480):** run `scripts/dev_ground_perception.py` with
  `ui_mode: minimal`; verify layout fits 800×480 with no scroll, crosshair
  scales, PIP zooms with `+`/`-` unlocked and follows the bbox when locked,
  health boxes fill the height and track `system/health`, latency shows
  glass→control, status text shows ARMED / LOCK / LOST / ACQ correctly, and the
  `Enter` toggle and `a`/`d` arm keys POST the right requests.

---

## 8. Out of scope

- The "fire" control and any fire/commit command (deferred to a later change).
- Changing the camera or `/stream` resolution (stays 640×480).
- Auth / HTTPS / CORS.
- The verbose UI's latent SSE field mismatches (`active_tracker`, `centroid_x`);
  the minimal UI deliberately depends only on fields `server.py` actually emits
  (`tracker_health`, `bbox_*`, `latency_*`, `video_fps`, `health`).

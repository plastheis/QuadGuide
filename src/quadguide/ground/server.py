from __future__ import annotations
import asyncio
import json
import math
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from quadguide.core.clock import monotonic_ns
from quadguide.core.config import ARDUCOPTER_MODES
from quadguide.core.messages import (
    ArmCmd, BoundingBox, FailsafeActionWire, FireCmd, HealthReport, LockOnCmd,
    ProcessState,
)
from quadguide.ground import overlay

# custom_mode number → friendly name, for the HUD failsafe banner.
_MODE_NAMES = {v: k for k, v in ARDUCOPTER_MODES.items()}

_STATIC      = Path(__file__).parent / "static"
_MJPEG_RATE  = 1 / 15   # 15 Hz
_SSE_RATE    = 0.1       # 10 Hz
_HEALTH_RATE = 0.2       # 5 Hz

# ── Shared HUD state ────────────────────────────────────────────────────────
# Operator-toggleable view settings live on the server, not in each browser, so
# every connected client (kiosk + laptop) shows the same HUD. Clients apply a
# change optimistically and POST /ui; the authoritative state rides back out on
# the /telemetry SSE, which is what makes the other clients converge.
#
# `crosshair` is in the clients' 640x400 overlay-canvas pixel space (both UIs
# use that canvas size regardless of the camera's native resolution).
_UI_CANVAS_W, _UI_CANVAS_H = 640, 400
_CROSSHAIR_MIN  = 40
_CROSSHAIR_MAX  = min(_UI_CANVAS_W, _UI_CANVAS_H) - 20
_UI_DEFAULTS = {
    "crosshair": 160,    # px, in the 640x400 overlay canvas space
    "show_bbox": True,   # burn the tracker box into the MJPEG stream
    "show_osd":  True,   # on-video status text (ARMED / FIRE / LOCK / FAILSAFE)
}

# Pre-encoded black frame served before the camera is ready.
_NO_SIGNAL_JPEG: bytes = cv2.imencode(
    ".jpg", np.zeros((480, 640, 3), dtype=np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80]
)[1].tobytes()


def create_app(bus, frame_buffer, config: dict | None = None) -> FastAPI:
    acquire_crop = overlay.acquire_crop_from_config(config)
    ui_mode = (config or {}).get("ground", {}).get("ui_mode", "verbose")
    tonemap_mode = (config or {}).get("ground", {}).get("tonemap", "percentile")
    index_file = "minimal.html" if ui_mode == "minimal" else "index.html"

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        app.state.bus            = bus
        app.state.frame_buffer   = frame_buffer
        app.state.acquire_crop   = acquire_crop
        app.state.tonemap_mode   = tonemap_mode
        app.state.lockon_seq     = 0
        app.state.ui             = dict(_UI_DEFAULTS)
        app.state.ui_seq         = 0
        app.state.process_health: dict[str, str] = {}
        app.state.latency_window: deque = deque(maxlen=20)
        app.state.mjpeg_frames   = 0
        app.state.mjpeg_fps      = 0.0
        app.state.mjpeg_fps_ts   = monotonic_ns()
        task = asyncio.create_task(_health_task(app))
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(lifespan=_lifespan)

    @app.get("/")
    async def index():
        return FileResponse(_STATIC / index_file)

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

    @app.post("/reset_lockon")
    async def reset_lockon(request: Request):
        request.app.state.lockon_seq += 1
        cmd = LockOnCmd(
            timestamp_ns=monotonic_ns(),
            seq=request.app.state.lockon_seq,
            bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
        )
        request.app.state.bus.publish("lockon/cmd", cmd)
        return {"ok": True}

    @app.post("/arm")
    async def arm(body: _ArmBody, request: Request):
        cmd = ArmCmd(timestamp_ns=monotonic_ns(), armed=body.armed)
        request.app.state.bus.publish("arm/cmd", cmd)
        return {"ok": True}

    @app.post("/fire")
    async def fire(body: _FireBody, request: Request):
        cmd = FireCmd(timestamp_ns=monotonic_ns(), active=body.active)
        request.app.state.bus.publish("fire/cmd", cmd)
        return {"ok": True}

    @app.get("/ui")
    async def get_ui(request: Request):
        """Shared HUD state, so a client starts in sync instead of waiting a tick."""
        return _ui_payload(request.app)

    @app.post("/ui")
    async def set_ui(body: _UiBody, request: Request):
        """Merge a partial HUD update; unset fields keep their current value."""
        return _apply_ui(request.app, body.model_dump(exclude_unset=True))

    return app


def _apply_ui(app: FastAPI, patch: dict) -> dict:
    """Merge ``patch`` into the shared HUD state and bump its sequence number.

    Only bumps ``ui_seq`` when something actually changed, so a client can ignore
    its own echo (and any redundant broadcast) by comparing sequence numbers.
    """
    changed = False
    for key, value in patch.items():
        if value is None or key not in _UI_DEFAULTS:
            continue
        if key == "crosshair":
            value = max(_CROSSHAIR_MIN, min(_CROSSHAIR_MAX, int(value)))
        else:
            value = bool(value)
        if app.state.ui[key] != value:
            app.state.ui[key] = value
            changed = True
    if changed:
        app.state.ui_seq += 1
    return _ui_payload(app)


def _ui_payload(app: FastAPI) -> dict:
    return {**app.state.ui, "seq": app.state.ui_seq}


class _LockOnBody(BaseModel):
    x: float
    y: float
    w: float
    h: float


class _ArmBody(BaseModel):
    armed: bool


class _FireBody(BaseModel):
    active: bool


class _UiBody(BaseModel):
    """Partial HUD update — every field optional, absent fields left untouched."""
    crosshair: int | None = None
    show_bbox: bool | None = None
    show_osd: bool | None = None


async def _mjpeg(app: FastAPI):
    while True:
        await asyncio.sleep(_MJPEG_RATE)
        frame, _ = app.state.frame_buffer.read_latest()
        if frame is None:
            jpeg = _NO_SIGNAL_JPEG
        else:
            frame = overlay.to_display_bgr(frame, app.state.tonemap_mode)
            estimate = app.state.bus.latest("target/estimate")
            jpeg = overlay.draw_overlay(
                frame, estimate, app.state.acquire_crop,
                show_bbox=app.state.ui["show_bbox"],
            )
        app.state.mjpeg_frames += 1
        now = monotonic_ns()
        dt = (now - app.state.mjpeg_fps_ts) / 1e9
        if dt >= 1.0:
            app.state.mjpeg_fps = app.state.mjpeg_frames / dt
            app.state.mjpeg_frames = 0
            app.state.mjpeg_fps_ts = now
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


def _failsafe_name(fs) -> str:
    """failsafe/action → "none" | "disarm" | "set_mode" (never None, for the HUD)."""
    if fs is None:
        return "none"
    return FailsafeActionWire(fs.action).name.lower()


def _failsafe_mode_name(fs) -> str | None:
    """Friendly ArduCopter mode name for a latched SET_MODE, else None."""
    if fs is None or fs.action != FailsafeActionWire.SET_MODE:
        return None
    return _MODE_NAMES.get(int(fs.custom_mode), str(fs.custom_mode))


async def _sse(app: FastAPI):
    while True:
        await asyncio.sleep(_SSE_RATE)
        estimate = app.state.bus.latest("target/estimate")
        attitude = app.state.bus.latest("fc/attitude")
        imu      = app.state.bus.latest("fc/imu")
        accel    = app.state.bus.latest("guidance/accel")
        control  = app.state.bus.latest("control/cmd")
        fire_cmd = app.state.bus.latest("fire/cmd")
        arm_cmd  = app.state.bus.latest("arm/cmd")
        failsafe = app.state.bus.latest("failsafe/action")
        fc_status = app.state.bus.latest("fc/status")
        report   = app.state.bus.latest("system/health")

        # End-to-end glass→control latency from the propagated origin_ns.
        # Use control.timestamp_ns (the publish time) rather than now, so the
        # SSE polling delay is excluded by construction. None until a lock exists.
        lat_ms = ((control.timestamp_ns - control.origin_ns) / 1e6
                  if control and control.origin_ns > 0 else None)
        if lat_ms is not None:
            app.state.latency_window.append(lat_ms)
        avg_ms = (sum(app.state.latency_window) / len(app.state.latency_window)
                  if app.state.latency_window else None)
        if report is not None:
            app.state.process_health[report.process] = report.state.value
        # Derive tracker algo from health process key ("tracker_kcf" → "kcf")
        tracker_algo = next(
            (k[8:] for k in app.state.process_health if k.startswith("tracker_")), None
        )

        data = {
            # target/estimate
            "tracker_health": estimate.tracker_health.value  if estimate else None,
            "confidence":     estimate.confidence             if estimate else None,
            "bbox_x":         estimate.bbox.x                 if estimate else None,
            "bbox_y":         estimate.bbox.y                 if estimate else None,
            "bbox_w":         estimate.bbox.w                 if estimate else None,
            "bbox_h":         estimate.bbox.h                 if estimate else None,
            # tracker process name (derived from system/health)
            "tracker_algo":   tracker_algo,
            # fc/attitude
            "roll_deg":       math.degrees(attitude.roll_rad)       if attitude else None,
            "pitch_deg":      math.degrees(attitude.pitch_rad)      if attitude else None,
            "yaw_deg":        math.degrees(attitude.yaw_rad)        if attitude else None,
            "roll_rate_dps":  math.degrees(attitude.roll_rate_rps)  if attitude else None,
            "pitch_rate_dps": math.degrees(attitude.pitch_rate_rps) if attitude else None,
            "yaw_rate_dps":   math.degrees(attitude.yaw_rate_rps)   if attitude else None,
            # fc/imu
            "imu_ax": imu.ax if imu else None,
            "imu_ay": imu.ay if imu else None,
            "imu_az": imu.az if imu else None,
            "imu_gx": imu.gx if imu else None,
            "imu_gy": imu.gy if imu else None,
            "imu_gz": imu.gz if imu else None,
            # guidance/accel
            "accel_ax": accel.ax if accel else None,
            "accel_ay": accel.ay if accel else None,
            # control/cmd
            "ctrl_roll_deg":     control.roll_deg       if control else None,
            "ctrl_pitch_deg":    control.pitch_deg      if control else None,
            "ctrl_yaw_rate_dps": control.yaw_rate_dps   if control else None,
            "ctrl_throttle":     control.throttle_norm   if control else None,
            # fire/cmd
            "fire_active":       bool(fire_cmd and fire_cmd.active),
            # arm/cmd — the operator's COMMANDED arm intent (what gates the
            # failsafe latches), distinct from fc_armed below. Sourced from the
            # bus, not the kiosk's click state, so it survives a page reload.
            "armed":             bool(arm_cmd and arm_cmd.armed),
            # failsafe/action — latched terminal action from the control worker.
            "failsafe_action":   _failsafe_name(failsafe),
            "failsafe_mode":     _failsafe_mode_name(failsafe),
            # fc/status — ground-truth arm/mode from HEARTBEAT (None until first beat)
            "fc_armed":          (fc_status.armed       if fc_status else None),
            "fc_mode":           (fc_status.custom_mode if fc_status else None),
            # system/health
            "health": dict(app.state.process_health),
            # latency
            "latency_ms":     lat_ms,
            "latency_avg_ms": avg_ms,
            # video stream fps
            "video_fps": app.state.mjpeg_fps if app.state.mjpeg_fps > 0 else None,
            # shared HUD state — broadcast so a toggle on one client reaches all
            "ui": _ui_payload(app),
        }
        yield f"data: {json.dumps(data)}\n\n"


async def _health_task(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(_HEALTH_RATE)
        app.state.bus.publish(
            "system/health",
            HealthReport(monotonic_ns(), "ground", ProcessState.OK, ""),
        )

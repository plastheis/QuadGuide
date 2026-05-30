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
from quadguide.core.messages import ArmCmd, BoundingBox, HealthReport, LockOnCmd, ProcessState
from quadguide.ground import overlay

_STATIC      = Path(__file__).parent / "static"
_MJPEG_RATE  = 1 / 15   # 15 Hz
_SSE_RATE    = 0.1       # 10 Hz
_HEALTH_RATE = 0.2       # 5 Hz

# Pre-encoded black frame served before the camera is ready.
_NO_SIGNAL_JPEG: bytes = cv2.imencode(
    ".jpg", np.zeros((480, 640, 3), dtype=np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80]
)[1].tobytes()


def create_app(bus, frame_buffer) -> FastAPI:

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        app.state.bus            = bus
        app.state.frame_buffer   = frame_buffer
        app.state.lockon_seq     = 0
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

    return app


class _LockOnBody(BaseModel):
    x: float
    y: float
    w: float
    h: float


class _ArmBody(BaseModel):
    armed: bool


async def _mjpeg(app: FastAPI):
    while True:
        await asyncio.sleep(_MJPEG_RATE)
        frame, _ = app.state.frame_buffer.read_latest()
        if frame is None:
            jpeg = _NO_SIGNAL_JPEG
        else:
            estimate = app.state.bus.latest("target/estimate")
            jpeg = overlay.draw_overlay(frame, estimate)
        app.state.mjpeg_frames += 1
        now = monotonic_ns()
        dt = (now - app.state.mjpeg_fps_ts) / 1e9
        if dt >= 1.0:
            app.state.mjpeg_fps = app.state.mjpeg_frames / dt
            app.state.mjpeg_frames = 0
            app.state.mjpeg_fps_ts = now
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


async def _sse(app: FastAPI):
    while True:
        await asyncio.sleep(_SSE_RATE)
        estimate = app.state.bus.latest("target/estimate")
        lat_ms = estimate.latency_ns / 1e6 if estimate and estimate.latency_ns > 0 else None
        if lat_ms is not None:
            app.state.latency_window.append(estimate.latency_ns)
        avg_ms = (
            sum(app.state.latency_window) / len(app.state.latency_window) / 1e6
            if app.state.latency_window else None
        )
        attitude = app.state.bus.latest("fc/attitude")
        imu      = app.state.bus.latest("fc/imu")
        accel    = app.state.bus.latest("guidance/accel")
        control  = app.state.bus.latest("control/cmd")
        report   = app.state.bus.latest("system/health")
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
            # system/health
            "health": dict(app.state.process_health),
            # latency
            "latency_ms":     lat_ms,
            "latency_avg_ms": avg_ms,
            # video stream fps
            "video_fps": app.state.mjpeg_fps if app.state.mjpeg_fps > 0 else None,
        }
        yield f"data: {json.dumps(data)}\n\n"


async def _health_task(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(_HEALTH_RATE)
        app.state.bus.publish(
            "system/health",
            HealthReport(monotonic_ns(), "ground", ProcessState.OK, ""),
        )

from __future__ import annotations
import asyncio
import json
import math
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
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
            jpeg = _NO_SIGNAL_JPEG
        else:
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

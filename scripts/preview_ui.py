#!/usr/bin/env python3
"""Preview the ground UI (minimal or verbose) on any machine — no hardware/bus.

Runs the REAL FastAPI ground server (``create_app`` + ``/stream`` + ``/telemetry``
+ ``overlay.py``) backed by a fake bus and fake frame buffer that synthesise a
moving target, telemetry, and health. This never imports ``core.bus`` /
``core.frame_buffer`` (which need the Unix-only ``fcntl``), so it runs on Windows.

What you see is the production UI code, driven by dummy data. It is interactive:
  - ``Enter`` toggles lock-on (LOCK text + bbox + PIP-follow appear/disappear)
  - ``a`` / ``d`` show/hide ARMED
  - ``+`` / ``-`` zoom the crosshair and PIP

    python scripts/preview_ui.py            # minimal kiosk UI (default)
    python scripts/preview_ui.py --verbose  # the full HUD
    python scripts/preview_ui.py --port 9000
"""
from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import uvicorn

from quadguide.core.clock import monotonic_ns
from quadguide.core.messages import (
    AccelCmd, AttitudeState, BoundingBox, ControlCmd, HealthReport, IMUFrame,
    ProcessState, TrackerEstimate, TrackerHealth,
)
from quadguide.ground.server import create_app

_W, _H = 640, 480
_MODULES = ["camera", "tracker_kcf", "link", "guidance", "control"]
_FAKE_LATENCY_NS = 16_000_000  # 16 ms glass→control lineage


def _target_norm(t: float) -> tuple[float, float, float, float]:
    """Animated bbox (x, y, w, h) normalised — a gentle Lissajous drift."""
    cx = 0.5 + 0.25 * math.sin(t * 0.6)
    cy = 0.5 + 0.18 * math.cos(t * 0.9)
    w = 0.12 + 0.02 * math.sin(t * 1.3)
    h = 0.16 + 0.02 * math.cos(t * 1.1)
    return cx - w / 2, cy - h / 2, w, h


class _FakeFrameBuffer:
    """Synthesises a 640x480 BGR frame with a moving target each read."""

    def read_latest(self):
        t = monotonic_ns() / 1e9
        frame = np.full((_H, _W, 3), (28, 24, 20), dtype=np.uint8)
        for x in range(0, _W, 40):
            cv2.line(frame, (x, 0), (x, _H), (40, 36, 32), 1)
        for y in range(0, _H, 40):
            cv2.line(frame, (0, y), (_W, y), (40, 36, 32), 1)
        bx, by, bw, bh = _target_norm(t)
        cx, cy = int((bx + bw / 2) * _W), int((by + bh / 2) * _H)
        cv2.circle(frame, (cx, cy), max(4, int(bh * _H * 0.5)), (60, 180, 255), -1)
        cv2.circle(frame, (cx, cy), max(2, int(bh * _H * 0.22)), (255, 255, 255), -1)
        cv2.putText(frame, "SYNTHETIC PREVIEW", (8, _H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        return frame, monotonic_ns()


class _FakeBus:
    """Read-only telemetry source; publish() lets the lock toggle affect the UI."""

    def __init__(self) -> None:
        self.locked = True
        self._hi = 0  # rotates system/health so all 5 module boxes populate

    def latest(self, topic: str):
        now = monotonic_ns()
        t = now / 1e9
        if topic == "target/estimate":
            bx, by, bw, bh = _target_norm(t)
            health = TrackerHealth.NOMINAL if self.locked else TrackerHealth.NO_LOCK
            return TrackerEstimate(now, BoundingBox(bx, by, bw, bh), 0.93, health,
                                   origin_ns=now - _FAKE_LATENCY_NS)
        if topic == "control/cmd":
            origin = now - _FAKE_LATENCY_NS if self.locked else 0
            return ControlCmd(now, 6.0 * math.sin(t), -4.0 * math.cos(t * 0.8),
                              12.0 * math.sin(t * 0.5), 0.42, origin_ns=origin)
        if topic == "fc/attitude":
            return AttitudeState(now, 0.10 * math.sin(t), 0.08 * math.cos(t),
                                 (0.4 * t) % 6.28 - 3.14, 0.2 * math.cos(t),
                                 0.2 * math.sin(t), 0.1 * math.sin(t * 1.5))
        if topic == "fc/imu":
            return IMUFrame(now, 0.2 * math.sin(t), 0.2 * math.cos(t), 9.81,
                            0.05 * math.sin(t), 0.05 * math.cos(t), 0.02)
        if topic == "guidance/accel":
            origin = now - _FAKE_LATENCY_NS if self.locked else 0
            return AccelCmd(now, 1.5 * math.sin(t), 1.2 * math.cos(t), origin_ns=origin)
        if topic == "system/health":
            name = _MODULES[self._hi % len(_MODULES)]
            self._hi += 1
            return HealthReport(now, name, ProcessState.OK, "")
        return None

    def publish(self, topic: str, msg) -> None:
        # A zero-size lock-on bbox is "release" (reset); any non-zero bbox is "lock".
        if topic == "lockon/cmd":
            self.locked = bool(msg.bbox.w > 0 and msg.bbox.h > 0)

    def detach(self) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview the quadguide ground UI (no hardware)")
    ap.add_argument("--verbose", action="store_true",
                    help="Show the full HUD (default: minimal kiosk UI)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    ui_mode = "verbose" if args.verbose else "minimal"
    config = {"ground": {"ui_mode": ui_mode}}
    app = create_app(_FakeBus(), _FakeFrameBuffer(), config)
    print(f"preview ({ui_mode}) -> http://127.0.0.1:{args.port}/   (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

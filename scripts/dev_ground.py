#!/usr/bin/env python3
"""
Standalone ground-module dev launcher for RPi test bench.

Starts a camera capture loop in a background thread and runs the ground
web server in the foreground.  No tracker or flight-controller processes
are required — bus.latest() calls return None and the UI shows the raw
camera feed with no overlay.

Usage:
    python scripts/dev_ground.py [--port 8080] [--camera 0]
"""
from __future__ import annotations

import argparse
import threading

import cv2

from quadguide.core.bus import Bus
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.ground.worker import run as ground_run


def _camera_loop(cap: cv2.VideoCapture, fb: FrameBuffer, stop: threading.Event) -> None:
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        fb.write_frame(frame)
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="dev ground launcher")
    parser.add_argument("--port",   type=int, default=8080)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    bus = Bus()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fb  = FrameBuffer(width=640, height=480)

    stop = threading.Event()
    cam_thread = threading.Thread(target=_camera_loop, args=(cap, fb, stop), daemon=True)
    cam_thread.start()

    try:
        ground_run({"ground": {"port": args.port}}, bus, fb)
    finally:
        stop.set()
        cam_thread.join(timeout=2)
        fb.unlink()


if __name__ == "__main__":
    main()

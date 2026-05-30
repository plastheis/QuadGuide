#!/usr/bin/env python3
"""
Dev launcher for ground + perception pipeline (camera → tracker → ground).

The single tracker process is built from `tracker.import` in config.yaml. The
camera backend is forced to v4l2 (USBCamera, /dev/video0).

Usage:
    python scripts/dev_ground_perception.py [--port 8080] [--config configs/config.yaml]
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import signal

from quadguide.core.bus import Bus
from quadguide.core.config import load_config
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.ground.worker import run as ground_run
from quadguide.perception.camera.worker import run_from_config as camera_run
from quadguide.perception.tracker_worker import run_from_config as tracker_run


def main() -> None:
    parser = argparse.ArgumentParser(description="dev ground + perception launcher")
    parser.add_argument("--port",   type=int, default=8080)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config, {})
    config["platform"]["camera"]["backend"] = "v4l2"
    config["ground"] = {"port": args.port}

    w = config["platform"]["camera"]["width"]
    h = config["platform"]["camera"]["height"]

    bus = Bus()
    fb  = FrameBuffer(width=w, height=h)

    cam_proc = multiprocessing.Process(
        target=camera_run, args=(config, bus, fb), daemon=True, name="camera",
    )
    tracker_proc = multiprocessing.Process(
        target=tracker_run, args=(config, bus, fb), daemon=True, name="tracker",
    )

    cam_proc.start()
    tracker_proc.start()

    print(f"camera PID {cam_proc.pid}  tracker PID {tracker_proc.pid}")
    print(f"ground station → http://0.0.0.0:{args.port}")

    all_procs = [cam_proc, tracker_proc]
    try:
        ground_run(config, bus, fb)
    finally:
        for proc in all_procs:
            if proc.is_alive() and proc.pid is not None:
                os.kill(proc.pid, signal.SIGTERM)
        for proc in all_procs:
            proc.join(timeout=3)
        fb.unlink()
        bus.close()


if __name__ == "__main__":
    main()

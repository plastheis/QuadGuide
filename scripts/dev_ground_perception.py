#!/usr/bin/env python3
"""
Dev launcher for ground + perception pipeline (camera → CCV → fusion → ground).

The CCV tracker algorithm is selected by config.tracker.ccv (default: kcf).
Change it to "mosse" in config.yaml without touching this script.

Tests the communication path between the perception and ground modules:
  - operator clicks bbox in browser → ground POSTs /lockon → lockon/cmd on bus
  - CCV worker reads lockon/cmd, initialises tracker, publishes ccv_tracker/estimate
  - fusion worker receives ccv_tracker/estimate and publishes target/estimate
  - ground overlay and telemetry reflect the tracked bbox

NCV worker is not started; fusion runs in single-tracker passthrough mode,
forwarding ccv_tracker/estimate directly to target/estimate.

The camera backend is forced to v4l2 (USBCamera, /dev/video0) regardless of
what configs/config.yaml specifies.

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
from quadguide.perception.ccv_tracker_worker import run_from_config as ccv_run
from quadguide.perception.fusion.worker import run as fusion_run


def _camera_proc(config: dict, bus: Bus, fb: FrameBuffer) -> None:
    camera_run(config, bus, fb)


def _ccv_proc(config: dict, bus: Bus, fb: FrameBuffer) -> None:
    ccv_run(config, bus, fb)


def _fusion_proc(config: dict, bus: Bus, fb: FrameBuffer) -> None:
    fusion_run(config, bus, fb)


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

    cam_proc    = multiprocessing.Process(
        target=_camera_proc, args=(config, bus, fb), daemon=True, name="camera",
    )
    ccv_proc    = multiprocessing.Process(
        target=_ccv_proc, args=(config, bus, fb), daemon=True, name="ccv",
    )
    fusion_proc = multiprocessing.Process(
        target=_fusion_proc, args=(config, bus, fb), daemon=True, name="fusion",
    )

    cam_proc.start()
    ccv_proc.start()
    fusion_proc.start()
    ccv_algo = config["tracker"]["ccv"]
    print(f"camera PID {cam_proc.pid}  ccv({ccv_algo}) PID {ccv_proc.pid}  fusion PID {fusion_proc.pid}")
    print(f"ground station → http://0.0.0.0:{args.port}")

    try:
        ground_run(config, bus, fb)
    finally:
        for proc in (cam_proc, ccv_proc, fusion_proc):
            if proc.is_alive() and proc.pid is not None:
                os.kill(proc.pid, signal.SIGTERM)
        for proc in (cam_proc, ccv_proc, fusion_proc):
            proc.join(timeout=3)
        fb.unlink()
        bus.close()


if __name__ == "__main__":
    main()

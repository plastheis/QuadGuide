#!/usr/bin/env python3
"""
Dev launcher for ground + perception pipeline (camera → trackers → fusion → ground).

Tracker processes are started based on what is configured in config.yaml:
  - tracker.ccv present → CCV worker started
  - tracker.ncv present → NCV worker started
  - only one tracker configured → fusion runs in passthrough mode

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
from quadguide.core.config import cfg_tracker, load_config
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.ground.worker import run as ground_run
from quadguide.perception.camera.worker import run_from_config as camera_run
from quadguide.perception.fusion.worker import run as fusion_run


def main() -> None:
    parser = argparse.ArgumentParser(description="dev ground + perception launcher")
    parser.add_argument("--port",   type=int, default=8080)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config, {})
    config["platform"]["camera"]["backend"] = "v4l2"
    config["ground"] = {"port": args.port}

    tcfg = cfg_tracker(config)
    w = config["platform"]["camera"]["width"]
    h = config["platform"]["camera"]["height"]

    bus = Bus()
    fb  = FrameBuffer(width=w, height=h)

    tracker_procs: list[multiprocessing.Process] = []

    if tcfg.ccv is not None:
        from quadguide.perception.ccv_tracker_worker import run_from_config as ccv_run
        tracker_procs.append(multiprocessing.Process(
            target=ccv_run, args=(config, bus, fb), daemon=True, name=f"ccv({tcfg.ccv})",
        ))

    if tcfg.ncv is not None:
        from quadguide.perception.nanotrack.worker import run as ncv_run
        tracker_procs.append(multiprocessing.Process(
            target=ncv_run, args=(config, bus, fb), daemon=True, name=f"ncv({tcfg.ncv})",
        ))

    cam_proc = multiprocessing.Process(
        target=camera_run, args=(config, bus, fb), daemon=True, name="camera",
    )
    fusion_proc = multiprocessing.Process(
        target=fusion_run, args=(config, bus, fb), daemon=True, name="fusion",
    )

    cam_proc.start()
    for p in tracker_procs:
        p.start()
    fusion_proc.start()

    tracker_summary = "  ".join(f"{p.name} PID {p.pid}" for p in tracker_procs) or "no trackers (passthrough)"
    print(f"camera PID {cam_proc.pid}  {tracker_summary}  fusion PID {fusion_proc.pid}")
    print(f"ground station → http://0.0.0.0:{args.port}")

    all_procs = [cam_proc, *tracker_procs, fusion_proc]
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

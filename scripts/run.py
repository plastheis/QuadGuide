#!/usr/bin/env python3
"""Flight orchestrator — forks all workers and supervises them.

Any worker that exits (cleanly or with an error) triggers a SIGTERM to all
remaining workers, followed by a SIGKILL after a grace period.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import signal
import sys
import time


# ── Worker entry points ──────────────────────────────────────────────────────

def _ncv_run(config: dict, bus, frame_buffer) -> None:
    """Build inference runtime + tracker inside the child process, then run."""
    from quadguide.inference.factory import get_runtime
    from quadguide.perception.tracker_factories import get_ncv_tracker
    from quadguide.perception.ncv_tracker_worker import NCVTrackerWorker
    runtime = get_runtime(config)
    tracker = get_ncv_tracker(config, runtime)
    NCVTrackerWorker(tracker, bus, frame_buffer, config=config).run()


# ── Process management ───────────────────────────────────────────────────────

def _start_workers(config: dict, bus, frame_buffer, *, ground: bool = True) -> list[multiprocessing.Process]:
    from quadguide.core.config import cfg_tracker
    from quadguide.perception.camera.worker import run_from_config as camera_run
    from quadguide.perception.ccv_tracker_worker import run_from_config as ccv_run
    from quadguide.perception.fusion.worker import run as fusion_run
    from quadguide.link.worker import run as link_run
    from quadguide.guidance.worker import run as guidance_run
    from quadguide.control.worker import run as control_run
    from quadguide.ground.worker import run as ground_run

    tcfg = cfg_tracker(config)

    entries: list[tuple[str, object, tuple]] = [
        ("camera",   camera_run,   (config, bus, frame_buffer)),
    ]
    if tcfg.ccv is not None:
        entries.append(("ccv_tracker", ccv_run,  (config, bus, frame_buffer)))
    if tcfg.ncv is not None:
        entries.append(("ncv_tracker", _ncv_run, (config, bus, frame_buffer)))
    if tcfg.ccv is not None or tcfg.ncv is not None:
        entries.append(("fusion",      fusion_run, (config, bus, frame_buffer)))
    entries += [
        ("link",     link_run,     (config, bus)),
        ("guidance", guidance_run, (config, bus, frame_buffer)),
        ("control",  control_run,  (config, bus, frame_buffer)),
    ]
    if ground:
        entries.append(("ground", ground_run, (config, bus, frame_buffer)))

    procs = []
    for name, target, args in entries:
        p = multiprocessing.Process(target=target, args=args, name=name, daemon=False)
        p.start()
        procs.append(p)
    return procs


def _shutdown(procs: list[multiprocessing.Process], grace: float = 5.0) -> None:
    for p in procs:
        if p.is_alive():
            os.kill(p.pid, signal.SIGTERM)

    deadline = time.monotonic() + grace
    for p in procs:
        remaining = max(0.0, deadline - time.monotonic())
        p.join(timeout=remaining)

    for p in procs:
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)
            p.join()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuadGuide flight orchestrator")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        metavar="PATH",
        help="Path to YAML config file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value using dot-notation (e.g. --set guidance.N=5.0)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Override logging.level from config",
    )
    parser.add_argument(
        "--log-dir",
        metavar="PATH",
        help="Override logging.dir from config",
    )
    parser.add_argument(
        "--no-ground",
        action="store_true",
        help="Skip the ground station worker (no HTTP API for arm/lockon)",
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    overrides: dict[str, str] = {}
    for item in args.overrides:
        key, _, val = item.partition("=")
        if not key or not val:
            print(f"[orchestrator] bad --set argument: {item!r} (expected KEY=VALUE)", file=sys.stderr)
            return 2
        overrides[key] = val
    if args.log_level:
        overrides["logging.level"] = args.log_level
    if args.log_dir:
        overrides["logging.dir"] = args.log_dir

    from quadguide.core.bus import Bus
    from quadguide.core.config import cfg_bus, cfg_platform, load_config
    from quadguide.core.frame_buffer import FrameBuffer

    try:
        config = load_config(args.config, overrides)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[orchestrator] config error: {exc}", file=sys.stderr)
        return 2

    bcfg = cfg_bus(config)
    pcfg = cfg_platform(config)

    bus = Bus(ring_depth=bcfg.ring_depth)
    frame_buffer = FrameBuffer(pcfg.camera.width, pcfg.camera.height)

    procs = _start_workers(config, bus, frame_buffer, ground=not args.no_ground)
    print(f"[orchestrator] started {len(procs)} workers: {[p.name for p in procs]}")

    stop = False
    failed: multiprocessing.Process | None = None

    def _on_signal(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    while not stop:
        for p in procs:
            if p.exitcode is not None:
                failed = p
                stop = True
                break
        if not stop:
            time.sleep(0.1)

    if failed is not None:
        msg = (
            f"worker '{failed.name}' exited with code {failed.exitcode}"
            if failed.exitcode != 0
            else f"worker '{failed.name}' exited unexpectedly"
        )
        print(f"[orchestrator] {msg} — shutting down", file=sys.stderr)

    _shutdown(procs)
    bus.close()

    return 1 if failed is not None else 0


if __name__ == "__main__":
    sys.exit(main())

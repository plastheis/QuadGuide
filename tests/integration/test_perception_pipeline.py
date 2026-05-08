"""Integration test: camera + CCV + NCV tracker workers communicate over the bus.

Uses 'fork' explicitly (Linux default) so workers inherit shared-memory handles
and pipe fds without pickling. Non-picklable objects (_SyntheticCamera,
_MockRuntime) survive the fork because they are already in the parent's memory.
"""
from __future__ import annotations
import multiprocessing
import os
import signal
import time

import numpy as np
import pytest

from quadguide.core.bus import Bus
from quadguide.core.config import load_config
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.messages import TrackerEstimate, TrackerHealth
from quadguide.perception.camera.sources import CameraSource
from quadguide.perception.camera.worker import run as run_camera
from quadguide.perception.ccv_tracker_worker import CCVTrackerWorker
from quadguide.perception.kcf.tracker import KCFTracker
from quadguide.perception.ncv_tracker_worker import NCVTrackerWorker
from quadguide.perception.nanotrack.tracker import NanoTracker

import pathlib
CONFIG_PATH = str(pathlib.Path(__file__).parents[2] / "configs" / "config.yaml")


# ── test-only fakes ──────────────────────────────────────────────────────────

class _SyntheticCamera(CameraSource):
    """Generates 640×480 frames at max speed — no hardware required."""
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._width, self._height = width, height
        self._i = 0

    def open(self) -> None:
        pass

    def read(self) -> tuple[np.ndarray, int]:
        frame = np.full((self._height, self._width, 3), self._i % 255, dtype=np.uint8)
        self._i += 1
        return frame, time.monotonic_ns()

    def close(self) -> None:
        pass


class _MockRuntime:
    """Returns correctly-shaped zero arrays so postprocess never crashes."""

    def load(self, path: str):
        return {"path": path}

    def infer(self, model, inputs: dict) -> dict:
        if "input" in inputs:
            return {"features": np.zeros((1, 256, 6, 6), dtype=np.float32)}
        # head call
        return {
            "score": np.zeros((1, 1, 25, 25), dtype=np.float32),
            "bbox":  np.zeros((1, 4, 25, 25), dtype=np.float32),
        }

    def close(self) -> None:
        pass


# ── worker entry points (top-level for fork) ─────────────────────────────────

def _run_camera(source, fb, bus):
    run_camera(source, fb, bus)


def _run_ccv(tracker, fb, bus):
    CCVTrackerWorker(tracker, bus, fb).run()


def _run_ncv(tracker, fb, bus):
    NCVTrackerWorker(tracker, bus, fb).run()


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def bus_and_fb():
    config = load_config(CONFIG_PATH, {"platform.inference.device": "cpu"})
    bus = Bus(ring_depth=8)
    fb  = FrameBuffer(width=640, height=480)
    yield bus, fb, config
    bus.close()
    fb.unlink()


# ── test ─────────────────────────────────────────────────────────────────────

def test_ccv_and_ncv_publish_tracker_estimates(bus_and_fb):
    """All three workers run concurrently; both tracker topics receive valid messages."""
    bus, fb, config = bus_and_fb
    ctx = multiprocessing.get_context("fork")

    camera  = _SyntheticCamera()
    runtime = _MockRuntime()
    from quadguide.core.config import cfg_tracker
    tcfg    = cfg_tracker(config)

    kcf_tracker  = KCFTracker(tcfg.kcf)
    nano_tracker = NanoTracker(
        runtime, runtime.load("bb.onnx"), runtime.load("hd.onnx"), tcfg.nanotrack
    )

    procs = [
        ctx.Process(target=_run_camera, args=(camera, fb, bus)),
        ctx.Process(target=_run_ccv,    args=(kcf_tracker, fb, bus)),
        ctx.Process(target=_run_ncv,    args=(nano_tracker, fb, bus)),
    ]
    for p in procs:
        p.start()

    deadline = time.monotonic() + 2.0
    ccv_ok = ncv_ok = False
    while time.monotonic() < deadline and not (ccv_ok and ncv_ok):
        time.sleep(0.05)
        ccv_ok = bus.latest("ccv_tracker/estimate") is not None
        ncv_ok = bus.latest("ncv_tracker/estimate") is not None

    for p in procs:
        os.kill(p.pid, signal.SIGTERM)
    for p in procs:
        p.join(timeout=3.0)

    # Assertions
    ccv_msg = bus.latest("ccv_tracker/estimate")
    ncv_msg = bus.latest("ncv_tracker/estimate")

    assert ccv_msg is not None, "ccv_tracker/estimate: no message within 2 s"
    assert ncv_msg is not None, "ncv_tracker/estimate: no message within 2 s"

    assert isinstance(ccv_msg, TrackerEstimate), f"expected TrackerEstimate, got {type(ccv_msg)}"
    assert isinstance(ncv_msg, TrackerEstimate), f"expected TrackerEstimate, got {type(ncv_msg)}"

    assert ccv_msg.tracker_health in list(TrackerHealth), \
        f"ccv health {ccv_msg.tracker_health!r} is not a valid TrackerHealth"
    assert ncv_msg.tracker_health in list(TrackerHealth), \
        f"ncv health {ncv_msg.tracker_health!r} is not a valid TrackerHealth"

    # No lock-on sent, so both should be NO_LOCK
    assert ccv_msg.tracker_health == TrackerHealth.NO_LOCK
    assert ncv_msg.tracker_health == TrackerHealth.NO_LOCK

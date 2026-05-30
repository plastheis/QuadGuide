"""Integration test: camera + single tracker worker over the bus.

Uses 'fork' (Linux default) so workers inherit shared-memory handles and pipe
fds. The synthetic camera feeds a stable bright rectangle; the tracker is the
real cv2 TrackerKCF wrapped by OpenCVTrackerAdapter.
"""
from __future__ import annotations
import multiprocessing
import os
import pathlib
import signal
import time

import numpy as np
import pytest

from quadguide.core.bus import Bus
from quadguide.core.config import load_config
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.messages import BoundingBox, LockOnCmd, TrackerEstimate
from quadguide.perception.camera.sources import CameraSource
from quadguide.perception.camera.worker import run as run_camera
from quadguide.perception.tracker_worker import (
    OpenCVTrackerAdapter, TrackerWorker,
)

CONFIG_PATH = str(pathlib.Path(__file__).parents[2] / "configs" / "config.yaml")


class _SyntheticCamera(CameraSource):
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._width, self._height = width, height
        self._i = 0

    def open(self) -> None: pass
    def close(self) -> None: pass

    def read(self) -> tuple[np.ndarray, int]:
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame[100:300, 200:400] = 200
        self._i += 1
        return frame, time.monotonic_ns()


def _run_camera(source, fb, bus):
    run_camera(source, fb, bus)


def _run_tracker(tracker, fb, bus):
    TrackerWorker(tracker, bus, fb, cpu_core=None, config={}).run()


@pytest.fixture
def bus_and_fb():
    bus = Bus(ring_depth=8)
    fb = FrameBuffer(640, 480)
    yield bus, fb
    bus.close()
    fb.unlink()


@pytest.mark.skipif(os.name != "posix", reason="fork is Linux-only")
def test_tracker_worker_publishes_target_estimate(bus_and_fb):
    bus, fb = bus_and_fb
    ctx = multiprocessing.get_context("fork")
    camera_proc = ctx.Process(
        target=_run_camera, args=(_SyntheticCamera(), fb, bus),
        name="camera", daemon=False,
    )
    tracker = OpenCVTrackerAdapter("TrackerKCF", {})
    tracker_proc = ctx.Process(
        target=_run_tracker, args=(tracker, fb, bus),
        name="tracker", daemon=False,
    )

    camera_proc.start()
    tracker_proc.start()
    try:
        time.sleep(0.2)
        bus.publish("lockon/cmd", LockOnCmd(
            timestamp_ns=time.monotonic_ns(), seq=1,
            bbox=BoundingBox(200/640, 100/480, 200/640, 200/480),
        ))

        deadline = time.monotonic() + 2.0
        est = None
        while time.monotonic() < deadline:
            est = bus.latest("target/estimate")
            if est is not None:
                break
            time.sleep(0.05)

        assert isinstance(est, TrackerEstimate), "no estimate published"
    finally:
        for p in (camera_proc, tracker_proc):
            if p.is_alive():
                os.kill(p.pid, signal.SIGTERM)
            p.join(timeout=2.0)
            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)
                p.join()


def test_config_path_loads_cleanly():
    """The shipped config.yaml is loadable and produces TrackerConfig."""
    from quadguide.core.config import cfg_tracker, cfg_platform
    cfg = load_config(CONFIG_PATH, {})
    tcfg = cfg_tracker(cfg)
    assert tcfg.import_spec
    pcfg = cfg_platform(cfg)
    assert pcfg.realtime.control_cpu_core == 3

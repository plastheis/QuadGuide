"""Generic tracker worker.

Owns the lockon/cmd subscription, the SHM frame read, and the target/estimate
publish — but no tracking algorithm. The algorithm is chosen at startup from
config.tracker.import and built once via load_tracker(). The worker treats the
tracker as opaque; trackers satisfy a small structural protocol
(name/init/update/reset/close).
"""
from __future__ import annotations

import importlib
import os
import signal
import time
from collections import namedtuple

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import (
    BoundingBox, HealthReport, LockOnCmd, ProcessState,
    TrackerEstimate, TrackerHealth,
)

__all__ = [
    "OpenCVTrackerAdapter", "TrackerWorker",
    "load_tracker", "run_from_config",
]

_HEALTH_EVERY = 50
_IDLE_POLL_S  = 0.001  # CPU yield when no new frame; caps new-frame detection delay

_TrackerOutput = namedtuple("_TrackerOutput", "bbox confidence health")
_BBox          = namedtuple("_BBox",          "x y w h")


# ── OpenCV adapter ──────────────────────────────────────────────────────────

def _resolve_cv2_factory(class_name: str):
    import cv2
    if hasattr(cv2, class_name):
        return getattr(cv2, class_name).create
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, class_name):
        return getattr(cv2.legacy, class_name).create
    raise AttributeError(
        f"cv2 has no tracker named {class_name!r} on cv2 or cv2.legacy"
    )


class OpenCVTrackerAdapter:
    """Wraps a cv2 tracker (pixel tuples + success bool) in the structural
    output protocol the worker reads.
    """

    def __init__(self, class_name: str, params: dict) -> None:
        self._factory = _resolve_cv2_factory(class_name)
        self._params = params  # held; cv2 trackers take typed Params, no YAML bridge yet
        self._name = class_name.lower().removeprefix("tracker")
        self._tracker = None
        self._initialized = False

    def name(self) -> str:
        return self._name

    def init(self, frame, bbox) -> None:
        h, w = frame.shape[:2]
        self._tracker = self._factory()
        self._tracker.init(frame, (
            int(bbox.x * w),
            int(bbox.y * h),
            max(1, int(bbox.w * w)),
            max(1, int(bbox.h * h)),
        ))
        self._initialized = True

    def update(self, frame):
        if not self._initialized:
            return _TrackerOutput(_BBox(0.0, 0.0, 0.0, 0.0), 0.0, "no_lock")
        h, w = frame.shape[:2]
        ok, bbox_px = self._tracker.update(frame)
        if ok:
            x, y, bw, bh = bbox_px
            return _TrackerOutput(_BBox(x / w, y / h, bw / w, bh / h), 1.0, "nominal")
        return _TrackerOutput(_BBox(0.0, 0.0, 0.0, 0.0), 0.0, "lost")

    def reset(self) -> None:
        self._initialized = False
        self._tracker = None

    def close(self) -> None:
        pass


# ── Loader ──────────────────────────────────────────────────────────────────

def load_tracker(config: dict):
    """Construct the tracker selected by config['tracker']['import'].

    `cv2:Foo` → OpenCVTrackerAdapter wrapping cv2.Foo / cv2.legacy.Foo.
    Anything else → importlib.import_module(module) and cls(**params).
    """
    tcfg = config["tracker"]
    spec = tcfg["import"]
    params = tcfg.get("params") or {}

    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise ValueError(
            f"tracker.import must be 'module:Class', got {spec!r}"
        )

    if module_name == "cv2":
        return OpenCVTrackerAdapter(class_name, params)

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**params)


# ── Worker ──────────────────────────────────────────────────────────────────

class TrackerWorker:
    """IPC loop owning lockon/cmd subscription, SHM frame read, and
    target/estimate publish. Tracker is opaque — no per-implementation branching.
    """

    def __init__(
        self,
        tracker,
        bus: Bus,
        frame_buffer: FrameBuffer,
        cpu_core: int | None = None,
        config: dict | None = None,
    ) -> None:
        self._tracker   = tracker
        self._bus       = bus
        self._fb        = frame_buffer
        self._cpu_core  = cpu_core
        self._config    = config or {}
        self._last_seq: int | None = None
        self._stop      = False
        self._proc_name = f"tracker_{tracker.name()}"

    def run(self) -> None:
        from quadguide.core.config import cfg_diag
        from quadguide.core.diagtrace import DiagTrace

        log = setup_logging(self._proc_name, self._config)
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        if self._cpu_core is not None:
            try:
                os.sched_setaffinity(0, {self._cpu_core})
            except (AttributeError, OSError):
                pass

        dcfg = cfg_diag(self._config)
        trace = DiagTrace(self._proc_name, enabled=dcfg.trace,
                          dir=dcfg.trace_dir, max_rows=dcfg.trace_max_rows)

        log.info(f"{self._proc_name}: started")
        i = 0
        last_ts = -1
        last_health: str | None = None
        try:
            while not self._stop:
                self._check_lockon()
                frame, frame_ts = self._fb.read_latest()
                # New-frame gate: process each frame exactly once. When the camera
                # hasn't produced a new frame yet, yield the CPU instead of
                # re-timestamping the same frame thousands of times — that
                # reprocessing was what manufactured the latency sawtooth and pinned
                # the core. Adds at most _IDLE_POLL_S of new-frame detection delay.
                if frame is None or frame_ts == last_ts:
                    time.sleep(_IDLE_POLL_S)
                    continue
                last_ts = frame_ts

                out     = self._tracker.update(frame)
                now_ns  = monotonic_ns()
                # origin_ns is the capture timestamp this estimate derives from.
                # Async trackers (e.g. EdgeCV AcquireTrack) supply their own
                # source-frame origin so the lineage reflects inference lag; fall
                # back to this worker's frame_ts when the tracker provides none.
                tracker_origin = getattr(out, "origin_ns", 0)
                origin_ns = tracker_origin if tracker_origin else (
                    frame_ts if frame_ts > 0 else 0)
                est = TrackerEstimate(
                    timestamp_ns=now_ns,
                    bbox=BoundingBox(out.bbox.x, out.bbox.y, out.bbox.w, out.bbox.h),
                    confidence=float(out.confidence),
                    tracker_health=TrackerHealth(out.health),
                    origin_ns=origin_ns,
                )
                self._bus.publish("target/estimate", est)
                # First hop: stage == cumulative (in_ts == origin == frame_ts).
                trace.latency(now_ns, origin_ns or None, origin_ns)

                # Record every health transition (not just the periodic snapshot
                # below) so the trace captures the exact YOLO→NanoTrack handoff:
                # acquiring→nominal is a lock, nominal→acquiring/uncertain a
                # drop/re-acquire. Transitions are rare, so this is near-free.
                health = str(out.health)
                if health != last_health:
                    trace.health(now_ns, health, detail=f"conf={float(out.confidence):.2f}")
                    last_health = health

                i += 1
                if i % _HEALTH_EVERY == 0:
                    self._bus.publish(
                        "system/health",
                        HealthReport(monotonic_ns(), self._proc_name, ProcessState.OK, ""),
                    )
                    trace.state(monotonic_ns(), algo=self._tracker.name(),
                                health=str(out.health), confidence=float(out.confidence))
        finally:
            trace.flush()
            self._tracker.close()
            self._bus.detach()
            log.info(f"{self._proc_name}: stopped")

    def _check_lockon(self) -> None:
        cmd: LockOnCmd | None = self._bus.latest("lockon/cmd")
        if cmd is None or cmd.seq == self._last_seq:
            return
        self._last_seq = cmd.seq
        if cmd.bbox.w == 0.0 and cmd.bbox.h == 0.0:
            self._tracker.reset()
            return
        frame, _ = self._fb.read_latest()
        if frame is not None:
            self._tracker.init(frame, cmd.bbox)

    def _handle_sigterm(self, sig, frame) -> None:
        self._stop = True


def run_from_config(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    """Build the tracker selected by config.tracker.import and run it."""
    from quadguide.core.config import cfg_platform
    pcfg = cfg_platform(config)
    tracker = load_tracker(config)
    TrackerWorker(
        tracker, bus, frame_buffer,
        cpu_core=pcfg.realtime.tracker_cpu_core,
        config=config,
    ).run()

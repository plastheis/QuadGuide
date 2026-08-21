"""AcquireTrack worker bodies (spec 2026-06-14-acquire-track-design §3).

Two spawned children, each pinned to its own NPU core. Both free-run reading the
latest frame and the parent's control word; each infers ONLY when its `mode` is
active (mutual exclusion), keeping its RKNN context warm while idle.

Everything that touches a backend is constructed INSIDE the child (§7.4/§14.7);
the parent never imports a model. The per-iteration logic lives in the
``YoloWorker``/``NanoWorker`` ``step()`` methods so it is unit-testable in-process
without spawning; ``_yolo_main``/``_nanotrack_main`` are the thin spawn entrypoints.
"""

from __future__ import annotations

import time

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.runtime.shm.control_channel import AcquireControlChannel, Mode
from edgecv.runtime.shm.frame_ring import FrameRing
from edgecv.runtime.shm.nano_result import NanoResultChannel
from edgecv.runtime.shm.payload import PayloadChannel
from edgecv.runtime.worker import detach_resource_tracker, request_death_with_parent

_IDLE_SLEEP = 0.001


class YoloWorker:
    """Per-iteration YOLO detection logic (mode-gated).

    `detector` exposes ``detect(frame, roi) -> DetectorOutput`` with boxes
    normalised to the full frame (e.g. ``YoloDetectorAdapter``). Publishes all
    detections to the YOLO payload channel; the parent picks the winner.
    """

    def __init__(self, detector, frame_ring: FrameRing,
                 control: AcquireControlChannel, result: PayloadChannel) -> None:
        self._detector = detector
        self._frame_ring = frame_ring
        self._control = control
        self._result = result
        self._last_seq = 0

    def step(self) -> bool:
        snap = self._control.read_latest()
        if snap.mode not in (Mode.YOLO, Mode.BOTH):  # BOTH: concurrent verify
            return False
        fr = self._frame_ring.read_latest()
        if fr is None:
            return False
        frame, seq, ts = fr
        if seq <= self._last_seq:
            return False
        self._last_seq = seq
        det = self._detector.detect(frame, snap.crop)
        boxes = np.ascontiguousarray(det.boxes, np.float32)
        if boxes.size == 0:
            boxes = boxes.reshape(0, 4)
        scores = np.ascontiguousarray(det.scores, np.float32).reshape(-1)
        self._result.publish(
            {"boxes": boxes, "scores": scores,
             "src_ts": np.array([ts], np.float64)},
            seq,
        )
        return True


class NanoWorker:
    """Per-iteration NanoTrack tracking logic (mode-gated).

    `nanotrack` exposes ``init(frame, bbox)`` and ``update(frame) -> TrackResult``.
    Re-initialises whenever the control word's ``lock_gen`` changes (the parent
    bumps it on every lock / re-lock), then publishes each update to the
    NanoTrack result channel.
    """

    def __init__(self, nanotrack, frame_ring: FrameRing,
                 control: AcquireControlChannel, result: NanoResultChannel) -> None:
        self._nano = nanotrack
        self._frame_ring = frame_ring
        self._control = control
        self._result = result
        self._last_seq = 0
        self._last_lock_gen = 0
        self._inited = False

    def step(self) -> str:
        snap = self._control.read_latest()
        if snap.mode not in (Mode.NANO, Mode.BOTH):  # BOTH: concurrent verify
            return "idle"
        fr = self._frame_ring.read_latest()
        if fr is None:
            return "no_frame"
        frame, seq, ts = fr

        if snap.lock_gen != self._last_lock_gen and snap.lock_gen != 0:
            self._last_lock_gen = snap.lock_gen
            self._nano.init(frame, snap.lock_bbox)
            self._inited = True
            self._last_seq = seq
            self._publish(self._nano.update(frame), seq, ts)
            return "init"

        if not self._inited:
            return "wait_lock"
        if seq <= self._last_seq:
            return "stale"
        self._last_seq = seq
        self._publish(self._nano.update(frame), seq, ts)
        return "update"

    def _publish(self, res, seq: int, ts: float) -> None:
        bbox = res.bbox if res.bbox is not None else BoundingBox(0.0, 0.0, 0.0, 0.0)
        conf = res.confidence if res.confidence is not None else 0.0
        self._result.publish(bbox, confidence=conf, status=res.status,
                             src_seq=seq, src_ts=ts)


# ── in-child component factories (picklable; built inside the worker) ────────

def build_yolo_detector(cfg: dict):
    """Build a YoloDetectorAdapter inside the YOLO worker (loads model in-child)."""
    from edgecv.models.manifest import load_manifest
    from edgecv.trackers.hybrid.detector_adapter import YoloDetectorAdapter

    mf = load_manifest(cfg["manifest"])
    return YoloDetectorAdapter(manifest=mf, backend=cfg.get("backend", "auto"),
                               **cfg.get("kwargs", {}))


def build_nanotrack(cfg: dict):
    """Build NanoTrack inside the NanoTrack worker (loads split models in-child)."""
    from edgecv.models.manifest import load_manifest
    from edgecv.trackers.nn.nanotrack import NanoTrack

    mf = load_manifest(cfg["manifest"])
    return NanoTrack.from_manifest(mf, backend=cfg.get("backend", "auto"),
                                   **cfg.get("kwargs", {}))


# ── spawn entrypoints ───────────────────────────────────────────────────────

def _yolo_main(detector_factory, detector_config: dict,
               fr_name: str, ctrl_name: str, result_name: str,
               max_h: int, max_w: int, max_c: int, frame_slots: int,
               payload_capacity: int, payload_max_arrays: int, stop_event) -> None:
    """Runs in the spawned YOLO child. Builds the detector in-process."""
    request_death_with_parent()
    for n in (fr_name, ctrl_name, result_name):
        detach_resource_tracker(n)

    detector = detector_factory(detector_config)
    frame_ring = FrameRing.attach(fr_name, slots=frame_slots,
                                  max_h=max_h, max_w=max_w, max_c=max_c)
    control = AcquireControlChannel.attach(ctrl_name)
    result = PayloadChannel.attach(result_name, capacity_bytes=payload_capacity,
                                   max_arrays=payload_max_arrays)
    worker = YoloWorker(detector, frame_ring, control, result)
    try:
        while not stop_event.is_set():
            if not worker.step():
                time.sleep(_IDLE_SLEEP)
    finally:
        detector.close()
        frame_ring.close(unlink=False)
        control.close(unlink=False)
        result.close(unlink=False)


def _nanotrack_main(nano_factory, nano_config: dict,
                    fr_name: str, ctrl_name: str, result_name: str,
                    max_h: int, max_w: int, max_c: int, frame_slots: int,
                    stop_event) -> None:
    """Runs in the spawned NanoTrack child. Builds the tracker in-process."""
    request_death_with_parent()
    for n in (fr_name, ctrl_name, result_name):
        detach_resource_tracker(n)

    nano = nano_factory(nano_config)
    frame_ring = FrameRing.attach(fr_name, slots=frame_slots,
                                  max_h=max_h, max_w=max_w, max_c=max_c)
    control = AcquireControlChannel.attach(ctrl_name)
    result = NanoResultChannel.attach(result_name)
    worker = NanoWorker(nano, frame_ring, control, result)
    try:
        while not stop_event.is_set():
            if worker.step() in ("idle", "no_frame", "stale", "wait_lock"):
                time.sleep(_IDLE_SLEEP)
    finally:
        nano.close()
        frame_ring.close(unlink=False)
        control.close(unlink=False)
        result.close(unlink=False)

"""AcquireKalmanTrack — YOLO-acquire → Kalman tracking-by-detection hybrid.

Spec: docs/superpowers/specs/2026-06-16-yolo-kalman-track-design.md §6

Same acquisition policy as [[acquire_track]] (``AcquireTrack``) — YOLO scans a
fixed central crop before lock; an operator init command commits the current best
detection — but the *track* stage is an in-parent **constant-velocity Kalman
filter** (``trackers/nn/yolo_kalman.KalmanBoxState``) instead of a NanoTrack
worker. So there is exactly **one** spawned worker (YOLO); the Kalman filter is
microseconds of linear algebra and runs inline in the parent.

The win of the async layout: the parent **predicts every frame at full frame
rate** (cheap) and emits a smoothed, extrapolated box; the slow YOLO worker only
supplies **corrections** (async, a few frames behind), and the motion model
bridges the detection latency. This is the Kalman analogue of MAFiD's
filter-injection.

State machine (mirrors AcquireTrack; NanoTrack → Kalman):

    ACQUIRE  YOLO on central crop; report best candidate (non-driving, for HUD)
       │ init cmd → seed Kalman from best candidate (padded), else from sent box
       ▼
    LOCKED   Kalman predicts every frame; YOLO detects on an ROI around the
       │     prediction; new detections associate (IoU/Mahalanobis) and correct.
       │ drop_frames consecutive failed associations
       ▼
    REACQ    Kalman coasts (predict only); YOLO re-acquires on the FULL frame;
       │     a confident detection re-seeds the filter → LOCKED.
       ▼
    LOST     coast + full-frame search; reset to ACQUIRE after search timeout.

Implements the standard ``Tracker`` ABC, so it slots into QuadGuide via the
existing adapter exactly like AcquireTrack. All geometry is normalised (0–1).
"""

from __future__ import annotations

import multiprocessing as mp
import time
from enum import Enum

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.core.tracker import Tracker
from edgecv.runtime.orchestrator import Orchestrator, WorkerSpec
from edgecv.runtime.shm.control_channel import AcquireControlChannel, Mode
from edgecv.runtime.shm.frame_ring import FrameRing
from edgecv.runtime.shm.payload import PayloadChannel
from edgecv.trackers.nn.yolo_kalman import KalmanBoxState, associate_detection

_FULL_CROP = BoundingBox(0.0, 0.0, 1.0, 1.0)
_ZERO_BBOX = BoundingBox(0.0, 0.0, 0.0, 0.0)


class State(Enum):
    ACQUIRE = "acquire"
    LOCKED = "locked"
    REACQ = "reacq"
    LOST = "lost"


class AcquireKalmanTrack(Tracker):
    def __init__(
        self,
        yolo_manifest=None,
        *,
        backend: str = "auto",
        # acquisition
        acquire_crop: float = 0.5,
        lock_pad: float = 1.15,
        lock_min_score: float = 0.35,
        # locked tracking / association
        search_factor: float = 2.0,
        iou_min: float = 0.3,
        min_score: float = 0.25,
        use_maha_fallback: bool = True,
        coast_locked_frames: int = 8,
        drop_frames: int = 5,
        # re-acquire / loss
        lost_timeout_frames: int = 90,
        search_timeout_frames: int = 300,
        # kalman
        dt: float = 1.0,
        std_position: float = 1.0 / 20.0,
        std_velocity: float = 1.0 / 160.0,
        # passthrough / shm
        yolo_kwargs: dict | None = None,
        max_h: int = 1080,
        max_w: int = 1920,
        max_c: int = 3,
        frame_slots: int = 4,
        payload_capacity: int = 256 * 1024,
        payload_max_arrays: int = 8,
        spawn_workers: bool = True,
        mp_context: str = "spawn",
    ) -> None:
        self._acquire_crop = acquire_crop
        self._lock_pad = lock_pad
        self._lock_min_score = lock_min_score
        self._search_factor = search_factor
        self._iou_min = iou_min
        self._min_score = min_score
        self._use_maha = use_maha_fallback
        self._coast_locked_frames = coast_locked_frames
        self._drop_frames = drop_frames
        self._lost_timeout_frames = lost_timeout_frames
        self._search_timeout_frames = search_timeout_frames
        self._kf_kwargs = {"dt": dt, "std_position": std_position,
                           "std_velocity": std_velocity}

        self._max_h, self._max_w, self._max_c = max_h, max_w, max_c
        self._frame_h, self._frame_w = max_h, max_w

        # ── shared memory (parent owns + unlinks; the worker attaches only) ──
        self._frame_ring = FrameRing.create(slots=frame_slots, max_h=max_h,
                                             max_w=max_w, max_c=max_c, dtype="uint8")
        self._control = AcquireControlChannel.create()
        self._yolo_result = PayloadChannel.create(capacity_bytes=payload_capacity,
                                                  max_arrays=payload_max_arrays)

        # ── state ──
        self._state = State.ACQUIRE
        self._seq = 0
        self._kf: KalmanBoxState | None = None
        self._candidate: BoundingBox | None = None
        self._last_bbox: BoundingBox | None = None
        self._last_score: float | None = None
        self._coast = 0           # frames since last Kalman correction (LOCKED)
        self._assoc_misses = 0    # consecutive failed associations on new dets
        self._reacq_frames = 0
        self._lost_frames = 0
        self._last_yolo_seq = 0
        self._frame_ts = 0.0
        self._out = TrackResult(bbox=None, confidence=None,
                                status=TrackStatus.INITIALIZING,
                                timestamp=time.monotonic(), seq=0)
        self._closed = False

        # ── worker (single YOLO detector) ──
        self._orch: Orchestrator | None = None
        self._stop = None
        if spawn_workers:
            self._spawn(yolo_manifest, backend, yolo_kwargs or {}, frame_slots,
                        payload_capacity, payload_max_arrays, mp_context)
        self._publish_control()

    # ── construction ──────────────────────────────────────────────────────────
    def _spawn(self, yolo_manifest, backend, yolo_kwargs, frame_slots,
               payload_capacity, payload_max_arrays, mp_context) -> None:
        if yolo_manifest is None:
            raise ValueError(
                "AcquireKalmanTrack(spawn_workers=True) needs yolo_manifest (a "
                "path); pass spawn_workers=False for tests"
            )
        from edgecv.trackers.hybrid.acquire_workers import (
            _yolo_main,
            build_yolo_detector,
        )

        ctx = mp.get_context(mp_context)
        self._stop = ctx.Event()
        self._orch = Orchestrator(mp_context)
        yolo_cfg = {"manifest": str(yolo_manifest), "backend": backend,
                    "kwargs": yolo_kwargs}
        self._orch.add_worker(WorkerSpec(
            "yolo", _yolo_main,
            (build_yolo_detector, yolo_cfg, self._frame_ring.name,
             self._control.name, self._yolo_result.name,
             self._max_h, self._max_w, self._max_c, frame_slots,
             payload_capacity, payload_max_arrays, self._stop)))
        self._orch.start()

    # ── Tracker API ────────────────────────────────────────────────────────────
    def name(self) -> str:
        return "AcquireKalmanTrack"

    @property
    def status(self) -> TrackStatus:
        return self._out.status

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        """Operator init command. Non-zero bbox = "commit/lock now"; zero = reset.

        Locks the current best YOLO candidate (padded) and seeds the Kalman
        filter on it; if no candidate is available, seeds from the passed box.
        """
        if bbox.w <= 0.0 or bbox.h <= 0.0:
            self.reset()
            return
        seed = self._candidate if self._candidate is not None else bbox
        self._relock(seed)

    def reset(self) -> None:
        self._state = State.ACQUIRE
        self._kf = None
        self._candidate = None
        self._last_bbox = None
        self._last_score = None
        self._coast = 0
        self._assoc_misses = 0
        self._reacq_frames = 0
        self._lost_frames = 0
        self._set_out(None, None, TrackStatus.INITIALIZING)
        self._publish_control()

    def update(self, frame: np.ndarray) -> TrackResult:
        self._seq += 1
        h, w = frame.shape[0], frame.shape[1]
        self._frame_h, self._frame_w = h, w
        self._frame_ts = time.monotonic()
        self._frame_ring.publish(frame, self._seq, self._frame_ts)
        self._tick()
        self._publish_control()
        return self._out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stop is not None:
            self._stop.set()
        if self._orch is not None:
            self._orch.close()
        self._frame_ring.close(unlink=True)
        self._control.close(unlink=True)
        self._yolo_result.close(unlink=True)

    def __enter__(self) -> AcquireKalmanTrack:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── state machine ───────────────────────────────────────────────────────────
    def _tick(self) -> None:
        if self._state == State.ACQUIRE:
            self._tick_acquire()
        elif self._state == State.LOCKED:
            self._tick_locked()
        elif self._state == State.REACQ:
            self._tick_reacq()
        elif self._state == State.LOST:
            self._tick_lost()

    def _tick_acquire(self) -> None:
        res = self._read_yolo_new()
        if res is None:
            return
        _seq, boxes, scores, src_ts = res
        best = self._best_in_crop(boxes, scores, self._central_crop())
        if best is None:
            self._candidate = None
            self._set_out(None, None, TrackStatus.INITIALIZING, src_ts)
        else:
            bbox, score = best
            self._candidate = bbox
            self._last_bbox = bbox
            self._set_out(bbox, score, TrackStatus.INITIALIZING, src_ts)

    def _tick_locked(self) -> None:
        # Predict every frame (full rate) — this is the reported estimate.
        self._kf.predict()
        self._coast += 1

        res = self._read_yolo_new()
        if res is not None:
            _seq, boxes, scores, _src_ts = res
            matched = associate_detection(
                self._kf, boxes, scores, iou_min=self._iou_min,
                min_score=self._min_score, use_maha=self._use_maha)
            if matched is not None:
                box, score = matched
                self._kf.update(box)
                self._coast = 0
                self._assoc_misses = 0
                self._last_score = float(score)
            else:
                self._assoc_misses += 1
                if self._assoc_misses >= self._drop_frames:
                    self._enter_reacq()
                    self._set_out(self._kf.to_bbox(), None, TrackStatus.COASTING)
                    return

        out = self._kf.to_bbox()
        self._last_bbox = out
        if self._coast <= self._coast_locked_frames:
            self._set_out(out, self._last_score, TrackStatus.LOCKED)
        else:
            self._set_out(out, None, TrackStatus.COASTING)

    def _tick_reacq(self) -> None:
        self._kf.predict()
        self._reacq_frames += 1
        if self._try_relock():
            return
        self._set_out(self._kf.to_bbox(), None, TrackStatus.COASTING)
        if self._reacq_frames >= self._lost_timeout_frames:
            self._enter_lost()

    def _tick_lost(self) -> None:
        self._kf.predict()
        self._lost_frames += 1
        if self._try_relock():
            return
        self._set_out(self._kf.to_bbox(), None, TrackStatus.LOST)
        if (self._search_timeout_frames > 0
                and self._lost_frames >= self._search_timeout_frames):
            self.reset()

    # ── transitions ─────────────────────────────────────────────────────────────
    def _relock(self, raw_bbox: BoundingBox) -> None:
        locked = self._pad(raw_bbox)
        self._kf = KalmanBoxState(locked, **self._kf_kwargs)
        self._last_bbox = locked
        self._coast = 0
        self._assoc_misses = 0
        self._reacq_frames = 0
        self._lost_frames = 0
        self._state = State.LOCKED
        self._set_out(locked, None, TrackStatus.LOCKED)
        self._publish_control()

    def _enter_reacq(self) -> None:
        self._state = State.REACQ
        self._reacq_frames = 0

    def _enter_lost(self) -> None:
        self._state = State.LOST
        self._lost_frames = 0

    def _try_relock(self) -> bool:
        """Re-seed the filter on the most confident full-frame detection.

        No spatial gate (matches AcquireTrack re-lock): the next confident
        detection anywhere wins, so a target that reappears elsewhere recovers.
        """
        res = self._read_yolo_new()
        if res is None:
            return False
        _seq, boxes, scores, _src_ts = res
        cand = self._best_above(boxes, scores)
        if cand is None:
            return False
        self._relock(cand)
        return True

    # ── channel I/O ──────────────────────────────────────────────────────────────
    def _read_yolo_new(self):
        got = self._yolo_result.try_read()
        if got is None:
            return None
        seq, arrays = got
        if seq == self._last_yolo_seq:
            return None
        self._last_yolo_seq = seq
        boxes = arrays["boxes"].reshape(-1, 4) if arrays["boxes"].size else \
            np.empty((0, 4), np.float32)
        scores = arrays["scores"].reshape(-1)
        src_ts = float(arrays["src_ts"][0]) if "src_ts" in arrays else 0.0
        return seq, boxes, scores, src_ts

    def _publish_control(self) -> None:
        """YOLO is always active (mode=YOLO); only the crop varies by state."""
        if self._state == State.ACQUIRE:
            crop = self._central_crop()
        elif self._state == State.LOCKED and self._kf is not None:
            crop = self._roi_crop(self._kf.to_bbox())
        else:  # REACQ, LOST — full-frame re-acquire
            crop = _FULL_CROP
        self._control.publish(mode=Mode.YOLO, crop=crop, lock_gen=0,
                              lock_bbox=_ZERO_BBOX)

    # ── geometry helpers (normalised) ────────────────────────────────────────────
    def _set_out(self, bbox, conf, status, src_ts=None) -> None:
        # The estimate is for the just-captured frame (the Kalman predicts to
        # "now"), so the lineage timestamp is this frame's capture time.
        self._out = TrackResult(
            bbox=bbox, confidence=conf, status=status,
            timestamp=self._frame_ts if self._frame_ts else time.monotonic(),
            seq=self._seq,
        )

    def _central_crop(self) -> BoundingBox:
        h, w = self._frame_h, self._frame_w
        side = self._acquire_crop * min(h, w)
        wn, hn = side / w, side / h
        return BoundingBox(x=(1.0 - wn) / 2.0, y=(1.0 - hn) / 2.0, w=wn, h=hn)

    def _roi_crop(self, b: BoundingBox) -> BoundingBox:
        """Search ROI around the predicted box (expanded by search_factor).

        May extend past the unit square; the worker's ``crop_with_context``
        centres on it and clamps, so off-frame coords are harmless.
        """
        cx, cy = b.center
        nw = max(b.w, 1e-3) * self._search_factor
        nh = max(b.h, 1e-3) * self._search_factor
        return BoundingBox(x=cx - nw / 2.0, y=cy - nh / 2.0, w=nw, h=nh)

    def _pad(self, b: BoundingBox) -> BoundingBox:
        cx, cy = b.center
        nw, nh = b.w * self._lock_pad, b.h * self._lock_pad
        return BoundingBox(x=cx - nw / 2.0, y=cy - nh / 2.0, w=nw, h=nh)

    def _best_in_crop(self, boxes, scores, crop):
        best, best_score = None, -1.0
        for box, sc in zip(boxes, scores, strict=False):
            sc = float(sc)
            if sc < self._lock_min_score:
                continue
            cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
            if not (crop.x <= cx <= crop.x + crop.w and
                    crop.y <= cy <= crop.y + crop.h):
                continue
            if sc > best_score:
                best_score = sc
                best = (BoundingBox(float(box[0]), float(box[1]),
                                    float(box[2]), float(box[3])), sc)
        return best

    def _best_above(self, boxes, scores):
        best, best_score = None, -1.0
        for box, sc in zip(boxes, scores, strict=False):
            sc = float(sc)
            if sc < self._lock_min_score:
                continue
            if sc > best_score:
                best_score = sc
                best = BoundingBox(float(box[0]), float(box[1]),
                                   float(box[2]), float(box[3]))
        return best

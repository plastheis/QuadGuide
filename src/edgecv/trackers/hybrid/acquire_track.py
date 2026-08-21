"""AcquireTrack — YOLO-acquire → NanoTrack-track hybrid tracker.

Spec: docs/superpowers/specs/2026-06-14-acquire-track-design.md

A detector + NN-tracker handoff (not the CF-filter MAFiD fusion). YOLO acquires a
target on a fixed central crop; on an operator init command NanoTrack locks onto
the current best detection; on confidence drop YOLO re-acquires on the full frame
and re-locks on the next confident detection. YOLO and NanoTrack run
mutually-exclusive in their own
spawned workers, each on its own NPU core. The state machine runs inline in
``update()``; workers free-run and infer only when their control ``mode`` is active.

Implements the standard ``Tracker`` ABC, so it slots into QuadGuide via the
existing adapter. All geometry is in normalised (0–1) coordinates.
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
from edgecv.runtime.shm.nano_result import NanoResultChannel
from edgecv.runtime.shm.payload import PayloadChannel

_FULL_CROP = BoundingBox(0.0, 0.0, 1.0, 1.0)


class State(Enum):
    ACQUIRE = "acquire"
    LOCKED = "locked"
    REACQ = "reacq"
    LOST = "lost"


class AcquireTrack(Tracker):
    def __init__(
        self,
        yolo_manifest=None,
        nanotrack_manifest=None,
        *,
        backend: str = "auto",
        acquire_crop: float = 0.5,
        lock_pad: float = 1.15,
        lock_min_score: float = 0.35,
        drop_score: float = 0.35,
        drop_frames: int = 5,
        lost_timeout_frames: int = 90,
        search_timeout_frames: int = 300,
        yolo_kwargs: dict | None = None,
        nanotrack_kwargs: dict | None = None,
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
        self._drop_score = drop_score
        self._drop_frames = drop_frames
        self._lost_timeout_frames = lost_timeout_frames
        self._search_timeout_frames = search_timeout_frames

        self._max_h, self._max_w, self._max_c = max_h, max_w, max_c
        self._frame_h, self._frame_w = max_h, max_w

        # ── shared memory (parent owns + unlinks; workers attach only) ──
        self._frame_ring = FrameRing.create(slots=frame_slots, max_h=max_h,
                                             max_w=max_w, max_c=max_c, dtype="uint8")
        self._control = AcquireControlChannel.create()
        self._yolo_result = PayloadChannel.create(capacity_bytes=payload_capacity,
                                                  max_arrays=payload_max_arrays)
        self._nano_result = NanoResultChannel.create()

        # ── state ──
        self._state = State.ACQUIRE
        self._seq = 0
        self._lock_gen = 0
        self._lock_bbox = BoundingBox(0.0, 0.0, 0.0, 0.0)
        self._candidate: BoundingBox | None = None
        self._last_bbox: BoundingBox | None = None
        self._miss = 0
        self._coast_frames = 0
        self._lost_frames = 0
        self._last_yolo_seq = 0
        self._last_nano_src = 0
        self._out = TrackResult(bbox=None, confidence=None,
                                status=TrackStatus.INITIALIZING,
                                timestamp=time.monotonic(), seq=0)
        self._closed = False

        # ── workers ──
        self._orch: Orchestrator | None = None
        self._stop = None
        if spawn_workers:
            self._spawn(yolo_manifest, nanotrack_manifest, backend,
                        yolo_kwargs or {}, nanotrack_kwargs or {},
                        frame_slots, payload_capacity, payload_max_arrays, mp_context)

        # publish the initial ACQUIRE control word
        self._publish_control()

    # ── construction helpers ────────────────────────────────────────────────
    def _spawn(self, yolo_manifest, nano_manifest, backend, yolo_kwargs,
               nano_kwargs, frame_slots, payload_capacity, payload_max_arrays,
               mp_context) -> None:
        if yolo_manifest is None or nano_manifest is None:
            raise ValueError(
                "AcquireTrack(spawn_workers=True) needs yolo_manifest and "
                "nanotrack_manifest (paths); pass spawn_workers=False for tests"
            )
        from edgecv.trackers.hybrid.acquire_workers import (
            _nanotrack_main,
            _yolo_main,
            build_nanotrack,
            build_yolo_detector,
        )

        ctx = mp.get_context(mp_context)
        self._stop = ctx.Event()
        self._orch = Orchestrator(mp_context)
        yolo_cfg = {"manifest": str(yolo_manifest), "backend": backend,
                    "kwargs": yolo_kwargs}
        nano_cfg = {"manifest": str(nano_manifest), "backend": backend,
                    "kwargs": nano_kwargs}
        self._orch.add_worker(WorkerSpec(
            "yolo", _yolo_main,
            (build_yolo_detector, yolo_cfg, self._frame_ring.name,
             self._control.name, self._yolo_result.name,
             self._max_h, self._max_w, self._max_c, frame_slots,
             payload_capacity, payload_max_arrays, self._stop)))
        self._orch.add_worker(WorkerSpec(
            "nanotrack", _nanotrack_main,
            (build_nanotrack, nano_cfg, self._frame_ring.name,
             self._control.name, self._nano_result.name,
             self._max_h, self._max_w, self._max_c, frame_slots, self._stop)))
        self._orch.start()

    # ── Tracker API ──────────────────────────────────────────────────────────
    def name(self) -> str:
        return "AcquireTrack"

    @property
    def status(self) -> TrackStatus:
        return self._out.status

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        """Operator init command. Non-zero bbox = "commit/lock now"; zero = reset.

        Locks the current best YOLO candidate (padded); if none is available,
        seeds NanoTrack from the passed crop box (spec §11.3).
        """
        if bbox.w <= 0.0 or bbox.h <= 0.0:
            self.reset()
            return
        seed = self._candidate if self._candidate is not None else bbox
        self._relock(seed)

    def reset(self) -> None:
        self._state = State.ACQUIRE
        self._candidate = None
        self._last_bbox = None
        self._miss = 0
        self._coast_frames = 0
        self._lost_frames = 0
        self._out = TrackResult(bbox=None, confidence=None,
                                status=TrackStatus.INITIALIZING,
                                timestamp=time.monotonic(), seq=self._seq)
        self._publish_control()

    def update(self, frame: np.ndarray) -> TrackResult:
        self._seq += 1
        h, w = frame.shape[0], frame.shape[1]
        self._frame_h, self._frame_w = h, w
        ts = time.monotonic()
        self._frame_ring.publish(frame, self._seq, ts)
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
        self._nano_result.close(unlink=True)

    def __enter__(self) -> AcquireTrack:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── state machine ─────────────────────────────────────────────────────────
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
        _seq, boxes, scores, src_seq, src_ts = res
        best = self._best_in_crop(boxes, scores, self._central_crop())
        if best is None:
            self._candidate = None
            self._set_out(None, None, TrackStatus.INITIALIZING, src_seq, src_ts)
        else:
            bbox, score = best
            self._candidate = bbox
            self._last_bbox = bbox
            self._set_out(bbox, score, TrackStatus.INITIALIZING, src_seq, src_ts)

    def _tick_locked(self) -> None:
        sample = self._read_nano_new()
        if sample is None:
            return
        conf, bbox = sample.confidence, sample.bbox
        self._last_bbox = bbox
        if conf < self._drop_score:
            self._miss += 1
            if self._miss >= self._drop_frames:
                self._enter_reacq()
                self._set_out(self._last_bbox, conf, TrackStatus.COASTING,
                              sample.src_seq, sample.src_ts)
                return
            self._set_out(bbox, conf, TrackStatus.COASTING,
                          sample.src_seq, sample.src_ts)
        else:
            self._miss = 0
            self._set_out(bbox, conf, TrackStatus.LOCKED,
                          sample.src_seq, sample.src_ts)

    def _tick_reacq(self) -> None:
        # Full-frame YOLO re-acquire: the next confident detection re-seeds
        # NanoTrack; until then we coast on the last-known box. No new detection
        # within lost_timeout_frames ⇒ LOST.
        self._coast_frames += 1
        if self._try_relock():
            return
        self._set_out(self._last_bbox, None, TrackStatus.COASTING)
        if self._coast_frames >= self._lost_timeout_frames:
            self._enter_lost()

    def _tick_lost(self) -> None:
        self._lost_frames += 1
        if self._try_relock():
            return
        self._set_out(self._last_bbox, None, TrackStatus.LOST)
        if (self._search_timeout_frames > 0
                and self._lost_frames >= self._search_timeout_frames):
            self.reset()

    # ── transitions ───────────────────────────────────────────────────────────
    def _relock(self, raw_bbox: BoundingBox) -> None:
        self._lock_bbox = self._pad(raw_bbox)
        self._last_bbox = self._lock_bbox
        self._lock_gen += 1
        self._miss = 0
        self._coast_frames = 0
        self._lost_frames = 0
        self._state = State.LOCKED
        self._set_out(self._lock_bbox, None, TrackStatus.LOCKED)
        self._publish_control()

    def _enter_reacq(self) -> None:
        self._state = State.REACQ
        self._coast_frames = 0

    def _enter_lost(self) -> None:
        self._state = State.LOST
        self._lost_frames = 0

    def _try_relock(self) -> bool:
        res = self._read_yolo_new()
        if res is None:
            return False
        _seq, boxes, scores, _src_seq, _src_ts = res
        cand = self._best_above(boxes, scores)
        if cand is None:
            return False
        self._relock(cand)
        return True

    # ── channel I/O ───────────────────────────────────────────────────────────
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
        return seq, boxes, scores, seq, src_ts

    def _read_nano_new(self):
        sample = self._nano_result.read_latest()
        if sample is None or sample.src_seq == self._last_nano_src:
            return None
        self._last_nano_src = sample.src_seq
        return sample

    def _publish_control(self) -> None:
        if self._state == State.LOCKED:
            self._control.publish(mode=Mode.NANO, crop=_FULL_CROP,
                                  lock_gen=self._lock_gen, lock_bbox=self._lock_bbox)
        elif self._state == State.ACQUIRE:
            self._control.publish(mode=Mode.YOLO, crop=self._central_crop(),
                                  lock_gen=self._lock_gen, lock_bbox=self._lock_bbox)
        else:  # REACQ, LOST — full-frame YOLO re-acquire
            self._control.publish(mode=Mode.YOLO, crop=_FULL_CROP,
                                  lock_gen=self._lock_gen, lock_bbox=self._lock_bbox)

    # ── geometry helpers (normalised) ─────────────────────────────────────────
    def _set_out(self, bbox, conf, status, src_seq=None, src_ts=None) -> None:
        self._out = TrackResult(
            bbox=bbox, confidence=conf, status=status,
            timestamp=src_ts if src_ts else time.monotonic(),
            seq=src_seq if src_seq is not None else self._seq,
        )

    def _central_crop(self) -> BoundingBox:
        h, w = self._frame_h, self._frame_w
        side = self._acquire_crop * min(h, w)
        wn, hn = side / w, side / h
        return BoundingBox(x=(1.0 - wn) / 2.0, y=(1.0 - hn) / 2.0, w=wn, h=hn)

    def _pad(self, b: BoundingBox) -> BoundingBox:
        cx, cy = b.x + b.w / 2.0, b.y + b.h / 2.0
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
        """Highest-scoring detection with score ≥ lock_min_score, anywhere in the
        full frame. Used to re-seed NanoTrack during re-acquire (no spatial gate —
        the next confident detection wins)."""
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

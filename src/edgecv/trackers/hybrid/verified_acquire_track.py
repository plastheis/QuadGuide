"""VerifiedAcquireTrack — AcquireTrack + concurrent YOLO lock verification.

Spec: docs/superpowers/specs/2026-06-16-acquire-track-lock-verification-design.md

Extends AcquireTrack to fix a single-object-tracker failure mode: NanoTrack
drifting onto background/clutter while keeping high confidence. Plain AcquireTrack
runs NanoTrack alone during LOCKED, so its confidence is the only loss signal —
confident drift never trips ``drop_score`` and the tracker reports the wrong box.

This subclass keeps YOLO running concurrently with NanoTrack during LOCKED (both
already own a dedicated NPU core and keep their RKNN context warm, so concurrency
costs no re-init) and uses it as an independent verifier: a locked box is
*supported* if it overlaps a YOLO detection. Unsupported for ``verify_miss_frames``
consecutive checks ⇒ declare drift and re-acquire — anchored on the last
verified-good position, not the drifted box (re-acquiring on the drift would just
re-lock the clutter).

Everything else — operator-gated lock, the confidence-drop path, the
full-frame re-acquire/coast/LOST machinery — is inherited unchanged. The two loss
signals are complementary: confidence catches fast loss, verification catches
confident drift. Set ``verify=False`` to fall back to exact AcquireTrack behaviour.

Scope: the distractor is background/clutter. YOLO verifies *class*, not identity,
so a same-class distractor near the target would still be accepted as support;
that needs appearance ReID or a motion prior (spec §9, out of scope here).
"""

from __future__ import annotations

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.runtime.shm.control_channel import Mode
from edgecv.trackers.hybrid.acquire_track import _FULL_CROP, AcquireTrack, State


class VerifiedAcquireTrack(AcquireTrack):
    """AcquireTrack with concurrent YOLO verification of the locked box."""

    def __init__(
        self,
        *args,
        verify: bool = True,
        verify_min_iou: float = 0.2,
        verify_min_score: float = 0.25,
        verify_miss_frames: int = 5,
        **kwargs,
    ) -> None:
        # Set verification state BEFORE super().__init__: the base constructor
        # publishes an initial control word via the overridden _publish_control,
        # which reads self._verify.
        self._verify = bool(verify)
        self._verify_min_iou = float(verify_min_iou)
        self._verify_min_score = float(verify_min_score)
        self._verify_miss_frames = int(verify_miss_frames)
        self._verify_miss = 0
        self._last_good_bbox: BoundingBox | None = None
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "VerifiedAcquireTrack"

    # ── overrides ───────────────────────────────────────────────────────────
    def _publish_control(self) -> None:
        # Identical to AcquireTrack except LOCKED publishes BOTH (NanoTrack tracks
        # + YOLO verifies, full-frame search so a drifted box anywhere is seen).
        if self._state == State.LOCKED:
            mode = Mode.BOTH if self._verify else Mode.NANO
            self._control.publish(mode=mode, crop=_FULL_CROP,
                                  lock_gen=self._lock_gen, lock_bbox=self._lock_bbox)
        elif self._state == State.ACQUIRE:
            self._control.publish(mode=Mode.YOLO, crop=self._central_crop(),
                                  lock_gen=self._lock_gen, lock_bbox=self._lock_bbox)
        else:  # REACQ, LOST — full-frame YOLO re-acquire
            self._control.publish(mode=Mode.YOLO, crop=_FULL_CROP,
                                  lock_gen=self._lock_gen, lock_bbox=self._lock_bbox)

    def _tick_locked(self) -> None:
        sample = self._read_nano_new()
        if sample is None:
            return
        conf, bbox = sample.confidence, sample.bbox
        self._last_bbox = bbox

        # Confidence-drop path (identical to AcquireTrack): fast loss.
        if conf < self._drop_score:
            self._miss += 1
            if self._miss >= self._drop_frames:
                self._enter_reacq()
                self._set_out(self._last_bbox, conf, TrackStatus.COASTING,
                              sample.src_seq, sample.src_ts)
                return
            self._set_out(bbox, conf, TrackStatus.COASTING,
                          sample.src_seq, sample.src_ts)
            return
        self._miss = 0

        # Concurrent YOLO verification: catch confident drift onto clutter. Only
        # acts on fresh YOLO results (_read_yolo_new returns None otherwise), so it
        # is YOLO-paced; verify_miss counts checks, not NanoTrack frames.
        if self._verify:
            yres = self._read_yolo_new()
            if yres is not None:
                _seq, boxes, scores, _src_seq, _src_ts = yres
                if self._supported(bbox, boxes, scores):
                    self._verify_miss = 0
                    self._last_good_bbox = bbox
                else:
                    self._verify_miss += 1
                    if self._verify_miss >= self._verify_miss_frames:
                        # Re-acquire around the last verified-good position; the
                        # current box is the drift. update() republishes control.
                        anchor = self._last_good_bbox or bbox
                        self._last_bbox = anchor
                        self._enter_reacq()
                        self._set_out(anchor, conf, TrackStatus.COASTING,
                                      sample.src_seq, sample.src_ts)
                        return

        self._set_out(bbox, conf, TrackStatus.LOCKED, sample.src_seq, sample.src_ts)

    def _relock(self, raw_bbox: BoundingBox) -> None:
        super()._relock(raw_bbox)
        # A fresh lock is good by definition; reset the verifier.
        self._last_good_bbox = self._lock_bbox
        self._verify_miss = 0

    def reset(self) -> None:
        super().reset()
        self._verify_miss = 0
        self._last_good_bbox = None

    # ── verification helpers ────────────────────────────────────────────────
    def _supported(self, bbox: BoundingBox, boxes, scores) -> bool:
        """True if some detection (score ≥ verify_min_score) overlaps bbox by
        IoU ≥ verify_min_iou. No detections ⇒ unsupported (occlusion/drift)."""
        if boxes.shape[0] == 0:
            return False
        keep = scores >= self._verify_min_score
        if not keep.any():
            return False
        return float(self._max_iou(bbox, boxes[keep])) >= self._verify_min_iou

    @staticmethod
    def _max_iou(b: BoundingBox, boxes: np.ndarray) -> float:
        """Max IoU of normalised xywh ``b`` against an (N,4) xywh array."""
        bx2, by2 = b.x + b.w, b.y + b.h
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
        iw = np.clip(np.minimum(bx2, x2) - np.maximum(b.x, x1), 0.0, None)
        ih = np.clip(np.minimum(by2, y2) - np.maximum(b.y, y1), 0.0, None)
        inter = iw * ih
        area_b = max(b.w, 0.0) * max(b.h, 0.0)
        area_o = np.clip(boxes[:, 2], 0.0, None) * np.clip(boxes[:, 3], 0.0, None)
        union = area_b + area_o - inter
        ious = inter / np.maximum(union, 1e-9)
        return float(ious.max()) if ious.size else 0.0

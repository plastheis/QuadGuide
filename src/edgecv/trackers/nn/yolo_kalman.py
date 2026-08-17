"""YOLO + Kalman single-object tracker (tracking-by-detection).

Spec: docs/superpowers/specs/2026-06-16-yolo-kalman-track-design.md

A lightweight, SORT-style tracking-by-detection tracker for a SINGLE target. It
runs the class-agnostic ``yolo11n`` P2/P3/P4 detector every frame and maintains
one constant-velocity Kalman filter over the target box. Each ``update()``:

1. **predict** the box forward one frame (constant-velocity motion model);
2. **associate** the best YOLO detection to that prediction (IoU gate, with a
   Mahalanobis fallback for fast motion);
3. **correct** the filter with the matched detection, or **coast** on the
   prediction when nothing associates.

This is the single-object specialisation of SORT (Bewley et al., ICIP 2016): one
track, so no Hungarian assignment — just pick the best detection. The Kalman
filter buys two things over the plain proximity association in ``YoloTracker``:
inter-frame *prediction* (so association survives fast motion and brief misses)
and *smoothing* (the reported box is the filtered estimate, not the raw, jittery
detection). It bridges detection gaps by coasting on the predicted state.

Implements the standard ``Tracker`` ABC, so it slots into QuadGuide via the
existing adapter. All geometry is in normalised (0–1) coordinates; the Kalman
state lives in that same normalised space, with velocities in units-per-frame.
"""

from __future__ import annotations

import time

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.trackers.nn.base import UNSET, NNTracker, resolve_pp
from edgecv.trackers.nn.yolo import YoloDetector

# chi-square 0.95 quantile, 2 dof — Mahalanobis gate on the box centre.
_CHI2_2DOF_95 = 5.9915


class KalmanBoxState:
    """Constant-velocity Kalman filter for one axis-aligned box.

    State (8-dim, normalised): ``[cx, cy, w, h, vcx, vcy, vw, vh]`` — box centre,
    size, and their per-frame velocities. Measurement is the box itself,
    ``z = [cx, cy, w, h]`` (linear ``H``). Process and measurement noise scale
    with the target height (DeepSORT convention) so the filter adapts to target
    size without per-scene retuning.

    Pure/self-contained (numpy only) so it is reusable by an async YOLO-worker
    variant (predict in the parent at full frame-rate, correct on async
    detections) — see the spec §6.
    """

    def __init__(self, box: BoundingBox, *, dt: float = 1.0,
                 std_position: float = 1.0 / 20.0,
                 std_velocity: float = 1.0 / 160.0) -> None:
        self._std_p = std_position
        self._std_v = std_velocity

        # F: constant-velocity transition (x' = x + v·dt).  H: observe the box.
        self._F = np.eye(8, dtype=np.float64)
        self._F[:4, 4:] = dt * np.eye(4)
        self._H = np.eye(4, 8, dtype=np.float64)

        cx, cy = box.center
        self.mean = np.array([cx, cy, box.w, box.h, 0, 0, 0, 0], np.float64)
        # Init covariance: generous on velocity (unknown at birth), per DeepSORT.
        h = max(box.h, 1e-3)
        std = np.array([
            2 * self._std_p * h, 2 * self._std_p * h,
            2 * self._std_p * h, 2 * self._std_p * h,
            10 * self._std_v * h, 10 * self._std_v * h,
            10 * self._std_v * h, 10 * self._std_v * h,
        ])
        self.cov = np.diag(std ** 2)

    # ── core filter steps ────────────────────────────────────────────────────
    def predict(self) -> None:
        """Advance the state one frame: x ← Fx, P ← FPFᵀ + Q."""
        h = max(self.mean[3], 1e-3)
        q = np.square(np.array([
            self._std_p * h, self._std_p * h, self._std_p * h, self._std_p * h,
            self._std_v * h, self._std_v * h, self._std_v * h, self._std_v * h,
        ]))
        self.mean = self._F @ self.mean
        self.cov = self._F @ self.cov @ self._F.T + np.diag(q)

    def _project(self) -> tuple[np.ndarray, np.ndarray]:
        """Project state into measurement space: (Hx, HPHᵀ + R)."""
        h = max(self.mean[3], 1e-3)
        r = np.diag(np.square(np.array([
            self._std_p * h, self._std_p * h, self._std_p * h, self._std_p * h])))
        proj_mean = self._H @ self.mean
        proj_cov = self._H @ self.cov @ self._H.T + r
        return proj_mean, proj_cov

    def update(self, box: BoundingBox) -> None:
        """Correct the state with a measured box (standard Kalman gain step)."""
        proj_mean, proj_cov = self._project()
        cx, cy = box.center
        z = np.array([cx, cy, box.w, box.h], np.float64)
        kalman_gain = self.cov @ self._H.T @ np.linalg.inv(proj_cov)
        self.mean = self.mean + kalman_gain @ (z - proj_mean)
        self.cov = (np.eye(8) - kalman_gain @ self._H) @ self.cov

    # ── readouts ─────────────────────────────────────────────────────────────
    def gating_distance(self, box: BoundingBox) -> float:
        """Squared Mahalanobis distance of a box CENTRE to the prediction."""
        proj_mean, proj_cov = self._project()
        cx, cy = box.center
        d = np.array([cx - proj_mean[0], cy - proj_mean[1]])
        return float(d @ np.linalg.inv(proj_cov[:2, :2]) @ d)

    def to_bbox(self) -> BoundingBox:
        """Current filtered estimate as a normalised BoundingBox."""
        cx, cy, w, h = self.mean[:4]
        w, h = max(float(w), 0.0), max(float(h), 0.0)
        return BoundingBox(x=float(cx) - w / 2.0, y=float(cy) - h / 2.0, w=w, h=h)


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """IoU of two normalised xywh top-left boxes."""
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0.0 else 0.0


def associate_detection(kf: KalmanBoxState, boxes, scores, *,
                        iou_min: float, min_score: float,
                        use_maha: bool) -> tuple[BoundingBox, float] | None:
    """Pick the detection that best matches the Kalman prediction, or None.

    Single-object SORT association against the current predicted box
    (``kf.to_bbox()``): IoU gate is primary (choose the highest IoU, score breaks
    ties); the Mahalanobis centre gate is the fallback for fast motion / size
    jumps where the predicted and detected boxes don't yet overlap. Shared by the
    inline :class:`YoloKalmanTracker` and the async ``AcquireKalmanTrack`` so the
    association policy has a single definition.

    Returns ``(BoundingBox, score)`` or ``None``.
    """
    pred = kf.to_bbox()
    best_iou, best = -1.0, None
    for box_n, sc in zip(boxes, scores, strict=False):
        sc = float(sc)
        if sc < min_score:
            continue
        b = BoundingBox(float(box_n[0]), float(box_n[1]),
                        float(box_n[2]), float(box_n[3]))
        iou = _iou(pred, b)
        if iou >= iou_min and (iou > best_iou or
                               (iou == best_iou and best is not None
                                and sc > best[1])):
            best_iou, best = iou, (b, sc)
    if best is not None:
        return best

    if not use_maha:
        return None
    best_d, cand = _CHI2_2DOF_95, None
    for box_n, sc in zip(boxes, scores, strict=False):
        sc = float(sc)
        if sc < min_score:
            continue
        b = BoundingBox(float(box_n[0]), float(box_n[1]),
                        float(box_n[2]), float(box_n[3]))
        d = kf.gating_distance(b)
        if d < best_d:
            best_d, cand = d, (b, sc)
    return cand


class YoloKalmanTracker(NNTracker):
    """Single-object tracking-by-detection: YOLO every frame + a Kalman filter.

    Parameters
    ----------
    iou_min
        IoU gate: a detection must overlap the predicted box by at least this to
        associate. SORT default ≈0.3.
    min_score
        Minimum YOLO confidence for a detection to be eligible.
    max_age
        Consecutive misses to coast (COASTING) before declaring LOST.
    use_maha_fallback
        When no detection clears the IoU gate (fast motion / size jump), fall
        back to the Mahalanobis centre gate so a nearby detection still re-locks.
    dt, std_position, std_velocity
        Kalman motion/noise parameters (per-frame); see ``KalmanBoxState``.
    """

    def __init__(self, manifest=None, *, backend="auto", model=None,
                 iou_min: float = 0.3, min_score: float = 0.25,
                 max_age: int = 30, use_maha_fallback: bool = True,
                 dt: float = 1.0, std_position: float = 1.0 / 20.0,
                 std_velocity: float = 1.0 / 160.0,
                 conf_thresh=UNSET, iou_thresh=UNSET,
                 input_size=UNSET, color=UNSET, scale=UNSET,
                 output_format=UNSET, strides=UNSET, reg_max=UNSET) -> None:
        super().__init__(manifest, backend=backend, model=model)
        pp = self._preprocessing
        # The detector shares this tracker's model handle (no second NPU context).
        self._detector = YoloDetector(
            model=self._model,
            input_size=resolve_pp(input_size, pp, "input", 640),
            color=resolve_pp(color, pp, "color", "rgb"),
            scale=resolve_pp(scale, pp, "scale", 1.0 / 255.0),
            output_format=resolve_pp(output_format, pp, "output_format", "yolov8"),
            conf_thresh=resolve_pp(conf_thresh, pp, "conf_thresh", 0.25),
            iou_thresh=resolve_pp(iou_thresh, pp, "iou_thresh", 0.45),
            strides=resolve_pp(strides, pp, "strides", (8, 16, 32)),
            reg_max=resolve_pp(reg_max, pp, "reg_max", 16))

        self._iou_min = iou_min
        self._min_score = min_score
        self._max_age = max_age
        self._use_maha = use_maha_fallback
        self._kf_kwargs = {"dt": dt, "std_position": std_position,
                           "std_velocity": std_velocity}

        self._kf: KalmanBoxState | None = None
        self._misses = 0

    def name(self) -> str:
        return "YOLO+Kalman"

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        self._kf = KalmanBoxState(bbox, **self._kf_kwargs)
        self._status = TrackStatus.LOCKED
        self._misses = 0
        self._seq = 0

    def update(self, frame: np.ndarray) -> TrackResult:
        assert self._kf is not None, "init() first"
        self._seq += 1

        # 1. PREDICT one frame forward.
        self._kf.predict()
        pred = self._kf.to_bbox()

        # 2. ASSOCIATE: detect full-frame, pick the best detection for this track.
        det = self._detector.detect(frame)
        matched = associate_detection(
            self._kf, det.boxes, det.scores, iou_min=self._iou_min,
            min_score=self._min_score, use_maha=self._use_maha)

        if matched is None:
            # 3a. COAST on the prediction; escalate to LOST past max_age.
            self._misses += 1
            self._status = (TrackStatus.LOST if self._misses > self._max_age
                            else TrackStatus.COASTING)
            return TrackResult(bbox=pred, confidence=None, status=self._status,
                               timestamp=time.monotonic(), seq=self._seq)

        # 3b. CORRECT with the matched detection.
        box, score = matched
        self._kf.update(box)
        self._misses = 0
        self._status = TrackStatus.LOCKED
        return TrackResult(bbox=self._kf.to_bbox(), confidence=float(score),
                           status=self._status, timestamp=time.monotonic(),
                           seq=self._seq)

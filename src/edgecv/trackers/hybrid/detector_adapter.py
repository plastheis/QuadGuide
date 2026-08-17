"""NN detector adapter (MAFiD spec §4.1, §7).

Wraps a specific NN detector for use inside a hybrid detector worker.
The adapter owns cropping, resizing, and postprocessing; the hybrid worker
only calls detect() and build_filter().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.fusion.calibrator import SigmoidCalibrator
from edgecv.fusion.policy import DetectorOutput
from edgecv.trackers.nn.nanotrack import NanoTrack
from edgecv.trackers.nn.preprocess import crop_with_context
from edgecv.trackers.nn.yolo import YoloDetector


class NNDetectorAdapter(ABC):
    """Wraps a specific NN detector for use inside a hybrid detector worker."""

    @abstractmethod
    def detect(self, frame: np.ndarray, search_roi: BoundingBox) -> DetectorOutput:
        """Crop frame to search_roi, run NN inference, return detection boxes+scores.

        Boxes are normalised 0-1 relative to the full frame.
        The adapter owns cropping, resizing, and postprocessing; the hybrid
        worker only calls detect() and build_filter().
        """

    def close(self) -> None:
        """Release backend resources (model handles, etc.). Default no-op."""


class YoloDetectorAdapter(NNDetectorAdapter):
    """Wraps YoloDetector for use inside a hybrid detector worker.

    The search_roi passed to detect() is the ALREADY-EXPANDED crop region
    (the caller applies search_padding before publishing). The adapter crops
    that region directly to input_size -- no further expansion.
    """

    # Default calibrator for YOLO objectness scores.
    default_calibrator = SigmoidCalibrator(centre=0.4, steepness=12.0)

    def __init__(self, manifest=None, *, backend="auto", model=None,
                 input_size: int = 640,
                 conf_thresh: float = 0.25,
                 iou_thresh: float = 0.45,
                 search_padding: float = 2.0,
                 **yolo_kwargs):
        # search_padding is stored for diagnostics; the hybrid's ROI channel
        # already expanded the crop region before publishing it.
        self._search_padding = search_padding
        self._detector = YoloDetector(
            manifest=manifest, backend=backend, model=model,
            input_size=input_size, conf_thresh=conf_thresh,
            iou_thresh=iou_thresh, **yolo_kwargs)
        self._input_size = input_size

    def detect(self, frame: np.ndarray, search_roi: BoundingBox) -> DetectorOutput:
        """Crop to search_roi -> resize to input_size -> detect -> map boxes back.

        search_roi is a BoundingBox (normalised 0-1) defining the crop region.
        The hybrid already expanded it by search_padding; the adapter crops it
        as-is.
        """
        h_img, w_img = frame.shape[:2]
        pix = search_roi.to_pixels(w_img, h_img)
        cx, cy = pix.center
        side_w, side_h = pix.w, pix.h
        n = self._input_size
        crop, xf = crop_with_context(frame, (cx, cy), (side_h, side_w), (n, n))
        det = self._detector.detect(crop)

        # Map crop-normalised boxes back to full-frame normalised coords
        frame_boxes, frame_scores = [], []
        for box_n, sc in zip(det.boxes, det.scores):
            ox1 = box_n[0] * n
            oy1 = box_n[1] * n
            ox2 = (box_n[0] + box_n[2]) * n
            oy2 = (box_n[1] + box_n[3]) * n
            fx1, fy1 = xf.to_frame((ox1, oy1))
            fx2, fy2 = xf.to_frame((ox2, oy2))
            frame_boxes.append([
                fx1 / w_img, fy1 / h_img,
                (fx2 - fx1) / w_img, (fy2 - fy1) / h_img])
            frame_scores.append(sc)

        return DetectorOutput(
            boxes=np.array(frame_boxes, np.float32) if frame_boxes
                  else np.empty((0, 4), np.float32),
            scores=np.array(frame_scores, np.float32),
            meta={"search_roi": search_roi})

    def close(self) -> None:
        self._detector.close()


class NanoTrackDetectorAdapter(NNDetectorAdapter):
    """Wraps NanoTrack (Siamese tracker) for use inside a hybrid detector worker.

    Unlike YOLO which is a stateless detector, NanoTrack is a stateful Siamese
    tracker. The adapter maintains a persistent NanoTrack instance: it is
    initialized on first detect() using the search_roi centre as the target, then
    calls update() on subsequent frames. When the parent process accepts a
    candidate filter (CF→NN mutual assistance), the adapter supports
    request_refresh() to re-initialise the template at the new location.
    """

    # Default calibrator for NanoTrack foreground probability scores (0-1).
    default_calibrator = SigmoidCalibrator(centre=0.5, steepness=10.0)

    def __init__(self, config: dict) -> None:
        # Two construction paths:
        #  • injected models (backbone/head/model in config) — used by tests;
        #  • manifest + backend — the on-device path. The backend (rknn on the
        #    Rockchip NPU, onnx on the host) is resolved and the split backbone +
        #    head loaded HERE, inside the worker process (ARCHITECTURE.md §7.4).
        manifest = config.get("manifest")
        backend = config.get("backend", "auto")
        backbone = config.get("backbone")
        head_model = config.get("head")
        model = config.get("model")
        score_lock = config.get("score_lock", 0.6)
        score_lost = config.get("score_lost", 0.35)

        if backbone is None and head_model is None and manifest is not None:
            self._nanotrack = NanoTrack.from_manifest(
                manifest, backend=backend,
                score_lock=score_lock, score_lost=score_lost,
            )
        else:
            self._nanotrack = NanoTrack(
                manifest=manifest,
                backend=backend,
                model=model,
                backbone=backbone,
                head=head_model,
                score_lock=score_lock,
                score_lost=score_lost,
            )
        self._initialized: bool = False
        self._needs_refresh: bool = False
        self._last_score: float = 0.0

    def request_refresh(self) -> None:
        """Signal that the template should be re-initialised on the next detect() call.

        Called by the worker when the parent accepts a candidate filter and
        signals a template generation increment.
        """
        self._needs_refresh = True

    def detect(self, frame: np.ndarray, search_roi: BoundingBox) -> DetectorOutput:
        """Run NanoTrack on the full frame.

        On first call (or after request_refresh): initialises NanoTrack's
        template from the search_roi centre. On subsequent calls: runs
        NanoTrack.update() to track the target.

        Returns a single-detection DetectorOutput with NanoTrack's tracked bbox
        and foreground confidence score.

        Args:
            frame: Full-resolution frame (H, W, C) from the FrameRing.
            search_roi: Search ROI bounding box (normalised 0-1). Used for
                        initialisation location; the NanoTrack template is
                        built from this region's centre.

        Returns:
            DetectorOutput with exactly one detection (the tracked bbox).
        """
        h_img, w_img = frame.shape[:2]

        if not self._initialized or self._needs_refresh:
            # First call or template refresh: build a fresh exemplar at the
            # search_roi centre (the CF tracker's current best estimate).
            # Use a reasonable initial box size at the ROI centre.
            roi_pix = search_roi.to_pixels(w_img, h_img)
            cx, cy = roi_pix.center
            # Initial bbox: use the ROI centre with a default size proportional
            # to the search region, clamped reasonably.
            init_w = min(roi_pix.w * 0.4 / w_img, 0.3)
            init_h = min(roi_pix.h * 0.4 / h_img, 0.3)
            init_bbox = BoundingBox(
                x=search_roi.x + (search_roi.w - init_w) / 2.0,
                y=search_roi.y + (search_roi.h - init_h) / 2.0,
                w=init_w, h=init_h,
            )
            self._nanotrack.init(frame, init_bbox)
            self._initialized = True
            self._needs_refresh = False
            # Run one update to get an initial tracked position
            result = self._nanotrack.update(frame)
            self._last_score = result.confidence
        else:
            result = self._nanotrack.update(frame)
            self._last_score = result.confidence

        box = result.bbox

        # Guard: return empty detection if NanoTrack lost the target
        if result.status == TrackStatus.LOST:
            return DetectorOutput(
                boxes=np.empty((0, 4), np.float32),
                scores=np.empty((0,), np.float32),
                meta={"search_roi": search_roi, "status": "lost"},
            )

        return DetectorOutput(
            boxes=np.array([[box.x, box.y, box.w, box.h]], np.float32),
            scores=np.array([result.confidence], np.float32),
            meta={"search_roi": search_roi},
        )

    def close(self) -> None:
        self._nanotrack.close()

"""Class-agnostic YOLO detector + standalone single-object tracker
(ARCHITECTURE.md §6.2; MAFiD local-detection mode, sensors-23-07082 §3.3).

YoloDetector.detect -> DetectorOutput is the reusable primitive a future hybrid
worker calls. YoloTracker wraps it for standalone use."""

from __future__ import annotations

import time

import numpy as np

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.fusion.calibrator import SigmoidCalibrator
from edgecv.fusion.policy import DetectorOutput
from edgecv.trackers.nn.base import (
    UNSET,
    NNTracker,
    manifest_preprocessing,
    resolve_model,
    resolve_pp,
)
from edgecv.trackers.nn.preprocess import (
    class_agnostic_nms,
    crop_with_context,
    decode_yolo_dfl,
    letterbox,
    to_input,
)


class YoloDetector:
    default_calibrator = SigmoidCalibrator(centre=0.4, steepness=12.0)


    """Boxes in the returned DetectorOutput are (N,4) normalised xywh top-left,
    normalised to the image passed to detect()."""

    def __init__(self, manifest=None, *, backend="auto", model=None,
                 input_size=UNSET, color=UNSET, scale=UNSET,
                 output_format=UNSET, conf_thresh=UNSET, iou_thresh=UNSET,
                 strides=UNSET, reg_max=UNSET) -> None:
        self._owns_model = model is None
        self._model = resolve_model(manifest, backend, model)
        pp = manifest_preprocessing(manifest)   # {} when a model= is injected
        self._input_size = resolve_pp(input_size, pp, "input", 640)
        self._color = resolve_pp(color, pp, "color", "rgb")
        self._scale = resolve_pp(scale, pp, "scale", 1.0 / 255.0)
        self._output_format = resolve_pp(output_format, pp, "output_format", "yolov8")
        self._conf = resolve_pp(conf_thresh, pp, "conf_thresh", 0.25)
        self._iou = resolve_pp(iou_thresh, pp, "iou_thresh", 0.45)
        # rknn_dfl (separated per-scale head) extras; ignored by the fused formats.
        self._strides = tuple(resolve_pp(strides, pp, "strides", (8, 16, 32)))
        self._reg_max = resolve_pp(reg_max, pp, "reg_max", 16)
        self._spec = self._model.io_spec.inputs[0]
        self._out_name = self._model.io_spec.outputs[0].name

    def detect(self, image: np.ndarray) -> DetectorOutput:
        h_img, w_img = image.shape[0], image.shape[1]
        n = self._input_size
        lb, xf = letterbox(image, (n, n))
        inp = to_input(lb, self._spec, color=self._color, scale=self._scale)
        out = self._model.infer({self._spec.name: inp})
        if self._output_format == "rknn_dfl":
            outputs = [np.asarray(out[o.name]) for o in self._model.io_spec.outputs]
            xyxy, score = decode_yolo_dfl(outputs, self._strides,
                                          reg_max=self._reg_max, conf_thresh=self._conf)
        else:
            xyxy, score = self._decode_fused(
                np.asarray(out[self._out_name], np.float32))
        if len(score) == 0:
            return DetectorOutput(boxes=np.empty((0, 4), np.float32),
                                  scores=np.empty((0,), np.float32))
        kept = class_agnostic_nms(xyxy, score, self._iou)
        xyxy, score = xyxy[kept], score[kept]
        # invert letterbox -> original px -> normalised xywh top-left
        boxes = np.empty((len(kept), 4), np.float32)
        for i, b in enumerate(xyxy):
            ox1, oy1, ox2, oy2 = xf.to_orig_xyxy((b[0], b[1], b[2], b[3]))
            boxes[i] = [ox1 / w_img, oy1 / h_img, (ox2 - ox1) / w_img, (oy2 - oy1) / h_img]
        return DetectorOutput(boxes=boxes, scores=score.astype(np.float32))

    def _decode_fused(self, raw: np.ndarray):
        """Decode a single fused head tensor → (xyxy_px, score) before NMS."""
        # v8/v26 one-to-many head is (1, 4+nc, N) channels-first -> transpose to rows;
        # v5/"decoded" are already (1, N, k).
        preds = raw[0].T if self._output_format == "yolov8" else raw[0]
        if preds.shape[0] == 0:
            return np.empty((0, 4), np.float32), np.empty((0,), np.float32)
        if self._output_format == "yolov5":
            # yolov5 row layout: [cx, cy, w, h | obj | cls_0..cls_{nc-1}]
            xywh, obj, cls = preds[:, :4], preds[:, 4], preds[:, 5:]
            score = obj * (cls.max(axis=1) if cls.shape[1] > 0 else 1.0)
        elif self._output_format == "yolov8":
            # anchor-free, NO objectness: [cx, cy, w, h | cls_0..cls_{nc-1}]
            xywh, cls = preds[:, :4], preds[:, 4:]
            score = (cls.max(axis=1) if cls.shape[1] > 0
                     else np.zeros((preds.shape[0],), np.float32))
        else:  # "decoded": model already emits xywh + score
            xywh, score = preds[:, :4], preds[:, 4]
        keep = score >= self._conf
        xywh, score = xywh[keep], score[keep]
        if len(score) == 0:
            return np.empty((0, 4), np.float32), np.empty((0,), np.float32)
        cxs, cys, ws, hs = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        # YOLO outputs raw unbounded [cx,cy,w,h] — w/h can be negative from model
        # noise. Clamp to zero before xyxy conversion to avoid degenerate boxes.
        ws = np.maximum(ws, 0.0)
        hs = np.maximum(hs, 0.0)
        xyxy = np.stack([cxs - ws / 2, cys - hs / 2, cxs + ws / 2, cys + hs / 2], axis=1)
        return xyxy, score

    def close(self) -> None:
        if self._owns_model:
            self._model.close()


class YoloTracker(NNTracker):
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 search_factor=2.0, assoc_sigma=0.5, assoc_threshold=0.1,
                 conf_thresh=UNSET, iou_thresh=UNSET, max_misses=5,
                 input_size=UNSET, color=UNSET, scale=UNSET,
                 output_format=UNSET) -> None:
        super().__init__(manifest, backend=backend, model=model)
        pp = self._preprocessing
        input_size = resolve_pp(input_size, pp, "input", 640)
        color = resolve_pp(color, pp, "color", "rgb")
        scale = resolve_pp(scale, pp, "scale", 1.0 / 255.0)
        output_format = resolve_pp(output_format, pp, "output_format", "yolov8")
        conf_thresh = resolve_pp(conf_thresh, pp, "conf_thresh", 0.25)
        iou_thresh = resolve_pp(iou_thresh, pp, "iou_thresh", 0.45)
        self._detector = YoloDetector(
            model=self._model, input_size=input_size, color=color, scale=scale,
            output_format=output_format, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        self._search_factor = search_factor
        self._assoc_sigma = assoc_sigma
        self._assoc_threshold = assoc_threshold
        self._max_misses = max_misses
        self._input_size = input_size
        self._box: BoundingBox | None = None
        self._init_box: BoundingBox | None = None  # frozen at init for stable sigma
        self._misses = 0

    def name(self) -> str:
        return "YOLO"

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        self._box = bbox
        self._init_box = bbox  # frozen for stable sigma reference
        self._status = TrackStatus.LOCKED
        self._misses = 0
        self._seq = 0

    def update(self, frame: np.ndarray) -> TrackResult:
        assert self._box is not None, "init() first"
        h_img, w_img = frame.shape[0], frame.shape[1]
        pix = self._box.to_pixels(w_img, h_img)
        cx, cy = pix.center
        side = self._search_factor * max(pix.w, pix.h)
        n = self._input_size
        crop, xf = crop_with_context(frame, (cx, cy), (side, side), (n, n))
        det = self._detector.detect(crop)

        best, best_w = None, -1.0
        # Use init box size for sigma — stable, not affected by detection drift.
        init_pix = self._init_box.to_pixels(w_img, h_img)
        sigma = self._assoc_sigma * max(init_pix.w, init_pix.h) + 1e-6
        for box_n, sc in zip(det.boxes, det.scores, strict=False):
            # crop-normalised xywh -> crop-out px -> frame px via xf.to_frame.
            # to_frame treats indices as pixel centres, so corners carry a
            # sub-pixel (<1px) drift — negligible for proximity association.
            ox1, oy1 = box_n[0] * n, box_n[1] * n
            ox2, oy2 = (box_n[0] + box_n[2]) * n, (box_n[1] + box_n[3]) * n
            fx1, fy1 = xf.to_frame((ox1, oy1))
            fx2, fy2 = xf.to_frame((ox2, oy2))
            dcx, dcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
            dist2 = (dcx - cx) ** 2 + (dcy - cy) ** 2
            w = float(sc) * float(np.exp(-0.5 * dist2 / (sigma * sigma)))
            if w > best_w:
                best_w, best = w, (fx1, fy1, fx2 - fx1, fy2 - fy1, float(sc))

        # Require minimum association weight — reject weak/distant detections
        if best is None or best_w < self._assoc_threshold:
            self._misses += 1
            self._status = TrackStatus.LOST if self._misses > self._max_misses \
                else TrackStatus.COASTING
            self._seq += 1
            return TrackResult(bbox=self._box, confidence=None, status=self._status,
                               timestamp=time.monotonic(), seq=self._seq)

        fx, fy, fw, fh, score = best
        self._box = BoundingBox.from_pixels(PixelBox(x=fx, y=fy, w=fw, h=fh), w_img, h_img)
        self._misses = 0
        self._status = TrackStatus.LOCKED
        self._seq += 1
        return TrackResult(bbox=self._box, confidence=best_w, status=self._status,
                           timestamp=time.monotonic(), seq=self._seq)

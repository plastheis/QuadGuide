from __future__ import annotations
import math
import time

import numpy as np

from quadguide.core.config import NanotrackConfig
from quadguide.core.messages import BoundingBox, TrackerEstimate, TrackerHealth
from quadguide.perception.nanotrack.postprocess import decode_response
from quadguide.perception.nanotrack.preprocess import (
    get_exemplar_crop, get_search_crop, normalise,
)

__all__ = ["NanoTracker"]

_STRIDE = 8  # NanoTrack feature stride (backbone downsampling factor)


class NanoTracker:
    """NanoTrack inference pipeline. No OpenCV, no bus/IPC imports.

    Backbone encodes both the exemplar template (stored in self._z_feat) and
    search crops on each update. Head scores and regresses bboxes. Search-crop
    coordinates are mapped back to full-frame normalised coords using the stored
    scale factor computed at init time.
    """

    def __init__(self, runtime, backbone_model, head_model, config: NanotrackConfig) -> None:
        self._runtime  = runtime
        self._backbone = backbone_model
        self._head     = head_model
        self._cfg      = config
        self._z_feat:    dict[str, np.ndarray] | None = None
        self._last_bbox: BoundingBox | None            = None
        self._scale:     float | None                  = None
        self._s_z:       float | None                  = None
        self._initialized = False

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        """Extract exemplar features and store search-region scale."""
        h, w = frame.shape[:2]
        bw_px = bbox.w * w
        bh_px = bbox.h * h
        p     = (bw_px + bh_px) / 2
        s_z   = math.sqrt((bw_px + p) * (bh_px + p))  # exemplar crop size in frame px
        self._scale     = (self._cfg.instance_sz / self._cfg.exemplar_sz)
        self._s_z       = s_z
        self._last_bbox = bbox

        crop         = get_exemplar_crop(frame, bbox, self._cfg.exemplar_sz)
        self._z_feat = self._runtime.infer(self._backbone, {"input": normalise(crop)})
        self._initialized = True

    def update(self, frame: np.ndarray) -> TrackerEstimate:
        if not self._initialized:
            return TrackerEstimate(
                timestamp_ns=time.monotonic_ns(),
                bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
                confidence=0.0,
                tracker_health=TrackerHealth.NO_LOCK,
            )

        h, w = frame.shape[:2]

        # Encode search region
        s_crop = get_search_crop(
            frame, self._last_bbox, self._scale, self._cfg.instance_sz
        )
        x_feat = self._runtime.infer(self._backbone, {"input": normalise(s_crop)})

        # Head: takes exemplar and search features
        z_arr  = list(self._z_feat.values())[0]
        x_arr  = list(x_feat.values())[0]
        out       = self._runtime.infer(self._head, {"z": z_arr, "x": x_arr})
        score_map = out["score"]  # (1, 1, H, W)
        bbox_map  = out["bbox"]   # (1, 4, H, W) ltrb

        (cx_n, cy_n, w_n, h_n), conf = decode_response(
            score_map, bbox_map, stride=_STRIDE, instance_sz=self._cfg.instance_sz
        )

        # Map search-crop coords back to full-frame normalised coords.
        # Search crop is centred at last_bbox centre; its pixel extent = s_z * scale.
        s_x = self._s_z * self._scale  # search region size in frame pixels
        search_cx_norm = self._last_bbox.x + self._last_bbox.w / 2
        search_cy_norm = self._last_bbox.y + self._last_bbox.h / 2

        target_cx_px = (cx_n - 0.5) * s_x + search_cx_norm * w
        target_cy_px = (cy_n - 0.5) * s_x + search_cy_norm * h
        target_w_px  = w_n * s_x
        target_h_px  = h_n * s_x

        bbox_out = BoundingBox(
            x=max(0.0, (target_cx_px - target_w_px / 2) / w),
            y=max(0.0, (target_cy_px - target_h_px / 2) / h),
            w=min(1.0, target_w_px / w),
            h=min(1.0, target_h_px / h),
        )
        self._last_bbox = bbox_out

        health = (
            TrackerHealth.NOMINAL
            if conf >= self._cfg.score_threshold
            else TrackerHealth.UNCERTAIN
        )
        return TrackerEstimate(
            timestamp_ns=time.monotonic_ns(),
            bbox=bbox_out,
            confidence=conf,
            tracker_health=health,
        )

    def close(self) -> None:
        self._runtime.close()

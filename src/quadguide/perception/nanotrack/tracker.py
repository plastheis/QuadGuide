"""NanoTrack tracker — direct port of OpenCV TrackerNano C++ logic.

Tracking logic matches opencv_contrib/modules/tracking/src/trackerNano.cpp exactly:
  - Template extracted at sx = sz * (255/127) pixels, run through backbone, centre-cropped.
  - Preprocessing: raw [0,255] float32 RGB. No ImageNet normalisation.
  - Score: per-cell softmax over 2-channel cls output.
  - Selection: scale/ratio-change penalty + Hanning window bias, then argmax.
  - Size: EMA update with lr = best_penalty * best_score * LR.
  - Position: direct update with scale correction.
"""
from __future__ import annotations
import time

import cv2
import numpy as np

from quadguide.core.config import NanotrackConfig
from quadguide.core.messages import BoundingBox, TrackerEstimate, TrackerHealth

__all__ = ["NanoTracker"]

# ---------------------------------------------------------------------------
# Hyperparameters — matched to OpenCV TrackerNano source
# ---------------------------------------------------------------------------
_EXEMPLAR_SZ      = 127
_INSTANCE_SZ      = 255
_TOTAL_STRIDE     = 16
_CONTEXT_AMOUNT   = 0.5
_PENALTY_K        = 0.055
_LR               = 0.37
_WINDOW_INFLUENCE = 0.455
_MIN_SIZE_PX      = 10

# Model-specific constants (verified against ONNX output shapes):
#   backbone: 255×255 → [1, 96, 16, 16]   (feat_size = 16)
#   head:     → [1, 2, 15, 15] cls, [1, 4, 15, 15] loc   (score_size = 15)
_SCORE_SIZE = 15
_FEAT_SIZE  = 16
_T_LO       = _FEAT_SIZE // 4       # 4
_T_HI       = _FEAT_SIZE - _T_LO   # 12


# ---------------------------------------------------------------------------
# Module-level helpers (stateless, no quadguide imports)
# ---------------------------------------------------------------------------

def _size_cal(w, h):
    pad = (w + h) * 0.5
    return np.sqrt(np.maximum((w + pad) * (h + pad), 1e-6))


def _get_subwindow(
    img: np.ndarray,
    cx: float, cy: float,
    original_sz: int,
    resize_sz: int,
) -> np.ndarray:
    """Crop original_sz×original_sz pixels centred at (cx, cy).

    Pads out-of-bounds regions with the per-channel mean of the full frame.
    Matches OpenCV TrackerNano ::getSubwindow exactly.
    """
    avg  = np.mean(img, axis=(0, 1))
    c    = (original_sz + 1) // 2
    xmin = int(cx) - c
    ymin = int(cy) - c
    xmax = xmin + original_sz - 1
    ymax = ymin + original_sz - 1

    pad_l = max(0, -xmin);      pad_t = max(0, -ymin)
    pad_r = max(0, xmax - img.shape[1] + 1)
    pad_b = max(0, ymax - img.shape[0] + 1)

    xmin += pad_l;  xmax += pad_l
    ymin += pad_t;  ymax += pad_t

    if pad_l or pad_t or pad_r or pad_b:
        src = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                                 cv2.BORDER_CONSTANT, value=avg)
    else:
        src = img

    crop = src[ymin:ymax + 1, xmin:xmax + 1]
    return cv2.resize(crop, (resize_sz, resize_sz))


def _preprocess(patch: np.ndarray) -> np.ndarray:
    """BGR uint8 → float32 NCHW RGB in [0, 255]. No ImageNet normalisation.

    Matches cv2.dnn.blobFromImage(scale=1.0, mean=Scalar(), swapRB=True),
    which is what OpenCV TrackerNano uses.
    """
    rgb = patch[:, :, ::-1].astype(np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class NanoTracker:
    """NanoTrack inference pipeline.

    Internal state (cx, cy, w, h) is kept in frame pixel coordinates,
    matching the reference. Input and output use quadguide's normalised
    BoundingBox convention.
    """

    def __init__(
        self,
        runtime,
        backbone_model,
        head_model,
        config: NanotrackConfig,
    ) -> None:
        self._runtime  = runtime
        self._backbone = backbone_model
        self._head     = head_model
        self._cfg      = config

        self._cx:    float              = 0.0
        self._cy:    float              = 0.0
        self._w:     float              = 0.0
        self._h:     float              = 0.0
        self._t_feat: np.ndarray | None = None
        self._initialized               = False

        self._build_hanning_and_grid()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _build_hanning_and_grid(self) -> None:
        S  = _SCORE_SIZE
        hv = np.hanning(S + 2)[1:-1].astype(np.float32)
        self._window = np.outer(hv, hv)  # [S, S], not normalised (matches OpenCV)

        # OpenCV TrackerNano grid formula (trackerNano.cpp ::generateGrids):
        #   grid[i] = (i - S//2) * TOTAL_STRIDE + INSTANCE_SZ // 2
        half = S // 2
        xs   = (np.arange(S, dtype=np.float32) - half) * _TOTAL_STRIDE + _INSTANCE_SZ // 2
        self._grid_x, self._grid_y = np.meshgrid(xs, xs)  # [S, S] each

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "nanotrack"

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        """Extract template features from frame at bbox."""
        h_img, w_img = frame.shape[:2]
        self._cx = (bbox.x + bbox.w / 2) * w_img
        self._cy = (bbox.y + bbox.h / 2) * h_img
        self._w  = bbox.w * w_img
        self._h  = bbox.h * h_img

        s_sum = self._w + self._h
        wz    = self._w + _CONTEXT_AMOUNT * s_sum
        hz    = self._h + _CONTEXT_AMOUNT * s_sum
        sz    = int(np.sqrt(wz * hz))
        sx    = int(sz * (_INSTANCE_SZ / _EXEMPLAR_SZ))

        crop         = _get_subwindow(frame, self._cx, self._cy, sx, _INSTANCE_SZ)
        z_raw        = self._runtime.infer(self._backbone, {"input": _preprocess(crop)})
        z_feat       = list(z_raw.values())[0]                       # [1, C, 16, 16]
        self._t_feat = z_feat[:, :, _T_LO:_T_HI, _T_LO:_T_HI]      # [1, C,  8,  8]
        self._initialized = True

    def update(self, frame: np.ndarray) -> TrackerEstimate:
        if not self._initialized:
            return TrackerEstimate(
                timestamp_ns=time.monotonic_ns(),
                bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
                confidence=0.0,
                tracker_health=TrackerHealth.NO_LOCK,
            )

        h_img, w_img = frame.shape[:2]
        cx, cy, w, h = self._cx, self._cy, self._w, self._h

        # Search region metrics
        s_sum   = w + h
        wc      = w + _CONTEXT_AMOUNT * s_sum
        hc      = h + _CONTEXT_AMOUNT * s_sum
        sz      = np.sqrt(wc * hc)
        scale_z = _EXEMPLAR_SZ / sz                   # search-image → frame-pixel ratio
        sx      = sz * (_INSTANCE_SZ / _EXEMPLAR_SZ)  # search crop size in frame pixels
        tw_s    = w * scale_z                          # target w in search-image pixels
        th_s    = h * scale_z                          # target h in search-image pixels

        # Search crop → backbone
        crop   = _get_subwindow(frame, cx, cy, int(sx), _INSTANCE_SZ)
        s_raw  = self._runtime.infer(self._backbone, {"input": _preprocess(crop)})
        s_feat = list(s_raw.values())[0]              # [1, C, 16, 16]

        # Head
        out  = self._runtime.infer(self._head, {"input1": self._t_feat, "input2": s_feat})
        cls  = out["output1"]   # [1, 2, S, S]
        loc  = out["output2"]   # [1, 4, S, S]  ltrb in search-image pixel units

        # Per-cell softmax over 2 class channels
        cls0  = cls[0]                                             # [2, S, S]
        mx    = cls0.max(axis=0, keepdims=True)
        exp   = np.exp(cls0 - mx)
        score = (exp / exp.sum(axis=0, keepdims=True))[1]         # [S, S] fg probability

        # Decode all cells using precomputed grid
        l_map, t_map = loc[0, 0], loc[0, 1]
        r_map, b_map = loc[0, 2], loc[0, 3]
        pred_x1 = self._grid_x - l_map
        pred_y1 = self._grid_y - t_map
        pred_x2 = self._grid_x + r_map
        pred_y2 = self._grid_y + b_map
        pred_w  = pred_x2 - pred_x1
        pred_h  = pred_y2 - pred_y1

        # Scale/ratio change penalty — suppresses large jumps
        def _rmax(v):
            return np.maximum(v, 1.0 / np.maximum(v, 1e-6))

        sc      = _rmax(_size_cal(pred_w, pred_h) / _size_cal(tw_s, th_s))
        rc      = _rmax((pred_w / np.maximum(pred_h, 1e-6)) / (w / max(h, 1e-6)))
        penalty = np.exp(-(sc * rc - 1.0) * _PENALTY_K)

        # Hanning window blends toward centre, penalty dampens scale jumps
        pscore  = score * penalty * (1.0 - _WINDOW_INFLUENCE) + self._window * _WINDOW_INFLUENCE
        bi, bj  = np.unravel_index(np.argmax(pscore), pscore.shape)

        best_score   = float(score[bi, bj])
        best_penalty = float(penalty[bi, bj])

        pred_xs = (pred_x1[bi, bj] + pred_x2[bi, bj]) / 2.0
        pred_ys = (pred_y1[bi, bj] + pred_y2[bi, bj]) / 2.0
        pred_bw = float(pred_w[bi, bj])
        pred_bh = float(pred_h[bi, bj])

        # Map from search-image coords back to frame pixels
        diff_x = (pred_xs - _INSTANCE_SZ // 2) / scale_z
        diff_y = (pred_ys - _INSTANCE_SZ // 2) / scale_z
        out_w  = pred_bw / scale_z
        out_h  = pred_bh / scale_z

        # Size EMA (damps oscillation); position direct (matches OpenCV)
        lr     = best_penalty * best_score * _LR
        new_cx = float(np.clip(cx + diff_x, 0, w_img))
        new_cy = float(np.clip(cy + diff_y, 0, h_img))
        new_w  = float(np.clip(out_w * lr + w * (1.0 - lr), _MIN_SIZE_PX, w_img))
        new_h  = float(np.clip(out_h * lr + h * (1.0 - lr), _MIN_SIZE_PX, h_img))

        self._cx, self._cy, self._w, self._h = new_cx, new_cy, new_w, new_h

        bbox_out = BoundingBox(
            x=max(0.0, (new_cx - new_w / 2) / w_img),
            y=max(0.0, (new_cy - new_h / 2) / h_img),
            w=min(1.0, new_w / w_img),
            h=min(1.0, new_h / h_img),
        )
        health = (
            TrackerHealth.NOMINAL
            if best_score >= self._cfg.score_threshold
            else TrackerHealth.UNCERTAIN
        )
        return TrackerEstimate(
            timestamp_ns=time.monotonic_ns(),
            bbox=bbox_out,
            confidence=best_score,
            tracker_health=health,
        )

    def reset(self) -> None:
        self._initialized = False

    def close(self) -> None:
        self._runtime.close()

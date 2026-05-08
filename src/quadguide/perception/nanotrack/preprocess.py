from __future__ import annotations
import math

import numpy as np

from quadguide.core.messages import BoundingBox

__all__ = ["get_exemplar_crop", "get_search_crop", "normalise"]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_exemplar_crop(
    frame: np.ndarray, bbox: BoundingBox, exemplar_sz: int
) -> np.ndarray:
    """Return an exemplar_sz × exemplar_sz BGR uint8 crop centred on bbox.

    Uses SiamTrack context padding: p = (w + h) / 2, s = sqrt((w+p)*(h+p)).
    The context area gives the backbone enough background to build a reliable template.
    """
    h, w = frame.shape[:2]
    bw_px = bbox.w * w
    bh_px = bbox.h * h
    p     = (bw_px + bh_px) / 2
    s     = math.sqrt((bw_px + p) * (bh_px + p))
    cx    = (bbox.x + bbox.w / 2) * w
    cy    = (bbox.y + bbox.h / 2) * h
    return _crop_and_resize(frame, cx, cy, s, exemplar_sz)


def get_search_crop(
    frame: np.ndarray, bbox: BoundingBox, scale: float, instance_sz: int
) -> np.ndarray:
    """Return an instance_sz × instance_sz BGR uint8 search crop.

    The search region is scale × the exemplar crop size, centred on bbox.
    Caller computes scale = instance_sz / exemplar_sz at init time.
    """
    h, w = frame.shape[:2]
    bw_px = bbox.w * w
    bh_px = bbox.h * h
    p     = (bw_px + bh_px) / 2
    s_z   = math.sqrt((bw_px + p) * (bh_px + p))
    s_x   = s_z * scale
    cx    = (bbox.x + bbox.w / 2) * w
    cy    = (bbox.y + bbox.h / 2) * h
    return _crop_and_resize(frame, cx, cy, s_x, instance_sz)


def normalise(crop: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 crop to float32 (1, 3, H, W) with ImageNet normalisation."""
    rgb = crop[:, :, ::-1].astype(np.float32) / 255.0  # BGR → RGB, scale to [0,1]
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD        # ImageNet normalise
    return rgb.transpose(2, 0, 1)[np.newaxis]           # (H,W,C) → (1,C,H,W)


def _crop_and_resize(
    frame: np.ndarray, cx: float, cy: float, size: float, out_sz: int
) -> np.ndarray:
    """Crop a square of `size` pixels centred at (cx, cy) and resize to out_sz × out_sz.

    Out-of-bound regions are filled with the per-channel mean of the full frame.
    """
    from PIL import Image

    h, w = frame.shape[:2]
    half  = size / 2
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = int(round(cx + half)), int(round(cy + half))

    pad_l = max(0, -x1);  pad_r = max(0, x2 - w)
    pad_t = max(0, -y1);  pad_b = max(0, y2 - h)

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = max(0, min(w, x2)), max(0, min(h, y2))
    crop = frame[y1c:y2c, x1c:x2c]

    if pad_l or pad_r or pad_t or pad_b:
        mean_color = tuple(int(v) for v in frame.mean(axis=(0, 1)))  # (B, G, R)
        canvas_h = crop.shape[0] + pad_t + pad_b
        canvas_w = crop.shape[1] + pad_l + pad_r
        canvas   = np.full((canvas_h, canvas_w, 3), mean_color, dtype=np.uint8)
        canvas[pad_t:pad_t + crop.shape[0], pad_l:pad_l + crop.shape[1]] = crop
        crop = canvas

    pil  = Image.fromarray(crop[:, :, ::-1])      # BGR → RGB for PIL
    pil  = pil.resize((out_sz, out_sz), Image.Resampling.BILINEAR)
    return np.array(pil)[:, :, ::-1]              # RGB → BGR, uint8

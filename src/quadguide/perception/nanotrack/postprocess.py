from __future__ import annotations
import numpy as np

__all__ = ["decode_response"]


def decode_response(
    score_map: np.ndarray,  # (1, 1, H, W) raw logits
    bbox_map: np.ndarray,   # (1, 4, H, W) ltrb offsets in feature-map units
    stride: int,
    instance_sz: int,
) -> tuple[tuple[float, float, float, float], float]:
    """Decode NanoTrack head outputs into a target location and confidence.

    Returns:
        coords: (cx_norm, cy_norm, w_norm, h_norm) — all normalised to [0, 1]
                relative to the search crop of size instance_sz × instance_sz.
        conf:   sigmoid of the peak score value, in [0, 1].

    Coordinate convention:
        Peak cell index (cy_idx, cx_idx) maps to pixel centre
        (cx_px, cy_px) = ((cx_idx + 0.5) * stride, (cy_idx + 0.5) * stride).
        LTRB offsets l, t, r, b (in feature-map units, scaled by stride) give:
            x1 = cx_px - l,  y1 = cy_px - t
            x2 = cx_px + r,  y2 = cy_px + b
        Normalised centre and size relative to instance_sz:
            cx_norm = (x1 + x2) / 2 / instance_sz
            cy_norm = (y1 + y2) / 2 / instance_sz
            w_norm  = (x2 - x1) / instance_sz
            h_norm  = (y2 - y1) / instance_sz
    """
    score = score_map[0, 0]  # (H, W)
    h_map, w_map = score.shape
    flat_idx = int(np.argmax(score))
    cy_idx   = flat_idx // w_map
    cx_idx   = flat_idx % w_map

    peak_val = float(score[cy_idx, cx_idx])
    conf     = float(1.0 / (1.0 + np.exp(-peak_val)))  # sigmoid

    cx_px = (cx_idx + 0.5) * stride
    cy_px = (cy_idx + 0.5) * stride

    l = float(bbox_map[0, 0, cy_idx, cx_idx]) * stride
    t = float(bbox_map[0, 1, cy_idx, cx_idx]) * stride
    r = float(bbox_map[0, 2, cy_idx, cx_idx]) * stride
    b = float(bbox_map[0, 3, cy_idx, cx_idx]) * stride

    x1, y1 = cx_px - l, cy_px - t
    x2, y2 = cx_px + r, cy_px + b

    cx_norm = (x1 + x2) / 2 / instance_sz
    cy_norm = (y1 + y2) / 2 / instance_sz
    w_norm  = max(0.0, (x2 - x1)) / instance_sz
    h_norm  = max(0.0, (y2 - y1)) / instance_sz

    return (cx_norm, cy_norm, w_norm, h_norm), conf

"""FilterState serialisation for IPC (MAFiD spec §6).

Flatten FilterState (arrays + bbox + meta) to dict[str, np.ndarray] for the
PayloadChannel, and deserialise back.
"""

from __future__ import annotations

import json

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.trackers.cf.base import FilterState


def _serialise_filter_state(fs: FilterState) -> dict[str, np.ndarray]:
    """Flatten a FilterState into a dict of numpy arrays for the PayloadChannel.

    The meta dict is JSON-encoded and stored as a uint8 byte array.
    The bbox is stored as a float32 [x, y, w, h] array.
    """
    out = dict(fs.arrays)  # A, B, ... as complex64 arrays
    out["fs_bbox"] = np.array([fs.bbox.x, fs.bbox.y, fs.bbox.w, fs.bbox.h], np.float32)
    out["fs_meta"] = np.frombuffer(json.dumps(fs.meta).encode(), np.uint8)
    return out


def _deserialise_filter_state(data: dict[str, np.ndarray]) -> FilterState | None:
    """Rebuild a FilterState from a payload dict.

    Keys prefixed with 'fs_' and known payload metadata keys are excluded
    from the arrays dict. The meta dict is decoded from JSON.

    Returns None if the data is incomplete (worker crashed mid-publish).
    """
    _SKIP_KEYS = {"fs_bbox", "fs_meta", "detector_out_boxes",
                  "detector_out_scores", "detect_time"}
    arrays = {k: v for k, v in data.items() if k not in _SKIP_KEYS}

    bbox_arr = data.get("fs_bbox")
    if bbox_arr is None or bbox_arr.size < 4:
        return None
    x, y, w, h = float(bbox_arr[0]), float(bbox_arr[1]), float(bbox_arr[2]), float(bbox_arr[3])
    if w <= 0.0 or h <= 0.0:
        return None
    bbox = BoundingBox(x=x, y=y, w=w, h=h)

    meta_arr = data.get("fs_meta")
    if meta_arr is None or meta_arr.size == 0:
        return None
    try:
        meta_bytes = meta_arr.tobytes().rstrip(b"\x00")
        meta = json.loads(meta_bytes.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not arrays:
        return None
    return FilterState(arrays=arrays, bbox=bbox, meta=meta)

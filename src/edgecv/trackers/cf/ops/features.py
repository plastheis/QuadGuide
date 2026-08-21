"""CF feature extractors with a numpy reference + optional fast backend.

ARCHITECTURE.md §6.1: trackers compose ``raw`` / ``hog`` / ``colornames``
features. Each is implemented here in pure numpy (the always-available
reference). The per-cell HOG loop is the one genuinely interpreter-bound
hotspot and is the intended drop-in point for a numba ``@njit`` fast path:
``feature_backends()`` reports ``"numba"`` when it is installed so a future
accelerated implementation can be selected behind the same signature without
touching tracker code (see the C-core analysis — accelerate one op, not the
whole core).
"""

from __future__ import annotations

import importlib.util

import numpy as np

# Rec. 601 luma weights, applied at the RGB -> gray boundary.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def feature_backends() -> list[str]:
    """Feature backends importable here; numpy is always present."""
    names = ["numpy"]
    if importlib.util.find_spec("numba") is not None:
        names.append("numba")
    return names


def _to_gray(patch: np.ndarray) -> np.ndarray:
    p = np.asarray(patch)
    if p.ndim == 3 and p.shape[2] >= 3:
        return p[..., :3].astype(np.float32) @ _LUMA
    if p.ndim == 3 and p.shape[2] == 1:
        return p[..., 0].astype(np.float32)
    return p.astype(np.float32)


def extract_raw(patch: np.ndarray) -> np.ndarray:
    """Grayscale pixels normalised to [0, 1], shape (H, W, 1)."""
    gray = _to_gray(patch) / 255.0
    return gray.astype(np.float32)[..., None]


def extract_hog(patch: np.ndarray, cell_size: int = 4, n_bins: int = 9) -> np.ndarray:
    """Unsigned-orientation HOG, shape (H//cell, W//cell, n_bins).

    Reference numpy implementation: central-difference gradients, magnitude-
    weighted voting into ``n_bins`` orientation bins over 0..pi, accumulated per
    ``cell_size``x``cell_size`` cell and L2-normalised per cell.
    """
    gray = _to_gray(patch)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx) % np.pi
    bin_idx = np.minimum((ang / (np.pi / n_bins)).astype(np.int32), n_bins - 1)

    h, w = gray.shape
    cy, cx = h // cell_size, w // cell_size
    out = np.zeros((cy, cx, n_bins), np.float32)
    for by in range(cy):
        ys = slice(by * cell_size, (by + 1) * cell_size)
        for bx in range(cx):
            xs = slice(bx * cell_size, (bx + 1) * cell_size)
            np.add.at(out[by, bx], bin_idx[ys, xs].ravel(), mag[ys, xs].ravel())

    norm = np.sqrt((out * out).sum(axis=2, keepdims=True)) + 1e-6
    return (out / norm).astype(np.float32)


def colornames(patch: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Map an (H, W, 3) RGB patch to Color-Names features via a lookup table.

    ``w2c`` is the standard 32768 x K table indexed by the 5-bit-per-channel
    quantised RGB value: ``idx = (R>>3) + 32*(G>>3) + 1024*(B>>3)``. The full
    11-dim table ships as a data artifact; this op is table-agnostic so it is
    testable with any K.
    """
    p = np.asarray(patch)
    if p.ndim != 3 or p.shape[2] < 3:
        raise ValueError("colornames expects an (H, W, 3) RGB patch")
    r = p[..., 0].astype(np.int32) >> 3
    g = p[..., 1].astype(np.int32) >> 3
    b = p[..., 2].astype(np.int32) >> 3
    idx = r + 32 * g + 1024 * b
    return np.asarray(w2c)[idx]

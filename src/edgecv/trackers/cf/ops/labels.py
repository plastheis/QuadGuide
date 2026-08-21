"""Regression-target labels for CF training (ARCHITECTURE.md §6.1).

The desired correlation output: a 2D Gaussian peaked at the window centre. CF
trackers form ``G = fft2(gaussian2d_labels(...))`` and train the filter so a
matched patch produces this response. Peaked at centre means target displacement
reads directly as ``peak - centre`` (no fftshift wrap).
"""

from __future__ import annotations

import numpy as np


def gaussian2d_labels(size: tuple[int, int], sigma: float) -> np.ndarray:
    """Peak-normalised 2D Gaussian of shape ``size`` = (h, w), centred at (h//2, w//2)."""
    h, w = size
    ys = np.arange(h) - h // 2
    xs = np.arange(w) - w // 2
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.exp(-(xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
    return g.astype(np.float32)

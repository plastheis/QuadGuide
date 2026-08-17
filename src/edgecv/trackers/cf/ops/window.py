"""Cosine (Hann) window for CF feature pre-processing (ARCHITECTURE.md §6.1).

A separable 2D Hann window tapers a feature patch to zero at the borders before
the FFT, suppressing the wrap-around discontinuity that would otherwise inject
spurious high frequencies into the correlation response.
"""

from __future__ import annotations

import numpy as np


def _hann(n: int) -> np.ndarray:
    if n < 2:
        return np.ones(max(n, 0), dtype=np.float64)
    return np.hanning(n)


def cos_window(size: tuple[int, int]) -> np.ndarray:
    """Return a separable 2D Hann window of shape ``size`` = (height, width)."""
    h, w = size
    return np.outer(_hann(h), _hann(w)).astype(np.float32)

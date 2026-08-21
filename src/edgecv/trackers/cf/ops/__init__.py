"""Shared correlation-filter ops (ARCHITECTURE.md §6.1).

Fundamental numpy ops that concrete CF trackers compose: FFT helpers, feature
extractors (raw / hog / colornames), the cosine (Hann) window, and PSR. Each op
has a numpy reference that always works; FFT and HOG expose optional accelerated
backends (scipy/pyFFTW, numba) selected behind a stable signature, so the base
wheel needs only numpy and a device build can opt into faster paths without any
tracker change.
"""

from edgecv.trackers.cf.ops.features import (
    colornames,
    extract_hog,
    extract_raw,
    feature_backends,
)
from edgecv.trackers.cf.ops.fft import fft2, fft_backends, fft_size, ifft2, set_fft_backend
from edgecv.trackers.cf.ops.labels import gaussian2d_labels
from edgecv.trackers.cf.ops.psr import psr
from edgecv.trackers.cf.ops.window import cos_window

__all__ = [
    "colornames",
    "cos_window",
    "extract_hog",
    "extract_raw",
    "feature_backends",
    "fft2",
    "fft_backends",
    "fft_size",
    "gaussian2d_labels",
    "ifft2",
    "psr",
    "set_fft_backend",
]

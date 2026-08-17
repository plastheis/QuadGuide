"""FFT ops with a swappable backend (ARCHITECTURE.md §6.1, §12).

The numpy backend is always available and is the reference. ``scipy.fft`` and
``pyFFTW`` are optional accelerated backends selected behind these stable
signatures; nothing imports them at module load, so the base wheel needs only
numpy. Transforms run over the **spatial axes (0, 1)**, leaving a trailing
channel axis untouched — the CF convention (one transform per feature channel).

Consistency rule: ``build_filter`` (in a worker) and ``evaluate`` (in the
caller) must go through this same module so a candidate filter is never
penalised by cross-backend numerical drift (extends ARCHITECTURE.md §14.6).
"""

from __future__ import annotations

import importlib.util

import numpy as np

_VALID = ("auto", "numpy", "scipy", "pyfftw")
_backend = "auto"


def _available(name: str) -> bool:
    spec = "scipy.fft" if name == "scipy" else name
    try:
        return importlib.util.find_spec(spec) is not None
    except ModuleNotFoundError:
        # find_spec raises (rather than returning None) when a parent package
        # of a dotted name is itself absent — treat that as "not available".
        return False


def fft_backends() -> list[str]:
    """Backends importable in this environment; numpy is always present."""
    names = ["numpy"]
    if _available("scipy"):
        names.append("scipy")
    if _available("pyfftw"):
        names.append("pyfftw")
    return names


def set_fft_backend(name: str) -> None:
    """Select the FFT backend. ``auto`` prefers scipy, falling back to numpy."""
    if name not in _VALID:
        raise ValueError(f"unknown fft backend {name!r}; choose from {_VALID}")
    if name not in ("auto", "numpy") and not _available(name):
        raise ValueError(f"fft backend {name!r} is not installed")
    global _backend
    _backend = name


def _resolve() -> str:
    if _backend != "auto":
        return _backend
    return "scipy" if _available("scipy") else "numpy"


def fft2(x: np.ndarray) -> np.ndarray:
    """2D forward FFT over the spatial axes (0, 1) of ``x``."""
    backend = _resolve()
    if backend == "scipy":
        import scipy.fft as sf

        return sf.fft2(x, axes=(0, 1))
    if backend == "pyfftw":
        import pyfftw.interfaces.numpy_fft as pf

        return pf.fft2(x, axes=(0, 1))
    return np.fft.fft2(x, axes=(0, 1))


def ifft2(x: np.ndarray) -> np.ndarray:
    """2D inverse FFT over the spatial axes (0, 1) of ``x``."""
    backend = _resolve()
    if backend == "scipy":
        import scipy.fft as sf

        return sf.ifft2(x, axes=(0, 1))
    if backend == "pyfftw":
        import pyfftw.interfaces.numpy_fft as pf

        return pf.ifft2(x, axes=(0, 1))
    return np.fft.ifft2(x, axes=(0, 1))


def fft_size(n: int) -> int:
    """Smallest efficient transform length >= n. Numpy reference: next power of two.

    CF templates are a fixed size after init and transform every frame, so rounding
    the crop up to a power of two keeps the FFT fast without changing the algorithm.
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

"""Peak-to-Sidelobe Ratio (ARCHITECTURE.md §3, §6.1).

PSR measures how sharply a CF response peaks above its surroundings; it serves
as both per-frame confidence and the lock signal, and is the quantity the
fusion gate (§8) compares between incumbent and candidate filters. It is
invariant to additive offset and positive scaling of the response, so the score
is stable regardless of feature normalisation.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def psr(response: np.ndarray, window: int = 11) -> float:
    """PSR of ``response``: (peak - sidelobe_mean) / sidelobe_std.

    A ``window``x``window`` square centred on the peak is excluded from the
    sidelobe statistics.
    """
    r = np.asarray(response, dtype=np.float64)
    peak = float(r.max())
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)

    half = window // 2
    mask = np.ones(r.shape, dtype=bool)
    y0, y1 = max(0, py - half), min(r.shape[0], py + half + 1)
    x0, x1 = max(0, px - half), min(r.shape[1], px + half + 1)
    mask[y0:y1, x0:x1] = False

    sidelobe = r[mask]
    mean = float(sidelobe.mean())
    std = float(sidelobe.std())
    return (peak - mean) / (std + _EPS)

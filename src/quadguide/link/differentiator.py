from __future__ import annotations
import math


def _angle_diff(a: float, b: float) -> float:
    """Shortest-path angular difference a - b, result in (-π, π]."""
    diff = b - a
    # Wrap to (-π, π]
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff <= -math.pi:
        diff += 2 * math.pi
    return diff


class AttitudeDifferentiator:
    def __init__(self, alpha: float):
        self._alpha  = alpha
        self._roll   = 0.0
        self._pitch  = 0.0
        self._yaw    = 0.0
        self._ns     = 0
        self._rr     = 0.0  # filtered roll rate
        self._pr     = 0.0  # filtered pitch rate
        self._yr     = 0.0  # filtered yaw rate
        self._ready  = False

    def update(self, roll: float, pitch: float, yaw: float, now_ns: int
               ) -> tuple[float, float, float]:
        if not self._ready:
            self._roll, self._pitch, self._yaw, self._ns = roll, pitch, yaw, now_ns
            self._ready = True
            return 0.0, 0.0, 0.0

        dt = (now_ns - self._ns) * 1e-9
        if dt <= 0.0:
            return self._rr, self._pr, self._yr

        raw_rr = (roll  - self._roll)              / dt
        raw_pr = (pitch - self._pitch)             / dt
        raw_yr = _angle_diff(yaw, self._yaw)       / dt

        self._rr = self._alpha * raw_rr + (1.0 - self._alpha) * self._rr
        self._pr = self._alpha * raw_pr + (1.0 - self._alpha) * self._pr
        self._yr = self._alpha * raw_yr + (1.0 - self._alpha) * self._yr

        self._roll, self._pitch, self._yaw, self._ns = roll, pitch, yaw, now_ns
        return self._rr, self._pr, self._yr

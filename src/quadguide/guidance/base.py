from __future__ import annotations
from typing import Protocol

from quadguide.core.messages import IMUFrame, LockOnCmd, TargetEstimate


class GuidanceMethod(Protocol):
    """Strategy interface for guidance algorithms.

    Each method consumes the same inputs but uses what it needs. Returns
    body-frame lateral/longitudinal acceleration commands in m/s² that the
    control worker maps to roll/pitch via the small-angle tilt approximation.
    """

    def compute(
        self,
        est: TargetEstimate,
        imu: IMUFrame,
        lockon_cmd: LockOnCmd | None,
        now_ns: int,
    ) -> tuple[float, float]: ...

    def name(self) -> str: ...

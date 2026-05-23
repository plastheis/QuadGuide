from __future__ import annotations

from quadguide.core.config import PronavConfig
from quadguide.core.messages import IMUFrame, LockOnCmd, TargetEstimate
from quadguide.guidance.closing_vel import ClosingVelEstimator
from quadguide.guidance.los import LOSRateEstimator


def pronav(
    los_rate: tuple[float, float],
    closing_vel: float,
    N: float,
) -> tuple[float, float]:
    """Proportional navigation: a_cmd = N * V_c * los_rate."""
    return N * closing_vel * los_rate[0], N * closing_vel * los_rate[1]


class ProNavGuidance:
    """Proportional navigation: a = N * V_c * LOS_rate.

    Requires LOS rate (image-plane derivative with body-rate derotation) and
    a closing velocity estimate from bounding-box area growth.
    """

    def __init__(self, cfg: PronavConfig, fov_horizontal_rad: float, aspect: float) -> None:
        self._cfg = cfg
        self._los = LOSRateEstimator(fov_horizontal_rad, aspect)
        self._cv = ClosingVelEstimator()

    def name(self) -> str:
        return "pronav"

    def compute(
        self,
        est: TargetEstimate,
        imu: IMUFrame,
        lockon_cmd: LockOnCmd | None,
        now_ns: int,
    ) -> tuple[float, float]:
        los_r = self._los.update(est.centroid_norm, imu, lockon_cmd, now_ns)
        v_c = self._cv.update(est.bbox, now_ns, self._cfg)
        return pronav(los_r, v_c, self._cfg.N)

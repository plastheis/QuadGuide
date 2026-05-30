from __future__ import annotations

from quadguide.core.config import PurePursuitConfig
from quadguide.core.messages import IMUFrame, LockOnCmd, TrackerEstimate
from quadguide.guidance._centroid import bbox_centroid_norm


class PurePursuitGuidance:
    """Pure pursuit: command acceleration straight toward the target LOS.

        a = K * LOS_angle

    The centroid_norm is converted to a physical LOS angle (radians) via the
    camera FoV, then scaled by K (m/s² per radian). No LOS rate, no closing
    velocity, no body-rate derotation — the simplest possible homing law.
    Maps cleanly into the control attitude_cmd downstream via a ≈ g·θ.
    """

    def __init__(self, cfg: PurePursuitConfig, fov_horizontal_rad: float, aspect: float) -> None:
        self._K = cfg.K
        # centroid_norm spans (-1, 1) across the image. Half-FoV per side.
        self._scale_x = fov_horizontal_rad * 0.5
        self._scale_y = (fov_horizontal_rad / aspect) * 0.5

    def name(self) -> str:
        return "pure_pursuit"

    def compute(
        self,
        est: TrackerEstimate,
        imu: IMUFrame,
        lockon_cmd: LockOnCmd | None,
        now_ns: int,
    ) -> tuple[float, float]:
        cx, cy = bbox_centroid_norm(est.bbox)
        return self._K * cx * self._scale_x, self._K * cy * self._scale_y

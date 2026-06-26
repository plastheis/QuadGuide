from __future__ import annotations

from quadguide.core.config import PurePursuitConfig
from quadguide.core.messages import IMUFrame, LockOnCmd, TrackerEstimate
from quadguide.guidance._centroid import bbox_centroid_norm


def _soft_deadband(c: float, db: float) -> float:
    """Shifted deadband: zero inside ±db, continuous past the edge.

    Outside the band the threshold is *subtracted* (not the value clamped),
    so the response rises from 0 at |c|=db with the original unit slope — no
    step at the edge. Keeps boresight jitter from twitching the command while
    preserving accuracy at distance (only the small db offset is lost).
    """
    if c > db:
        return c - db
    if c < -db:
        return c + db
    return 0.0


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
        self._deadband = cfg.deadband
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
        # Center deadband: ignore sub-db boresight jitter so a locked-but-jittering
        # box near centre doesn't continuously micro-tilt the airframe.
        cx = _soft_deadband(cx, self._deadband)
        cy = _soft_deadband(cy, self._deadband)
        # Image axes → body-frame accel for the bore-up mount: the camera's
        # horizontal axis (cx) is a *lateral* (body +Y) offset, nulled by roll
        # (AccelCmd.ay); the vertical axis (cy) is a *fore/aft* (body +X) offset,
        # nulled by pitch (AccelCmd.ax). attitude_cmd then maps ax→pitch, ay→roll.
        ax = self._K * cy * self._scale_y   # forward / pitch ← vertical image error
        ay = self._K * cx * self._scale_x   # lateral / roll  ← horizontal image error
        return ax, ay

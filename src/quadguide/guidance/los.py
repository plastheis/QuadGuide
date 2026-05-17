from __future__ import annotations
import math

from quadguide.core.messages import AttitudeState, LockOnCmd


def _rot_matrix(roll: float, pitch: float, yaw: float) -> tuple:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr),
        (sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr),
        (-sp,      cp * sr,                 cp * cr               ),
    )


def _body_rate_correction(
    att: AttitudeState,
    fov_h: float,
    aspect: float,
) -> tuple[float, float]:
    R = _rot_matrix(att.roll_rad, att.pitch_rad, att.yaw_rad)
    p, q, r = att.roll_rate_rps, att.pitch_rate_rps, att.yaw_rate_rps

    # Camera boresight in inertial frame: R @ [0, 0, 1] = R column 2
    bx = R[0][2]; by = R[1][2]; bz = R[2][2]

    # Angular velocity in inertial frame: R @ [p, q, r]
    wx = R[0][0] * p + R[0][1] * q + R[0][2] * r
    wy = R[1][0] * p + R[1][1] * q + R[1][2] * r
    wz = R[2][0] * p + R[2][1] * q + R[2][2] * r

    # LOS angular rate in inertial = omega_i x boresight_i
    lx_i = wy * bz - wz * by
    ly_i = wz * bx - wx * bz

    # Project back to body/image frame: R.T @ [lx_i, ly_i, 0]
    lx_b = R[0][0] * lx_i + R[1][0] * ly_i
    ly_b = R[0][1] * lx_i + R[1][1] * ly_i

    # Scale from rad/s to centroid_norm/s (centroid_norm spans 2 across fov_h)
    fov_v = fov_h / aspect
    return lx_b * (2.0 / fov_h), ly_b * (2.0 / fov_v)


class LOSRateEstimator:
    """Line-of-sight rate estimator with lock-on seq reset and body-rate correction."""

    def __init__(self, fov_horizontal_rad: float, aspect: float) -> None:
        self._fov_h = fov_horizontal_rad
        self._aspect = aspect
        self._prev_centroid: tuple[float, float] | None = None
        self._prev_ts_ns: int = 0
        self._last_lockon_seq: int | None = None

    def update(
        self,
        centroid_norm: tuple[float, float],
        att: AttitudeState,
        lockon_cmd: LockOnCmd | None,
        now_ns: int,
    ) -> tuple[float, float]:
        if lockon_cmd is not None and lockon_cmd.seq != self._last_lockon_seq:
            self._last_lockon_seq = lockon_cmd.seq
            self._prev_centroid = centroid_norm
            self._prev_ts_ns = now_ns
            return (0.0, 0.0)

        if self._prev_centroid is None:
            self._prev_centroid = centroid_norm
            self._prev_ts_ns = now_ns
            return (0.0, 0.0)

        dt = (now_ns - self._prev_ts_ns) * 1e-9
        if dt <= 0.0:
            return (0.0, 0.0)

        raw_x = (centroid_norm[0] - self._prev_centroid[0]) / dt
        raw_y = (centroid_norm[1] - self._prev_centroid[1]) / dt

        corr_x, corr_y = _body_rate_correction(att, self._fov_h, self._aspect)

        self._prev_centroid = centroid_norm
        self._prev_ts_ns = now_ns

        return raw_x - corr_x, raw_y - corr_y

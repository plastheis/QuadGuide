from __future__ import annotations

from quadguide.core.messages import IMUFrame, LockOnCmd


def _body_rate_correction(
    imu: IMUFrame,
    fov_h: float,
    aspect: float,
) -> tuple[float, float]:
    """Image-plane LOS rate induced by body rotation, in centroid_norm/s.

    Bore-sight is body +Z. With pitch about body Y the image sweeps in x;
    with roll about body X the image sweeps in y (opposite sign). Yaw about
    +Z rotates the image about its centre, contributing nothing at the centroid.
    """
    fov_v = fov_h / aspect
    return imu.gy * (2.0 / fov_h), -imu.gx * (2.0 / fov_v)


class LOSRateEstimator:
    """Line-of-sight rate estimator with lock-on seq reset and body-rate derotation.

    Body rates come directly from the FC's 0x80 IMU gyro (NED body axes),
    NOT from differentiated Euler angles.
    """

    def __init__(self, fov_horizontal_rad: float, aspect: float) -> None:
        self._fov_h = fov_horizontal_rad
        self._aspect = aspect
        self._prev_centroid: tuple[float, float] | None = None
        self._prev_ts_ns: int = 0
        self._last_lockon_seq: int | None = None

    def update(
        self,
        centroid_norm: tuple[float, float],
        imu: IMUFrame,
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

        corr_x, corr_y = _body_rate_correction(imu, self._fov_h, self._aspect)

        self._prev_centroid = centroid_norm
        self._prev_ts_ns = now_ns

        return raw_x - corr_x, raw_y - corr_y

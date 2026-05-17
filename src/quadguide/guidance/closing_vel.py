from __future__ import annotations
import logging

from quadguide.core.messages import BoundingBox

log = logging.getLogger(__name__)


class ClosingVelEstimator:
    """Estimates closing velocity from rate of change of bounding box area.

    Positive = target getting closer. Falls back to cfg.closing_vel_fallback
    when the EMA-smoothed area rate is below cfg.closing_vel_min_area_rate.
    """

    def __init__(self) -> None:
        self._prev_area: float | None = None
        self._prev_ts_ns: int = 0
        self._ema_area_rate: float = 0.0

    def update(self, bbox: BoundingBox, now_ns: int, cfg) -> float:
        area = bbox.w * bbox.h

        if self._prev_area is None:
            self._prev_area = area
            self._prev_ts_ns = now_ns
            return cfg.closing_vel_fallback

        dt = (now_ns - self._prev_ts_ns) * 1e-9
        if dt <= 0.0:
            return cfg.closing_vel_fallback

        raw_rate = (area - self._prev_area) / dt
        self._ema_area_rate = (
            cfg.closing_vel_ema_alpha * raw_rate
            + (1.0 - cfg.closing_vel_ema_alpha) * self._ema_area_rate
        )

        self._prev_area = area
        self._prev_ts_ns = now_ns

        if abs(self._ema_area_rate) < cfg.closing_vel_min_area_rate:
            log.debug("closing_vel: using fallback")
            return cfg.closing_vel_fallback

        return self._ema_area_rate * cfg.closing_vel_area_scale

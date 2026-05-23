from __future__ import annotations

from quadguide.core.bus import Bus
from quadguide.core.config import WatchdogConfig
from quadguide.core.health import Watchdog


def build_watchdog(cfg: WatchdogConfig, bus: Bus) -> Watchdog:
    return Watchdog(
        [
            ("target/estimate", cfg.target_estimate_ms),
            ("fc/attitude",     cfg.fc_attitude_ms),
            ("fc/imu",          cfg.fc_imu_ms),
            ("guidance/accel",  cfg.guidance_accel_ms),
        ],
        bus,
    )

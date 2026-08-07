from __future__ import annotations

from quadguide.core.bus import Bus
from quadguide.core.config import WatchdogConfig
from quadguide.core.health import Watchdog


def build_watchdog(cfg: WatchdogConfig, bus: Bus) -> Watchdog:
    """Topics whose staleness means a genuine FAULT (a worker or the FC died).

    ``guidance/accel`` is deliberately NOT here. Guidance stops publishing it by
    DESIGN whenever the tracker is not driving (``guidance/worker.py`` skips
    LOST/NO_LOCK/ACQUIRING), so watchdogging it made a normal target loss look
    like a fault: the watchdog latch (accel stale at guidance_accel_ms + its own
    hold_ms) beat the target_loss latch every time, so target_loss.hold_ms was
    effectively capped and the trip was misattributed to the watchdog. It also
    meant arming before the operator locked on immediately latched the failsafe.

    Accel freshness still gates the attitude command — the control worker checks
    it directly against ``guidance_accel_ms`` and levels roll/pitch on a stale
    accel — it just no longer counts as a failsafe condition.
    """
    return Watchdog(
        [
            ("target/estimate", cfg.target_estimate_ms),
            ("fc/attitude",     cfg.fc_attitude_ms),
            ("fc/imu",          cfg.fc_imu_ms),
        ],
        bus,
    )

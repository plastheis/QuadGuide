from __future__ import annotations
import signal

from quadguide.core.bus import Bus
from quadguide.core.clock import RateLimiter, monotonic_ns
from quadguide.core.config import cfg_airframe, cfg_guidance, cfg_platform, cfg_watchdog
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.health import FailsafeState, HealthFault
from quadguide.core.logging import setup_logging
from quadguide.core.messages import ControlCmd, HealthReport, ProcessState
from quadguide.control.attitude_cmd import compute as attitude_cmd_compute
from quadguide.control.limiter import failsafe_cmd, saturate, slew_rate
from quadguide.control.watchdog import build_watchdog
from quadguide.platform.adapter import PlatformAdapter

__all__ = ["run"]

_HEALTH_EVERY = 20   # iterations; 100 Hz / 20 = 5 Hz health rate
_DT = 1.0 / 100      # nominal loop period (s); fixed, not measured per-loop


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    log = setup_logging("control", config)
    gcfg = cfg_guidance(config)
    acfg = cfg_airframe(config)
    pcfg = cfg_platform(config)
    wcfg = cfg_watchdog(config)

    platform = PlatformAdapter(config)
    platform.set_realtime(pcfg.realtime.control_cpu_core, pcfg.realtime.control_fifo_prio)

    watchdog = build_watchdog(wcfg, bus)
    rate = RateLimiter(hz=100)

    prev_cmd: ControlCmd | None = None
    state = FailsafeState.NOMINAL
    stop = False

    def _on_sigterm(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    i = 0
    log.info(
        "control: started (100 Hz, core=%d, sched_fifo=%s)",
        pcfg.realtime.control_cpu_core,
        pcfg.realtime.control_sched_fifo,
    )

    while not stop:
        rate.sleep()
        i += 1

        try:
            watchdog.check_all()
            state = FailsafeState.NOMINAL
        except HealthFault as e:
            state = FailsafeState.LEVEL
            cmd = failsafe_cmd(gcfg.throttle_hold)
            bus.publish("control/cmd", cmd)
            log.warning("control: failsafe — %s", e)
            prev_cmd = cmd
            if i % _HEALTH_EVERY == 0:
                bus.publish(
                    "system/health",
                    HealthReport(monotonic_ns(), "control", ProcessState.FAILSAFE, str(e)),
                )
            continue

        accel = bus.latest("guidance/accel")

        if accel is None:
            continue

        roll, pitch = attitude_cmd_compute(accel)
        roll, pitch = saturate(roll, pitch, acfg.control_limits)
        roll, pitch = slew_rate(roll, pitch, prev_cmd, acfg.control_limits, _DT)

        cmd = ControlCmd(monotonic_ns(), roll, pitch, 0.0, gcfg.throttle_hold)
        bus.publish("control/cmd", cmd)
        prev_cmd = cmd

        if i % _HEALTH_EVERY == 0:
            bus.publish(
                "system/health",
                HealthReport(monotonic_ns(), "control", ProcessState.OK, ""),
            )

    bus.detach()
    log.info("control: stopped (last state=%s)", state.value)

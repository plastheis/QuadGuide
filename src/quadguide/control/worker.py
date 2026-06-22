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
from quadguide.control.limiter import saturate, slew_rate
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

    from quadguide.core.config import cfg_diag
    from quadguide.core.diagtrace import DiagTrace

    platform = PlatformAdapter(config)
    platform.set_realtime(pcfg.realtime.control_cpu_core, pcfg.realtime.control_fifo_prio)

    watchdog = build_watchdog(wcfg, bus)
    rate = RateLimiter(hz=100)

    dcfg = cfg_diag(config)
    trace = DiagTrace("control", enabled=dcfg.trace,
                      dir=dcfg.trace_dir, max_rows=dcfg.trace_max_rows)

    prev_cmd: ControlCmd | None = None
    state = FailsafeState.NOMINAL
    stop = False

    def _on_sigterm(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    armed = False
    in_failsafe = False
    i = 0
    log.info(
        "control: started (100 Hz, core=%d, sched_fifo=%s, throttle_hold=%.2f)",
        pcfg.realtime.control_cpu_core,
        pcfg.realtime.control_sched_fifo,
        gcfg.throttle_hold,
    )

    while not stop:
        rate.sleep()
        i += 1
        now_ns = monotonic_ns()

        # Track arm state from ground station
        arm_cmd = bus.latest("arm/cmd")
        now_armed = bool(arm_cmd and arm_cmd.armed)
        if now_armed != armed:
            armed = now_armed
            if armed:
                log.info("control: ARMED — throttle gated by fire button")
            else:
                log.info("control: DISARMED — throttle=0, commands suppressed")

        # Track fire state from ground station
        fire_cmd = bus.latest("fire/cmd")
        fire_active = bool(fire_cmd and fire_cmd.active)

        # Watchdog
        try:
            watchdog.check_all()
            if in_failsafe:
                log.info("control: failsafe cleared")
                in_failsafe = False
            state = FailsafeState.NOMINAL
            fault = None
        except HealthFault as e:
            fault = e
            state = FailsafeState.LEVEL
            if not in_failsafe:
                log.warning("control: entering failsafe — %s", e)
                in_failsafe = True

        accel = bus.latest("guidance/accel")

        # Choose throttle: armed + fire active → throttle_hold; else 0
        thr = gcfg.throttle_hold if (armed and fire_active) else 0.0

        # Choose attitude: only apply guidance when armed, no failsafe, and accel present
        if armed and fault is None and accel is not None:
            roll, pitch = attitude_cmd_compute(accel)
            roll, pitch = saturate(roll, pitch, acfg.control_limits)
            roll, pitch = slew_rate(roll, pitch, prev_cmd, acfg.control_limits, _DT)
        else:
            roll, pitch = 0.0, 0.0
            prev_cmd = None  # reset slew baseline so re-entry starts from level

        # origin_ns tracks the capture lineage regardless of whether the command is
        # applied — so end-to-end latency is observable even disarmed (during bench).
        origin_ns = accel.origin_ns if accel is not None else 0
        cmd = ControlCmd(now_ns, roll, pitch, 0.0, thr, origin_ns=origin_ns)
        bus.publish("control/cmd", cmd)
        prev_cmd = cmd
        if accel is not None and accel.origin_ns > 0:
            trace.latency(now_ns, accel.timestamp_ns, accel.origin_ns)

        if i % _HEALTH_EVERY == 0:
            proc_state = ProcessState.FAILSAFE if in_failsafe else ProcessState.OK
            detail = str(fault) if fault is not None else ""
            bus.publish(
                "system/health",
                HealthReport(monotonic_ns(), "control", proc_state, detail),
            )
            trace.state(monotonic_ns(), armed=armed, fire_active=fire_active,
                        in_failsafe=in_failsafe, fault=detail, throttle=thr)

    trace.flush()
    bus.detach()
    log.info("control: stopped (last state=%s)", state.value)

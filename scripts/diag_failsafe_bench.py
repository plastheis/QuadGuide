#!/usr/bin/env python3
"""Bench diagnosis for the target-loss failsafe — no camera, no FC.

Forks the REAL control worker with the REAL config, then plays synthetic bus
traffic that reproduces "operator locks on, then the target leaves frame":

  phase LOCK   target/estimate=nominal + fc/attitude + fc/imu + guidance/accel
  phase LOST   target/estimate=lost, guidance/accel STOPS
               (exactly what guidance/worker.py does — it `continue`s on LOST,
                so guidance/accel goes stale ~guidance_accel_ms later)
  phase BACK   target reacquired — proves the latch is sticky
  phase SAFE   operator disarm — proves the latch clears

Prints a sampled timeline so nothing depends on log interleaving.

Scenarios and their expected verdicts (configs/rpi4b.yaml, target_loss
hold_ms=300 / watchdog hold_ms=200):

    armed      LOST while armed      → SET_MODE LAND @ ~300 ms, target_loss=True
    disarmed   LOST while disarmed   → NO FAILSAFE (the arm gate — by design)
    nolock     armed, never locked   → NO FAILSAFE (no_lock is not LOST)
    fcloss     FC link dies          → SET_MODE LAND @ ~450 ms, watchdog=True

Check which condition latched in the control log:
    grep "FAILSAFE latched" {logging.dir}/control.log

Usage:
    python scripts/diag_failsafe_bench.py --scenario armed
    python scripts/diag_failsafe_bench.py --config configs/rk3588.yaml --scenario fcloss
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.config import load_config, cfg_failsafe
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.messages import (
    AccelCmd, ArmCmd, AttitudeState, BoundingBox, FailsafeActionWire,
    FireCmd, IMUFrame, TrackerEstimate, TrackerHealth,
)
from quadguide.control import worker as control_worker

_TICK = 0.02


def _fc(bus):
    now = monotonic_ns()
    bus.publish("fc/attitude", AttitudeState(now, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    bus.publish("fc/imu", IMUFrame(now, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))


def _track(bus, health, conf):
    now = monotonic_ns()
    bus.publish("target/estimate", TrackerEstimate(
        now, BoundingBox(0.45, 0.45, 0.1, 0.1), conf, health, origin_ns=now))


def _accel(bus):
    now = monotonic_ns()
    bus.publish("guidance/accel", AccelCmd(now, 0.0, 0.0, origin_ns=now))


def run_phase(bus, rows, t0, label, secs, *, health, publish_accel, armed,
              publish_fc=True):
    """Drive the bus for `secs`, sampling failsafe/action + control/cmd each tick."""
    end = time.monotonic() + secs
    while time.monotonic() < end:
        bus.publish("arm/cmd", ArmCmd(monotonic_ns(), armed))
        bus.publish("fire/cmd", FireCmd(monotonic_ns(), True))
        if publish_fc:
            _fc(bus)
        if health is not None:
            _track(bus, health, 0.99 if health is TrackerHealth.NOMINAL else 0.0)
        if publish_accel:
            _accel(bus)
        time.sleep(_TICK)
        fs = bus.latest("failsafe/action")
        cc = bus.latest("control/cmd")
        rows.append((
            time.monotonic() - t0, label, armed,
            health.value if health else "-",
            FailsafeActionWire(fs.action).name if fs else "-",
            fs.custom_mode if fs else 0,
            round(cc.throttle_norm, 2) if cc else None,
        ))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rpi4b.yaml")
    ap.add_argument("--scenario", default="armed",
                    choices=["armed", "disarmed", "nolock", "fcloss"])
    ap.add_argument("--log-dir", default=None,
                    help="where the forked control worker writes control.log "
                         "(default: a bench dir, NOT the flight log dir)")
    args = ap.parse_args()

    cfg = load_config(args.config, {})
    # Never write into the flight log dir from a bench run.
    logdir = args.log_dir or os.path.join(os.getcwd(), "quadguide-trace", "failsafe-bench")
    os.makedirs(logdir, exist_ok=True)
    cfg.setdefault("logging", {})["dir"] = logdir
    cfg["platform"]["realtime"]["control_sched_fifo"] = False

    f = cfg_failsafe(cfg)
    print(f"config: {args.config}   scenario: {args.scenario}")
    print(f"  target_loss: enabled={f.target_loss.enabled} action={f.target_loss.action.value} "
          f"mode={f.target_loss.mode} hold_ms={f.target_loss.hold_ms}")
    print(f"  watchdog:    enabled={f.watchdog.enabled} action={f.watchdog.action.value} "
          f"mode={f.watchdog.mode} hold_ms={f.watchdog.hold_ms}")
    print(f"  watchdog timeouts: {cfg['watchdog']}\n")

    armed = args.scenario != "disarmed"
    bus = Bus(ring_depth=cfg.get("bus", {}).get("ring_depth", 8))
    fb = FrameBuffer(64, 64, 3, n_slots=2)
    proc = multiprocessing.Process(
        target=control_worker.run, args=(cfg, bus, fb), daemon=True)
    proc.start()
    time.sleep(0.4)

    rows: list = []
    t0 = time.monotonic()
    try:
        if args.scenario == "fcloss":
            # genuine fault: the FC link dies while locked and tracking.
            # The watchdog MUST still catch this (fc/attitude + fc/imu stale).
            run_phase(bus, rows, t0, "LOCK", 1.5,
                      health=TrackerHealth.NOMINAL, publish_accel=True, armed=armed)
            run_phase(bus, rows, t0, "FCLOSS", 2.0,
                      health=TrackerHealth.NOMINAL, publish_accel=True,
                      armed=armed, publish_fc=False)
        elif args.scenario == "nolock":
            # never locked on: tracker reports no_lock, guidance never publishes
            run_phase(bus, rows, t0, "NOLOCK", 2.0,
                      health=TrackerHealth.NO_LOCK, publish_accel=False, armed=armed)
        else:
            run_phase(bus, rows, t0, "LOCK", 1.5,
                      health=TrackerHealth.NOMINAL, publish_accel=True, armed=armed)
            run_phase(bus, rows, t0, "LOST", 2.0,
                      health=TrackerHealth.LOST, publish_accel=False, armed=armed)
            run_phase(bus, rows, t0, "BACK", 1.0,
                      health=TrackerHealth.NOMINAL, publish_accel=True, armed=armed)
        run_phase(bus, rows, t0, "SAFE", 0.6,
                  health=TrackerHealth.NOMINAL, publish_accel=True, armed=False)
    finally:
        proc.terminate()
        proc.join(timeout=2)
        bus.close()
        fb.close()

    # ── report: print only the transitions, plus phase boundaries ──────────
    print(f"{'t(s)':>7}  {'phase':<7} {'armed':<6} {'health':<9} "
          f"{'failsafe':<9} {'mode':>4}  thr")
    prev = None
    for r in rows:
        key = (r[1], r[2], r[3], r[4], r[5])
        if key != prev:
            print(f"{r[0]:7.3f}  {r[1]:<7} {str(r[2]):<6} {r[3]:<9} "
                  f"{r[4]:<9} {r[5]:>4}  {r[6]}")
            prev = key

    # ── verdict ────────────────────────────────────────────────────────────
    lost_rows = [r for r in rows if r[1] in ("LOST", "NOLOCK", "FCLOSS")]
    trip = next((r for r in lost_rows if r[4] not in ("-", "NONE")), None)
    t_phase = lost_rows[0][0] if lost_rows else 0.0
    print()
    if trip:
        print(f"VERDICT: failsafe fired {(trip[0] - t_phase) * 1000:.0f} ms into "
              f"the loss → {trip[4]} custom_mode={trip[5]}")
    else:
        print("VERDICT: NO FAILSAFE — nothing latched for the whole loss window")
    back = [r for r in rows if r[1] == "BACK"]
    if back:
        print(f"  latched after reacquire: {back[-1][4]} "
              f"({'sticky — correct' if back[-1][4] not in ('-', 'NONE') else 'CLEARED — not sticky'})")
    safe = [r for r in rows if r[1] == "SAFE"]
    if safe:
        print(f"  after operator disarm:   {safe[-1][4]} "
              f"({'cleared — correct' if safe[-1][4] in ('-', 'NONE') else 'STILL LATCHED'})")
    return 0


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    sys.exit(main())

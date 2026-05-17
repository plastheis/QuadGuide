#!/usr/bin/env python3
"""CRSF RC uplink transmitter test (bus-based).

Starts the link worker and publishes ControlCmd/ArmCmd through the bus at a
fixed rate. The link worker handles the µs→ticks conversion and serial writes.

Usage:
    python scripts/test_link_tx.py [--rate 50]
        [--arm] [--pre-arm-secs 2] [--arm-secs 2]
        [--roll-deg 0] [--pitch-deg 0] [--throttle-norm 0.0] [--yaw-rate-dps 0]

Defaults for serial port and baud are read from configs/config.yaml.

When --arm is set, the script runs a safe arming sequence:
  Phase 1 (--pre-arm-secs): throttle=0, disarmed  — establish uplink
  Phase 2 (--arm-secs):     throttle=0, armed      — FC arms with throttle held low
  Phase 3 (ongoing):        commanded throttle, armed
"""
import argparse
import multiprocessing
import os
import sys
import time

sys.path.insert(0, "src")

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.config import load_config
from quadguide.core.messages import ArmCmd, ControlCmd
from quadguide.link import worker as link_worker

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")


def main():
    cfg        = load_config(_CONFIG_PATH, {})
    serial_cfg = cfg["platform"]["serial"]

    parser = argparse.ArgumentParser(description="CRSF RC uplink test (via bus)")
    parser.add_argument("--rate",          type=float, default=50.0,
                        help="Publish rate in Hz (default: 50)")
    parser.add_argument("--arm",           action="store_true",
                        help="Run arming sequence then hold armed")
    parser.add_argument("--pre-arm-secs",  type=float, default=2.0,
                        help="Seconds to send disarmed uplink before arming (default: 2)")
    parser.add_argument("--arm-secs",      type=float, default=2.0,
                        help="Seconds to hold armed with throttle-min before applying throttle (default: 2)")
    parser.add_argument("--roll-deg",      type=float, default=0.0,
                        help="Roll command in degrees (default: 0)")
    parser.add_argument("--pitch-deg",     type=float, default=0.0,
                        help="Pitch command in degrees (default: 0)")
    parser.add_argument("--throttle-norm", type=float, default=0.0,
                        help="Throttle 0.0–1.0 (default: 0.0)")
    parser.add_argument("--yaw-rate-dps",  type=float, default=0.0,
                        help="Yaw rate command in deg/s (default: 0)")
    args = parser.parse_args()

    ring_depth = cfg.get("bus", {}).get("ring_depth", 8)
    bus        = Bus(ring_depth=ring_depth)

    link_proc = multiprocessing.Process(
        target=link_worker.run, args=(cfg, bus), daemon=True
    )
    link_proc.start()

    interval = 1.0 / args.rate
    start    = time.monotonic()
    count    = 0
    next_t   = start

    if args.arm:
        arm_end      = start + args.pre_arm_secs
        throttle_end = arm_end + args.arm_secs
        print(f"Serial: {serial_cfg['port']} @ {serial_cfg['baud']} baud  rate={args.rate:.0f} Hz")
        print(f"Arming: {args.pre_arm_secs:.0f}s disarmed → "
              f"{args.arm_secs:.0f}s armed/throttle-min → throttle={args.throttle_norm:.2f}")
        print("Press Ctrl+C to stop.\n")
    else:
        arm_end = throttle_end = start
        print(f"Serial: {serial_cfg['port']} @ {serial_cfg['baud']} baud  rate={args.rate:.0f} Hz")
        print(f"roll={args.roll_deg:+.1f}° pitch={args.pitch_deg:+.1f}° "
              f"thr={args.throttle_norm:.2f} yaw={args.yaw_rate_dps:+.1f}dps  [DISARMED]")
        print("Press Ctrl+C to stop.\n")

    try:
        while True:
            now = time.monotonic()
            if now < next_t:
                continue

            t  = now - start
            ts = monotonic_ns()

            if args.arm and now < arm_end:
                armed        = False
                throttle_now = 0.0
                phase        = "PHASE 1: uplink  "
            elif args.arm and now < throttle_end:
                armed        = True
                throttle_now = 0.0
                phase        = "PHASE 2: arming  "
            else:
                armed        = args.arm
                throttle_now = args.throttle_norm
                phase        = "PHASE 3: running " if args.arm else "         running "

            bus.publish("arm/cmd", ArmCmd(ts, armed))
            bus.publish("control/cmd", ControlCmd(
                timestamp_ns=ts,
                roll_deg=args.roll_deg,
                pitch_deg=args.pitch_deg,
                yaw_rate_dps=args.yaw_rate_dps,
                throttle_norm=throttle_now,
            ))
            count += 1
            print(
                f"\r[t={t:6.2f}s] {phase} "
                f"roll={args.roll_deg:+6.1f}°  pitch={args.pitch_deg:+6.1f}°  "
                f"thr={throttle_now:.2f}  yaw={args.yaw_rate_dps:+6.1f}dps  "
                f"armed={armed}  frames={count}",
                end="", flush=True,
            )
            next_t += interval

    except KeyboardInterrupt:
        pass
    finally:
        link_proc.terminate()
        link_proc.join(timeout=2)
        bus.close()

    elapsed = time.monotonic() - start
    print(f"\n\n{count} frames in {elapsed:.1f}s ({count/elapsed:.1f} Hz)" if elapsed > 0 else "")


if __name__ == "__main__":
    main()

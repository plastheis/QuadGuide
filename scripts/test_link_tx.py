#!/usr/bin/env python3
"""CRSF RC uplink transmitter test.

Sends CRSF RC_CHANNELS_PACKED frames at a fixed rate. Use this to verify the
companion → FC uplink is working. Once a steady uplink is established, the FC
exits failsafe and begins sending attitude telemetry back (visible in test_link_rx.py).

Usage:
    python scripts/test_link_tx.py [--port /dev/ttyS0] [--baud 420000] [--rate 50]
        [--arm] [--pre-arm-secs 2] [--arm-secs 2]
        [--roll 992] [--pitch 992] [--throttle 172] [--yaw 992]

Defaults for --port and --baud are read from configs/config.yaml.

When --arm is set, the script runs a safe arming sequence:
  Phase 1 (--pre-arm-secs): throttle=min, CH5=low  — establish uplink, FC exits failsafe
  Phase 2 (--arm-secs):     throttle=min, CH5=high  — FC arms with throttle held down
  Phase 3 (ongoing):        throttle=commanded, CH5=high

CH5 is the arm channel. Use --arm to set it high (1811 = armed).
Channel values are in CRSF ticks: 172 (1000µs) – 992 (1500µs) – 1811 (2000µs).
"""
import argparse
import os
import sys
import time

import serial

sys.path.insert(0, "src")

from quadguide.core.config import load_config
from quadguide.link.crsf import build_frame, pack_channels, CRSF_RC_CHANNELS

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")

_MIN_THROTTLE = 172
_ARM_HIGH     = 1811
_ARM_LOW      = 172


def _make_frame(roll, pitch, throttle, yaw, ch5):
    channels = [roll, pitch, throttle, yaw, ch5, *([992] * 11)]
    return build_frame(CRSF_RC_CHANNELS, pack_channels(channels))


def main():
    cfg = load_config(_CONFIG_PATH, {})
    serial_cfg = cfg["platform"]["serial"]

    parser = argparse.ArgumentParser(description="CRSF RC uplink test transmitter")
    parser.add_argument("--port",         default=serial_cfg["port"],
                        help=f"Serial port (default from config: {serial_cfg['port']})")
    parser.add_argument("--baud",         type=int,   default=serial_cfg["baud"],
                        help=f"Baud rate (default from config: {serial_cfg['baud']})")
    parser.add_argument("--rate",         type=float, default=50.0,
                        help="Transmit rate in Hz (default: 50)")
    parser.add_argument("--arm",          action="store_true",
                        help="Run arming sequence then hold armed.")
    parser.add_argument("--pre-arm-secs", type=float, default=2.0,
                        help="Seconds to send disarmed uplink before arming (default: 2)")
    parser.add_argument("--arm-secs",     type=float, default=2.0,
                        help="Seconds to hold arm switch up with throttle-min before applying throttle (default: 2)")
    parser.add_argument("--roll",         type=int, default=992, help="CH1 ticks (default: 992)")
    parser.add_argument("--pitch",        type=int, default=992, help="CH2 ticks (default: 992)")
    parser.add_argument("--throttle",     type=int, default=172, help="CH3 ticks (default: 172 = min)")
    parser.add_argument("--yaw",          type=int, default=992, help="CH4 ticks (default: 992)")
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"ERROR: Cannot open {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    interval = 1.0 / args.rate
    start    = time.monotonic()
    count    = 0
    next_t   = start

    if args.arm:
        arm_end      = start + args.pre_arm_secs
        throttle_end = arm_end + args.arm_secs
        print(f"Transmitting on {args.port} @ {args.baud} baud, {args.rate:.0f} Hz")
        print(f"Arming sequence: {args.pre_arm_secs:.0f}s disarmed → "
              f"{args.arm_secs:.0f}s armed/throttle-min → throttle={args.throttle}")
        print("Press Ctrl+C to stop.\n")
    else:
        arm_end = throttle_end = start  # skip sequence, go straight to commanded state
        ch5 = _ARM_LOW
        print(f"Transmitting on {args.port} @ {args.baud} baud, {args.rate:.0f} Hz")
        print(f"CH1={args.roll} CH2={args.pitch} CH3={args.throttle} "
              f"CH4={args.yaw} CH5={ch5} [DISARMED]")
        print("Press Ctrl+C to stop.\n")

    try:
        while True:
            now = time.monotonic()
            if now < next_t:
                continue

            t = now - start

            if args.arm and now < arm_end:
                # Phase 1: uplink established, arm switch low, throttle min
                ch5      = _ARM_LOW
                throttle = _MIN_THROTTLE
                phase    = "PHASE 1: uplink  "
            elif args.arm and now < throttle_end:
                # Phase 2: arm switch high, throttle still min
                ch5      = _ARM_HIGH
                throttle = _MIN_THROTTLE
                phase    = "PHASE 2: arming  "
            else:
                # Phase 3 (or no-arm mode): commanded values
                ch5      = _ARM_HIGH if args.arm else _ARM_LOW
                throttle = args.throttle
                phase    = "PHASE 3: running " if args.arm else "         running "

            ser.write(_make_frame(args.roll, args.pitch, throttle, args.yaw, ch5))
            count += 1
            print(
                f"\r[t={t:6.2f}s] {phase} "
                f"ch1={args.roll:4d} ch2={args.pitch:4d} "
                f"ch3={throttle:4d} ch4={args.yaw:4d} "
                f"ch5={ch5:4d}  frames={count}",
                end="", flush=True,
            )
            next_t += interval

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

    elapsed = time.monotonic() - start
    print(f"\n\n{count} frames in {elapsed:.1f}s ({count/elapsed:.1f} Hz)" if elapsed > 0 else "")


if __name__ == "__main__":
    main()

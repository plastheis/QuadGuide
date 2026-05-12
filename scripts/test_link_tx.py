#!/usr/bin/env python3
"""CRSF RC uplink transmitter test.

Sends CRSF RC_CHANNELS_PACKED frames at a fixed rate. Use this to verify the
companion → FC uplink is working. Once a steady uplink is established, the FC
exits failsafe and begins sending attitude telemetry back (visible in test_link_rx.py).

Usage:
    python scripts/test_link_tx.py [--port /dev/ttyS0] [--baud 420000] [--rate 50]
        [--arm] [--roll 992] [--pitch 992] [--throttle 172] [--yaw 992]

Defaults for --port and --baud are read from configs/config.yaml.

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


def main():
    cfg = load_config(_CONFIG_PATH, {})
    serial_cfg = cfg["platform"]["serial"]

    parser = argparse.ArgumentParser(description="CRSF RC uplink test transmitter")
    parser.add_argument("--port",     default=serial_cfg["port"],
                        help=f"Serial port (default from config: {serial_cfg['port']})")
    parser.add_argument("--baud",     type=int,   default=serial_cfg["baud"],
                        help=f"Baud rate (default from config: {serial_cfg['baud']})")
    parser.add_argument("--rate",     type=float, default=50.0,
                        help="Transmit rate in Hz (default: 50)")
    parser.add_argument("--arm",      action="store_true",
                        help="Set CH5 high (armed). Default: CH5 low (disarmed).")
    parser.add_argument("--roll",     type=int, default=992, help="CH1 ticks (default: 992)")
    parser.add_argument("--pitch",    type=int, default=992, help="CH2 ticks (default: 992)")
    parser.add_argument("--throttle", type=int, default=172, help="CH3 ticks (default: 172 = min)")
    parser.add_argument("--yaw",      type=int, default=992, help="CH4 ticks (default: 992)")
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"ERROR: Cannot open {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    ch5   = 1811 if args.arm else 172
    arm_s = "ARMED" if args.arm else "DISARMED"
    interval = 1.0 / args.rate

    print(f"Transmitting on {args.port} @ {args.baud} baud, {args.rate:.0f} Hz")
    print(f"CH1={args.roll} CH2={args.pitch} CH3={args.throttle} "
          f"CH4={args.yaw} CH5={ch5} [{arm_s}]")
    print("Press Ctrl+C to stop.\n")

    channels = [
        args.roll, args.pitch, args.throttle, args.yaw, ch5,
        *([992] * 11),
    ]
    frame = build_frame(CRSF_RC_CHANNELS, pack_channels(channels))

    start  = time.monotonic()
    count  = 0
    next_t = start

    try:
        while True:
            now = time.monotonic()
            if now >= next_t:
                ser.write(frame)
                count += 1
                t = now - start
                print(
                    f"\r[t={t:7.3f}s] TX: "
                    f"ch1={args.roll:4d} ch2={args.pitch:4d} "
                    f"ch3={args.throttle:4d} ch4={args.yaw:4d} "
                    f"ch5={ch5:4d}  {arm_s}   frames={count}",
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

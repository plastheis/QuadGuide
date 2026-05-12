#!/usr/bin/env python3
"""Live CRSF attitude telemetry monitor.

Parses CRSF frames from the FC's UART and prints decoded attitude + derived
body rates. Use this to verify CRSF telemetry is reaching the companion computer
before starting the full stack.

Usage:
    python scripts/test_link_rx.py --port /dev/ttyS0 [--baud 420000] [--duration 10] [--verbose]

With --verbose: also prints raw hex bytes and flags CRC errors.
"""
import argparse
import math
import sys
import time

import serial

# Allow running from repo root without installing
sys.path.insert(0, "src")

from quadguide.link.crsf import CRSFParser, CRSF_ATTITUDE
from quadguide.link.differentiator import AttitudeDifferentiator


def main():
    parser = argparse.ArgumentParser(description="CRSF attitude monitor")
    parser.add_argument("--port",     default="/dev/ttyS0")
    parser.add_argument("--baud",     type=int, default=420000)
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds (default: run forever)")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print raw hex bytes and flag CRC errors")
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"ERROR: Cannot open {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Listening on {args.port} @ {args.baud} baud"
          + (" (verbose)" if args.verbose else ""))
    print("Waiting for CRSF attitude frames (FC must be receiving uplink)...\n")

    crsf_parser = CRSFParser()
    diff        = AttitudeDifferentiator(alpha=1.0)
    frame_count = 0
    start       = time.monotonic()
    raw_buf     = bytearray()

    try:
        while True:
            if args.duration and (time.monotonic() - start) >= args.duration:
                break

            chunk = ser.read(64)
            if not chunk:
                continue

            for byte in chunk:
                if args.verbose:
                    raw_buf.append(byte)

                frame = crsf_parser.feed(byte)

                if frame is None:
                    continue

                if args.verbose:
                    hex_str = " ".join(f"{b:02x}" for b in raw_buf)
                    print(f"  raw hex: {hex_str}  CRC OK")
                    raw_buf.clear()

                if frame.type != CRSF_ATTITUDE:
                    if args.verbose:
                        print(f"  [type=0x{frame.type:02x} skipped]")
                    continue

                import struct
                pitch_raw, roll_raw, yaw_raw = struct.unpack(">hhh", frame.payload[:6])
                roll_rad  = roll_raw  * 1e-4
                pitch_rad = pitch_raw * 1e-4
                yaw_rad   = yaw_raw   * 1e-4
                rr, pr, yr = diff.update(roll_rad, pitch_rad, yaw_rad, frame.timestamp_ns)

                t = time.monotonic() - start
                print(
                    f"[t={t:7.3f}s] "
                    f"roll={math.degrees(roll_rad):7.2f}°  "
                    f"pitch={math.degrees(pitch_rad):7.2f}°  "
                    f"yaw={math.degrees(yaw_rad):7.2f}°  "
                    f"rates: p={rr:+.3f} q={pr:+.3f} r={yr:+.3f} rad/s"
                )
                frame_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

    elapsed = time.monotonic() - start
    print(f"\n{frame_count} attitude frames in {elapsed:.1f}s "
          f"({frame_count/elapsed:.1f} Hz)" if elapsed > 0 else "")


if __name__ == "__main__":
    main()

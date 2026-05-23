#!/usr/bin/env python3
"""Live CRSF telemetry monitor (bus-based).

Starts the link worker and reads fc/attitude and fc/imu from the bus, printing
all 9 telemetry values: 3 attitude angles, 3 accelerometer values, 3 body rates.
Use this to verify CRSF telemetry is reaching the companion computer before
starting the full stack. The FC must be receiving an uplink (e.g. from
test_link_tx.py) to send telemetry back.

Usage:
    python scripts/test_link_rx.py [--duration 10]

Defaults for serial port and baud are read from configs/config.yaml.
"""
import argparse
import math
import multiprocessing
import os
import sys
import time

sys.path.insert(0, "src")

from quadguide.core.bus import Bus
from quadguide.core.config import load_config
from quadguide.link import worker as link_worker

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")


def main():
    cfg        = load_config(_CONFIG_PATH, {})
    serial_cfg = cfg["platform"]["serial"]

    parser = argparse.ArgumentParser(description="CRSF telemetry monitor (via bus)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds (default: run forever)")
    args = parser.parse_args()

    ring_depth = cfg.get("bus", {}).get("ring_depth", 8)
    bus        = Bus(ring_depth=ring_depth)

    link_proc = multiprocessing.Process(
        target=link_worker.run, args=(cfg, bus), daemon=True
    )
    link_proc.start()

    print(f"Listening via link worker on {serial_cfg['port']} @ {serial_cfg['baud']} baud")
    print("Waiting for fc/attitude + fc/imu on bus (FC must be receiving uplink)...\n")
    print(
        f"  {'t(s)':>7}  "
        f"{'roll':>8} {'pitch':>8} {'yaw':>8}  "
        f"{'ax(m/s²)':>9} {'ay':>9} {'az':>9}  "
        f"{'gx(r/s)':>8} {'gy':>8} {'gz':>8}"
    )
    print("  " + "─" * 100)

    frame_count  = 0
    start        = time.monotonic()
    last_att_ns  = None

    try:
        while True:
            if args.duration and (time.monotonic() - start) >= args.duration:
                break

            att = bus.latest("fc/attitude")
            if att is None or att.timestamp_ns == last_att_ns:
                time.sleep(0.005)
                continue

            last_att_ns = att.timestamp_ns
            imu = bus.latest("fc/imu")
            t   = time.monotonic() - start

            if imu is not None:
                print(
                    f"  {t:>7.3f}s  "
                    f"{math.degrees(att.roll_rad):>+8.2f}°"
                    f" {math.degrees(att.pitch_rad):>+8.2f}°"
                    f" {math.degrees(att.yaw_rad):>+8.2f}°  "
                    f"{imu.ax:>+9.3f}"
                    f" {imu.ay:>+9.3f}"
                    f" {imu.az:>+9.3f}  "
                    f"{imu.gx:>+8.3f}"
                    f" {imu.gy:>+8.3f}"
                    f" {imu.gz:>+8.3f}"
                )
            else:
                print(
                    f"  {t:>7.3f}s  "
                    f"{math.degrees(att.roll_rad):>+8.2f}°"
                    f" {math.degrees(att.pitch_rad):>+8.2f}°"
                    f" {math.degrees(att.yaw_rad):>+8.2f}°  "
                    f"  [waiting for fc/imu]"
                )
            frame_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        link_proc.terminate()
        link_proc.join(timeout=2)
        bus.close()

    elapsed = time.monotonic() - start
    print(
        f"\n{frame_count} frames in {elapsed:.1f}s "
        f"({frame_count/elapsed:.1f} Hz)" if elapsed > 0 else ""
    )


if __name__ == "__main__":
    main()

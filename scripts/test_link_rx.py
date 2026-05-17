#!/usr/bin/env python3
"""Live CRSF attitude telemetry monitor (bus-based).

Starts the link worker and reads fc/attitude from the bus. Use this to verify
CRSF telemetry is reaching the companion computer before starting the full stack.
The FC must be receiving an uplink (e.g. from test_link_tx.py) to send attitude
telemetry back.

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

    parser = argparse.ArgumentParser(description="CRSF attitude monitor (via bus)")
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
    print("Waiting for fc/attitude on bus (FC must be receiving uplink)...\n")

    frame_count  = 0
    start        = time.monotonic()
    last_seen_ns = None

    try:
        while True:
            if args.duration and (time.monotonic() - start) >= args.duration:
                break

            att = bus.latest("fc/attitude")
            if att is None or att.timestamp_ns == last_seen_ns:
                time.sleep(0.005)
                continue

            last_seen_ns = att.timestamp_ns
            t = time.monotonic() - start
            print(
                f"[t={t:7.3f}s] "
                f"roll={math.degrees(att.roll_rad):7.2f}°  "
                f"pitch={math.degrees(att.pitch_rad):7.2f}°  "
                f"yaw={math.degrees(att.yaw_rad):7.2f}°  "
                f"rates: p={att.roll_rate_rps:+.3f} q={att.pitch_rate_rps:+.3f} "
                f"r={att.yaw_rate_rps:+.3f} rad/s"
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
        f"\n{frame_count} attitude frames in {elapsed:.1f}s "
        f"({frame_count/elapsed:.1f} Hz)" if elapsed > 0 else ""
    )


if __name__ == "__main__":
    main()

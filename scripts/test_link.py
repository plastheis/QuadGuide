#!/usr/bin/env python3
"""Combined MAVLink2 link test — single link worker, RX and TX together.

Starts one link worker process that owns the serial port. The main process
publishes ControlCmd/ArmCmd through the bus and prints incoming fc/attitude
telemetry as it arrives. Use this instead of running test_link_rx.py and
test_link_tx.py simultaneously (which would open the port twice).

Usage:
    python scripts/test_link.py [options]

    python scripts/test_link.py                         # idle, disarmed
    python scripts/test_link.py --arm                   # arming sequence, then idle
    python scripts/test_link.py --arm --throttle-norm 0.15   # arm and hover
    python scripts/test_link.py --roll-deg 10           # roll right, disarmed

Options:
    --rate FLOAT          TX publish rate in Hz (default: 50)
    --arm                 Run arming sequence: uplink → arm → commanded throttle
    --pre-arm-secs FLOAT  Seconds of disarmed uplink before arming (default: 2)
    --arm-secs FLOAT      Seconds armed at zero throttle before applying throttle (default: 2)
    --roll-deg FLOAT      Roll command (degrees, default: 0)
    --pitch-deg FLOAT     Pitch command (degrees, default: 0)
    --throttle-norm FLOAT Throttle 0.0–1.0 (default: 0.0)
    --yaw-rate-dps FLOAT  Yaw rate command (deg/s, default: 0)
    --display-hz FLOAT    Screen refresh rate (default: 5)

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
from quadguide.core.clock import monotonic_ns
from quadguide.core.config import load_config
from quadguide.core.messages import ArmCmd, ControlCmd, ProcessState
from quadguide.link import worker as link_worker

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")

_W = 100  # terminal width for separator lines


def _sep(char="─"):
    print(char * _W)


def _event(t: float, msg: str) -> None:
    print(f"  *** [t={t:7.2f}s] {msg}")


def main():
    cfg        = load_config(_CONFIG_PATH, {})
    serial_cfg = cfg["platform"]["serial"]

    parser = argparse.ArgumentParser(
        description="MAVLink2 link test — single link worker, combined RX+TX"
    )
    parser.add_argument("--rate",          type=float, default=50.0,
                        help="TX publish rate in Hz (default: 50)")
    parser.add_argument("--display-hz",    type=float, default=5.0,
                        help="Screen refresh rate in Hz (default: 5)")
    parser.add_argument("--arm",           action="store_true",
                        help="Run arming sequence then hold armed")
    parser.add_argument("--pre-arm-secs",  type=float, default=2.0,
                        help="Seconds disarmed before arming (default: 2)")
    parser.add_argument("--arm-secs",      type=float, default=2.0,
                        help="Seconds armed at zero throttle before applying throttle (default: 2)")
    parser.add_argument("--roll-deg",      type=float, default=0.0)
    parser.add_argument("--pitch-deg",     type=float, default=0.0)
    parser.add_argument("--throttle-norm", type=float, default=0.0,
                        help="Throttle 0.0–1.0 (default: 0.0)")
    parser.add_argument("--yaw-rate-dps",  type=float, default=0.0)
    args = parser.parse_args()

    ring_depth = cfg.get("bus", {}).get("ring_depth", 8)
    bus        = Bus(ring_depth=ring_depth)

    link_proc = multiprocessing.Process(
        target=link_worker.run, args=(cfg, bus), daemon=True
    )
    link_proc.start()

    tx_interval      = 1.0 / args.rate
    display_interval = 1.0 / args.display_hz
    start            = time.monotonic()

    if args.arm:
        arm_end      = start + args.pre_arm_secs
        throttle_end = arm_end + args.arm_secs
    else:
        arm_end = throttle_end = start   # skip sequence, stay in "running" state

    # ── state ──────────────────────────────────────────────────────────────
    next_tx          = start
    next_display     = start
    tx_count         = 0
    rx_count         = 0
    last_seen_ns     = None
    prev_link_state  = None
    prev_phase       = None
    latest_att       = None
    latest_imu       = None

    # rolling Hz counters (reset every second)
    hz_window_start = start
    rx_in_window    = 0
    tx_in_window    = 0
    rx_hz           = 0.0
    tx_hz           = 0.0

    # ── header ─────────────────────────────────────────────────────────────
    _sep("═")
    print(f"  MAVLink2 link test  |  {serial_cfg['port']} @ {serial_cfg['baud']} baud"
          f"  |  TX {args.rate:.0f} Hz  |  press Ctrl+C to stop")
    if args.arm:
        print(f"  Arming: {args.pre_arm_secs:.0f}s uplink → {args.arm_secs:.0f}s armed/thr=0"
              f" → thr={args.throttle_norm:.2f}")
    _sep("═")
    print(
        f"  {'t(s)':>7}  {'phase':<14}  "
        f"{'── TX cmd ──':^32}  "
        f"{'─ attitude (deg) ─':^28}  "
        f"{'─ accel (m/s²) ─':^28}  "
        f"{'─ rates (rad/s) ─':^28}  "
        f"{'Hz':>8}"
    )
    print(
        f"  {'':>7}  {'':14}  "
        f"{'roll':>7} {'pitch':>7} {'thr':>5} {'yaw/s':>7}  "
        f"{'roll':>8} {'pitch':>8} {'yaw':>8}  "
        f"{'ax':>8} {'ay':>8} {'az':>8}  "
        f"{'gx':>8} {'gy':>8} {'gz':>8}  "
        f"{'rx/tx':>8}"
    )
    _sep()

    try:
        while True:
            now = time.monotonic()
            t   = now - start
            ts  = monotonic_ns()

            # ── arming phase ───────────────────────────────────────────────
            if args.arm and now < arm_end:
                phase        = "uplink"
                armed        = False
                throttle_now = 0.0
            elif args.arm and now < throttle_end:
                phase        = "arming"
                armed        = True
                throttle_now = 0.0
            else:
                phase        = "armed" if args.arm else "disarmed"
                armed        = args.arm
                throttle_now = args.throttle_norm

            # ── print phase-change events immediately ──────────────────────
            if phase != prev_phase:
                if prev_phase is not None:
                    _event(t, f"phase: {prev_phase} → {phase}  armed={armed}")
                prev_phase = phase

            # ── publish TX ─────────────────────────────────────────────────
            if now >= next_tx:
                bus.publish("arm/cmd", ArmCmd(ts, armed))
                bus.publish("control/cmd", ControlCmd(
                    timestamp_ns=ts,
                    roll_deg=args.roll_deg,
                    pitch_deg=args.pitch_deg,
                    yaw_rate_dps=args.yaw_rate_dps,
                    throttle_norm=throttle_now,
                ))
                tx_count      += 1
                tx_in_window  += 1
                next_tx       += tx_interval

            # ── update Hz counters every second ────────────────────────────
            elapsed_window = now - hz_window_start
            if elapsed_window >= 1.0:
                rx_hz          = rx_in_window / elapsed_window
                tx_hz          = tx_in_window / elapsed_window
                rx_in_window   = 0
                tx_in_window   = 0
                hz_window_start = now

            # ── check link health — print state changes immediately ────────
            health = bus.latest("system/health")
            if health and health.process == "link" and health.state != prev_link_state:
                if health.state == ProcessState.OK:
                    _event(t, "LINK  OK")
                else:
                    _event(t, f"LINK  {health.state.value.upper()}")
                prev_link_state = health.state

            # ── read latest RX attitude + IMU ─────────────────────────────
            att = bus.latest("fc/attitude")
            if att is not None and att.timestamp_ns != last_seen_ns:
                latest_att   = att
                last_seen_ns = att.timestamp_ns
                rx_count     += 1
                rx_in_window += 1
            imu = bus.latest("fc/imu")
            if imu is not None:
                latest_imu = imu

            # ── periodic display line ──────────────────────────────────────
            if now >= next_display:
                next_display += display_interval

                tx_str = (
                    f"{args.roll_deg:>+7.1f}°"
                    f" {args.pitch_deg:>+7.1f}°"
                    f" {throttle_now:>5.2f}"
                    f" {args.yaw_rate_dps:>+7.1f}°/s"
                )

                if latest_att is None:
                    rx_str = f"  {'[waiting for telemetry]':<88}"
                else:
                    a = latest_att
                    att_str = (
                        f"  {math.degrees(a.roll_rad):>+8.2f}°"
                        f" {math.degrees(a.pitch_rad):>+8.2f}°"
                        f" {math.degrees(a.yaw_rad):>+8.2f}°"
                    )
                    if latest_imu is not None:
                        m = latest_imu
                        imu_str = (
                            f"  {m.ax:>+8.3f}"
                            f" {m.ay:>+8.3f}"
                            f" {m.az:>+8.3f}"
                            f"  {m.gx:>+8.3f}"
                            f" {m.gy:>+8.3f}"
                            f" {m.gz:>+8.3f}"
                        )
                    else:
                        imu_str = f"  {'[no fc/imu]':<58}"
                    rx_str = att_str + imu_str

                hz_str = f"{rx_hz:4.0f}/{tx_hz:<4.0f}"
                print(
                    f"  {t:>7.2f}s  {phase:<14}  {tx_str}{rx_str}  {hz_str}"
                )

    except KeyboardInterrupt:
        pass
    finally:
        link_proc.terminate()
        link_proc.join(timeout=2)
        bus.close()

    elapsed = time.monotonic() - start
    _sep()
    print(
        f"  stopped after {elapsed:.1f}s  |  "
        f"TX {tx_count} frames ({tx_count/elapsed:.1f} Hz)  |  "
        f"RX {rx_count} frames ({rx_count/elapsed:.1f} Hz)"
    )
    _sep("═")


if __name__ == "__main__":
    main()

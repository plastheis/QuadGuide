#!/usr/bin/env python3
"""Drive the link worker against ArduPilot SITL and verify failsafe MODE commands.

Publishes failsafe/action(SET_MODE, <mode>) through the bus and asserts SITL's
HEARTBEAT custom_mode (surfaced on fc/status) changes to the commanded ArduCopter
mode. This exercises encode_set_mode + _ModeController end-to-end over MAVLink.

Prereqs: ArduCopter SITL reachable over TCP. Example:
    sim_vehicle.py -v ArduCopter --out=tcp:127.0.0.1:5760

Usage:
    QUADGUIDE_SITL=127.0.0.1:5760 python scripts/test_failsafe_sitl.py
"""
import os
import sys
import time
import multiprocessing

sys.path.insert(0, "src")

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.config import load_config, ARDUCOPTER_MODES
from quadguide.core.messages import ArmCmd, FailsafeCmd, FailsafeActionWire
from quadguide.link import worker as link_worker

_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "rpi4b.yaml")


def _sitl_config():
    host, _, port = os.environ.get("QUADGUIDE_SITL", "127.0.0.1:5760").partition(":")
    cfg = load_config(_CONFIG, {})
    cfg["platform"]["serial"]["mode"] = "tcp"
    cfg["platform"]["serial"]["tcp_host"] = host
    cfg["platform"]["serial"]["tcp_port"] = int(port)
    return cfg


def _wait_mode(bus, target_mode: int, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        st = bus.latest("fc/status")
        if st is not None and st.custom_mode == target_mode:
            return True
        time.sleep(0.1)
    return False


def _publish_mode(bus, mode_name: str, secs: float = 3.0):
    custom = ARDUCOPTER_MODES[mode_name]
    end = time.monotonic() + secs
    while time.monotonic() < end:
        bus.publish("arm/cmd", ArmCmd(monotonic_ns(), True))
        bus.publish("failsafe/action",
                    FailsafeCmd(monotonic_ns(), FailsafeActionWire.SET_MODE, custom))
        time.sleep(0.05)
    return custom


def main() -> int:
    cfg = _sitl_config()
    bus = Bus(ring_depth=cfg.get("bus", {}).get("ring_depth", 8))
    link = multiprocessing.Process(target=link_worker.run, args=(cfg, bus), daemon=True)
    link.start()
    ok = True
    try:
        # wait for the first heartbeat (fc/status appears)
        if not _wait_mode(bus, ARDUCOPTER_MODES["LAND"], timeout_s=1.0):
            pass  # not expected yet; just gives SITL a moment
        for mode_name in ("LAND", "STABILIZE", "ALTHOLD"):
            target = _publish_mode(bus, mode_name)
            got = _wait_mode(bus, target)
            print(f"  {mode_name:10s} custom_mode={target:2d} -> {'OK' if got else 'FAIL'}")
            ok = ok and got
    finally:
        link.terminate()
        link.join(timeout=2)
        bus.close()
    print("SITL failsafe MODE test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

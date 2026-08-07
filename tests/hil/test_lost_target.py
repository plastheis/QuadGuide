"""HIL: target-loss / watchdog failsafe drives the FC into the configured mode.

Requires ArduPilot SITL reachable over TCP. Skipped unless QUADGUIDE_SITL is set
(e.g. QUADGUIDE_SITL=127.0.0.1:5760). Runs on Linux/WSL or the SBC — not Windows.
"""
import os
import sys
import time
import multiprocessing

import pytest

sys.path.insert(0, "src")

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.config import load_config, ARDUCOPTER_MODES
from quadguide.core.messages import ArmCmd, FailsafeCmd, FailsafeActionWire
from quadguide.link import worker as link_worker

pytestmark = pytest.mark.skipif(
    "QUADGUIDE_SITL" not in os.environ,
    reason="set QUADGUIDE_SITL=host:port with ArduCopter SITL running",
)

_CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "rpi4b.yaml")


def _sitl_config():
    host, _, port = os.environ["QUADGUIDE_SITL"].partition(":")
    cfg = load_config(_CONFIG, {})
    cfg["platform"]["serial"].update(mode="tcp", tcp_host=host, tcp_port=int(port))
    return cfg


def test_failsafe_set_mode_reaches_sitl():
    cfg = _sitl_config()
    bus = Bus(ring_depth=8)
    link = multiprocessing.Process(target=link_worker.run, args=(cfg, bus), daemon=True)
    link.start()
    try:
        target = ARDUCOPTER_MODES["LAND"]
        deadline = time.monotonic() + 12.0
        reached = False
        while time.monotonic() < deadline and not reached:
            bus.publish("arm/cmd", ArmCmd(monotonic_ns(), True))
            bus.publish("failsafe/action",
                        FailsafeCmd(monotonic_ns(), FailsafeActionWire.SET_MODE, target))
            st = bus.latest("fc/status")
            reached = st is not None and st.custom_mode == target
            time.sleep(0.05)
        assert reached, "SITL did not enter LAND after failsafe SET_MODE"
    finally:
        link.terminate()
        link.join(timeout=2)
        bus.close()

# MAVLink2 / ArduPilot Link Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CRSF UART link to the madflight ESP32 FC with a MAVLink2 link to an ArduPilot H743 FC — reading attitude/IMU telemetry and streaming `SET_ATTITUDE_TARGET` setpoints in GUIDED_NOGPS — with zero changes to bus contracts or non-link workers.

**Architecture:** pymavlink is used as a pure codec (constructed with `file=None`); the existing `SerialPort`/`TCPSerialPort` async transport, reconnect loop, health loop, and latency trace are preserved. The link worker maps `ATTITUDE`→`fc/attitude`, `RAW_IMU`→`fc/imu`, streams `SET_ATTITUDE_TARGET` @ 50 Hz, arms over MAVLink (edge-triggered), and holds heading by baking a latched yaw into the commanded quaternion. HIL retargets to ArduPilot SITL over TCP.

**Tech Stack:** Python 3.14, asyncio, pymavlink (ardupilotmega v2.0 dialect), pytest. Reference spec: `docs/superpowers/specs/2026-06-21-mavlink-link-migration-design.md`.

**Note on test scope:** Per project memory, the bus/runtime are Linux-only (`fcntl`). Every task below is a **pure-codec / pure-logic** unit test that runs on the Windows dev box (no bus, no real serial). SITL integration is manual (see Task 12).

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/quadguide/link/mavlink_codec.py` | Codec-mode MAVLink object, `euler_to_quaternion`, message-id/mask constants |
| Rewrite | `src/quadguide/link/fc.py` | Semantic map: decode `ATTITUDE`/`RAW_IMU`/`HEARTBEAT`; encode `SET_ATTITUDE_TARGET`/arm/`SET_MESSAGE_INTERVAL`/heartbeat |
| Rewrite | `src/quadguide/link/worker.py` | `_LinkState`, `_ArmController`, `latch_yaw`, RX/TX/heartbeat/stream-setup loops, reconnect wiring |
| Delete | `src/quadguide/link/crsf.py` | CRSF framing — pymavlink owns framing now |
| Modify | `requirements.txt` | add `pymavlink` |
| Modify | `configs/config.yaml`, `configs/rk3588.yaml` | serial baud/port/tcp_port, new `link` fields, remove `channels` + `diff_lowpass_alpha` |
| Delete | `CRSF_PROTOCOL.md` | retired |
| Modify | `ARCHITECTURE.md`, `QUADGUIDE_HIL_INTEGRATION.md` | link section + HIL→SITL |
| Create | `tests/unit/test_mavlink_codec.py` | quaternion + codec |
| Rewrite | `tests/unit/test_fc.py` | decode/encode round-trips |
| Rewrite | `tests/unit/test_link_worker.py` | rx mapping, arm controller, yaw latch |
| Delete | `tests/unit/test_crsf.py` | CRSF parser tests retired |

---

## Task 1: Add pymavlink dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, under `# Core runtime`, after the `pyserial>=3.5` line, add:

```
pymavlink>=2.4        # MAVLink2 codec for the ArduPilot link (link/mavlink_codec.py)
```

- [ ] **Step 2: Install into the venv**

Run: `.venv/Scripts/python.exe -m pip install -r requirements.txt`
Expected: `pymavlink` (and its deps `lxml`, `fastcrc`) install successfully.

- [ ] **Step 3: Verify import + codec construction**

Run:
```bash
.venv/Scripts/python.exe -c "from pymavlink.dialects.v20 import ardupilotmega as m; mav=m.MAVLink(None); print('ok', mav.set_attitude_target_encode.__name__)"
```
Expected: `ok set_attitude_target_encode`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add pymavlink dependency for MAVLink link"
```

---

## Task 2: mavlink_codec.py — quaternion + codec factory

**Files:**
- Create: `src/quadguide/link/mavlink_codec.py`
- Test: `tests/unit/test_mavlink_codec.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_mavlink_codec.py`:

```python
import math
import pytest

from quadguide.link.mavlink_codec import (
    ATT_TARGET_IGNORE_RATES, MSG_ID_ATTITUDE, MSG_ID_RAW_IMU,
    euler_to_quaternion, make_mav,
)


def test_make_mav_sets_source_ids_and_robust_parsing():
    mav = make_mav(1, 191)
    assert mav.srcSystem == 1
    assert mav.srcComponent == 191
    assert mav.robust_parsing is True


def test_constants():
    assert ATT_TARGET_IGNORE_RATES == 0x07
    assert MSG_ID_ATTITUDE == 30
    assert MSG_ID_RAW_IMU == 27


def test_quaternion_identity():
    assert euler_to_quaternion(0.0, 0.0, 0.0) == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-9)


def test_quaternion_roll_90():
    q = euler_to_quaternion(math.pi / 2, 0.0, 0.0)
    assert q == pytest.approx((math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0), abs=1e-9)


def test_quaternion_pitch_90():
    q = euler_to_quaternion(0.0, math.pi / 2, 0.0)
    assert q == pytest.approx((math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0), abs=1e-9)


def test_quaternion_yaw_90():
    q = euler_to_quaternion(0.0, 0.0, math.pi / 2)
    assert q == pytest.approx((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mavlink_codec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quadguide.link.mavlink_codec'`

- [ ] **Step 3: Write the implementation**

Create `src/quadguide/link/mavlink_codec.py`:

```python
"""MAVLink2 codec helpers for the ArduPilot link.

pymavlink is used as a *codec only* — the MAVLink object is built with no
connection (`file=None`) and only parses incoming bytes / serializes outgoing
messages. The transport (UART or TCP) stays in serial_port.py / tcp_serial.py,
so the RX/TX loops and reconnect machinery are transport-agnostic.
"""
from __future__ import annotations
import math

from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mavlink2

# Telemetry message ids we request via SET_MESSAGE_INTERVAL.
MSG_ID_ATTITUDE = mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE  # 30
MSG_ID_RAW_IMU = mavutil.mavlink.MAVLINK_MSG_ID_RAW_IMU    # 27

# SET_ATTITUDE_TARGET type_mask: ignore the three body-rate fields (bits 0,1,2).
# ArduPilot GUIDED_NOGPS ignores them regardless, so attitude — yaw included — is
# carried entirely by the quaternion + thrust.
ATT_TARGET_IGNORE_RATES = 0x07

# Companion identity (quadguide is an onboard controller, not an autopilot).
MAV_TYPE_COMPANION = mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER  # 18
MAV_AUTOPILOT_NONE = mavutil.mavlink.MAV_AUTOPILOT_INVALID        # 8


def make_mav(system_id: int, component_id: int) -> mavlink2.MAVLink:
    """Build a codec-mode MAVLink2 object (no connection — parse/serialize only)."""
    mav = mavlink2.MAVLink(None, srcSystem=system_id, srcComponent=component_id)
    # Garbage bytes on a real UART must not raise — swallow framing/CRC errors.
    mav.robust_parsing = True
    return mav


def euler_to_quaternion(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """Aircraft ZYX euler (radians) → quaternion (w, x, y, z) in MAVLink order."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mavlink_codec.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/mavlink_codec.py tests/unit/test_mavlink_codec.py
git commit -m "feat(link): MAVLink codec factory + euler_to_quaternion"
```

---

## Task 3: fc.py — decoders (ATTITUDE / RAW_IMU / HEARTBEAT)

**Files:**
- Create (replaces CRSF content): `src/quadguide/link/fc.py`
- Test: `tests/unit/test_fc.py` (new content; old CRSF tests removed in Task 9)

This task creates `fc.py` with the decoder functions only. Encoders are added in Task 4 (same file).

- [ ] **Step 1: Write the failing test**

Replace the entire contents of `tests/unit/test_fc.py` with:

```python
import math
import pytest
from pymavlink import mavutil

from quadguide.link.mavlink_codec import make_mav
from quadguide.link.fc import decode_attitude, decode_heartbeat, decode_imu
from quadguide.core.messages import AttitudeState, IMUFrame

_G = 9.80665


@pytest.fixture
def mav():
    return make_mav(1, 191)


# ── decode_attitude ──────────────────────────────────────────────────────────

def test_decode_attitude_maps_angles_and_rates(mav):
    msg = mav.attitude_encode(0, 0.05, 0.1, -0.02, 0.5, -0.25, 1.0)
    att = decode_attitude(msg, recv_ns=123)
    assert isinstance(att, AttitudeState)
    assert att.timestamp_ns == 123
    assert att.roll_rad == pytest.approx(0.05)
    assert att.pitch_rad == pytest.approx(0.1)
    assert att.yaw_rad == pytest.approx(-0.02)
    assert att.roll_rate_rps == pytest.approx(0.5)
    assert att.pitch_rate_rps == pytest.approx(-0.25)
    assert att.yaw_rate_rps == pytest.approx(1.0)


# ── decode_imu ───────────────────────────────────────────────────────────────

def test_decode_imu_scales_accel_mg_to_mps2(mav):
    # accel in milli-g: 1000 mG = 1 g = 9.80665 m/s²
    msg = mav.raw_imu_encode(0, 1000, -500, 250, 0, 0, 0, 0, 0, 0)
    imu = decode_imu(msg, recv_ns=7)
    assert isinstance(imu, IMUFrame)
    assert imu.timestamp_ns == 7
    assert imu.ax == pytest.approx(_G, rel=1e-4)
    assert imu.ay == pytest.approx(-0.5 * _G, rel=1e-4)
    assert imu.az == pytest.approx(0.25 * _G, rel=1e-4)


def test_decode_imu_scales_gyro_mrad_to_rad(mav):
    # gyro in milli-rad/s: 1571 mrad/s ≈ π/2 rad/s
    msg = mav.raw_imu_encode(0, 0, 0, 0, 1571, -785, 100, 0, 0, 0)
    imu = decode_imu(msg, recv_ns=0)
    assert imu.gx == pytest.approx(1.571, rel=1e-4)
    assert imu.gy == pytest.approx(-0.785, rel=1e-4)
    assert imu.gz == pytest.approx(0.1, rel=1e-4)


def test_decode_imu_accepts_scaled_imu2(mav):
    msg = mav.scaled_imu2_encode(0, 0, 0, 1000, 0, 0, 0, 0, 0, 0)
    imu = decode_imu(msg, recv_ns=0)
    assert imu.az == pytest.approx(_G, rel=1e-4)


# ── decode_heartbeat ─────────────────────────────────────────────────────────

def test_decode_heartbeat_armed(mav):
    msg = mav.heartbeat_encode(
        mavutil.mavlink.MAV_TYPE_QUADROTOR,
        mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
        mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED, 20, 0,
    )
    armed, mode = decode_heartbeat(msg)
    assert armed is True
    assert mode == 20


def test_decode_heartbeat_disarmed(mav):
    msg = mav.heartbeat_encode(
        mavutil.mavlink.MAV_TYPE_QUADROTOR,
        mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA, 0, 0, 0,
    )
    armed, mode = decode_heartbeat(msg)
    assert armed is False
    assert mode == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_fc.py -v`
Expected: FAIL with `ImportError: cannot import name 'decode_attitude'` (old `fc.py` still has the CRSF API).

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `src/quadguide/link/fc.py` with the decoders (encoders added in Task 4):

```python
"""Semantic map between ArduPilot MAVLink messages and quadguide bus dataclasses.

Decoders take a parsed pymavlink message plus the monotonic receive timestamp
(stamped by the RX loop) and return the bus dataclass. Encoders (Task 4) take a
codec `mav` object and return packed MAVLink2 bytes.
"""
from __future__ import annotations

from pymavlink import mavutil

from quadguide.core.messages import AttitudeState, IMUFrame

_G_MPS2 = 9.80665


def decode_attitude(msg, recv_ns: int) -> AttitudeState:
    """ATTITUDE (#30) → AttitudeState. Angles rad, rates rad/s, body frame — native."""
    return AttitudeState(
        timestamp_ns=recv_ns,
        roll_rad=msg.roll,
        pitch_rad=msg.pitch,
        yaw_rad=msg.yaw,
        roll_rate_rps=msg.rollspeed,
        pitch_rate_rps=msg.pitchspeed,
        yaw_rate_rps=msg.yawspeed,
    )


def decode_imu(msg, recv_ns: int) -> IMUFrame:
    """RAW_IMU (#27) / SCALED_IMU2 (#116) → IMUFrame.

    ArduPilot units: accel in milli-g (1000 = 1 g), gyro in milli-rad/s, body
    NED/FRD. Returned units: m/s² and rad/s.
    """
    return IMUFrame(
        timestamp_ns=recv_ns,
        ax=(msg.xacc / 1000.0) * _G_MPS2,
        ay=(msg.yacc / 1000.0) * _G_MPS2,
        az=(msg.zacc / 1000.0) * _G_MPS2,
        gx=msg.xgyro / 1000.0,
        gy=msg.ygyro / 1000.0,
        gz=msg.zgyro / 1000.0,
    )


def decode_heartbeat(msg) -> tuple[bool, int]:
    """HEARTBEAT → (armed, custom_mode)."""
    armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    return armed, msg.custom_mode
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_fc.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/fc.py tests/unit/test_fc.py
git commit -m "feat(link): decode ATTITUDE/RAW_IMU/HEARTBEAT to bus dataclasses"
```

---

## Task 4: fc.py — encoders (SET_ATTITUDE_TARGET / arm / stream / heartbeat)

**Files:**
- Modify: `src/quadguide/link/fc.py` (append encoder functions)
- Test: `tests/unit/test_fc.py` (append encoder tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_fc.py`:

```python
import math as _math
from quadguide.link.mavlink_codec import (
    ATT_TARGET_IGNORE_RATES, MSG_ID_ATTITUDE, euler_to_quaternion,
)
from quadguide.link.fc import (
    encode_arm, encode_attitude_target, encode_heartbeat,
    encode_set_message_interval,
)
from quadguide.core.messages import ControlCmd


def _roundtrip(data: bytes):
    """Parse packed MAVLink2 bytes back into a message via a fresh codec."""
    rx = make_mav(1, 1)
    out = None
    for b in data:
        m = rx.parse_char(bytes([b]))
        if m is not None:
            out = m
    return out


# ── encode_attitude_target ───────────────────────────────────────────────────

def test_encode_attitude_target_mask_thrust_and_quaternion(mav):
    cmd = ControlCmd(0, roll_deg=0.0, pitch_deg=0.0, yaw_rate_dps=0.0, throttle_norm=0.4)
    msg = _roundtrip(encode_attitude_target(
        mav, cmd, yaw_hold=0.0, target_sys=1, target_comp=1,
        max_roll_deg=35.0, max_pitch_deg=35.0, now_ms=0))
    assert msg.get_type() == "SET_ATTITUDE_TARGET"
    assert msg.type_mask == ATT_TARGET_IGNORE_RATES
    assert msg.thrust == pytest.approx(0.4)
    assert list(msg.q) == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)


def test_encode_attitude_target_clamps_roll_to_limit(mav):
    cmd = ControlCmd(0, roll_deg=90.0, pitch_deg=0.0, yaw_rate_dps=0.0, throttle_norm=0.0)
    msg = _roundtrip(encode_attitude_target(
        mav, cmd, 0.0, 1, 1, max_roll_deg=35.0, max_pitch_deg=35.0, now_ms=0))
    expected = euler_to_quaternion(_math.radians(35.0), 0.0, 0.0)
    assert list(msg.q) == pytest.approx(list(expected), abs=1e-5)


def test_encode_attitude_target_clamps_thrust(mav):
    cmd = ControlCmd(0, 0.0, 0.0, 0.0, throttle_norm=2.0)
    msg = _roundtrip(encode_attitude_target(mav, cmd, 0.0, 1, 1, 35.0, 35.0, 0))
    assert msg.thrust == pytest.approx(1.0)


def test_encode_attitude_target_none_is_level_zero_thrust(mav):
    msg = _roundtrip(encode_attitude_target(mav, None, 0.0, 1, 1, 35.0, 35.0, 0))
    assert msg.thrust == pytest.approx(0.0)
    assert list(msg.q) == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)


def test_encode_attitude_target_bakes_yaw_hold(mav):
    cmd = ControlCmd(0, 0.0, 0.0, 0.0, 0.0)
    msg = _roundtrip(encode_attitude_target(
        mav, cmd, yaw_hold=_math.pi / 2, target_sys=1, target_comp=1,
        max_roll_deg=35.0, max_pitch_deg=35.0, now_ms=0))
    assert list(msg.q) == pytest.approx(
        [_math.sqrt(0.5), 0.0, 0.0, _math.sqrt(0.5)], abs=1e-6)


# ── encode_arm ───────────────────────────────────────────────────────────────

def test_encode_arm_arms(mav):
    msg = _roundtrip(encode_arm(mav, True, 1, 1))
    assert msg.get_type() == "COMMAND_LONG"
    assert msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    assert msg.param1 == pytest.approx(1.0)


def test_encode_arm_disarms(mav):
    msg = _roundtrip(encode_arm(mav, False, 1, 1))
    assert msg.param1 == pytest.approx(0.0)


# ── encode_set_message_interval ──────────────────────────────────────────────

def test_encode_set_message_interval_converts_hz_to_us(mav):
    msg = _roundtrip(encode_set_message_interval(mav, MSG_ID_ATTITUDE, 50.0, 1, 1))
    assert msg.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL
    assert msg.param1 == pytest.approx(MSG_ID_ATTITUDE)
    assert msg.param2 == pytest.approx(20000.0)  # 1e6 / 50


# ── encode_heartbeat ─────────────────────────────────────────────────────────

def test_encode_heartbeat_is_onboard_controller(mav):
    msg = _roundtrip(encode_heartbeat(mav))
    assert msg.get_type() == "HEARTBEAT"
    assert msg.type == mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_fc.py -v`
Expected: FAIL with `ImportError: cannot import name 'encode_attitude_target'`

- [ ] **Step 3: Write the implementation**

Append to `src/quadguide/link/fc.py`:

```python
import math

from quadguide.core.messages import ControlCmd
from quadguide.link.mavlink_codec import (
    ATT_TARGET_IGNORE_RATES, MAV_AUTOPILOT_NONE, MAV_TYPE_COMPANION,
    euler_to_quaternion,
)


def encode_attitude_target(
    mav, cmd: ControlCmd | None, yaw_hold: float,
    target_sys: int, target_comp: int,
    max_roll_deg: float, max_pitch_deg: float, now_ms: int,
) -> bytes:
    """Build a SET_ATTITUDE_TARGET (#82) frame.

    roll/pitch come from `cmd` (clamped to limits) and yaw from the latched
    `yaw_hold`, all baked into the quaternion (type_mask ignores body rates).
    thrust = clamped throttle_norm. `cmd is None` → level attitude, zero thrust.
    """
    if cmd is None:
        roll_rad = pitch_rad = 0.0
        thrust = 0.0
    else:
        roll_rad = math.radians(_clamp(cmd.roll_deg, -max_roll_deg, max_roll_deg))
        pitch_rad = math.radians(_clamp(cmd.pitch_deg, -max_pitch_deg, max_pitch_deg))
        thrust = _clamp(cmd.throttle_norm, 0.0, 1.0)
    q = euler_to_quaternion(roll_rad, pitch_rad, yaw_hold)
    msg = mav.set_attitude_target_encode(
        now_ms, target_sys, target_comp, ATT_TARGET_IGNORE_RATES,
        list(q), 0.0, 0.0, 0.0, thrust,
    )
    return msg.pack(mav)


def encode_arm(mav, arm: bool, target_sys: int, target_comp: int) -> bytes:
    """COMMAND_LONG / MAV_CMD_COMPONENT_ARM_DISARM. param1: 1=arm, 0=disarm."""
    msg = mav.command_long_encode(
        target_sys, target_comp,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1.0 if arm else 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    return msg.pack(mav)


def encode_set_message_interval(
    mav, msg_id: int, rate_hz: float, target_sys: int, target_comp: int
) -> bytes:
    """COMMAND_LONG / MAV_CMD_SET_MESSAGE_INTERVAL. param2 is the interval in µs."""
    interval_us = 0.0 if rate_hz <= 0 else 1_000_000.0 / rate_hz
    msg = mav.command_long_encode(
        target_sys, target_comp,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        float(msg_id), interval_us, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    return msg.pack(mav)


def encode_heartbeat(mav) -> bytes:
    """Companion HEARTBEAT so the FC sees quadguide as a live onboard controller."""
    msg = mav.heartbeat_encode(
        MAV_TYPE_COMPANION, MAV_AUTOPILOT_NONE, 0, 0,
        mavutil.mavlink.MAV_STATE_ACTIVE,
    )
    return msg.pack(mav)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_fc.py -v`
Expected: PASS (17 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/fc.py tests/unit/test_fc.py
git commit -m "feat(link): encode SET_ATTITUDE_TARGET, arm, stream-interval, heartbeat"
```

---

## Task 5: worker.py — _ArmController + latch_yaw (pure logic)

**Files:**
- Create (replaces CRSF worker): `src/quadguide/link/worker.py`
- Test: `tests/unit/test_link_worker.py` (new content; old CRSF tests replaced)

This task creates `worker.py` with only the pure helpers (`_ArmController`, `latch_yaw`) and an empty `_LinkState`. The async loops + `run()` are added in Tasks 6–7.

- [ ] **Step 1: Write the failing test**

Replace the entire contents of `tests/unit/test_link_worker.py` with:

```python
import pytest
from pymavlink import mavutil

from quadguide.link.worker import _ArmController, _LinkState, latch_yaw


# ── _ArmController ───────────────────────────────────────────────────────────

def test_arm_controller_silent_when_steady_disarmed():
    arm = _ArmController(retry_count=3, resend_every_ticks=2)
    assert arm.on_arm_state(False) is None
    assert arm.on_arm_state(False) is None


def test_arm_controller_emits_arm_on_rising_edge():
    arm = _ArmController(retry_count=3, resend_every_ticks=2)
    assert arm.on_arm_state(True) is True


def test_arm_controller_emits_disarm_on_falling_edge():
    arm = _ArmController(retry_count=3, resend_every_ticks=2)
    arm.on_arm_state(True)
    arm.on_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
               mavutil.mavlink.MAV_RESULT_ACCEPTED)
    assert arm.on_arm_state(False) is False


def test_arm_controller_resends_until_retries_exhausted():
    arm = _ArmController(retry_count=2, resend_every_ticks=2)
    assert arm.on_arm_state(True) is True   # edge
    assert arm.on_arm_state(True) is None   # tick 1
    assert arm.on_arm_state(True) is True   # tick 2 → resend (retries 2→1)
    assert arm.on_arm_state(True) is None   # tick 1
    assert arm.on_arm_state(True) is True   # tick 2 → resend (retries 1→0)
    assert arm.on_arm_state(True) is None   # exhausted
    assert arm.on_arm_state(True) is None


def test_arm_controller_stops_after_ack():
    arm = _ArmController(retry_count=5, resend_every_ticks=2)
    assert arm.on_arm_state(True) is True
    arm.on_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
               mavutil.mavlink.MAV_RESULT_ACCEPTED)
    assert arm.on_arm_state(True) is None
    assert arm.on_arm_state(True) is None


def test_arm_controller_ignores_unrelated_ack():
    arm = _ArmController(retry_count=5, resend_every_ticks=1)
    arm.on_arm_state(True)
    arm.on_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
               mavutil.mavlink.MAV_RESULT_ACCEPTED)
    assert arm.on_arm_state(True) is True   # still pending → resends


# ── latch_yaw ────────────────────────────────────────────────────────────────

def test_latch_yaw_latches_on_arm_edge():
    assert latch_yaw(armed=True, prev_armed=False, last_yaw=0.7, held=0.0) == pytest.approx(0.7)


def test_latch_yaw_holds_between_ticks():
    assert latch_yaw(True, True, 1.2, 0.7) == pytest.approx(0.7)


def test_latch_yaw_zero_when_no_attitude_yet():
    assert latch_yaw(True, False, None, 0.0) == 0.0


def test_latch_yaw_keeps_held_while_disarmed():
    assert latch_yaw(False, True, 0.9, 0.7) == pytest.approx(0.7)


# ── _LinkState ───────────────────────────────────────────────────────────────

def test_link_state_defaults():
    s = _LinkState()
    assert s.have_heartbeat is False
    assert s.have_raw_imu is False
    assert s.last_yaw is None
    assert s.target_system == 0
    assert s.target_component == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_link_worker.py -v`
Expected: FAIL with `ImportError: cannot import name '_ArmController'` (old worker still has CRSF API).

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `src/quadguide/link/worker.py` with the pure helpers (loops added next task):

```python
from __future__ import annotations

from pymavlink import mavutil


class _LinkState:
    """Per-connection mutable state shared between the RX/TX loops."""

    def __init__(self) -> None:
        self.target_system: int = 0
        self.target_component: int = 0
        self.have_heartbeat: bool = False
        self.have_raw_imu: bool = False
        self.last_yaw: float | None = None
        self.fc_armed: bool = False
        self.fc_mode: int = -1


class _ArmController:
    """Edge-triggered MAVLink arm/disarm with bounded retransmits until ACK.

    Call `on_arm_state(desired)` once per TX tick with the latest arm/cmd state.
    It returns the arm value (True/False) to transmit this tick, or None to send
    nothing. On a new edge it emits immediately, then re-emits every
    `resend_every_ticks` ticks up to `retry_count` times until `on_ack` confirms.
    """

    def __init__(self, retry_count: int, resend_every_ticks: int) -> None:
        self._desired: bool = False          # assume disarmed at startup; no spurious cmd
        self._acked: bool = True
        self._retries_left: int = 0
        self._ticks: int = 0
        self._retry_count = retry_count
        self._resend_every = resend_every_ticks

    def on_arm_state(self, desired: bool) -> bool | None:
        if desired != self._desired:
            self._desired = desired
            self._acked = False
            self._retries_left = self._retry_count
            self._ticks = 0
            return desired
        if self._acked or self._retries_left <= 0:
            return None
        self._ticks += 1
        if self._ticks >= self._resend_every:
            self._ticks = 0
            self._retries_left -= 1
            return self._desired
        return None

    def on_ack(self, command: int, result: int) -> None:
        if (command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
                and result == mavutil.mavlink.MAV_RESULT_ACCEPTED):
            self._acked = True


def latch_yaw(
    armed: bool, prev_armed: bool, last_yaw: float | None, held: float
) -> float:
    """Hold-heading: latch the current yaw on the disarmed→armed edge; else keep."""
    if armed and not prev_armed:
        return last_yaw if last_yaw is not None else 0.0
    return held
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_link_worker.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/worker.py tests/unit/test_link_worker.py
git commit -m "feat(link): edge-triggered arm controller + hold-heading yaw latch"
```

---

## Task 6: worker.py — _rx_loop (message → bus mapping)

**Files:**
- Modify: `src/quadguide/link/worker.py` (add `_rx_loop`, `_on_heartbeat`)
- Test: `tests/unit/test_link_worker.py` (append RX tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_link_worker.py`:

```python
import asyncio
import logging

from quadguide.link.mavlink_codec import make_mav
from quadguide.link.worker import _rx_loop
from quadguide.core.messages import AttitudeState, IMUFrame


class _FakeSerial:
    """Async byte-stream stub that yields a fixed sequence once."""
    def __init__(self, data: bytes):
        self._data = data

    async def read_stream(self):
        for b in self._data:
            yield b


class _FakeBus:
    def __init__(self):
        self.published: list[tuple[str, object]] = []

    def publish(self, topic, msg):
        self.published.append((topic, msg))

    def latest(self, topic):
        return None


def _enc(fn) -> bytes:
    """Pack a message on a fresh FC-side codec (sys=1, comp=1)."""
    m = make_mav(1, 1)
    return fn(m)


def _run_rx(data: bytes):
    serial = _FakeSerial(data)
    bus = _FakeBus()
    state = _LinkState()
    arm = _ArmController(retry_count=5, resend_every_ticks=25)
    log = logging.getLogger("test")
    mav = make_mav(1, 191)
    asyncio.run(_rx_loop(serial, mav, state, bus, arm, log))
    return bus, state, arm


def test_rx_publishes_attitude_and_tracks_yaw():
    data = _enc(lambda m: m.attitude_encode(0, 0.05, 0.1, -0.02, 0.0, 0.0, 0.0).pack(m))
    bus, state, _ = _run_rx(data)
    atts = [msg for t, msg in bus.published if t == "fc/attitude"]
    assert len(atts) == 1 and isinstance(atts[0], AttitudeState)
    assert atts[0].roll_rad == pytest.approx(0.05)
    assert state.last_yaw == pytest.approx(-0.02)


def test_rx_publishes_imu_from_raw_imu():
    data = _enc(lambda m: m.raw_imu_encode(0, 0, 0, 1000, 0, 0, 0, 0, 0, 0).pack(m))
    bus, state, _ = _run_rx(data)
    imus = [msg for t, msg in bus.published if t == "fc/imu"]
    assert len(imus) == 1 and isinstance(imus[0], IMUFrame)
    assert state.have_raw_imu is True


def test_rx_scaled_imu2_is_fallback_only_until_raw_imu():
    m1 = _enc(lambda m: m.scaled_imu2_encode(0, 0, 0, 1000, 0, 0, 0, 0, 0, 0).pack(m))
    m2 = _enc(lambda m: m.raw_imu_encode(0, 0, 0, 1000, 0, 0, 0, 0, 0, 0).pack(m))
    m3 = _enc(lambda m: m.scaled_imu2_encode(0, 0, 0, 1000, 0, 0, 0, 0, 0, 0).pack(m))
    bus, state, _ = _run_rx(m1 + m2 + m3)
    imus = [msg for t, msg in bus.published if t == "fc/imu"]
    assert len(imus) == 2  # scaled (fallback) + raw; the post-raw scaled is ignored


def test_rx_heartbeat_learns_fc_target_ids():
    data = _enc(lambda m: m.heartbeat_encode(
        mavutil.mavlink.MAV_TYPE_QUADROTOR,
        mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA, 0, 0, 0).pack(m))
    _, state, _ = _run_rx(data)
    assert state.have_heartbeat is True
    assert state.target_system == 1
    assert state.target_component == 1


def test_rx_ignores_gcs_heartbeat():
    data = _enc(lambda m: m.heartbeat_encode(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0).pack(m))
    _, state, _ = _run_rx(data)
    assert state.have_heartbeat is False


def test_rx_command_ack_acks_pending_arm():
    arm = _ArmController(retry_count=5, resend_every_ticks=25)
    arm.on_arm_state(True)  # rising edge → pending, not acked
    data = _enc(lambda m: m.command_ack_encode(
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        mavutil.mavlink.MAV_RESULT_ACCEPTED).pack(m))
    serial = _FakeSerial(data)
    bus = _FakeBus()
    state = _LinkState()
    log = logging.getLogger("test")
    mav = make_mav(1, 191)
    asyncio.run(_rx_loop(serial, mav, state, bus, arm, log))
    assert arm.on_arm_state(True) is None  # acked → nothing more to send
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_link_worker.py -v`
Expected: FAIL with `ImportError: cannot import name '_rx_loop'`

- [ ] **Step 3: Write the implementation**

Add to the top of `src/quadguide/link/worker.py` (imports) and append the RX functions. The new import block at the top becomes:

```python
from __future__ import annotations
import asyncio
import logging

from pymavlink import mavutil

from quadguide.core.clock import monotonic_ns
from quadguide.link.fc import decode_attitude, decode_heartbeat, decode_imu
```

Append after `latch_yaw`:

```python
def _on_heartbeat(msg, state: _LinkState, log: logging.Logger) -> None:
    """Learn FC ids on the first heartbeat; log arm/mode transitions."""
    if not state.have_heartbeat:
        state.target_system = msg.get_srcSystem()
        state.target_component = msg.get_srcComponent()
        state.have_heartbeat = True
        log.info("FC HEARTBEAT: sys=%d comp=%d", state.target_system, state.target_component)
    armed, mode = decode_heartbeat(msg)
    if armed != state.fc_armed:
        log.info("FC arm state → %s", "ARMED" if armed else "DISARMED")
        state.fc_armed = armed
    if mode != state.fc_mode:
        log.info("FC custom_mode → %d", mode)
        state.fc_mode = mode


async def _rx_loop(serial, mav, state: _LinkState, bus,
                   arm_ctrl: _ArmController, log: logging.Logger) -> None:
    async for byte in serial.read_stream():
        msg = mav.parse_char(bytes([byte]))
        if msg is None:
            continue
        t = msg.get_type()
        if t == "ATTITUDE":
            state.last_yaw = msg.yaw
            bus.publish("fc/attitude", decode_attitude(msg, monotonic_ns()))
        elif t == "RAW_IMU":
            state.have_raw_imu = True
            bus.publish("fc/imu", decode_imu(msg, monotonic_ns()))
        elif t == "SCALED_IMU2":
            if not state.have_raw_imu:          # fallback until RAW_IMU arrives
                bus.publish("fc/imu", decode_imu(msg, monotonic_ns()))
        elif t == "HEARTBEAT":
            if msg.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID:  # ignore GCS
                _on_heartbeat(msg, state, log)
        elif t == "COMMAND_ACK":
            arm_ctrl.on_ack(msg.command, msg.result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_link_worker.py -v`
Expected: PASS (17 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/worker.py tests/unit/test_link_worker.py
git commit -m "feat(link): MAVLink RX loop maps ATTITUDE/IMU/HEARTBEAT/ACK"
```

---

## Task 7: worker.py — TX/heartbeat/stream loops + reconnect wiring

**Files:**
- Modify: `src/quadguide/link/worker.py` (add loops, `_link_cfg`, `_serial_factory`, `_run_async`, `run`)

This task wires the async loops. The loop *bodies* delegate to functions already unit-tested in Tasks 4–6; this is orchestration glue, so it has no new unit test. Verification is the import/smoke check in Step 3 plus the full suite in Task 9.

- [ ] **Step 1: Add remaining imports**

Extend the import block at the top of `src/quadguide/link/worker.py` to add:

```python
import signal

from quadguide.core.config import cfg_airframe, cfg_diag
from quadguide.core.diagtrace import DiagTrace
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, ProcessState
from quadguide.link.mavlink_codec import (
    MSG_ID_ATTITUDE, MSG_ID_RAW_IMU, make_mav,
)
from quadguide.link.fc import (
    encode_arm, encode_attitude_target, encode_heartbeat,
    encode_set_message_interval,
)
from quadguide.link.serial_port import SerialPort
from quadguide.link.tcp_serial import TCPSerialPort
```

- [ ] **Step 2: Append the loops and entry points**

Append to `src/quadguide/link/worker.py`:

```python
_NS_PER_MS = 1_000_000


def _link_cfg(config: dict) -> dict:
    """Flatten the link + airframe-limit fields the loops need into one dict."""
    link = config["link"]
    acfg = cfg_airframe(config)
    return {
        "tx_rate_hz": link["tx_rate_hz"],
        "stream_rate_hz": link.get("stream_rate_hz", 50),
        "system_id": link.get("system_id", 1),
        "component_id": link.get("component_id", 191),
        "target_system": link.get("target_system", 1),
        "target_component": link.get("target_component", 1),
        "arm_retry_count": link.get("arm_retry_count", 5),
        "heartbeat_wait_s": link.get("heartbeat_wait_s", 5.0),
        "max_roll_deg": acfg.control_limits.max_roll_deg,
        "max_pitch_deg": acfg.control_limits.max_pitch_deg,
    }


async def _tx_loop(serial, mav, bus, state: _LinkState, arm_ctrl: _ArmController,
                   lc: dict, log: logging.Logger, trace) -> None:
    interval = 1.0 / lc["tx_rate_hz"]
    prev_armed = False
    yaw_hold = 0.0
    while True:
        cmd = bus.latest("control/cmd")
        arm_cmd = bus.latest("arm/cmd")
        armed = bool(arm_cmd and arm_cmd.armed)

        to_send = arm_ctrl.on_arm_state(armed)
        if to_send is not None and state.have_heartbeat:
            await serial.write(encode_arm(mav, to_send,
                                          state.target_system, state.target_component))
            log.info("arm command → %s", "ARM" if to_send else "DISARM")

        yaw_hold = latch_yaw(armed, prev_armed, state.last_yaw, yaw_hold)
        prev_armed = armed

        now = monotonic_ns()
        tsys = state.target_system or lc["target_system"]
        tcomp = state.target_component or lc["target_component"]
        await serial.write(encode_attitude_target(
            mav, cmd, yaw_hold, tsys, tcomp,
            lc["max_roll_deg"], lc["max_pitch_deg"], now // _NS_PER_MS))
        # Actuation point: glass→TX latency for the command just sent to the FC.
        if cmd is not None and cmd.origin_ns > 0:
            trace.latency(now, cmd.timestamp_ns, cmd.origin_ns)
        await asyncio.sleep(interval)


async def _heartbeat_loop(serial, mav) -> None:
    while True:
        await serial.write(encode_heartbeat(mav))
        await asyncio.sleep(1.0)


async def _stream_setup_loop(serial, mav, state: _LinkState, lc: dict,
                             log: logging.Logger) -> None:
    """Wait (bounded) for the first FC heartbeat, then request the telemetry streams.

    Runs once per connection and returns. The RX loop is the sole reader, so this
    polls `state.have_heartbeat` rather than reading bytes itself.
    """
    waited = 0.0
    while not state.have_heartbeat and waited < lc["heartbeat_wait_s"]:
        await asyncio.sleep(0.1)
        waited += 0.1
    tsys = state.target_system or lc["target_system"]
    tcomp = state.target_component or lc["target_component"]
    for mid in (MSG_ID_ATTITUDE, MSG_ID_RAW_IMU):
        await serial.write(encode_set_message_interval(
            mav, mid, lc["stream_rate_hz"], tsys, tcomp))
    log.info("requested ATTITUDE+RAW_IMU @ %d Hz from sys=%d comp=%d",
             lc["stream_rate_hz"], tsys, tcomp)


async def _health_loop(bus, state: _LinkState, trace) -> None:
    while True:
        mode = str(state.fc_mode) if state.have_heartbeat else ""
        bus.publish("system/health",
                    HealthReport(monotonic_ns(), "link", ProcessState.OK, mode))
        trace.health(monotonic_ns(), "ok", mode)
        await asyncio.sleep(0.2)


def _serial_factory(config: dict, log: logging.Logger):
    """Pick the link transport from platform.serial.mode.

    "uart" → MAVLink2 over the real UART; "tcp" → MAVLink2 over a TCP socket to
    ArduPilot SITL. Both satisfy the same async port interface, so the loops are
    transport-agnostic.
    """
    scfg = config["platform"]["serial"]
    mode = scfg.get("mode", "uart")
    if mode == "tcp":
        host = scfg["tcp_host"]
        port = scfg["tcp_port"]
        log.info(f"HIL: MAVLink2 over TCP → {host}:{port} (SITL)")
        return (lambda: TCPSerialPort(host, port), f"tcp {host}:{port}")
    if mode != "uart":
        raise ValueError(f"Unknown serial mode {mode!r}. Valid values: 'uart', 'tcp'")
    dev = scfg["port"]
    baud = scfg["baud"]
    return (lambda: SerialPort(dev, baud), f"uart {dev} @ {baud}")


async def _run_async(config: dict, bus) -> None:
    log = setup_logging("link", config)
    lc = _link_cfg(config)
    make_serial, transport = _serial_factory(config, log)

    dcfg = cfg_diag(config)
    trace = DiagTrace("link", enabled=dcfg.trace,
                      dir=dcfg.trace_dir, max_rows=dcfg.trace_max_rows)

    loop = asyncio.get_running_loop()

    def _on_sigterm(*_):
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGTERM, _on_sigterm)

    while True:
        serial = make_serial()
        mav = make_mav(lc["system_id"], lc["component_id"])
        state = _LinkState()
        arm_ctrl = _ArmController(
            lc["arm_retry_count"], max(1, int(lc["tx_rate_hz"] // 2)))
        tasks: list[asyncio.Task] = []
        try:
            await serial.open()
            log.info(f"Link opened ({transport})")
            tasks = [
                asyncio.create_task(_rx_loop(serial, mav, state, bus, arm_ctrl, log)),
                asyncio.create_task(_tx_loop(serial, mav, bus, state, arm_ctrl, lc, log, trace)),
                asyncio.create_task(_heartbeat_loop(serial, mav)),
                asyncio.create_task(_stream_setup_loop(serial, mav, state, lc, log)),
                asyncio.create_task(_health_loop(bus, state, trace)),
            ]
            await asyncio.gather(*tasks)

        except ConnectionError as exc:
            log.error(f"Serial error: {exc}")
            bus.publish("system/health",
                        HealthReport(monotonic_ns(), "link", ProcessState.DEGRADED, ""))

        except asyncio.CancelledError:
            break

        finally:
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            serial.close()

        log.info("Reconnecting in 500 ms...")
        await asyncio.sleep(0.5)

    trace.flush()
    bus.detach()
    log.info("Link worker stopped.")


def run(config: dict, bus) -> None:
    asyncio.run(_run_async(config, bus))
```

Note: `_stream_setup_loop` returns after sending the requests; `asyncio.gather` keeps running because the other four loops never return. On a real FC the requests reach the autopilot once. This is intentional.

- [ ] **Step 3: Smoke-check import + worker module surface**

Run:
```bash
.venv/Scripts/python.exe -c "from quadguide.link import worker; print('ok', worker.run.__name__, worker._link_cfg.__name__)"
```
Expected: `ok run _link_cfg` (no import errors)

- [ ] **Step 4: Re-run the link worker unit tests (still green)**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_link_worker.py -v`
Expected: PASS (17 tests — pure helpers + RX loop unaffected by the new glue)

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/worker.py
git commit -m "feat(link): TX/heartbeat/stream-setup/health loops + reconnect wiring"
```

---

## Task 8: Update configs

**Files:**
- Modify: `configs/config.yaml`
- Modify: `configs/rk3588.yaml`

- [ ] **Step 1: Rewrite the serial block in `configs/config.yaml`**

Replace lines 12–19 (`serial:` block) with:

```yaml
  serial:
    # MAVLink2 link to the ArduPilot FC. HIL: set mode=tcp (+ camera backend) to
    # drive the stack from ArduPilot SITL instead of real hardware.
    mode: uart            # "uart" (real FC) | "tcp" (HIL — MAVLink2 over TCP to SITL)
    port: /dev/ttyAMA0    # used when mode=uart — UART device path, change per SBC
    baud: 115200          # H743 MAVLink2 baud — TODO confirm SERIALn_BAUD (115200 vs 921600)
    tcp_host: "127.0.0.1" # used when mode=tcp — ArduPilot SITL host
    tcp_port: 5760        # used when mode=tcp — SITL MAVLink TCP port
    rx_pin: "GPIO15"      # wiring reference only — not used by driver
    tx_pin: "GPIO14"      # wiring reference only — not used by driver
```

- [ ] **Step 2: Rewrite the link block in `configs/config.yaml`**

Replace the entire `link:` block (lines 59–74, `tx_rate_hz` through `yaw_rate_scale`) with:

```yaml
link:
  tx_rate_hz: 50          # SET_ATTITUDE_TARGET stream rate (Hz) — keep constant to hold GUIDED setpoints
  stream_rate_hz: 50      # requested ATTITUDE + RAW_IMU telemetry rate (Hz)
  system_id: 1            # quadguide MAVLink source system id
  component_id: 191       # MAV_COMP_ID_ONBOARD_COMPUTER
  target_system: 1        # FC system id (corrected at runtime from the first HEARTBEAT)
  target_component: 1     # MAV_COMP_ID_AUTOPILOT1
  arm_retry_count: 5      # arm/disarm COMMAND_LONG retransmits before giving up
  heartbeat_wait_s: 5.0   # max wait for the first FC HEARTBEAT before requesting streams
```

- [ ] **Step 3: Rewrite the serial block in `configs/rk3588.yaml`**

Replace lines 17–27 (`serial:` block) with:

```yaml
  serial:
    # MAVLink2 link to the ArduPilot FC. This board exposes only /dev/ttyFIQ0 out
    # of the box; enable a UART overlay (e.g. uart3/uart5) so a /dev/ttyS* node
    # appears, then set it here. mode=tcp drives the stack from ArduPilot SITL.
    mode: tcp            # "uart" (real FC) | "tcp" (HIL — MAVLink2 over TCP to SITL)
    port: /dev/ttyS6      # used when mode=uart — set to your enabled UART node
    baud: 115200          # H743 MAVLink2 baud — TODO confirm SERIALn_BAUD (115200 vs 921600)
    tcp_host: "192.168.86.46"   # used when mode=tcp — host running ArduPilot SITL
    tcp_port: 5760              # used when mode=tcp — SITL MAVLink TCP port
    rx_pin: "UART_RX"     # wiring reference only — not used by driver
    tx_pin: "UART_TX"     # wiring reference only — not used by driver
```

- [ ] **Step 4: Rewrite the link block in `configs/rk3588.yaml`**

Replace the entire `link:` block (lines 93–105) with:

```yaml
link:
  tx_rate_hz: 50          # SET_ATTITUDE_TARGET stream rate (Hz) — keep constant to hold GUIDED setpoints
  stream_rate_hz: 50      # requested ATTITUDE + RAW_IMU telemetry rate (Hz)
  system_id: 1            # quadguide MAVLink source system id
  component_id: 191       # MAV_COMP_ID_ONBOARD_COMPUTER
  target_system: 1        # FC system id (corrected at runtime from the first HEARTBEAT)
  target_component: 1     # MAV_COMP_ID_AUTOPILOT1
  arm_retry_count: 5      # arm/disarm COMMAND_LONG retransmits before giving up
  heartbeat_wait_s: 5.0   # max wait for the first FC HEARTBEAT before requesting streams
```

- [ ] **Step 5: Verify both configs still load**

Run:
```bash
.venv/Scripts/python.exe -c "from quadguide.core.config import load_config, cfg_platform; [print(p, cfg_platform(load_config(p, {})).serial.mode) for p in ('configs/config.yaml','configs/rk3588.yaml')]"
```
Expected: prints both paths with their `serial.mode` (`uart`, `tcp`) and no exception.

- [ ] **Step 6: Commit**

```bash
git add configs/config.yaml configs/rk3588.yaml
git commit -m "config: MAVLink link fields; drop CRSF channel calibration"
```

---

## Task 9: Delete CRSF code + run full suite

**Files:**
- Delete: `src/quadguide/link/crsf.py`
- Delete: `tests/unit/test_crsf.py`
- Delete: `CRSF_PROTOCOL.md`

- [ ] **Step 1: Confirm nothing still imports crsf**

Run: `git grep -n "link.crsf\|link import crsf\|from quadguide.link.crsf" -- src tests scripts`
Expected: **no output** (Tasks 3–7 removed all uses). If anything prints, fix that file to use the MAVLink API before deleting.

- [ ] **Step 2: Delete the files**

```bash
git rm src/quadguide/link/crsf.py tests/unit/test_crsf.py CRSF_PROTOCOL.md
```

- [ ] **Step 3: Run the full unit suite**

Run: `.venv/Scripts/python.exe -m pytest tests/unit -v`
Expected: PASS, with no collection errors. The MAVLink tests (`test_mavlink_codec.py`, `test_fc.py`, `test_link_worker.py`) pass; no `test_crsf.py` is collected.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(link): remove CRSF code, tests, and protocol doc"
```

---

## Task 10: Update ARCHITECTURE.md + HIL doc

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `QUADGUIDE_HIL_INTEGRATION.md`

- [ ] **Step 1: Update the §1 overview sentence**

In `ARCHITECTURE.md`, replace:

```
sends roll/pitch setpoints
to a madflight flight controller over UART using the CRSF protocol (420000
baud, bidirectional).
```

with:

```
sends roll/pitch attitude setpoints
to an ArduPilot flight controller (H743) over UART using MAVLink2
(SET_ATTITUDE_TARGET, GUIDED_NOGPS).
```

- [ ] **Step 2: Update the §1 guidance-input sentence**

Replace:

```
together with raw body-rate and acceleration data from the FC over the CRSF `0x80` IMU
frame, are the primary guidance inputs.
```

with:

```
together with body-rate and acceleration data from the FC's MAVLink ATTITUDE
(#30) and RAW_IMU (#27) telemetry, are the primary guidance inputs.
```

- [ ] **Step 3: Update the §2.5 failsafe sentence**

Replace:

```
The link worker keeps the CRSF uplink at a constant 50 Hz regardless
of upstream health so the FC never enters its own RX failsafe.
```

with:

```
The link worker streams SET_ATTITUDE_TARGET at a constant 50 Hz regardless of
upstream health so the FC holds its GUIDED_NOGPS setpoint instead of timing out.
```

- [ ] **Step 4: Rewrite the §6.4 link/ section**

Replace the entire §6.4 block:

```
### 6.4 link/

CRSF parser/serializer + UART worker. RX path decodes `0x1E` (ATTITUDE),
`0x80` (custom IMU) into `fc/attitude` and `fc/imu`. TX path runs at a fixed
50 Hz, reading `control/cmd` and `arm/cmd`, mapping to channel ticks via
the configured `link.channels` table. The 50 Hz cadence is **constant**
regardless of upstream health — drop it and the FC enters its own RX
failsafe.
```

with:

```
### 6.4 link/

MAVLink2 codec + UART worker for an ArduPilot FC. `mavlink_codec.py` builds a
codec-mode pymavlink object (`file=None`) and holds `euler_to_quaternion` +
message-id/mask constants; `fc.py` maps messages ⇄ bus dataclasses; `worker.py`
runs the RX/TX/heartbeat/stream-setup/health loops over the transport-agnostic
`SerialPort`/`TCPSerialPort`. RX decodes `ATTITUDE` (#30, native body rates) →
`fc/attitude` and `RAW_IMU` (#27) → `fc/imu`, and tracks armed/mode from
`HEARTBEAT`. TX streams `SET_ATTITUDE_TARGET` at a **constant** 50 Hz (roll/pitch
from `control/cmd`, yaw held by a latched heading baked into the quaternion,
thrust = `throttle_norm`); arming is edge-triggered via
`MAV_CMD_COMPONENT_ARM_DISARM`. The pilot's RC switch owns the RC↔GUIDED_NOGPS
toggle — quadguide never changes mode. Drop the 50 Hz stream and the FC abandons
the GUIDED setpoint. Requires FC params `SERIALn_PROTOCOL=2` and `GUID_OPTIONS`
bit 3 (direct thrust).
```

- [ ] **Step 5: Update the §12 known constraint**

Replace:

```
1. **CRSF TX cadence is fixed at 50 Hz.** Drop it and the FC enters its own
   RX failsafe — independent of quadguide's watchdogs.
```

with:

```
1. **SET_ATTITUDE_TARGET cadence is fixed at 50 Hz.** Drop it and the FC
   abandons its GUIDED_NOGPS setpoint — independent of quadguide's watchdogs.
   ArduPilot also needs `GUID_OPTIONS` bit 3 set so `thrust` is direct (0–1),
   not climb rate.
```

- [ ] **Step 6: Rewrite QUADGUIDE_HIL_INTEGRATION.md for SITL**

Replace the file's "How to Run a HIL Session" → step 1 ("Start the dev machine simulator") and the CRSF-bridge references with ArduPilot SITL. Specifically, replace the `### 1. Start the dev machine simulator` section body with:

```markdown
### 1. Start ArduPilot SITL

```bash
sim_vehicle.py -v ArduCopter -f quad --console --map
# SITL exposes MAVLink2 on tcp:127.0.0.1:5760
```

In the SITL console/MAVProxy, put the vehicle in GUIDED_NOGPS so it accepts
quadguide's SET_ATTITUDE_TARGET (quadguide still arms + streams setpoints; it
never changes mode):

```
mode GUIDED_NOGPS
```
```

And in the toggle/overview tables, change the "UART to FC / TCPSerialPort … Raw
CRSF bytes" row to "UART to FC / TCPSerialPort → TCP socket / Bidirectional /
MAVLink2, same as UART", and change `tcp_port: 42000` references to `5760`.

- [ ] **Step 7: Verify no stale CRSF references remain in the two docs**

Run: `git grep -n -i "crsf\|0x80\|0x1E\|channel ticks" -- ARCHITECTURE.md QUADGUIDE_HIL_INTEGRATION.md`
Expected: no output (or only historical mentions you intentionally keep — there should be none).

- [ ] **Step 8: Commit**

```bash
git add ARCHITECTURE.md QUADGUIDE_HIL_INTEGRATION.md
git commit -m "docs: ARCHITECTURE + HIL updated for MAVLink/SITL link"
```

---

## Task 11: Update the link plan reference (optional bookkeeping)

**Files:**
- Modify: `README.md` (only if it mentions CRSF)

- [ ] **Step 1: Check README for CRSF mentions**

Run: `git grep -n -i "crsf\|madflight" -- README.md`
Expected: lists any lines to update; if none, skip to Task 12.

- [ ] **Step 2: Update any matched lines**

For each matched line, replace the CRSF/madflight phrasing with the MAVLink/ArduPilot equivalent (e.g. "CRSF to a madflight FC" → "MAVLink2 to an ArduPilot FC"). Keep surrounding wording intact.

- [ ] **Step 3: Commit (only if changed)**

```bash
git add README.md
git commit -m "docs: README mentions MAVLink/ArduPilot link"
```

---

## Task 12: Manual SITL integration verification (Linux)

**Not a unit test** — this runs the real stack against ArduPilot SITL and must be done on Linux (the bus is Linux-only). Record results; do not block the merge of the unit-tested code on hardware availability.

- [ ] **Step 1: Start SITL**

```bash
sim_vehicle.py -v ArduCopter -f quad --console
# Set the direct-thrust option once:
param set GUID_OPTIONS 8
mode GUIDED_NOGPS
```

- [ ] **Step 2: Run quadguide against SITL**

```bash
python scripts/run.py --config configs/config.yaml \
  --set platform.serial.mode=tcp \
  --set platform.serial.tcp_host=127.0.0.1 \
  --set platform.serial.tcp_port=5760 \
  --no-ground   # or keep ground for the HUD
```

- [ ] **Step 3: Confirm telemetry**

Expected in the link log: `FC HEARTBEAT: sys=1 comp=1`, then `requested ATTITUDE+RAW_IMU @ 50 Hz`. Confirm `fc/attitude` and `fc/imu` update (HUD telemetry or a `--log` trace).

- [ ] **Step 4: Confirm arm + setpoint**

Arm via the ground UI (or `python -c` publishing `arm/cmd`). Expected: link log `arm command → ARM`; SITL console shows the vehicle arm. With a lock-on active, the SITL vehicle should tilt per the roll/pitch setpoint.

- [ ] **Step 5: Confirm reconnect**

Kill and restart SITL. Expected: link reports DEGRADED, then re-handshakes (`FC HEARTBEAT` again) within ~1 s of SITL coming back.

- [ ] **Step 6: Record results in the PR description** (no commit needed).

---

## Self-Review Notes (for the implementer)

- **FC params are out of scope for code**: `SERIALn_PROTOCOL=2`, `SERIALn_BAUD` (match `serial.baud`), `GUID_OPTIONS=8`, and an RC switch mapped to GUIDED_NOGPS are set manually on the H743 (see spec §"FC-Side Parameters"). The `baud` placeholder (`115200`) is intentionally a TODO per the approved spec.
- **No bus-contract changes**: `core/messages.py` is untouched; `control/cmd` still carries `yaw_rate_dps` (always 0), consumed only as the hold-heading rationale.
- **Type consistency check**: `_rx_loop(serial, mav, state, bus, arm_ctrl, log)`, `_ArmController.on_arm_state/on_ack`, `latch_yaw(armed, prev_armed, last_yaw, held)`, and `encode_attitude_target(mav, cmd, yaw_hold, target_sys, target_comp, max_roll_deg, max_pitch_deg, now_ms)` signatures are identical across the tasks that define and call them.

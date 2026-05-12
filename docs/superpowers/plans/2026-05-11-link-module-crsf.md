# Link Module — CRSF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stubbed MSP link module with a working CRSF implementation: full-duplex UART to ESP-FC, bidirectional RC channels + attitude telemetry, finite-difference body rates, and an operator arm/disarm endpoint.

**Architecture:** The link worker runs three asyncio coroutines — RX decodes CRSF attitude frames and publishes `fc/attitude` + `fc/imu`; TX sends RC_CHANNELS_PACKED at 50 Hz (must start immediately on open to prevent FC failsafe); health posts at 5 Hz. Attitude is differentiated in-process before bus publish. A new `arm/cmd` bus topic carries operator arm intent from ground station to link worker.

**Tech Stack:** Python 3.11, asyncio, pyserial 3.5, FastAPI (ground endpoint), struct (bit-packing), pytest

---

## File Map

| Action | Path |
|---|---|
| Modify | `src/quadguide/core/messages.py` |
| Modify | `configs/config.yaml` |
| Create | `src/quadguide/link/crsf.py` |
| Create | `src/quadguide/link/differentiator.py` |
| Create | `src/quadguide/link/espfc.py` |
| Create | `src/quadguide/link/serial_port.py` |
| Create | `src/quadguide/link/worker.py` |
| Delete | `src/quadguide/link/msp.py` |
| Modify | `src/quadguide/ground/server.py` |
| Modify | `src/quadguide/ground/static/index.html` |
| Create | `scripts/test_link_rx.py` |
| Create | `scripts/test_link_tx.py` |
| Modify | `ARCHITECTURE.md` |
| Modify | `tests/unit/test_messages.py` |
| Create | `tests/unit/test_crsf.py` |
| Create | `tests/unit/test_differentiator.py` |
| Create | `tests/unit/test_espfc.py` |
| Create | `tests/unit/test_link_worker.py` |
| Modify | `tests/unit/test_ground_server.py` |

---

## Task 1: Add ArmCmd to core/messages.py

**Files:**
- Modify: `src/quadguide/core/messages.py`
- Modify: `tests/unit/test_messages.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_messages.py`:

```python
from quadguide.core.messages import ArmCmd, FMT_ARM_CMD
import struct

class TestArmCmd:
    def test_format_size(self):
        assert struct.calcsize(FMT_ARM_CMD) == 9  # Q(8) + B(1)

    def test_round_trip_armed_true(self):
        msg = ArmCmd(timestamp_ns=1_000_000, armed=True)
        r = ArmCmd.unpack(msg.pack())
        assert r.timestamp_ns == 1_000_000
        assert r.armed is True

    def test_round_trip_armed_false(self):
        msg = ArmCmd(timestamp_ns=2_000_000, armed=False)
        r = ArmCmd.unpack(msg.pack())
        assert r.armed is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_messages.py::TestArmCmd -v
```

Expected: ImportError — `ArmCmd` not defined.

- [ ] **Step 3: Implement ArmCmd in messages.py**

In `src/quadguide/core/messages.py`, add after the existing imports and before `__all__`:

```python
FMT_ARM_CMD = "!QB"
# Q(8) + armed(B=1) = 9 bytes
```

Add `"ArmCmd"` and `"FMT_ARM_CMD"` to `__all__`.

Add the `_ST_ARM_CMD` struct after the other `_ST_*` definitions:

```python
_ST_ARM_CMD = struct.Struct(FMT_ARM_CMD)
```

Add the `ArmCmd` dataclass after `HealthReport`:

```python
@dataclass(frozen=True)
class ArmCmd:
    timestamp_ns: int
    armed: bool

    def pack(self) -> bytes:
        return _ST_ARM_CMD.pack(self.timestamp_ns, int(self.armed))

    @classmethod
    def unpack(cls, data: bytes) -> ArmCmd:
        ts, armed_b = _ST_ARM_CMD.unpack(data)
        return cls(timestamp_ns=ts, armed=bool(armed_b))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_messages.py -v
```

Expected: all pass, including existing tests.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/core/messages.py tests/unit/test_messages.py
git commit -m "feat(core): add ArmCmd message for arm/disarm bus topic"
```

---

## Task 2: Update configs/config.yaml

**Files:**
- Modify: `configs/config.yaml`
- Modify: `src/quadguide/core/bus.py` (if `arm/cmd` needs declaring — check IPC pre-declaration)

- [ ] **Step 1: Update serial section and add link section**

In `configs/config.yaml`, replace the `serial:` block under `platform:`:

```yaml
  serial:
    port: /dev/ttyS0      # UART device path — change per SBC (e.g. /dev/ttyAMA0 on RPi4)
    baud: 420000          # CRSF standard — do not change without matching FC config
    rx_pin: "GPIO15"      # wiring reference only — not used by driver
    tx_pin: "GPIO14"      # wiring reference only — not used by driver
```

Add a new top-level `link:` section (after `watchdog:`, before `mission:`):

```yaml
link:
  tx_rate_hz: 50          # RC channels uplink rate to FC (Hz) — must stay constant to keep FC out of failsafe
  diff_lowpass_alpha: 0.3 # LP filter alpha for attitude-derived body rates (0=heavy smooth, 1=raw)
```

- [ ] **Step 2: Declare arm/cmd topic in bus pre-declaration**

Open `src/quadguide/core/bus.py` and find where topics are pre-declared (the IPC table initialisation). Add `"arm/cmd"` with message type `ArmCmd` and the same ring_depth as other topics.

The exact location depends on bus.py implementation — look for a dict or list of `(topic, message_class, ring_depth)` tuples or similar, and add:

```python
("arm/cmd",    ArmCmd,    cfg_bus.ring_depth),
```

Import `ArmCmd` at the top of `bus.py` if not already imported.

- [ ] **Step 3: Verify config loads without error**

```bash
python -c "
from quadguide.core.config import load_config
cfg = load_config('configs/config.yaml', {})
assert cfg['link']['tx_rate_hz'] == 50
assert cfg['link']['diff_lowpass_alpha'] == 0.3
assert cfg['platform']['serial']['baud'] == 420000
print('config OK')
"
```

Expected: `config OK`

- [ ] **Step 4: Commit**

```bash
git add configs/config.yaml src/quadguide/core/bus.py
git commit -m "config: switch serial to CRSF 420kbaud, add link section, declare arm/cmd topic"
```

---

## Task 3: CRSF frame primitives

**Files:**
- Create: `src/quadguide/link/crsf.py`
- Create: `tests/unit/test_crsf.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_crsf.py`:

```python
import struct
import pytest
from quadguide.link.crsf import (
    crc8, build_frame, pack_channels,
    CRSFFrame, CRSF_SYNC, CRSF_ATTITUDE, CRSF_RC_CHANNELS,
)


# --- CRC8 ---

def _ref_crc8(data: bytes) -> int:
    """Reference implementation to validate against."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def test_crc8_matches_reference_single_byte():
    data = bytes([0x1E])
    assert crc8(data) == _ref_crc8(data)


def test_crc8_matches_reference_multi_byte():
    data = bytes([0x16]) + bytes(22)
    assert crc8(data) == _ref_crc8(data)


def test_crc8_different_inputs_differ():
    assert crc8(b'\x1e\x00\x00') != crc8(b'\x1e\x00\x01')


# --- build_frame ---

def test_build_frame_sync_byte():
    frame = build_frame(CRSF_ATTITUDE, bytes(6))
    assert frame[0] == CRSF_SYNC  # 0xC8


def test_build_frame_length_field():
    # len = type(1) + payload_len + crc(1)
    frame = build_frame(CRSF_ATTITUDE, bytes(6))
    assert frame[1] == 8   # 1 + 6 + 1


def test_build_frame_type_byte():
    frame = build_frame(CRSF_ATTITUDE, bytes(6))
    assert frame[2] == CRSF_ATTITUDE


def test_build_frame_total_length():
    # sync(1) + len(1) + type(1) + payload(6) + crc(1) = 10
    frame = build_frame(CRSF_ATTITUDE, bytes(6))
    assert len(frame) == 10


def test_build_frame_rc_channels_length():
    # sync(1) + len(1) + type(1) + payload(22) + crc(1) = 26
    frame = build_frame(CRSF_RC_CHANNELS, bytes(22))
    assert len(frame) == 26
    assert frame[1] == 24  # 1 + 22 + 1


def test_build_frame_crc_appended():
    payload = struct.pack(">hhh", 100, 200, 300)
    frame = build_frame(CRSF_ATTITUDE, payload)
    expected_crc = _ref_crc8(bytes([CRSF_ATTITUDE]) + payload)
    assert frame[-1] == expected_crc


# --- pack_channels ---

def test_pack_channels_produces_22_bytes():
    assert len(pack_channels([992] * 16)) == 22


def test_pack_channels_center_values():
    packed = pack_channels([992] * 16)
    bits = int.from_bytes(packed, "little")
    for i in range(16):
        assert (bits >> (i * 11)) & 0x7FF == 992


def test_pack_channels_min_max():
    channels = [172, 1811] + [992] * 14
    packed = pack_channels(channels)
    bits = int.from_bytes(packed, "little")
    assert (bits >> 0) & 0x7FF == 172    # ch1 min
    assert (bits >> 11) & 0x7FF == 1811  # ch2 max


def test_pack_channels_all_independent():
    # Each channel occupies its own 11-bit slot; changing ch3 should not affect ch1
    base = [992] * 16
    modified = list(base)
    modified[2] = 500
    packed_base = pack_channels(base)
    packed_mod  = pack_channels(modified)
    bits_base = int.from_bytes(packed_base, "little")
    bits_mod  = int.from_bytes(packed_mod, "little")
    assert (bits_base >> 0) & 0x7FF == (bits_mod >> 0) & 0x7FF  # ch1 unchanged
    assert (bits_mod >> 22) & 0x7FF == 500                       # ch3 changed
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_crsf.py -v
```

Expected: ModuleNotFoundError — `quadguide.link.crsf` doesn't exist.

- [ ] **Step 3: Implement CRSF primitives in link/crsf.py**

Create `src/quadguide/link/crsf.py`:

```python
from __future__ import annotations
import time
from dataclasses import dataclass
from enum import IntEnum

CRSF_SYNC        = 0xC8
CRSF_ATTITUDE    = 0x1E
CRSF_RC_CHANNELS = 0x16

_MAX_LEN = 62  # max valid len field value (payload ≤ 60, +type+crc = 62)


def _make_crc8_table(poly: int) -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
        table.append(crc)
    return table


_CRC8_TABLE = _make_crc8_table(0xD5)


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _CRC8_TABLE[crc ^ b]
    return crc


@dataclass
class CRSFFrame:
    type: int
    payload: bytes
    timestamp_ns: int


def build_frame(frame_type: int, payload: bytes) -> bytes:
    length = len(payload) + 2  # type(1) + crc(1)
    crc_input = bytes([frame_type]) + payload
    return bytes([CRSF_SYNC, length, frame_type]) + payload + bytes([crc8(crc_input)])


def pack_channels(channels: list[int]) -> bytes:
    assert len(channels) == 16
    bits = 0
    for i, ch in enumerate(channels):
        bits |= (ch & 0x7FF) << (i * 11)
    return bits.to_bytes(22, "little")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_crsf.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/crsf.py tests/unit/test_crsf.py
git commit -m "feat(link): add CRSF frame primitives — crc8, build_frame, pack_channels"
```

---

## Task 4: CRSF parser

**Files:**
- Modify: `src/quadguide/link/crsf.py` (add CRSFParser)
- Modify: `tests/unit/test_crsf.py` (add parser tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_crsf.py`:

```python
from quadguide.link.crsf import CRSFParser


class TestCRSFParser:
    def _feed_all(self, parser: CRSFParser, data: bytes) -> list[CRSFFrame]:
        results = []
        for b in data:
            frame = parser.feed(b)
            if frame is not None:
                results.append(frame)
        return results

    def test_parses_attitude_frame(self):
        payload = struct.pack(">hhh", 100, 200, 300)
        frame_bytes = build_frame(CRSF_ATTITUDE, payload)
        parser = CRSFParser()
        frames = self._feed_all(parser, frame_bytes)
        assert len(frames) == 1
        assert frames[0].type == CRSF_ATTITUDE
        assert frames[0].payload == payload

    def test_parses_rc_channels_frame(self):
        payload = pack_channels([992] * 16)
        frame_bytes = build_frame(CRSF_RC_CHANNELS, payload)
        parser = CRSFParser()
        frames = self._feed_all(parser, frame_bytes)
        assert len(frames) == 1
        assert frames[0].type == CRSF_RC_CHANNELS

    def test_returns_none_until_frame_complete(self):
        payload = struct.pack(">hhh", 0, 0, 0)
        frame_bytes = build_frame(CRSF_ATTITUDE, payload)
        parser = CRSFParser()
        # All but the last byte should return None
        for b in frame_bytes[:-1]:
            assert parser.feed(b) is None
        # Last byte completes the frame
        assert parser.feed(frame_bytes[-1]) is not None

    def test_ignores_non_sync_bytes(self):
        parser = CRSFParser()
        for b in [0x00, 0x01, 0xFF, 0xAA, 0x42]:
            assert parser.feed(b) is None

    def test_rejects_crc_mismatch(self):
        payload = struct.pack(">hhh", 100, 200, 300)
        frame_bytes = bytearray(build_frame(CRSF_ATTITUDE, payload))
        frame_bytes[-1] ^= 0xFF  # corrupt CRC
        parser = CRSFParser()
        frames = self._feed_all(parser, bytes(frame_bytes))
        assert len(frames) == 0

    def test_recovers_after_crc_mismatch(self):
        # Bad frame followed by a good frame — parser must recover
        payload = struct.pack(">hhh", 100, 200, 300)
        bad = bytearray(build_frame(CRSF_ATTITUDE, payload))
        bad[-1] ^= 0xFF
        good = build_frame(CRSF_ATTITUDE, payload)
        parser = CRSFParser()
        frames = self._feed_all(parser, bytes(bad) + good)
        assert len(frames) == 1

    def test_parses_two_consecutive_frames(self):
        payload = struct.pack(">hhh", 1, 2, 3)
        frame_bytes = build_frame(CRSF_ATTITUDE, payload) * 2
        parser = CRSFParser()
        frames = self._feed_all(parser, frame_bytes)
        assert len(frames) == 2

    def test_resets_on_oversized_length(self):
        # Feed a sync byte followed by an invalid length (> 62)
        parser = CRSFParser()
        assert parser.feed(CRSF_SYNC) is None
        assert parser.feed(63) is None  # invalid len, should reset
        # Next valid frame should still parse
        payload = struct.pack(">hhh", 0, 0, 0)
        frames = self._feed_all(parser, build_frame(CRSF_ATTITUDE, payload))
        assert len(frames) == 1

    def test_frame_has_timestamp(self):
        payload = struct.pack(">hhh", 0, 0, 0)
        frame_bytes = build_frame(CRSF_ATTITUDE, payload)
        parser = CRSFParser()
        frames = self._feed_all(parser, frame_bytes)
        assert frames[0].timestamp_ns > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_crsf.py::TestCRSFParser -v
```

Expected: AttributeError — `CRSFParser` not defined.

- [ ] **Step 3: Implement CRSFParser in link/crsf.py**

Append to `src/quadguide/link/crsf.py`:

```python
class _State(IntEnum):
    WAIT_SYNC    = 0
    READ_LEN     = 1
    READ_TYPE    = 2
    READ_PAYLOAD = 3
    READ_CRC     = 4


class CRSFParser:
    def __init__(self):
        self._state     = _State.WAIT_SYNC
        self._len       = 0
        self._type      = 0
        self._payload   = bytearray()
        self._remaining = 0

    def feed(self, byte: int) -> CRSFFrame | None:
        if self._state == _State.WAIT_SYNC:
            if byte == CRSF_SYNC:
                self._state = _State.READ_LEN

        elif self._state == _State.READ_LEN:
            if byte < 2 or byte > _MAX_LEN:
                self._reset()
            else:
                self._len       = byte
                self._remaining = byte - 2  # payload bytes = len - type(1) - crc(1)
                self._state     = _State.READ_TYPE

        elif self._state == _State.READ_TYPE:
            self._type    = byte
            self._payload = bytearray()
            self._state   = _State.READ_PAYLOAD if self._remaining > 0 else _State.READ_CRC

        elif self._state == _State.READ_PAYLOAD:
            self._payload.append(byte)
            self._remaining -= 1
            if self._remaining == 0:
                self._state = _State.READ_CRC

        elif self._state == _State.READ_CRC:
            self._state = _State.WAIT_SYNC
            expected = crc8(bytes([self._type]) + self._payload)
            if byte == expected:
                return CRSFFrame(
                    type=self._type,
                    payload=bytes(self._payload),
                    timestamp_ns=time.monotonic_ns(),
                )
            # CRC mismatch — drop frame silently

        return None

    def _reset(self) -> None:
        self._state   = _State.WAIT_SYNC
        self._payload = bytearray()
```

- [ ] **Step 4: Run all crsf tests**

```bash
pytest tests/unit/test_crsf.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/crsf.py tests/unit/test_crsf.py
git commit -m "feat(link): add CRSFParser state machine with CRC8 verification"
```

---

## Task 5: AttitudeDifferentiator

**Files:**
- Create: `src/quadguide/link/differentiator.py`
- Create: `tests/unit/test_differentiator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_differentiator.py`:

```python
import math
import pytest
from quadguide.link.differentiator import AttitudeDifferentiator


def test_first_call_returns_zero_rates():
    diff = AttitudeDifferentiator(alpha=1.0)
    rates = diff.update(0.1, 0.2, 0.3, now_ns=0)
    assert rates == (0.0, 0.0, 0.0)


def test_rate_calculation_roll():
    diff = AttitudeDifferentiator(alpha=1.0)
    diff.update(0.0, 0.0, 0.0, now_ns=0)
    rates = diff.update(1.0, 0.0, 0.0, now_ns=int(1e9))  # 1 rad in 1 second
    assert rates[0] == pytest.approx(1.0, rel=1e-5)
    assert rates[1] == pytest.approx(0.0, abs=1e-9)
    assert rates[2] == pytest.approx(0.0, abs=1e-9)


def test_rate_calculation_pitch():
    diff = AttitudeDifferentiator(alpha=1.0)
    diff.update(0.0, 0.0, 0.0, now_ns=0)
    rates = diff.update(0.0, 2.0, 0.0, now_ns=int(2e9))  # 2 rad in 2 seconds
    assert rates[1] == pytest.approx(1.0, rel=1e-5)


def test_rate_calculation_half_second():
    diff = AttitudeDifferentiator(alpha=1.0)
    diff.update(0.0, 0.0, 0.0, now_ns=0)
    rates = diff.update(1.0, 0.0, 0.0, now_ns=int(500e6))  # 1 rad in 0.5 seconds
    assert rates[0] == pytest.approx(2.0, rel=1e-5)


def test_yaw_wraparound_positive_to_negative():
    diff = AttitudeDifferentiator(alpha=1.0)
    # Yaw crosses +π → -π boundary; shortest-path change is -0.2 rad
    diff.update(0.0, 0.0, math.pi - 0.1, now_ns=0)
    rates = diff.update(0.0, 0.0, -(math.pi - 0.1), now_ns=int(1e9))
    assert rates[2] == pytest.approx(-0.2, abs=1e-5)


def test_yaw_wraparound_negative_to_positive():
    diff = AttitudeDifferentiator(alpha=1.0)
    diff.update(0.0, 0.0, -(math.pi - 0.1), now_ns=0)
    rates = diff.update(0.0, 0.0, math.pi - 0.1, now_ns=int(1e9))
    assert rates[2] == pytest.approx(0.2, abs=1e-5)


def test_lowpass_filter_alpha_1_is_unfiltered():
    # alpha=1.0 means no filtering: output equals raw rate
    diff = AttitudeDifferentiator(alpha=1.0)
    diff.update(0.0, 0.0, 0.0, now_ns=0)
    diff.update(2.0, 0.0, 0.0, now_ns=int(1e9))  # raw = 2.0
    rates = diff.update(4.0, 0.0, 0.0, now_ns=int(2e9))  # raw = 2.0
    assert rates[0] == pytest.approx(2.0, rel=1e-5)


def test_lowpass_filter_smoothing():
    # alpha=0.5: filtered = 0.5*raw + 0.5*prev_filtered
    diff = AttitudeDifferentiator(alpha=0.5)
    diff.update(0.0, 0.0, 0.0, now_ns=0)
    # Step 1: raw=2.0, filtered = 0.5*2.0 + 0.5*0.0 = 1.0
    diff.update(2.0, 0.0, 0.0, now_ns=int(1e9))
    # Step 2: raw=2.0, filtered = 0.5*2.0 + 0.5*1.0 = 1.5
    rates = diff.update(4.0, 0.0, 0.0, now_ns=int(2e9))
    assert rates[0] == pytest.approx(1.5, abs=1e-6)


def test_zero_dt_returns_previous_rates():
    diff = AttitudeDifferentiator(alpha=1.0)
    diff.update(0.0, 0.0, 0.0, now_ns=0)
    diff.update(1.0, 0.0, 0.0, now_ns=int(1e9))  # sets rate to 1.0
    # Same timestamp — dt=0, should return previous filtered rates unchanged
    rates = diff.update(2.0, 0.0, 0.0, now_ns=int(1e9))
    assert rates[0] == pytest.approx(1.0, rel=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_differentiator.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement AttitudeDifferentiator**

Create `src/quadguide/link/differentiator.py`:

```python
from __future__ import annotations
import math


def _angle_diff(a: float, b: float) -> float:
    """Shortest-path angular difference a - b, result in (-π, π]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class AttitudeDifferentiator:
    def __init__(self, alpha: float):
        self._alpha  = alpha
        self._roll   = 0.0
        self._pitch  = 0.0
        self._yaw    = 0.0
        self._ns     = 0
        self._rr     = 0.0  # filtered roll rate
        self._pr     = 0.0  # filtered pitch rate
        self._yr     = 0.0  # filtered yaw rate
        self._ready  = False

    def update(self, roll: float, pitch: float, yaw: float, now_ns: int
               ) -> tuple[float, float, float]:
        if not self._ready:
            self._roll, self._pitch, self._yaw, self._ns = roll, pitch, yaw, now_ns
            self._ready = True
            return 0.0, 0.0, 0.0

        dt = (now_ns - self._ns) * 1e-9
        if dt <= 0.0:
            return self._rr, self._pr, self._yr

        raw_rr = (roll  - self._roll)              / dt
        raw_pr = (pitch - self._pitch)             / dt
        raw_yr = _angle_diff(yaw, self._yaw)       / dt

        self._rr = self._alpha * raw_rr + (1.0 - self._alpha) * self._rr
        self._pr = self._alpha * raw_pr + (1.0 - self._alpha) * self._pr
        self._yr = self._alpha * raw_yr + (1.0 - self._alpha) * self._yr

        self._roll, self._pitch, self._yaw, self._ns = roll, pitch, yaw, now_ns
        return self._rr, self._pr, self._yr
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_differentiator.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/differentiator.py tests/unit/test_differentiator.py
git commit -m "feat(link): add AttitudeDifferentiator with yaw-wrap and LP filter"
```

---

## Task 6: espfc — decode and encode

**Files:**
- Create: `src/quadguide/link/espfc.py`
- Create: `tests/unit/test_espfc.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_espfc.py`:

```python
import math
import struct
import pytest
from quadguide.link.crsf import build_frame, pack_channels, CRSF_ATTITUDE, CRSF_RC_CHANNELS, CRSFFrame
from quadguide.link.differentiator import AttitudeDifferentiator
from quadguide.link.espfc import decode_attitude, encode_rc, us_to_ticks, ticks_to_us
from quadguide.core.messages import AttitudeState, ControlCmd, IMUFrame


# ── us_to_ticks / ticks_to_us ─────────────────────────────────────────────

def test_us_to_ticks_center():
    assert us_to_ticks(1500.0) == 992

def test_us_to_ticks_full_high():
    assert us_to_ticks(2000.0) == 1792

def test_us_to_ticks_full_low():
    assert us_to_ticks(1000.0) == 192

def test_us_to_ticks_clamps_high():
    assert us_to_ticks(3000.0) == 1811

def test_us_to_ticks_clamps_low():
    assert us_to_ticks(0.0) == 172

def test_ticks_to_us_center():
    assert ticks_to_us(992) == pytest.approx(1500.0)

def test_ticks_to_us_full_high():
    assert ticks_to_us(1792) == pytest.approx(2000.0)

def test_ticks_to_us_full_low():
    assert ticks_to_us(192) == pytest.approx(1000.0)


# ── decode_attitude ───────────────────────────────────────────────────────

def _make_attitude_frame(pitch_raw: int, roll_raw: int, yaw_raw: int) -> CRSFFrame:
    payload = struct.pack(">hhh", pitch_raw, roll_raw, yaw_raw)
    frame = build_frame(CRSF_ATTITUDE, payload)
    # Build CRSFFrame directly (timestamp doesn't matter for unit test)
    return CRSFFrame(type=CRSF_ATTITUDE, payload=payload, timestamp_ns=int(1e9))


def test_decode_attitude_angles():
    frame = _make_attitude_frame(pitch_raw=1000, roll_raw=500, yaw_raw=-200)
    diff = AttitudeDifferentiator(alpha=1.0)
    att, imu = decode_attitude(frame, diff)
    assert isinstance(att, AttitudeState)
    assert att.pitch_rad == pytest.approx(0.1,   rel=1e-5)   # 1000 * 1e-4
    assert att.roll_rad  == pytest.approx(0.05,  rel=1e-5)   # 500  * 1e-4
    assert att.yaw_rad   == pytest.approx(-0.02, rel=1e-5)   # -200 * 1e-4


def test_decode_attitude_first_call_body_rates_zero():
    frame = _make_attitude_frame(1000, 500, 200)
    diff = AttitudeDifferentiator(alpha=1.0)
    att, _ = decode_attitude(frame, diff)
    assert att.roll_rate_rps  == pytest.approx(0.0, abs=1e-9)
    assert att.pitch_rate_rps == pytest.approx(0.0, abs=1e-9)
    assert att.yaw_rate_rps   == pytest.approx(0.0, abs=1e-9)


def test_decode_attitude_imu_gyro_matches_att_rates():
    diff = AttitudeDifferentiator(alpha=1.0)
    decode_attitude(_make_attitude_frame(0, 0, 0), diff)
    # Second frame: 1 second later, 1 rad change in roll
    frame2 = CRSFFrame(type=CRSF_ATTITUDE,
                       payload=struct.pack(">hhh", 0, 10000, 0),
                       timestamp_ns=int(2e9))
    att, imu = decode_attitude(frame2, diff)
    assert imu.gx == pytest.approx(att.roll_rate_rps,  rel=1e-5)
    assert imu.gy == pytest.approx(att.pitch_rate_rps, rel=1e-5)
    assert imu.gz == pytest.approx(att.yaw_rate_rps,   rel=1e-5)


def test_decode_attitude_imu_accel_zero():
    frame = _make_attitude_frame(0, 0, 0)
    diff = AttitudeDifferentiator(alpha=1.0)
    _, imu = decode_attitude(frame, diff)
    assert isinstance(imu, IMUFrame)
    assert imu.ax == 0.0
    assert imu.ay == 0.0
    assert imu.az == 0.0


# ── encode_rc ─────────────────────────────────────────────────────────────

def _decode_channels(frame_bytes: bytes) -> list[int]:
    """Helper: extract 16 CRSF channel values from a built RC_CHANNELS frame."""
    payload = frame_bytes[3:25]   # skip sync(1), len(1), type(1); payload is 22 bytes
    bits = int.from_bytes(payload, "little")
    return [(bits >> (i * 11)) & 0x7FF for i in range(16)]


def test_encode_rc_none_cmd_center_roll_pitch_yaw():
    channels = _decode_channels(encode_rc(None, armed=False))
    assert channels[0] == 992   # ch1 roll  — neutral
    assert channels[1] == 992   # ch2 pitch — neutral
    assert channels[3] == 992   # ch4 yaw   — neutral


def test_encode_rc_none_cmd_min_throttle():
    channels = _decode_channels(encode_rc(None, armed=False))
    assert channels[2] == 172   # ch3 throttle — minimum


def test_encode_rc_none_cmd_disarmed():
    channels = _decode_channels(encode_rc(None, armed=False))
    assert channels[4] == 172   # ch5 — disarmed


def test_encode_rc_armed_sets_ch5_max():
    channels = _decode_channels(encode_rc(None, armed=True))
    assert channels[4] == 1811  # ch5 — armed


def test_encode_rc_ch6_to_ch16_neutral():
    channels = _decode_channels(encode_rc(None, armed=False))
    for i in range(5, 16):
        assert channels[i] == 992


def test_encode_rc_roll_right():
    # roll_deg = +90 → 2000 µs → 1792 ticks
    cmd = ControlCmd(timestamp_ns=0, roll_deg=90.0, pitch_deg=0.0,
                     yaw_rate_dps=0.0, throttle_norm=0.0)
    channels = _decode_channels(encode_rc(cmd, armed=False))
    assert channels[0] == 1792


def test_encode_rc_roll_left():
    cmd = ControlCmd(timestamp_ns=0, roll_deg=-90.0, pitch_deg=0.0,
                     yaw_rate_dps=0.0, throttle_norm=0.0)
    channels = _decode_channels(encode_rc(cmd, armed=False))
    assert channels[0] == 192


def test_encode_rc_throttle_half():
    # throttle_norm=0.5 → 1500 µs → 992 ticks
    cmd = ControlCmd(timestamp_ns=0, roll_deg=0.0, pitch_deg=0.0,
                     yaw_rate_dps=0.0, throttle_norm=0.5)
    channels = _decode_channels(encode_rc(cmd, armed=False))
    assert channels[2] == 992


def test_encode_rc_throttle_full():
    cmd = ControlCmd(timestamp_ns=0, roll_deg=0.0, pitch_deg=0.0,
                     yaw_rate_dps=0.0, throttle_norm=1.0)
    channels = _decode_channels(encode_rc(cmd, armed=False))
    assert channels[2] == 1792


def test_encode_rc_returns_valid_crsf_frame():
    data = encode_rc(None, armed=False)
    assert data[0] == 0xC8              # sync
    assert data[2] == CRSF_RC_CHANNELS  # type
    assert len(data) == 26              # sync+len+type+22+crc
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_espfc.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement espfc.py**

Create `src/quadguide/link/espfc.py`:

```python
from __future__ import annotations
import struct

from quadguide.core.messages import AttitudeState, ControlCmd, IMUFrame
from quadguide.link.crsf import CRSF_RC_CHANNELS, CRSFFrame, build_frame, pack_channels
from quadguide.link.differentiator import AttitudeDifferentiator

_ROLL_PITCH_SCALE = 90.0   # ±90° maps to ±500 µs full deflection
_YAW_RATE_SCALE   = 200.0  # ±200 dps maps to ±500 µs full deflection

_NEUTRAL      = 992   # 1500 µs — center
_THROTTLE_MIN = 172   # 1000 µs — minimum throttle
_ARM_HIGH     = 1811  # 2000 µs — armed
_ARM_LOW      = 172   # 1000 µs — disarmed


def decode_attitude(frame: CRSFFrame, diff: AttitudeDifferentiator
                    ) -> tuple[AttitudeState, IMUFrame]:
    pitch_raw, roll_raw, yaw_raw = struct.unpack(">hhh", frame.payload[:6])
    roll_rad  = roll_raw  * 1e-4
    pitch_rad = pitch_raw * 1e-4
    yaw_rad   = yaw_raw   * 1e-4
    rr, pr, yr = diff.update(roll_rad, pitch_rad, yaw_rad, frame.timestamp_ns)
    att = AttitudeState(
        timestamp_ns=frame.timestamp_ns,
        roll_rad=roll_rad, pitch_rad=pitch_rad, yaw_rad=yaw_rad,
        roll_rate_rps=rr, pitch_rate_rps=pr, yaw_rate_rps=yr,
    )
    imu = IMUFrame(
        timestamp_ns=frame.timestamp_ns,
        ax=0.0, ay=0.0, az=0.0,
        gx=rr, gy=pr, gz=yr,
    )
    return att, imu


def encode_rc(cmd: ControlCmd | None, armed: bool) -> bytes:
    if cmd is None:
        ch_roll, ch_pitch, ch_throttle, ch_yaw = (
            _NEUTRAL, _NEUTRAL, _THROTTLE_MIN, _NEUTRAL
        )
    else:
        ch_roll     = us_to_ticks(1500.0 + _clamp(cmd.roll_deg    / _ROLL_PITCH_SCALE, -1, 1) * 500.0)
        ch_pitch    = us_to_ticks(1500.0 + _clamp(cmd.pitch_deg   / _ROLL_PITCH_SCALE, -1, 1) * 500.0)
        ch_throttle = us_to_ticks(1000.0 + _clamp(cmd.throttle_norm,                   0, 1) * 1000.0)
        ch_yaw      = us_to_ticks(1500.0 + _clamp(cmd.yaw_rate_dps / _YAW_RATE_SCALE,  -1, 1) * 500.0)
    channels = [
        ch_roll, ch_pitch, ch_throttle, ch_yaw,
        _ARM_HIGH if armed else _ARM_LOW,  # ch5 arm
        *([_NEUTRAL] * 11),                # ch6–16
    ]
    return build_frame(CRSF_RC_CHANNELS, pack_channels(channels))


def us_to_ticks(us: float) -> int:
    return int(_clamp((us - 1500.0) * 8.0 / 5.0 + 992.0, 172.0, 1811.0))


def ticks_to_us(ticks: int) -> float:
    return (ticks - 992) * 5.0 / 8.0 + 1500.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_espfc.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/espfc.py tests/unit/test_espfc.py
git commit -m "feat(link): add espfc decode_attitude and encode_rc for CRSF"
```

---

## Task 7: SerialPort

**Files:**
- Create: `src/quadguide/link/serial_port.py`

No unit test — pyserial requires a real device. Coverage comes from `scripts/test_link_rx.py` and `scripts/test_link_tx.py`.

- [ ] **Step 1: Implement SerialPort**

Create `src/quadguide/link/serial_port.py`:

```python
from __future__ import annotations
import asyncio
from typing import AsyncGenerator

import serial


class SerialPort:
    def __init__(self, port: str, baud: int):
        self._port      = port
        self._baud      = baud
        self._ser: serial.Serial | None = None
        self._connected = False

    async def open(self) -> None:
        loop = asyncio.get_running_loop()
        self._ser = await loop.run_in_executor(
            None, lambda: serial.Serial(self._port, self._baud, timeout=0.05)
        )
        self._connected = True

    async def read_stream(self) -> AsyncGenerator[int, None]:
        loop = asyncio.get_running_loop()
        while self._connected:
            try:
                data = await loop.run_in_executor(None, self._ser.read, 64)
            except serial.SerialException as exc:
                self._connected = False
                raise ConnectionError(str(exc)) from exc
            for b in data:
                yield b

    async def write(self, data: bytes) -> None:
        if not self._connected or self._ser is None:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._ser.write, data)
        except serial.SerialException:
            self._connected = False

    def close(self) -> None:
        self._connected = False
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        return self._connected
```

- [ ] **Step 2: Verify import**

```bash
python -c "from quadguide.link.serial_port import SerialPort; print('SerialPort OK')"
```

Expected: `SerialPort OK`

- [ ] **Step 3: Commit**

```bash
git add src/quadguide/link/serial_port.py
git commit -m "feat(link): add async SerialPort wrapper with reconnect support"
```

---

## Task 8: Link worker

**Files:**
- Create: `src/quadguide/link/worker.py`
- Create: `tests/unit/test_link_worker.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_link_worker.py`:

```python
import asyncio
import logging
import struct

from quadguide.link.crsf import build_frame, CRSFParser, CRSF_ATTITUDE
from quadguide.link.differentiator import AttitudeDifferentiator
from quadguide.link.worker import _rx_loop
from quadguide.core.messages import AttitudeState, IMUFrame


class _FakeSerial:
    """Async-generator serial stub that yields a fixed byte sequence once."""
    def __init__(self, data: bytes):
        self._data = data

    async def read_stream(self):
        for b in self._data:
            yield b


class _FakeBus:
    def __init__(self):
        self.published: list[tuple[str, object]] = []

    def publish(self, topic: str, msg) -> None:
        self.published.append((topic, msg))

    def latest(self, topic: str):
        return None


def test_rx_loop_publishes_attitude_and_imu():
    payload = struct.pack(">hhh", 1000, 500, -200)  # pitch, roll, yaw (raw int16)
    frame_bytes = build_frame(CRSF_ATTITUDE, payload)

    serial = _FakeSerial(frame_bytes)
    bus    = _FakeBus()
    diff   = AttitudeDifferentiator(alpha=1.0)
    parser = CRSFParser()
    log    = logging.getLogger("test")

    asyncio.run(_rx_loop(serial, parser, diff, bus, log))

    att_msgs = [m for t, m in bus.published if t == "fc/attitude"]
    imu_msgs = [m for t, m in bus.published if t == "fc/imu"]

    assert len(att_msgs) == 1
    assert len(imu_msgs) == 1
    assert isinstance(att_msgs[0], AttitudeState)
    assert isinstance(imu_msgs[0], IMUFrame)

    import pytest
    att = att_msgs[0]
    assert att.pitch_rad == pytest.approx(0.1,   rel=1e-4)
    assert att.roll_rad  == pytest.approx(0.05,  rel=1e-4)
    assert att.yaw_rad   == pytest.approx(-0.02, rel=1e-4)


def test_rx_loop_ignores_non_attitude_frames():
    from quadguide.link.crsf import pack_channels, CRSF_RC_CHANNELS
    frame_bytes = build_frame(CRSF_RC_CHANNELS, bytes(22))

    serial = _FakeSerial(frame_bytes)
    bus    = _FakeBus()
    diff   = AttitudeDifferentiator(alpha=1.0)
    parser = CRSFParser()
    log    = logging.getLogger("test")

    asyncio.run(_rx_loop(serial, parser, diff, bus, log))

    assert not any(t == "fc/attitude" for t, _ in bus.published)
    assert not any(t == "fc/imu" for t, _ in bus.published)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_link_worker.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement worker.py**

Create `src/quadguide/link/worker.py`:

```python
from __future__ import annotations
import asyncio
import logging
import signal

from quadguide.core.clock import monotonic_ns
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, ProcessState
from quadguide.link.crsf import CRSF_ATTITUDE, CRSFParser
from quadguide.link.differentiator import AttitudeDifferentiator
from quadguide.link.espfc import decode_attitude, encode_rc
from quadguide.link.serial_port import SerialPort


async def _rx_loop(serial, parser: CRSFParser,
                   diff: AttitudeDifferentiator, bus, log: logging.Logger) -> None:
    async for byte in serial.read_stream():
        frame = parser.feed(byte)
        if frame is None:
            continue
        if frame.type == CRSF_ATTITUDE:
            att, imu = decode_attitude(frame, diff)
            bus.publish("fc/attitude", att)
            bus.publish("fc/imu", imu)


async def _tx_loop(serial, bus, tx_rate_hz: float, log: logging.Logger) -> None:
    interval = 1.0 / tx_rate_hz
    while True:
        cmd     = bus.latest("control/cmd")
        arm_cmd = bus.latest("arm/cmd")
        armed   = arm_cmd.armed if arm_cmd else False
        await serial.write(encode_rc(cmd, armed))
        await asyncio.sleep(interval)


async def _health_loop(bus, log: logging.Logger) -> None:
    while True:
        bus.publish("system/health",
                    HealthReport(monotonic_ns(), "link", ProcessState.OK, ""))
        await asyncio.sleep(0.2)


async def _run_async(config: dict, bus) -> None:
    log        = setup_logging("link", config)
    diff       = AttitudeDifferentiator(config["link"]["diff_lowpass_alpha"])
    tx_rate_hz = config["link"]["tx_rate_hz"]
    port       = config["platform"]["serial"]["port"]
    baud       = config["platform"]["serial"]["baud"]

    loop = asyncio.get_running_loop()

    def _on_sigterm(*_):
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGTERM, _on_sigterm)

    while True:
        serial = SerialPort(port, baud)
        tasks: list[asyncio.Task] = []
        try:
            await serial.open()
            log.info(f"Serial opened {port} @ {baud}")

            tasks = [
                asyncio.create_task(_rx_loop(serial, CRSFParser(), diff, bus, log)),
                asyncio.create_task(_tx_loop(serial, bus, tx_rate_hz, log)),
                asyncio.create_task(_health_loop(bus, log)),
            ]
            # Block until any task raises; rx raises ConnectionError on disconnect
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

    bus.detach()
    log.info("Link worker stopped.")


def run(config: dict, bus) -> None:
    asyncio.run(_run_async(config, bus))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_link_worker.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/worker.py tests/unit/test_link_worker.py
git commit -m "feat(link): implement CRSF link worker with asyncio RX/TX/health coroutines"
```

---

## Task 9: Ground /arm endpoint and UI button

**Files:**
- Modify: `src/quadguide/ground/server.py`
- Modify: `src/quadguide/ground/static/index.html`
- Modify: `tests/unit/test_ground_server.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_ground_server.py`:

```python
from quadguide.core.messages import ArmCmd


def test_arm_returns_ok(client):
    resp = client.post("/arm", json={"armed": True})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_arm_publishes_arm_cmd_true(bus_client):
    bus, client = bus_client
    client.post("/arm", json={"armed": True})
    arm_msgs = [(t, m) for t, m in bus.published if t == "arm/cmd"]
    assert len(arm_msgs) == 1
    assert isinstance(arm_msgs[0][1], ArmCmd)
    assert arm_msgs[0][1].armed is True


def test_arm_publishes_arm_cmd_false(bus_client):
    bus, client = bus_client
    client.post("/arm", json={"armed": False})
    arm_msgs = [(t, m) for t, m in bus.published if t == "arm/cmd"]
    assert len(arm_msgs) == 1
    assert arm_msgs[0][1].armed is False


def test_arm_missing_field_returns_422(client):
    resp = client.post("/arm", json={})
    assert resp.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_ground_server.py::test_arm_returns_ok tests/unit/test_ground_server.py::test_arm_publishes_arm_cmd_true -v
```

Expected: 404 — endpoint doesn't exist.

- [ ] **Step 3: Add /arm endpoint to server.py**

In `src/quadguide/ground/server.py`, update the import line:

```python
from quadguide.core.messages import ArmCmd, BoundingBox, HealthReport, LockOnCmd, ProcessState
```

Add `_ArmBody` model after `_LockOnBody`:

```python
class _ArmBody(BaseModel):
    armed: bool
```

Add the endpoint inside `create_app`, after the `/lockon` route:

```python
    @app.post("/arm")
    async def arm(body: _ArmBody, request: Request):
        cmd = ArmCmd(timestamp_ns=monotonic_ns(), armed=body.armed)
        request.app.state.bus.publish("arm/cmd", cmd)
        return {"ok": True}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_ground_server.py -v
```

Expected: all pass, including pre-existing tests.

- [ ] **Step 5: Add ARM/DISARM button to index.html**

In `src/quadguide/ground/static/index.html`, add the following CSS inside `<style>` after the `.hentry` rules:

```css
  .btn { background: #222; border: 1px solid #444; color: #ccc; font-family: monospace;
         font-size: 11px; padding: 3px 10px; cursor: pointer; }
  .btn:hover { background: #333; }
  .btn.armed  { border-color: #f44; color: #f44; }
```

In the `<div id="hud">` section, replace the `<!-- CROSSHAIR SIZE -->` block:

```html
  <!-- CROSSHAIR SIZE -->
  <div class="section">
    <div class="sec-title">CROSSHAIR</div>
    <div class="row"><span class="lbl">size</span><span class="val" id="hud-size">160 px</span></div>
  </div>

  <!-- ARM CONTROL -->
  <div class="section">
    <div class="sec-title">ARM CONTROL</div>
    <div class="row"><span class="lbl">status</span><span class="val dim" id="h-arm-state">DISARMED</span></div>
    <div class="row" style="margin-top:5px;gap:6px;">
      <button class="btn" id="btn-arm">ARM</button>
      <button class="btn" id="btn-disarm">DISARM</button>
    </div>
  </div>
```

In the `<script>` block, add before the closing `</script>`:

```javascript
// ── Arm control ────────────────────────────────────────────────────────────

let _armed = false;

function sendArm(armed) {
  fetch('/arm', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({armed}),
  }).then(r => r.json()).then(() => {
    _armed = armed;
    const el = document.getElementById('h-arm-state');
    el.textContent = armed ? 'ARMED' : 'DISARMED';
    el.className = 'val ' + (armed ? 'danger' : 'dim');
    document.getElementById('btn-arm').classList.toggle('armed', armed);
  }).catch(() => {});
}

document.getElementById('btn-arm').addEventListener('click', () => sendArm(true));
document.getElementById('btn-disarm').addEventListener('click', () => sendArm(false));
```

Also add `a` / `d` key shortcuts in the existing `document.addEventListener('keydown', ...)` handler, inside the if-chain:

```javascript
  } else if (e.key === 'a') {
    sendArm(true);
  } else if (e.key === 'd') {
    sendArm(false);
  }
```

Update the controls hint line:

```html
<div id="controls">[ + ] grow &nbsp; [ - ] shrink &nbsp; [ Enter ] lock on &nbsp; [ Esc ] cancel &nbsp; [ a ] arm &nbsp; [ d ] disarm</div>
```

- [ ] **Step 6: Run all ground server tests**

```bash
pytest tests/unit/test_ground_server.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/quadguide/ground/server.py src/quadguide/ground/static/index.html tests/unit/test_ground_server.py
git commit -m "feat(ground): add /arm endpoint and ARM/DISARM UI button"
```

---

## Task 10: scripts/test_link_rx.py

**Files:**
- Create: `scripts/test_link_rx.py`

- [ ] **Step 1: Implement the script**

Create `scripts/test_link_rx.py`:

```python
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
```

- [ ] **Step 2: Verify it imports correctly**

```bash
python scripts/test_link_rx.py --help
```

Expected: prints usage without error.

- [ ] **Step 3: Commit**

```bash
git add scripts/test_link_rx.py
git commit -m "scripts: add test_link_rx.py CRSF attitude monitor"
```

---

## Task 11: scripts/test_link_tx.py

**Files:**
- Create: `scripts/test_link_tx.py`

- [ ] **Step 1: Implement the script**

Create `scripts/test_link_tx.py`:

```python
#!/usr/bin/env python3
"""CRSF RC uplink transmitter test.

Sends CRSF RC_CHANNELS_PACKED frames at a fixed rate. Use this to verify the
companion → FC uplink is working. Once a steady uplink is established, the FC
exits failsafe and begins sending attitude telemetry back (visible in test_link_rx.py).

Usage:
    python scripts/test_link_tx.py --port /dev/ttyS0 [--baud 420000] [--rate 50]
        [--arm] [--roll 992] [--pitch 992] [--throttle 172] [--yaw 992]

CH5 is the arm channel. Use --arm to set it high (1811 = armed).
Channel values are in CRSF ticks: 172 (1000µs) – 992 (1500µs) – 1811 (2000µs).
"""
import argparse
import sys
import time

import serial

sys.path.insert(0, "src")

from quadguide.link.crsf import build_frame, pack_channels, CRSF_RC_CHANNELS


def main():
    parser = argparse.ArgumentParser(description="CRSF RC uplink test transmitter")
    parser.add_argument("--port",     default="/dev/ttyS0")
    parser.add_argument("--baud",     type=int,   default=420000)
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
```

- [ ] **Step 2: Verify it imports correctly**

```bash
python scripts/test_link_tx.py --help
```

Expected: prints usage without error.

- [ ] **Step 3: Commit**

```bash
git add scripts/test_link_tx.py
git commit -m "scripts: add test_link_tx.py CRSF RC uplink test transmitter"
```

---

## Task 12: Remove msp.py and update ARCHITECTURE.md

**Files:**
- Delete: `src/quadguide/link/msp.py`
- Delete: `tests/unit/test_msp.py`
- Modify: `ARCHITECTURE.md`

- [ ] **Step 1: Delete msp.py and its test**

```bash
git rm src/quadguide/link/msp.py tests/unit/test_msp.py
```

- [ ] **Step 2: Verify no remaining MSP imports**

```bash
grep -r "from quadguide.link.msp\|import msp\|MSP" src/ --include="*.py"
```

Expected: no output. If any files import from msp, update them to use the new crsf/espfc modules.

- [ ] **Step 3: Run the full unit test suite to confirm nothing broke**

```bash
pytest tests/unit/ -v
```

Expected: all pass (msp tests removed, new tests present).

- [ ] **Step 4: Update ARCHITECTURE.md**

In `ARCHITECTURE.md`, find Section 1 "Project Overview" and update the description line:

Change:
```
sends roll/pitch setpoints to an ESP-FC flight controller
over UART using the MSP v2 protocol.
```

To:
```
sends roll/pitch setpoints to an ESP-FC flight controller
over UART using the CRSF protocol (420000 baud, bidirectional).
```

In the hardware stack diagram, update the link line:
```
  ESP-FC ──→ UART ──→
  link ──→ bus (attitude/body-rates derived)
```

In Section 4.1, replace the `[link worker]` block:

Change:
```
[link worker]                               [ground worker]
  rx loop:                                    subscribe all topics
    parse MSP frames from UART                serve web UI on :8080
    bus.publish("fc/attitude", att)           handle POST /lockon
    bus.publish("fc/imu", imu)                  → bus.publish("lockon/cmd", cmd)
  tx loop:                                    stream annotated MJPEG
    cmd = bus.latest("control/cmd")
    write MSP_SET_RAW_RC to UART
```

To:
```
[link worker]                               [ground worker]
  rx loop:                                    subscribe all topics
    parse CRSF frames from UART (420kbaud)    serve web UI on :8080
    decode ATTITUDE (0x1E)                    handle POST /lockon
    differentiate angles → body rates           → bus.publish("lockon/cmd", cmd)
    bus.publish("fc/attitude", att)           handle POST /arm
    bus.publish("fc/imu", imu)                  → bus.publish("arm/cmd", cmd)
  tx loop (50 Hz, starts immediately):       stream annotated MJPEG
    cmd     = bus.latest("control/cmd")
    arm_cmd = bus.latest("arm/cmd")
    write CRSF RC_CHANNELS_PACKED to UART
    (FC enters failsafe if uplink stops)
```

In Section 6.5 `link/`, replace the three file descriptions for `msp.py`, `espfc.py`, `serial_port.py` with the CRSF versions:

**`link/crsf.py`**
CRSF protocol implementation. No bus or serial dependencies.
- `CRSF_SYNC = 0xC8`, `CRSF_ATTITUDE = 0x1E`, `CRSF_RC_CHANNELS = 0x16`
- `crc8(data: bytes) → int` — CRC8 with poly 0xD5, precomputed lookup table
- `CRSFFrame` dataclass: `type`, `payload`, `timestamp_ns`
- `CRSFParser` — stateful byte-by-byte parser; states: WAIT_SYNC → READ_LEN → READ_TYPE → READ_PAYLOAD → READ_CRC; resets on bad length or CRC mismatch
- `build_frame(type, payload) → bytes` — `[0xC8][len][type][payload][crc]`
- `pack_channels(channels: list[int]) → bytes` — packs 16 × 11-bit values into 22 bytes, LSB-first

**`link/differentiator.py`**
`AttitudeDifferentiator(alpha: float)` — finite-difference body rate estimator with per-axis first-order LP filter. Yaw uses shortest-path angular difference to handle ±180° wrap. `alpha=1.0` = no filtering, `alpha→0` = heavy smoothing.

**`link/espfc.py`**
ESP-FC specific encoding/decoding.
- `decode_attitude(frame, diff) → (AttitudeState, IMUFrame)` — unpacks int16 pitch/roll/yaw (units: 100 µrad), calls differentiator for body rates, returns `AttitudeState` with angles+rates and `IMUFrame` with gx/gy/gz from diff, ax=ay=az=0
- `encode_rc(cmd, armed) → bytes` — maps ControlCmd to CRSF RC_CHANNELS_PACKED; CH1=roll, CH2=pitch, CH3=throttle, CH4=yaw, CH5=arm (1811 armed / 172 disarmed), CH6–16=992
- `us_to_ticks(us) → int`, `ticks_to_us(ticks) → float` — standard CRSF conversion

**`link/serial_port.py`**
Async UART wrapper. `open()` initialises pyserial in a thread executor. `read_stream()` is an async generator yielding one byte at a time; raises `ConnectionError` on serial disconnect. `write(data)` is non-blocking; silently drops on disconnect. `close()` shuts down pyserial.

In Section 7 IPC table, add the `arm/cmd` row:

```
| `arm/cmd`         | ArmCmd                | ground worker    | link worker                          | event-driven     |
```

Also update the `fc/attitude` and `fc/imu` descriptions to note that body rates are derived:

```
| `fc/attitude`     | AttitudeState         | link worker      | guidance, control (watchdog), ground | 50–100 Hz        |
| `fc/imu`          | IMUFrame              | link worker      | ground                               | 50–100 Hz (gx/gy/gz derived; ax=ay=az=0) |
```

Update the link protocol references in Section 11 Known Constraints:

Change:
```
**MSP_SET_RAW_RC rate** is capped at ~100Hz by ESP-FC. The control loop runs
at 100Hz to match. Faster commands will queue in the serial buffer.
```

To:
```
**CRSF uplink rate** defaults to 50 Hz (configurable via `link.tx_rate_hz`). The uplink
must be continuous — if it stops, the FC enters failsafe. Body rates in `fc/attitude`
and `fc/imu` are finite-difference approximations of Euler angles, not raw gyro data.
Raw gyro accuracy requires a direct IMU connection to the companion computer.
```

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/unit/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "docs(arch): update link module section from MSP to CRSF"
```

Then commit the deletions:

```bash
git commit -m "chore(link): remove msp.py (replaced by crsf.py)"
```

---

## Final Verification

- [ ] Run the complete unit test suite:

```bash
pytest tests/unit/ -v --tb=short
```

Expected: all tests pass, no skipped.

- [ ] Confirm all new link module files are importable:

```bash
python -c "
from quadguide.link.crsf import CRSFParser, build_frame, pack_channels
from quadguide.link.differentiator import AttitudeDifferentiator
from quadguide.link.espfc import decode_attitude, encode_rc
from quadguide.link.serial_port import SerialPort
from quadguide.link.worker import run
print('All link imports OK')
"
```

Expected: `All link imports OK`

- [ ] Verify test scripts print help:

```bash
python scripts/test_link_rx.py --help && python scripts/test_link_tx.py --help
```

Expected: both print usage.

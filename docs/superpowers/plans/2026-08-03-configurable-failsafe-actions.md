# Configurable Failsafe Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the shipped target-loss disarm into a per-condition failsafe model where target-loss and watchdog-staleness each select a configurable terminal action — `disarm` or an ArduPilot flight mode (LAND/ALTHOLD) — that latches and hands off to the FC.

**Architecture:** `control` is the safety authority + arbiter: it debounces each condition into a latch, arbitrates a single effective action, and publishes it on `failsafe/action`. `link` is the executor: `disarm` drives the existing `_ArmController`; `mode` drives a new `_ModeController` (DO_SET_MODE, retransmit-until-ACK) and suppresses the `SET_ATTITUDE_TARGET` stream. Both actions latch until the operator disarms (`arm/cmd` → False).

**Tech Stack:** Python 3, `pymavlink` (codec-only), `struct`-packed shared-memory bus, `pytest`. ArduPilot SITL over TCP for HIL.

## Global Constraints

- **Failsafe is opt-in per condition.** `failsafe:` is an optional YAML section (NOT in `_REQUIRED_SECTIONS`). Absent section or `enabled: false` → that condition off.
- **GPS-denied-safe mode allowlist:** `{LAND, ALTHOLD, STABILIZE}`. Any other mode name (RTL, LOITER, POSHOLD, …) is rejected at config load with a `ValueError`.
- **ArduCopter custom_mode numbers:** `STABILIZE=0, ALTHOLD=2, LAND=9` (the allowlisted three).
- **Arbitration precedence:** `DISARM` beats `MODE` (more conservative wins). Among latched `MODE` conditions, **target-loss beats watchdog**.
- **Latch clear gate:** a latch clears only when `arm/cmd` goes False (the operator's *commanded* disarm intent) — never on `fc/status`. No auto-resume; QuadGuide never auto-commands GUIDED_NOGPS.
- **Single-writer bus topics:** `control` is the sole writer of `failsafe/action`.
- **Platform:** the full stack runs on Linux only; the Windows dev box runs the `pytest` unit suite. The SITL HIL test (Task 7) runs on Linux/WSL or the SBC with ArduPilot SITL — never on Windows.
- **Legacy keys `disarm_on_lost` / `lost_hold_ms` are removed** and rejected at load (fail-fast — a silently-ignored failsafe is a hazard).
- TDD, DRY, YAGNI, frequent commits.

---

## File Structure

- `src/quadguide/core/config.py` — add `FailsafeAction`, `ConditionFailsafe`, restructured `FailsafeConfig`, `ARDUCOPTER_MODES`, `FAILSAFE_MODE_ALLOWLIST`, rewritten `cfg_failsafe`.
- `src/quadguide/core/messages.py` — `FailsafeActionWire` enum; `FailsafeCmd` carries `action` + `custom_mode`; `FMT_FAILSAFE_CMD` `!QB` → `!QBI`.
- `src/quadguide/core/bus.py` — rename topic `failsafe/disarm` → `failsafe/action`.
- `src/quadguide/link/fc.py` — new `encode_set_mode`.
- `src/quadguide/link/worker.py` — new `_ModeController`; `_tx_loop`/`_rx_loop` wire DISARM vs SET_MODE + attitude-stream suppression.
- `src/quadguide/control/failsafe.py` — `LostDisarmLatch` → generic `FailsafeLatch`; new pure `arbitrate_failsafe`.
- `src/quadguide/control/worker.py` — two latches, arbitration, publish `failsafe/action`, keep soft-LEVEL leveling.
- `src/quadguide/core/health.py` — add `FailsafeState.MODE`.
- `configs/rpi4b.yaml`, `configs/rk3588.yaml` — restructured `failsafe:` blocks.
- `scripts/test_failsafe_sitl.py`, `tests/hil/test_lost_target.py` — SITL validation.
- `docs/ARCHITECTURE.md` — failsafe section update.

**Note on intermediate state:** Task 2 changes the `FailsafeCmd` shape and topic name. `control/worker.py` and `link/worker.py` are not rewritten until Tasks 5 and 6, so between Task 2 and Task 6 the *running stack* is mid-refactor. The **unit suite stays green throughout** (no unit test runs those worker loops); full runtime correctness is restored by Task 6 and verified end-to-end by Task 7 (SITL).

---

### Task 1: Config layer — per-condition failsafe

**Files:**
- Modify: `src/quadguide/core/config.py` (replace `FailsafeConfig` + `cfg_failsafe`)
- Modify: `configs/rpi4b.yaml` (replace `failsafe:` block)
- Modify: `configs/rk3588.yaml` (replace `failsafe:` block)
- Test: `tests/unit/test_config.py`, `tests/unit/test_rpi4b_config.py`, `tests/unit/test_rk3588_config.py`

**Interfaces:**
- Produces: `FailsafeAction` (Enum: `DISARM="disarm"`, `MODE="mode"`); `ConditionFailsafe(enabled: bool, action: FailsafeAction, mode: str|None, custom_mode: int|None, hold_ms: int)`; `FailsafeConfig(target_loss: ConditionFailsafe, watchdog: ConditionFailsafe)`; `cfg_failsafe(d: dict) -> FailsafeConfig`; `ARDUCOPTER_MODES: dict[str,int]`; `FAILSAFE_MODE_ALLOWLIST: frozenset[str]`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_config.py`, update the import line (add `ConditionFailsafe`, `FailsafeAction`):

```python
from quadguide.core.config import (
    load_config,
    cfg_platform, cfg_airframe, cfg_tracker,
    cfg_guidance, cfg_watchdog, cfg_mission,
    cfg_logging, cfg_bus, cfg_diag, cfg_failsafe,
    BusConfig, DiagConfig, FailsafeConfig, ConditionFailsafe, FailsafeAction,
)
```

Replace the two existing `test_cfg_failsafe_*` methods (the ones asserting `disarm_on_lost`/`lost_hold_ms`) with:

```python
    def test_cfg_failsafe_defaults_when_section_absent(self):
        f = cfg_failsafe({})
        assert f.target_loss.enabled is False
        assert f.watchdog.enabled is False

    def test_cfg_failsafe_mode_action_resolves_custom_mode(self):
        d = {"failsafe": {"target_loss": {
            "enabled": True, "action": "mode", "mode": "LAND", "hold_ms": 400}}}
        f = cfg_failsafe(d)
        assert f.target_loss.enabled is True
        assert f.target_loss.action is FailsafeAction.MODE
        assert f.target_loss.mode == "LAND"
        assert f.target_loss.custom_mode == 9
        assert f.target_loss.hold_ms == 400

    def test_cfg_failsafe_disarm_action_has_no_mode(self):
        d = {"failsafe": {"target_loss": {"enabled": True, "action": "disarm"}}}
        f = cfg_failsafe(d)
        assert f.target_loss.action is FailsafeAction.DISARM
        assert f.target_loss.custom_mode is None

    def test_cfg_failsafe_rejects_gps_dependent_mode(self):
        d = {"failsafe": {"watchdog": {
            "enabled": True, "action": "mode", "mode": "RTL"}}}
        with pytest.raises(ValueError, match="RTL"):
            cfg_failsafe(d)

    def test_cfg_failsafe_mode_action_requires_mode_name(self):
        d = {"failsafe": {"target_loss": {"enabled": True, "action": "mode"}}}
        with pytest.raises(ValueError, match="requires a 'mode'"):
            cfg_failsafe(d)

    def test_cfg_failsafe_rejects_legacy_keys(self):
        d = {"failsafe": {"disarm_on_lost": True, "lost_hold_ms": 300}}
        with pytest.raises(ValueError, match="legacy"):
            cfg_failsafe(d)
```

In `tests/unit/test_rpi4b_config.py`, replace `test_rpi4b_target_loss_disarm_enabled` with:

```python
def test_rpi4b_failsafe_actions_land_on_loss_and_staleness():
    """rpi4b: both target-loss and watchdog failsafes LAND the aircraft."""
    from quadguide.core.config import cfg_failsafe, FailsafeAction
    config = load_config(str(CONFIG), {})
    f = cfg_failsafe(config)
    assert f.target_loss.enabled is True
    assert f.target_loss.action is FailsafeAction.MODE
    assert f.target_loss.mode == "LAND"
    assert f.target_loss.custom_mode == 9
    assert f.watchdog.enabled is True
    assert f.watchdog.action is FailsafeAction.MODE
    assert f.watchdog.custom_mode == 9
```

In `tests/unit/test_rk3588_config.py`, replace the failsafe test with:

```python
def test_rk3588_failsafe_disabled_for_parity():
    """rk3588 ships the per-condition failsafe structure but disabled by default."""
    from quadguide.core.config import cfg_failsafe
    config = load_config(str(CONFIG), {})
    f = cfg_failsafe(config)
    assert f.target_loss.enabled is False
    assert f.watchdog.enabled is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_config.py -k failsafe tests/unit/test_rpi4b_config.py tests/unit/test_rk3588_config.py -v`
Expected: FAIL (`ImportError` for `ConditionFailsafe`/`FailsafeAction`, and old `FailsafeConfig(disarm_on_lost=...)` gone).

- [ ] **Step 3: Implement the config types + loader**

In `src/quadguide/core/config.py`, add `from enum import Enum` near the top imports (after `from dataclasses import ...`). Replace the existing `FailsafeConfig` dataclass (the `disarm_on_lost`/`lost_hold_ms` one) with:

```python
class FailsafeAction(Enum):
    DISARM = "disarm"
    MODE   = "mode"


# ArduCopter flight-mode custom_mode numbers (subset).
ARDUCOPTER_MODES: dict[str, int] = {
    "STABILIZE": 0, "ACRO": 1, "ALTHOLD": 2, "AUTO": 3, "GUIDED": 4,
    "LOITER": 5, "RTL": 6, "CIRCLE": 7, "LAND": 9, "POSHOLD": 16,
    "BRAKE": 17, "GUIDED_NOGPS": 20, "SMART_RTL": 21,
}

# GPS-denied-safe failsafe targets. Everything else is rejected at config load.
FAILSAFE_MODE_ALLOWLIST: frozenset[str] = frozenset({"LAND", "ALTHOLD", "STABILIZE"})


@dataclass(frozen=True)
class ConditionFailsafe:
    enabled: bool = False
    action: FailsafeAction = FailsafeAction.DISARM
    mode: str | None = None          # friendly name when action=MODE
    custom_mode: int | None = None   # resolved ArduCopter number when action=MODE
    hold_ms: int = 300               # continuous trip before the latch engages


@dataclass(frozen=True)
class FailsafeConfig:
    target_loss: ConditionFailsafe
    watchdog: ConditionFailsafe
```

Replace the existing `cfg_failsafe` function with:

```python
_LEGACY_FAILSAFE_KEYS = ("disarm_on_lost", "lost_hold_ms")


def _condition_failsafe(raw: dict | None, default_hold_ms: int) -> ConditionFailsafe:
    raw = raw or {}
    action_str = str(raw.get("action", "disarm")).lower()
    try:
        action = FailsafeAction(action_str)
    except ValueError:
        raise ValueError(
            f"failsafe: unknown action {action_str!r}; expected 'disarm' or 'mode'"
        )
    mode = None
    custom_mode = None
    if action is FailsafeAction.MODE:
        raw_mode = raw.get("mode")
        if not raw_mode:
            raise ValueError("failsafe: action 'mode' requires a 'mode' name")
        mode = str(raw_mode).upper()
        if mode not in FAILSAFE_MODE_ALLOWLIST:
            raise ValueError(
                f"failsafe: mode {mode!r} is not a GPS-denied-safe failsafe mode "
                f"(allowed: {sorted(FAILSAFE_MODE_ALLOWLIST)})"
            )
        custom_mode = ARDUCOPTER_MODES[mode]
    return ConditionFailsafe(
        enabled=bool(raw.get("enabled", False)),
        action=action,
        mode=mode,
        custom_mode=custom_mode,
        hold_ms=int(raw.get("hold_ms", default_hold_ms)),
    )


def cfg_failsafe(d: dict) -> FailsafeConfig:
    """Per-condition failsafe actions. Absent section → both conditions off."""
    f = d.get("failsafe") or {}
    legacy = [k for k in _LEGACY_FAILSAFE_KEYS if k in f]
    if legacy:
        raise ValueError(
            f"failsafe: legacy key(s) {legacy} are no longer supported; use the "
            "per-condition structure (target_loss:/watchdog: each with enabled, "
            "action, mode, hold_ms)."
        )
    return FailsafeConfig(
        target_loss=_condition_failsafe(f.get("target_loss"), default_hold_ms=300),
        watchdog=_condition_failsafe(f.get("watchdog"), default_hold_ms=200),
    )
```

- [ ] **Step 4: Update the two YAML configs**

In `configs/rpi4b.yaml`, replace the entire `failsafe:` block with:

```yaml
failsafe:
  # Per-condition failsafe actions. Each condition selects a terminal action of
  # `disarm` (cut motors) or `mode` (command an ArduPilot flight mode and hand
  # off). Both LATCH until the operator disarms — cycle the ground arm switch
  # off→on to re-engage. QuadGuide does NOT auto-restore GUIDED_NOGPS; the
  # operator restores guided mode on the FC before re-arming.
  target_loss:
    # Tracker reports LOST (bare NanoTrack conf < score_lost) continuously for
    # hold_ms while armed → LAND. tracker_health is the confidence gate (see
    # tracker.params.score_lost); this only debounces + acts.
    enabled: true
    action: mode          # disarm | mode
    mode: LAND            # LAND | ALTHOLD | STABILIZE (GPS-denied-safe allowlist)
    hold_ms: 300          # continuous LOST before the latch trips
  watchdog:
    # Any watched telemetry/guidance topic stale past its watchdog timeout,
    # sustained for hold_ms while armed → LAND. Brief blips only level (roll/
    # pitch → 0) during the debounce; the terminal action needs sustained loss.
    enabled: true
    action: mode
    mode: LAND
    hold_ms: 200
```

In `configs/rk3588.yaml`, replace the entire `failsafe:` block with:

```yaml
failsafe:
  # Per-condition failsafe actions (see configs/rpi4b.yaml). Disabled on this
  # build for parity/docs; the RK3588 flight config typically runs acquire_track,
  # whose LOST is already heavily debounced. Enable per-condition if needed.
  target_loss:
    enabled: false
    action: mode
    mode: LAND
    hold_ms: 300
  watchdog:
    enabled: false
    action: mode
    mode: LAND
    hold_ms: 200
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_config.py tests/unit/test_rpi4b_config.py tests/unit/test_rk3588_config.py -v`
Expected: PASS (the failsafe tests; pre-existing unrelated camera assertions in `test_rpi4b_config.py` are out of scope — if one is already red on `main`, leave it).

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/core/config.py configs/rpi4b.yaml configs/rk3588.yaml \
        tests/unit/test_config.py tests/unit/test_rpi4b_config.py tests/unit/test_rk3588_config.py
git commit -m "feat: per-condition failsafe config (disarm|mode) with validated mode allowlist"
```

---

### Task 2: Message + bus topic — generalize `FailsafeCmd`

**Files:**
- Modify: `src/quadguide/core/messages.py`
- Modify: `src/quadguide/core/bus.py`
- Test: `tests/unit/test_messages.py`, `tests/unit/test_bus.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `FailsafeActionWire` (IntEnum: `NONE=0, DISARM=1, SET_MODE=2`); `FailsafeCmd(timestamp_ns: int, action: FailsafeActionWire, custom_mode: int = 0)` with `pack()`/`unpack()`; `FMT_FAILSAFE_CMD = "!QBI"` (13 bytes); bus topic name **`failsafe/action`**.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_messages.py`, add `FailsafeActionWire` to the messages import, then replace the entire `TestFailsafeCmd` class with:

```python
class TestFailsafeCmd:
    def test_format_size(self):
        assert struct.calcsize(FMT_FAILSAFE_CMD) == 13  # Q(8) + B(1) + I(4)

    def test_round_trip_disarm(self):
        msg = FailsafeCmd(timestamp_ns=1_000_000, action=FailsafeActionWire.DISARM)
        r = FailsafeCmd.unpack(msg.pack())
        assert r.timestamp_ns == 1_000_000
        assert r.action is FailsafeActionWire.DISARM
        assert r.custom_mode == 0

    def test_round_trip_set_mode_carries_custom_mode(self):
        msg = FailsafeCmd(timestamp_ns=2_000_000,
                          action=FailsafeActionWire.SET_MODE, custom_mode=9)
        r = FailsafeCmd.unpack(msg.pack())
        assert r.action is FailsafeActionWire.SET_MODE
        assert r.custom_mode == 9

    def test_round_trip_none(self):
        r = FailsafeCmd.unpack(
            FailsafeCmd(3_000_000, FailsafeActionWire.NONE).pack())
        assert r.action is FailsafeActionWire.NONE
```

In `tests/unit/test_bus.py` (line ~23), rename `"failsafe/disarm"` to `"failsafe/action"` in the expected-topics list.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_messages.py::TestFailsafeCmd tests/unit/test_bus.py -v`
Expected: FAIL (`ImportError` for `FailsafeActionWire`; size 9 ≠ 13; `failsafe/action` not a topic).

- [ ] **Step 3: Implement the message change**

In `src/quadguide/core/messages.py`:

Change the enum import (line ~3) to include `IntEnum`:

```python
from enum import Enum, IntEnum
```

Add `FailsafeActionWire` to `__all__` (in the enums group alongside `TrackerHealth`, `ProcessState`).

Add the enum near the other enums (after `ProcessState`):

```python
class FailsafeActionWire(IntEnum):
    """Wire encoding for the arbitrated failsafe action (control → link)."""
    NONE     = 0   # no failsafe active
    DISARM   = 1   # disarm the FC
    SET_MODE = 2   # DO_SET_MODE(custom_mode)
```

Change the format string + comment:

```python
FMT_FAILSAFE_CMD = "!QBI"
# Q(8) + action(B=1) + custom_mode(I=4) = 13 bytes
# Arbitrated failsafe action: control publishes, link executes. custom_mode is
# the ArduCopter mode number, meaningful only for SET_MODE (0 otherwise).
# Single-writer (control).
```

Replace the `FailsafeCmd` dataclass with:

```python
@dataclass(frozen=True)
class FailsafeCmd:
    """Arbitrated failsafe action. control publishes; link executes."""
    timestamp_ns: int
    action: FailsafeActionWire
    custom_mode: int = 0

    def pack(self) -> bytes:
        return _ST_FAILSAFE_CMD.pack(
            self.timestamp_ns, int(self.action), self.custom_mode)

    @classmethod
    def unpack(cls, data: bytes) -> FailsafeCmd:
        ts, action_b, custom_mode = _ST_FAILSAFE_CMD.unpack(data)
        return cls(timestamp_ns=ts,
                   action=FailsafeActionWire(action_b),
                   custom_mode=custom_mode)
```

- [ ] **Step 4: Rename the bus topic**

In `src/quadguide/core/bus.py`, in the `TOPICS` dict change the key:

```python
    "failsafe/action":      (FailsafeCmd,     FMT_FAILSAFE_CMD),
```

(The `FailsafeCmd, FMT_FAILSAFE_CMD` import at the top of `bus.py` is unchanged.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_messages.py tests/unit/test_bus.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/core/messages.py src/quadguide/core/bus.py \
        tests/unit/test_messages.py tests/unit/test_bus.py
git commit -m "feat: FailsafeCmd carries action+custom_mode; rename topic failsafe/action"
```

---

### Task 3: `encode_set_mode` — DO_SET_MODE encoder

**Files:**
- Modify: `src/quadguide/link/fc.py`
- Modify: `src/quadguide/link/worker.py` (import list only)
- Test: `tests/unit/test_fc.py`

**Interfaces:**
- Produces: `encode_set_mode(mav, custom_mode: int, target_sys: int, target_comp: int) -> bytes` — packs a `COMMAND_LONG`/`MAV_CMD_DO_SET_MODE` with `param1 = MAV_MODE_FLAG_CUSTOM_MODE_ENABLED`, `param2 = custom_mode`.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_fc.py`, add `encode_set_mode` to the `from quadguide.link.fc import (...)` block, then add:

```python
# ── encode_set_mode ──────────────────────────────────────────────────────────

def test_encode_set_mode_sets_custom_mode(mav):
    msg = _roundtrip(encode_set_mode(mav, 9, 1, 1))  # 9 = ArduCopter LAND
    assert msg.get_type() == "COMMAND_LONG"
    assert msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE
    assert msg.param1 == pytest.approx(
        float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED))
    assert msg.param2 == pytest.approx(9.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_fc.py::test_encode_set_mode_sets_custom_mode -v`
Expected: FAIL (`ImportError: cannot import name 'encode_set_mode'`)

- [ ] **Step 3: Implement the encoder**

In `src/quadguide/link/fc.py`, add after `encode_arm`:

```python
def encode_set_mode(mav, custom_mode: int, target_sys: int, target_comp: int) -> bytes:
    """COMMAND_LONG / MAV_CMD_DO_SET_MODE.

    param1 = MAV_MODE_FLAG_CUSTOM_MODE_ENABLED (interpret param2 as a custom
    mode), param2 = the ArduCopter custom_mode number (e.g. 9 = LAND).
    """
    msg = mav.command_long_encode(
        target_sys, target_comp,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
        float(custom_mode), 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    return msg.pack(mav)
```

In `src/quadguide/link/worker.py`, add `encode_set_mode` to the `from quadguide.link.fc import (...)` import block (so it is available in Task 6):

```python
from quadguide.link.fc import (
    decode_attitude, decode_heartbeat, decode_imu,
    encode_arm, encode_attitude_target, encode_heartbeat,
    encode_set_message_interval, encode_set_mode,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_fc.py -v`
Expected: PASS (all `test_fc.py` tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/fc.py src/quadguide/link/worker.py tests/unit/test_fc.py
git commit -m "feat: add encode_set_mode (DO_SET_MODE) MAVLink encoder"
```

---

### Task 4: `_ModeController` — edge-triggered DO_SET_MODE with retransmit-until-ACK

**Files:**
- Modify: `src/quadguide/link/worker.py` (add `_ModeController` class)
- Test: `tests/unit/test_link_worker.py`

**Interfaces:**
- Produces: `_ModeController(retry_count: int, resend_every_ticks: int)` with `on_mode_state(desired: int | None) -> int | None` and `on_ack(command: int, result: int) -> None`. `desired=None` means "no mode failsafe active" (clears state, returns None). A new/changed `desired` emits immediately, then re-emits every `resend_every_ticks` ticks up to `retry_count` times until an ACCEPTED `MAV_CMD_DO_SET_MODE` ACK.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_link_worker.py`, add `_ModeController` to the `from quadguide.link.worker import (...)` import, then add:

```python
# ── _ModeController ──────────────────────────────────────────────────────────

def test_mode_controller_silent_when_no_failsafe():
    mode = _ModeController(retry_count=3, resend_every_ticks=2)
    assert mode.on_mode_state(None) is None
    assert mode.on_mode_state(None) is None

def test_mode_controller_emits_on_new_desired_mode():
    mode = _ModeController(retry_count=3, resend_every_ticks=2)
    assert mode.on_mode_state(9) == 9

def test_mode_controller_resends_until_retries_exhausted():
    mode = _ModeController(retry_count=2, resend_every_ticks=2)
    assert mode.on_mode_state(9) == 9     # edge
    assert mode.on_mode_state(9) is None  # tick 1
    assert mode.on_mode_state(9) == 9     # tick 2 → resend (retries 2→1)
    assert mode.on_mode_state(9) is None  # tick 1
    assert mode.on_mode_state(9) == 9     # tick 2 → resend (retries 1→0)
    assert mode.on_mode_state(9) is None  # exhausted
    assert mode.on_mode_state(9) is None

def test_mode_controller_stops_after_ack():
    mode = _ModeController(retry_count=5, resend_every_ticks=2)
    assert mode.on_mode_state(9) == 9
    mode.on_ack(mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                mavutil.mavlink.MAV_RESULT_ACCEPTED)
    assert mode.on_mode_state(9) is None
    assert mode.on_mode_state(9) is None

def test_mode_controller_ignores_unrelated_ack():
    mode = _ModeController(retry_count=5, resend_every_ticks=1)
    mode.on_mode_state(9)
    mode.on_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                mavutil.mavlink.MAV_RESULT_ACCEPTED)
    assert mode.on_mode_state(9) == 9  # still pending → resends

def test_mode_controller_clears_when_failsafe_releases():
    mode = _ModeController(retry_count=5, resend_every_ticks=2)
    assert mode.on_mode_state(9) == 9
    assert mode.on_mode_state(None) is None   # failsafe released
    assert mode.on_mode_state(9) == 9         # re-trip re-emits
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_link_worker.py -k mode_controller -v`
Expected: FAIL (`ImportError: cannot import name '_ModeController'`)

- [ ] **Step 3: Implement `_ModeController`**

In `src/quadguide/link/worker.py`, add after the `_ArmController` class:

```python
class _ModeController:
    """Edge-triggered MAVLink DO_SET_MODE with bounded retransmits until ACK.

    Call `on_mode_state(desired)` once per TX tick with the desired custom_mode,
    or None when no mode failsafe is active. Returns the custom_mode to transmit
    this tick, or None to send nothing. On a new/changed desired mode it emits
    immediately, then re-emits every `resend_every_ticks` ticks up to
    `retry_count` times until `on_ack` confirms a DO_SET_MODE.
    """

    def __init__(self, retry_count: int, resend_every_ticks: int) -> None:
        self._desired: int | None = None
        self._acked: bool = True
        self._retries_left: int = 0
        self._ticks: int = 0
        self._retry_count = retry_count
        self._resend_every = resend_every_ticks

    def on_mode_state(self, desired: int | None) -> int | None:
        if desired is None:                  # no mode failsafe active
            self._desired = None
            self._acked = True
            return None
        if desired != self._desired:         # new/changed target mode → emit now
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
        if (command == mavutil.mavlink.MAV_CMD_DO_SET_MODE
                and result == mavutil.mavlink.MAV_RESULT_ACCEPTED):
            self._acked = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_link_worker.py -v`
Expected: PASS (all, including the six new `_ModeController` tests)

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/link/worker.py tests/unit/test_link_worker.py
git commit -m "feat: add _ModeController (DO_SET_MODE edge + retransmit-until-ACK)"
```

---

### Task 5: Control safety authority — generic latch + arbiter + worker wiring

**Files:**
- Modify: `src/quadguide/control/failsafe.py` (rename+generalize `LostDisarmLatch` → `FailsafeLatch`; add `arbitrate_failsafe`)
- Modify: `src/quadguide/core/health.py` (add `FailsafeState.MODE`)
- Modify: `src/quadguide/control/worker.py` (two latches, arbitration, publish `failsafe/action`)
- Test: `tests/unit/test_failsafe_latch.py` (rewrite)

**Interfaces:**
- Consumes: `ConditionFailsafe`, `FailsafeAction` (Task 1); `FailsafeCmd`, `FailsafeActionWire` (Task 2).
- Produces: `FailsafeLatch(enabled: bool, hold_ns: int)` with `update(now_ns: int, armed: bool, tripped: bool) -> bool`; `arbitrate_failsafe(tl_latched: bool, tl: ConditionFailsafe, wd_latched: bool, wd: ConditionFailsafe) -> tuple[FailsafeActionWire, int]`; `FailsafeState.MODE`.

- [ ] **Step 1: Rewrite the failing latch/arbiter tests**

Replace the entire contents of `tests/unit/test_failsafe_latch.py` with:

```python
from quadguide.control.failsafe import FailsafeLatch, arbitrate_failsafe
from quadguide.core.config import ConditionFailsafe, FailsafeAction
from quadguide.core.messages import FailsafeActionWire

MS = 1_000_000        # ns per ms
HOLD = 300 * MS


def _latch(enabled=True, hold_ns=HOLD):
    return FailsafeLatch(enabled=enabled, hold_ns=hold_ns)


# ── FailsafeLatch (generic debounce/latch on a `tripped` predicate) ──────────

def test_no_trip_before_hold():
    latch = _latch()
    assert latch.update(0, armed=True, tripped=True) is False
    assert latch.update(299 * MS, armed=True, tripped=True) is False


def test_trips_at_hold():
    latch = _latch()
    latch.update(0, armed=True, tripped=True)
    assert latch.update(300 * MS, armed=True, tripped=True) is True


def test_debounce_resets_on_non_tripped_tick():
    latch = _latch()
    latch.update(0, armed=True, tripped=True)
    latch.update(250 * MS, armed=True, tripped=False)   # resets debounce
    latch.update(300 * MS, armed=True, tripped=True)    # fresh run at 300ms
    assert latch.update(400 * MS, armed=True, tripped=True) is False  # 100ms in
    assert latch.update(600 * MS, armed=True, tripped=True) is True   # 300ms in


def test_latch_persists_through_recovery():
    latch = _latch()
    latch.update(0, armed=True, tripped=True)
    assert latch.update(300 * MS, armed=True, tripped=True) is True
    assert latch.update(400 * MS, armed=True, tripped=False) is True  # sticky


def test_cleared_by_operator_disarm():
    latch = _latch()
    latch.update(0, armed=True, tripped=True)
    assert latch.update(300 * MS, armed=True, tripped=True) is True
    assert latch.update(400 * MS, armed=False, tripped=True) is False  # disarm clears
    assert latch.update(500 * MS, armed=True, tripped=True) is False   # debounce reset


def test_disabled_never_trips():
    latch = _latch(enabled=False)
    latch.update(0, armed=True, tripped=True)
    assert latch.update(10_000 * MS, armed=True, tripped=True) is False


def test_not_armed_never_trips():
    latch = _latch()
    latch.update(0, armed=False, tripped=True)
    assert latch.update(300 * MS, armed=False, tripped=True) is False


# ── arbitrate_failsafe (precedence: DISARM > MODE; target-loss > watchdog) ────

_DISARM = ConditionFailsafe(enabled=True, action=FailsafeAction.DISARM)
_MODE_LAND = ConditionFailsafe(
    enabled=True, action=FailsafeAction.MODE, mode="LAND", custom_mode=9)
_MODE_ALTHOLD = ConditionFailsafe(
    enabled=True, action=FailsafeAction.MODE, mode="ALTHOLD", custom_mode=2)


def test_arbitrate_none_when_no_latch():
    action, mode = arbitrate_failsafe(False, _MODE_LAND, False, _MODE_LAND)
    assert action is FailsafeActionWire.NONE
    assert mode == 0


def test_arbitrate_mode_from_target_loss():
    action, mode = arbitrate_failsafe(True, _MODE_LAND, False, _MODE_LAND)
    assert action is FailsafeActionWire.SET_MODE
    assert mode == 9


def test_arbitrate_mode_from_watchdog_only():
    action, mode = arbitrate_failsafe(False, _MODE_LAND, True, _MODE_ALTHOLD)
    assert action is FailsafeActionWire.SET_MODE
    assert mode == 2


def test_arbitrate_disarm_beats_mode():
    action, mode = arbitrate_failsafe(True, _MODE_LAND, True, _DISARM)
    assert action is FailsafeActionWire.DISARM
    assert mode == 0


def test_arbitrate_target_loss_mode_wins_over_watchdog_mode():
    action, mode = arbitrate_failsafe(True, _MODE_LAND, True, _MODE_ALTHOLD)
    assert action is FailsafeActionWire.SET_MODE
    assert mode == 9  # target-loss precedence
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_failsafe_latch.py -v`
Expected: FAIL (`ImportError: cannot import name 'FailsafeLatch'` / `arbitrate_failsafe`)

- [ ] **Step 3: Rewrite `control/failsafe.py`**

Replace the entire contents of `src/quadguide/control/failsafe.py` with:

```python
from __future__ import annotations

from quadguide.core.config import ConditionFailsafe, FailsafeAction
from quadguide.core.messages import FailsafeActionWire

__all__ = ["FailsafeLatch", "arbitrate_failsafe"]


class FailsafeLatch:
    """Debounced trip -> latch machine (pure state machine, no bus/clock).

    The caller passes monotonic ``now_ns`` and a boolean ``tripped`` predicate
    each tick; ``update`` returns whether the latch is engaged. Semantics:

    * Only trips while ``armed`` (the operator's commanded arm intent, i.e.
      ``arm/cmd`` — never the FC's actual armed state).
    * Trips when ``tripped`` is True continuously for ``hold_ns``.
    * Any non-tripped tick resets the debounce.
    * Once latched, stays latched until ``armed`` goes False (operator disarm) —
      the manual re-engage gate.
    * Disabled -> always returns False.

    Action-agnostic: the trip predicate (``health == LOST`` for target-loss,
    ``any watched topic stale`` for the watchdog) and the resulting action are
    supplied by the control worker, not this class.
    """

    def __init__(self, enabled: bool, hold_ns: int) -> None:
        self._enabled = enabled
        self._hold_ns = hold_ns
        self._trip_since: int | None = None
        self._latched = False

    def update(self, now_ns: int, armed: bool, tripped: bool) -> bool:
        if not self._enabled:
            return False
        if not armed:                       # operator disarm clears latch + debounce
            self._latched = False
            self._trip_since = None
            return False
        if self._latched:                   # sticky until 'not armed' clears it above
            return True
        if tripped:
            if self._trip_since is None:
                self._trip_since = now_ns
            elif now_ns - self._trip_since >= self._hold_ns:
                self._latched = True
        else:
            self._trip_since = None          # any non-tripped tick resets the debounce
        return self._latched


def arbitrate_failsafe(
    tl_latched: bool, tl: ConditionFailsafe,
    wd_latched: bool, wd: ConditionFailsafe,
) -> tuple[FailsafeActionWire, int]:
    """Resolve one effective failsafe action from the two condition latches.

    Precedence: DISARM beats MODE (more conservative). Among latched MODE
    conditions, target-loss beats watchdog. Returns (action, custom_mode);
    custom_mode is 0 unless the action is SET_MODE.
    """
    active: list[ConditionFailsafe] = []
    if tl_latched:
        active.append(tl)          # target-loss first → precedence among modes
    if wd_latched:
        active.append(wd)
    if not active:
        return FailsafeActionWire.NONE, 0
    if any(c.action is FailsafeAction.DISARM for c in active):
        return FailsafeActionWire.DISARM, 0
    chosen = active[0]             # all MODE; first is target-loss if latched
    return FailsafeActionWire.SET_MODE, int(chosen.custom_mode or 0)
```

- [ ] **Step 4: Add `FailsafeState.MODE`**

In `src/quadguide/core/health.py`, extend the enum:

```python
class FailsafeState(Enum):
    NOMINAL  = "nominal"
    LEVEL    = "level"
    DISARMED = "disarmed"
    MODE     = "mode"        # handed off to an FC flight mode (LAND/ALTHOLD/…)
```

- [ ] **Step 5: Wire the control worker**

In `src/quadguide/control/worker.py`:

Update imports:

```python
from quadguide.control.failsafe import FailsafeLatch, arbitrate_failsafe
from quadguide.core.messages import (
    ControlCmd, FailsafeCmd, FailsafeActionWire, HealthReport, ProcessState,
)
from quadguide.core.messages import TrackerHealth
```

(Keep the other existing imports. `TrackerHealth` may already be imported transitively — add the explicit import if not present.)

Replace the single latch construction line

```python
    latch = LostDisarmLatch(fcfg.disarm_on_lost, fcfg.lost_hold_ms * 1_000_000)
```

with two latches:

```python
    tl_latch = FailsafeLatch(fcfg.target_loss.enabled,
                             fcfg.target_loss.hold_ms * 1_000_000)
    wd_latch = FailsafeLatch(fcfg.watchdog.enabled,
                             fcfg.watchdog.hold_ms * 1_000_000)
```

Replace the target-loss latch block (the section from `est = bus.latest("target/estimate")` through `bus.publish("failsafe/disarm", FailsafeCmd(now_ns, latched))`) with:

```python
        # Failsafe conditions → latch → arbitrated action. `fault is not None`
        # (from the staleness watchdog above) is the watchdog trip predicate; it
        # also drives the local leveling below, so brief blips only level while
        # the debounce runs — the terminal action needs sustained loss.
        stale = fault is not None
        est = bus.latest("target/estimate")
        health = est.tracker_health if est is not None else None
        tl_latched = tl_latch.update(now_ns, armed, health == TrackerHealth.LOST)
        wd_latched = wd_latch.update(now_ns, armed, stale)
        action, custom_mode = arbitrate_failsafe(
            tl_latched, fcfg.target_loss, wd_latched, fcfg.watchdog)
        any_latched = tl_latched or wd_latched
        effective_armed = armed and not any_latched
        if action == FailsafeActionWire.DISARM:
            state = FailsafeState.DISARMED
        elif action == FailsafeActionWire.SET_MODE:
            state = FailsafeState.MODE
        if any_latched and not latched_prev:
            log.warning("control: FAILSAFE latched — action=%s custom_mode=%d "
                        "(target_loss=%s watchdog=%s)",
                        action.name, custom_mode, tl_latched, wd_latched)
        elif latched_prev and not any_latched:
            log.info("control: failsafe cleared (operator disarm)")
        latched_prev = any_latched
        bus.publish("failsafe/action", FailsafeCmd(now_ns, action, custom_mode))
```

Update the health-report block near the end of the loop (the `if i % _HEALTH_EVERY == 0:` section) to use the new names — replace `latched` with `any_latched` and update the detail string:

```python
        if i % _HEALTH_EVERY == 0:
            proc_state = ProcessState.FAILSAFE if (in_failsafe or any_latched) else ProcessState.OK
            detail = str(fault) if fault is not None else (
                f"failsafe {action.name}" if any_latched else "")
            bus.publish(
                "system/health",
                HealthReport(monotonic_ns(), "control", proc_state, detail),
            )
            trace.state(monotonic_ns(), armed=armed, fire_active=fire_active,
                        in_failsafe=in_failsafe, disarm_latched=any_latched,
                        fault=detail, throttle=thr)
```

(The `throttle`/attitude gates already key on `effective_armed`, so they are unchanged. The existing `watchdog.check_all()` try/except that sets `fault`/soft-LEVEL stays exactly as-is — it runs *before* this block so `fault` is available.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_failsafe_latch.py -v && python -m pytest tests/unit -q`
Expected: PASS (new latch/arbiter tests, and the whole unit suite stays green — `control/worker.py` imports resolve; no unit test executes its loop).

- [ ] **Step 7: Commit**

```bash
git add src/quadguide/control/failsafe.py src/quadguide/core/health.py \
        src/quadguide/control/worker.py tests/unit/test_failsafe_latch.py
git commit -m "feat: control arbitrates per-condition failsafe (disarm|mode), latched"
```

---

### Task 6: Link executor — apply DISARM vs SET_MODE

**Files:**
- Modify: `src/quadguide/link/worker.py` (`_run_async`, `_tx_loop`, `_rx_loop`)
- Test: `tests/unit/test_link_worker.py` (update `_run_rx` helper + add a mode-ack rx test)

**Interfaces:**
- Consumes: `FailsafeCmd`/`FailsafeActionWire` on `failsafe/action` (Task 2); `_ModeController` (Task 4); `encode_set_mode` (Task 3).
- Produces: link executes the arbitrated action — DISARM via `_ArmController` (unchanged path), SET_MODE via `_ModeController` with the `SET_ATTITUDE_TARGET` stream suppressed.

- [ ] **Step 1: Update the failing rx test + helper, add a mode-ack test**

In `tests/unit/test_link_worker.py`, update the import to add `_ModeController` (already imported in Task 4) and `FailsafeActionWire`/`FailsafeCmd` if needed for the new test. The `_rx_loop` signature gains a `mode_ctrl` parameter, so update the `_run_rx` helper:

```python
def _run_rx(data: bytes):
    serial = _FakeSerial(data)
    bus = _FakeBus()
    state = _LinkState()
    arm = _ArmController(retry_count=5, resend_every_ticks=25)
    mode = _ModeController(retry_count=5, resend_every_ticks=25)
    log = logging.getLogger("test")
    mav = make_mav(1, 191)
    asyncio.run(_rx_loop(serial, mav, state, bus, arm, mode, log))
    return bus, state, arm
```

Also update `test_rx_command_ack_acks_pending_arm` (which calls `_rx_loop` directly) to pass a `_ModeController`:

```python
def test_rx_command_ack_acks_pending_arm():
    arm = _ArmController(retry_count=5, resend_every_ticks=25)
    mode = _ModeController(retry_count=5, resend_every_ticks=25)
    arm.on_arm_state(True)
    data = _enc(lambda m: m.command_ack_encode(
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        mavutil.mavlink.MAV_RESULT_ACCEPTED).pack(m))
    serial = _FakeSerial(data)
    bus = _FakeBus()
    state = _LinkState()
    log = logging.getLogger("test")
    mav = make_mav(1, 191)
    asyncio.run(_rx_loop(serial, mav, state, bus, arm, mode, log))
    assert arm.on_arm_state(True) is None
```

Add a new test proving a DO_SET_MODE ack routes to the mode controller:

```python
def test_rx_command_ack_acks_pending_mode():
    arm = _ArmController(retry_count=5, resend_every_ticks=25)
    mode = _ModeController(retry_count=5, resend_every_ticks=25)
    mode.on_mode_state(9)  # pending DO_SET_MODE(LAND)
    data = _enc(lambda m: m.command_ack_encode(
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        mavutil.mavlink.MAV_RESULT_ACCEPTED).pack(m))
    serial = _FakeSerial(data)
    bus = _FakeBus()
    state = _LinkState()
    log = logging.getLogger("test")
    mav = make_mav(1, 191)
    asyncio.run(_rx_loop(serial, mav, state, bus, arm, mode, log))
    assert mode.on_mode_state(9) is None  # acked → nothing more to send
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_link_worker.py -k "rx" -v`
Expected: FAIL (`_rx_loop()` got an unexpected/too-many args; `_rx_loop` still takes the old signature).

- [ ] **Step 3: Wire `_rx_loop` to route mode ACKs**

In `src/quadguide/link/worker.py`, change the `_rx_loop` signature and COMMAND_ACK handling:

```python
async def _rx_loop(serial, mav, state: _LinkState, bus,
                   arm_ctrl: _ArmController, mode_ctrl: _ModeController,
                   log: logging.Logger) -> None:
```

and in its `COMMAND_ACK` branch:

```python
        elif t == "COMMAND_ACK":
            arm_ctrl.on_ack(msg.command, msg.result)
            mode_ctrl.on_ack(msg.command, msg.result)
```

(Each controller ignores commands that aren't its own, so calling both is safe.)

- [ ] **Step 4: Wire `_tx_loop` to apply the action**

In `src/quadguide/link/worker.py`, change the `_tx_loop` signature to accept `mode_ctrl`:

```python
async def _tx_loop(serial, mav, bus, state: _LinkState, arm_ctrl: _ArmController,
                   mode_ctrl: _ModeController, lc: dict, log: logging.Logger, trace) -> None:
```

Replace the body of the `while True:` loop (the section that reads `failsafe/disarm`, computes `effective`, sends arm, latches yaw, and sends the attitude target) with:

```python
    while True:
        cmd = bus.latest("control/cmd")
        arm_cmd = bus.latest("arm/cmd")
        fs = bus.latest("failsafe/action")
        armed = bool(arm_cmd and arm_cmd.armed)
        action = fs.action if fs else FailsafeActionWire.NONE
        custom_mode = fs.custom_mode if fs else 0
        disarm = action == FailsafeActionWire.DISARM
        mode_active = action == FailsafeActionWire.SET_MODE
        effective = armed and not disarm

        now = monotonic_ns()
        tsys = state.target_system or lc["target_system"]
        tcomp = state.target_component or lc["target_component"]

        # DISARM action: drive the arm controller False (existing path).
        to_send = arm_ctrl.on_arm_state(effective)
        if to_send is not None and state.have_heartbeat:
            await serial.write(encode_arm(mav, to_send, tsys, tcomp))
            reason = " (target-loss failsafe)" if (not to_send and disarm) else ""
            log.info("arm command → %s%s", "ARM" if to_send else "DISARM", reason)

        # MODE action: command DO_SET_MODE and suppress the attitude stream.
        mode_to_send = mode_ctrl.on_mode_state(custom_mode if mode_active else None)
        if mode_to_send is not None and state.have_heartbeat:
            await serial.write(encode_set_mode(mav, mode_to_send, tsys, tcomp))
            log.info("mode command → custom_mode=%d (failsafe handoff)", mode_to_send)

        yaw_hold = latch_yaw(effective, prev_armed, state.last_yaw, yaw_hold)
        prev_armed = effective

        # Stream SET_ATTITUDE_TARGET unless a mode failsafe has handed off to the FC.
        if not mode_active:
            await serial.write(encode_attitude_target(
                mav, cmd, yaw_hold, tsys, tcomp,
                lc["max_roll_deg"], lc["max_pitch_deg"], now // _NS_PER_MS))
            if cmd is not None and cmd.origin_ns > 0:
                trace.latency(now, cmd.timestamp_ns, cmd.origin_ns)
        await asyncio.sleep(interval)
```

Add `FailsafeActionWire` to the messages import at the top of `worker.py`:

```python
from quadguide.core.messages import FCStatus, HealthReport, ProcessState, FailsafeActionWire
```

- [ ] **Step 5: Build and pass `_ModeController` into the loops**

In `_run_async`, where `arm_ctrl` is constructed, add the mode controller and pass both into the tasks:

```python
        arm_ctrl = _ArmController(
            lc["arm_retry_count"], max(1, int(lc["tx_rate_hz"] // 2)))
        mode_ctrl = _ModeController(
            lc["arm_retry_count"], max(1, int(lc["tx_rate_hz"] // 2)))
```

and update the two `create_task` lines:

```python
                asyncio.create_task(_rx_loop(serial, mav, state, bus, arm_ctrl, mode_ctrl, log)),
                asyncio.create_task(_tx_loop(serial, mav, bus, state, arm_ctrl, mode_ctrl, lc, log, trace)),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_link_worker.py -v && python -m pytest tests/unit -q`
Expected: PASS (all link-worker tests, and the full unit suite is green — the runtime path is now complete).

- [ ] **Step 7: Commit**

```bash
git add src/quadguide/link/worker.py tests/unit/test_link_worker.py
git commit -m "feat: link applies failsafe action — disarm or DO_SET_MODE (attitude stream suppressed)"
```

---

### Task 7: SITL validation — DO_SET_MODE against ArduPilot SITL

**Files:**
- Create: `scripts/test_failsafe_sitl.py`
- Create/replace: `tests/hil/test_lost_target.py` (currently empty)

**Interfaces:**
- Consumes: the link worker + bus; publishes `failsafe/action`; reads `fc/status`.
- Produces: a runnable SITL driver and a guarded pytest asserting the FC enters the commanded mode.

**Runs on Linux/WSL or the SBC with ArduPilot SITL only — never on Windows.** Start SITL first, e.g.:
`sim_vehicle.py -v ArduCopter --out=tcp:127.0.0.1:5760` (or your MAVProxy TCP out).

- [ ] **Step 1: Write the SITL driver script**

Create `scripts/test_failsafe_sitl.py`:

```python
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
```

- [ ] **Step 2: Write the guarded HIL pytest**

Replace the (empty) `tests/hil/test_lost_target.py` with:

```python
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
```

- [ ] **Step 3: Run against SITL (Linux/WSL/SBC)**

Start SITL, then run:

```bash
QUADGUIDE_SITL=127.0.0.1:5760 python scripts/test_failsafe_sitl.py
QUADGUIDE_SITL=127.0.0.1:5760 python -m pytest tests/hil/test_lost_target.py -v
```

Expected: script prints `LAND … OK`, `STABILIZE … OK`, `ALTHOLD … OK` and `PASS`; pytest passes. On Windows (no `QUADGUIDE_SITL`) the pytest is **skipped** (`python -m pytest tests/hil/test_lost_target.py -v` → 1 skipped), which is the expected local result.

- [ ] **Step 4: Commit**

```bash
git add scripts/test_failsafe_sitl.py tests/hil/test_lost_target.py
git commit -m "test: SITL validation of failsafe DO_SET_MODE command exchange"
```

---

### Task 8: Docs — update ARCHITECTURE.md

**Files:**
- Modify: `docs/ARCHITECTURE.md` (the failsafe section)

- [ ] **Step 1: Locate the current failsafe section**

Run: `grep -n -i "target-loss\|disarm\|failsafe" docs/ARCHITECTURE.md`
Read the surrounding section that describes the shipped target-loss disarm.

- [ ] **Step 2: Rewrite the failsafe section**

Replace the target-loss-disarm description with a per-condition-action description covering: the two conditions (target-loss, watchdog staleness); the two actions (`disarm`, `mode: <NAME>` with the GPS-denied allowlist LAND/ALTHOLD/STABILIZE); latch + hand-off semantics and the re-engage procedure (operator disarm clears; operator restores GUIDED_NOGPS + re-arms; no auto-resume); the arbitration precedence (DISARM > MODE, target-loss > watchdog); and the signal path `control (arbiter) → failsafe/action → link (executor: _ArmController | _ModeController, attitude stream suppressed on mode)`. Keep the prose consistent with the design spec `docs/superpowers/specs/2026-08-03-configurable-failsafe-actions-design.md`.

- [ ] **Step 3: Verify no stale references remain**

Run: `grep -n -i "failsafe/disarm\|disarm_on_lost\|lost_hold_ms\|LostDisarmLatch" docs/ARCHITECTURE.md`
Expected: no matches (all renamed to the new topic/config/class names).

- [ ] **Step 4: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: describe configurable per-condition failsafe actions in ARCHITECTURE.md"
```

---

## Final verification

- [ ] Run the full unit suite: `python -m pytest tests/unit -q` → all pass.
- [ ] Confirm no stale identifiers remain in source: `grep -rn "failsafe/disarm\|disarm_on_lost\|lost_hold_ms\|LostDisarmLatch" src/ configs/` → no matches.
- [ ] On a Linux/WSL/SBC host with ArduCopter SITL: `QUADGUIDE_SITL=127.0.0.1:5760 python scripts/test_failsafe_sitl.py` → PASS (LAND/STABILIZE/ALTHOLD all reached).

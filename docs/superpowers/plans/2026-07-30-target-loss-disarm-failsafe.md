# Target-Loss Disarm Failsafe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Disarm the FC over MAVLink when the tracker reports `LOST` (NanoTrack confidence below `score_lost`) continuously for a debounce window, latching until the operator re-arms.

**Architecture:** The control worker (safety authority) debounces `target/estimate.tracker_health == LOST` into a latch and publishes it on a new single-writer `failsafe/disarm` topic. The link worker (sole FC arm authority) reads that latch and commands DISARM via `effective_armed = arm/cmd AND NOT failsafe/disarm`, reusing the existing edge-triggered `_ArmController`. No existing wire format changes.

**Tech Stack:** Python 3.11, `struct`-packed shared-memory bus, `pymavlink`, `pytest`.

**Design spec:** `docs/superpowers/specs/2026-07-30-target-loss-disarm-failsafe-design.md`

## Global Constraints

- **Signal is `tracker_health`, not raw confidence.** The confidence gate already lives in EdgeCV (`nanotrack._status_from`, thresholds `score_lock`/`score_lost` set from `tracker.params`). QuadGuide only debounces + disarms.
- **Every bus topic stays single-writer.** control writes `failsafe/disarm`; ground keeps writing `arm/cmd`; link arbitrates. Do not add a second writer to `arm/cmd`.
- **Latch keys on `arm/cmd` (operator intent), never `fc/status`.** The FC's actual armed state goes False when we disarm; keying on that would instantly re-arm.
- **`failsafe:` config is optional and defaults OFF.** Read via `d.get("failsafe", …)`; do NOT add it to `_REQUIRED_SECTIONS` — existing configs must keep loading.
- **Debounce default `lost_hold_ms = 300`.**
- **Platform test split (this repo runs on Windows dev + Linux/Pi target):** pure-logic tests (`test_messages.py`, `test_config.py`, `test_failsafe_latch.py`, the per-board config tests) import no `fcntl` and run on Windows. `test_bus.py` imports the real `Bus` (`fcntl`) and runs only on Linux/Pi. Worker wiring (control, link) has no test harness yet (all `tests/hil/*` and `tests/integration/test_control_pipeline.py` are empty stubs) and is verified by an on-device bench check.
- **Do not "run the whole suite and expect green":** `tests/unit/test_rpi4b_config.py::test_rpi4b_is_usb_camera_flight_default` is already red (asserts `v4l2`/`720`; committed config is `gstreamer`/`800`) — pre-existing, out of scope. Run the *specific* test files named in each task.
- **Branch first:** this work happens on a feature branch (e.g. `feat/target-loss-disarm`), not `main`. Commit per task.

---

### Task 1: `FailsafeCmd` message + `failsafe/disarm` topic

**Files:**
- Modify: `src/quadguide/core/messages.py`
- Modify: `src/quadguide/core/bus.py`
- Test: `tests/unit/test_messages.py` (add), `tests/unit/test_bus.py` (update)

**Interfaces:**
- Produces: `FailsafeCmd(timestamp_ns: int, disarm: bool)` with `.pack() -> bytes` and `.unpack(bytes) -> FailsafeCmd`; module constant `FMT_FAILSAFE_CMD = "!QB"`; bus topic `"failsafe/disarm"` mapping to `(FailsafeCmd, FMT_FAILSAFE_CMD)`.

- [ ] **Step 1: Write the failing message test**

Add to the imports at the top of `tests/unit/test_messages.py`: `FailsafeCmd` and `FMT_FAILSAFE_CMD`. Then append this class:

```python
class TestFailsafeCmd:
    def test_format_size(self):
        assert struct.calcsize(FMT_FAILSAFE_CMD) == 9  # Q(8) + B(1)

    def test_round_trip_disarm_true(self):
        msg = FailsafeCmd(timestamp_ns=1_000_000, disarm=True)
        r = FailsafeCmd.unpack(msg.pack())
        assert r.timestamp_ns == 1_000_000
        assert r.disarm is True

    def test_round_trip_disarm_false(self):
        msg = FailsafeCmd(timestamp_ns=2_000_000, disarm=False)
        r = FailsafeCmd.unpack(msg.pack())
        assert r.disarm is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/unit/test_messages.py::TestFailsafeCmd -v`
Expected: FAIL — `ImportError: cannot import name 'FailsafeCmd'`.

- [ ] **Step 3: Implement the message**

In `src/quadguide/core/messages.py`:

Add `"FailsafeCmd"` and `"FMT_FAILSAFE_CMD"` to `__all__`.

After the `FMT_FC_STATUS` block, add:

```python
FMT_FAILSAFE_CMD = "!QB"
# Q(8) + disarm(B=1) = 9 bytes
# Latching target-loss disarm signal: control publishes, link arbitrates
# (effective_armed = arm/cmd AND NOT failsafe/disarm). Single-writer (control).
```

After `_ST_FC_STATUS = struct.Struct(FMT_FC_STATUS)`, add:

```python
_ST_FAILSAFE_CMD = struct.Struct(FMT_FAILSAFE_CMD)
```

At the end of the file, add:

```python
@dataclass(frozen=True)
class FailsafeCmd:
    """Latching target-loss disarm. control publishes; link arbitrates."""
    timestamp_ns: int
    disarm: bool

    def pack(self) -> bytes:
        return _ST_FAILSAFE_CMD.pack(self.timestamp_ns, int(self.disarm))

    @classmethod
    def unpack(cls, data: bytes) -> FailsafeCmd:
        ts, disarm_b = _ST_FAILSAFE_CMD.unpack(data)
        return cls(timestamp_ns=ts, disarm=bool(disarm_b))
```

- [ ] **Step 4: Run the message test to verify it passes**

Run: `python -m pytest tests/unit/test_messages.py::TestFailsafeCmd -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Register the topic on the bus**

In `src/quadguide/core/bus.py`, add `FailsafeCmd, FMT_FAILSAFE_CMD` to the `from quadguide.core.messages import (...)` block, and add this entry to the `TOPICS` dict (after `"fc/status"`):

```python
    "failsafe/disarm":      (FailsafeCmd,     FMT_FAILSAFE_CMD),
```

- [ ] **Step 6: Update the bus topic-registry test**

In `tests/unit/test_bus.py`, add `"failsafe/disarm"` to the `expected` set in `test_all_topics_registered`, and change `test_topics_constant_entry_count` from `== 10` to `== 11`.

- [ ] **Step 7: Run the bus test (Linux/Pi only)**

Run (on the Pi or a Linux CI — `test_bus.py` imports `fcntl`): `python -m pytest tests/unit/test_bus.py -v`
Expected: PASS. On Windows this step is skipped (import error is expected there and not a regression).

- [ ] **Step 8: Commit**

```bash
git add src/quadguide/core/messages.py src/quadguide/core/bus.py tests/unit/test_messages.py tests/unit/test_bus.py
git commit -m "feat: add FailsafeCmd message + failsafe/disarm bus topic"
```

---

### Task 2: `FailsafeConfig` + `cfg_failsafe`

**Files:**
- Modify: `src/quadguide/core/config.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `FailsafeConfig(disarm_on_lost: bool = False, lost_hold_ms: int = 300)` and `cfg_failsafe(d: dict) -> FailsafeConfig`.

- [ ] **Step 1: Write the failing config test**

In `tests/unit/test_config.py`, add `cfg_failsafe` and `FailsafeConfig` to the `from quadguide.core.config import (...)` block. Then add these two methods to `class TestAccessors`:

```python
    def test_cfg_failsafe_defaults_when_section_absent(self):
        assert cfg_failsafe({}) == FailsafeConfig(disarm_on_lost=False, lost_hold_ms=300)

    def test_cfg_failsafe_from_config(self):
        d = {"failsafe": {"disarm_on_lost": True, "lost_hold_ms": 500}}
        f = cfg_failsafe(d)
        assert f.disarm_on_lost is True
        assert f.lost_hold_ms == 500
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/unit/test_config.py::TestAccessors::test_cfg_failsafe_from_config tests/unit/test_config.py::TestAccessors::test_cfg_failsafe_defaults_when_section_absent -v`
Expected: FAIL — `ImportError: cannot import name 'cfg_failsafe'`.

- [ ] **Step 3: Implement the config type + accessor**

In `src/quadguide/core/config.py`, add the dataclass after `WatchdogConfig`:

```python
@dataclass(frozen=True)
class FailsafeConfig:
    disarm_on_lost: bool = False   # disarm the FC when the tracker reports LOST
    lost_hold_ms: int = 300        # continuous LOST required before disarm (debounce)
```

And add the accessor after `cfg_diag`:

```python
def cfg_failsafe(d: dict) -> FailsafeConfig:
    """Optional target-loss disarm failsafe. Absent section → feature off."""
    f = d.get("failsafe") or {}
    return FailsafeConfig(
        disarm_on_lost=f.get("disarm_on_lost", False),
        lost_hold_ms=f.get("lost_hold_ms", 300),
    )
```

Do NOT add `"failsafe"` to `_REQUIRED_SECTIONS`.

- [ ] **Step 4: Run the config test to verify it passes**

Run: `python -m pytest tests/unit/test_config.py -k failsafe -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/core/config.py tests/unit/test_config.py
git commit -m "feat: add optional FailsafeConfig (disarm_on_lost, lost_hold_ms)"
```

---

### Task 3: `LostDisarmLatch` (pure debounce/latch)

**Files:**
- Create: `src/quadguide/control/failsafe.py`
- Test: `tests/unit/test_failsafe_latch.py`

**Interfaces:**
- Consumes: `TrackerHealth` from `quadguide.core.messages`.
- Produces: `LostDisarmLatch(enabled: bool, hold_ns: int)` with `update(now_ns: int, armed: bool, health) -> bool`. `health` is a `TrackerHealth` or `None` (no estimate yet). Returns whether the disarm latch is engaged.

- [ ] **Step 1: Write the failing latch tests**

Create `tests/unit/test_failsafe_latch.py`:

```python
from quadguide.control.failsafe import LostDisarmLatch
from quadguide.core.messages import TrackerHealth

MS = 1_000_000        # ns per ms
HOLD = 300 * MS


def _latch(enabled=True, hold_ns=HOLD):
    return LostDisarmLatch(enabled=enabled, hold_ns=hold_ns)


def test_no_trip_before_hold():
    latch = _latch()
    assert latch.update(0, armed=True, health=TrackerHealth.LOST) is False
    assert latch.update(299 * MS, armed=True, health=TrackerHealth.LOST) is False


def test_trips_at_hold():
    latch = _latch()
    latch.update(0, armed=True, health=TrackerHealth.LOST)              # start debounce
    assert latch.update(300 * MS, armed=True, health=TrackerHealth.LOST) is True


def test_debounce_resets_on_non_lost_frame():
    latch = _latch()
    latch.update(0, armed=True, health=TrackerHealth.LOST)              # debounce starts at 0
    latch.update(250 * MS, armed=True, health=TrackerHealth.NOMINAL)    # non-LOST resets it
    latch.update(300 * MS, armed=True, health=TrackerHealth.LOST)       # new run starts at 300ms
    assert latch.update(400 * MS, armed=True, health=TrackerHealth.LOST) is False  # only 100ms in
    assert latch.update(600 * MS, armed=True, health=TrackerHealth.LOST) is True   # 300ms in → trip


def test_latch_persists_through_recovery():
    latch = _latch()
    latch.update(0, armed=True, health=TrackerHealth.LOST)
    assert latch.update(300 * MS, armed=True, health=TrackerHealth.LOST) is True
    assert latch.update(400 * MS, armed=True, health=TrackerHealth.NOMINAL) is True  # sticky


def test_cleared_by_operator_disarm():
    latch = _latch()
    latch.update(0, armed=True, health=TrackerHealth.LOST)
    assert latch.update(300 * MS, armed=True, health=TrackerHealth.LOST) is True
    assert latch.update(400 * MS, armed=False, health=TrackerHealth.LOST) is False   # disarm clears
    assert latch.update(500 * MS, armed=True, health=TrackerHealth.NOMINAL) is False  # clean slate


def test_disabled_never_trips():
    latch = _latch(enabled=False)
    latch.update(0, armed=True, health=TrackerHealth.LOST)
    assert latch.update(10_000 * MS, armed=True, health=TrackerHealth.LOST) is False


def test_not_armed_never_trips():
    latch = _latch()
    latch.update(0, armed=False, health=TrackerHealth.LOST)
    assert latch.update(300 * MS, armed=False, health=TrackerHealth.LOST) is False


def test_health_none_treated_as_not_lost():
    latch = _latch()
    latch.update(0, armed=True, health=TrackerHealth.LOST)
    latch.update(100 * MS, armed=True, health=None)                     # no estimate → resets
    assert latch.update(350 * MS, armed=True, health=TrackerHealth.LOST) is False  # fresh run
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_failsafe_latch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quadguide.control.failsafe'`.

- [ ] **Step 3: Implement the latch**

Create `src/quadguide/control/failsafe.py`:

```python
from __future__ import annotations

from quadguide.core.messages import TrackerHealth

__all__ = ["LostDisarmLatch"]


class LostDisarmLatch:
    """Debounced target-loss -> disarm latch (pure state machine, no bus/clock).

    The caller passes monotonic ``now_ns`` each tick; ``update`` returns whether
    the disarm latch is engaged. Semantics (design spec 2026-07-30):

    * Only trips while ``armed`` (the operator's commanded arm intent, i.e.
      ``arm/cmd`` — never the FC's actual armed state).
    * Trips when ``health == LOST`` continuously for ``hold_ns``.
    * Any non-LOST tick (including ``health is None``) resets the debounce.
    * Once latched, stays latched through health recovery; clears only when
      ``armed`` goes False (operator disarm) — the manual re-arm gate.
    * Disabled -> always returns False.
    """

    def __init__(self, enabled: bool, hold_ns: int) -> None:
        self._enabled = enabled
        self._hold_ns = hold_ns
        self._lost_since: int | None = None
        self._latched = False

    def update(self, now_ns: int, armed: bool, health) -> bool:
        if not self._enabled:
            return False
        if not armed:                       # operator disarm clears latch + debounce
            self._latched = False
            self._lost_since = None
            return False
        if self._latched:                   # sticky until 'not armed' clears it above
            return True
        if health == TrackerHealth.LOST:
            if self._lost_since is None:
                self._lost_since = now_ns
            elif now_ns - self._lost_since >= self._hold_ns:
                self._latched = True
        else:
            self._lost_since = None          # any non-LOST frame resets the debounce
        return self._latched
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_failsafe_latch.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/control/failsafe.py tests/unit/test_failsafe_latch.py
git commit -m "feat: add LostDisarmLatch debounce/latch for target-loss disarm"
```

---

### Task 4: Wire the latch into the control worker

**Files:**
- Modify: `src/quadguide/control/worker.py`

**Interfaces:**
- Consumes: `cfg_failsafe` (Task 2), `LostDisarmLatch` (Task 3), `FailsafeCmd` (Task 1).
- Produces: publishes `FailsafeCmd(now_ns, latched)` on `"failsafe/disarm"` every tick; applies `effective_armed = armed and not latched` to throttle + attitude gating.

> No new automated test — the control worker's `run()` loop has no harness (integration/HIL stubs are empty). Verification is the passing unit suite (Tasks 1–3) plus the on-device bench check in Task 7. Keep the change to pure wiring so all decision logic stays in the already-tested `LostDisarmLatch`.

- [ ] **Step 1: Add imports**

In `src/quadguide/control/worker.py`, change the config import line to include `cfg_failsafe`:

```python
from quadguide.core.config import cfg_airframe, cfg_guidance, cfg_platform, cfg_watchdog, cfg_failsafe
```

Change the messages import to include `FailsafeCmd`:

```python
from quadguide.core.messages import ControlCmd, FailsafeCmd, HealthReport, ProcessState
```

Add, next to the other `control.*` imports:

```python
from quadguide.control.failsafe import LostDisarmLatch
```

- [ ] **Step 2: Read config + build the latch**

After `wcfg = cfg_watchdog(config)` add:

```python
    fcfg = cfg_failsafe(config)
```

After `rate = RateLimiter(hz=100)` add:

```python
    latch = LostDisarmLatch(fcfg.disarm_on_lost, fcfg.lost_hold_ms * 1_000_000)
```

- [ ] **Step 3: Add the edge-tracking flag**

Change:

```python
    armed = False
    in_failsafe = False
    i = 0
```

to:

```python
    armed = False
    in_failsafe = False
    latched_prev = False
    i = 0
```

- [ ] **Step 4: Evaluate the latch (after the watchdog block)**

Immediately after the watchdog `except HealthFault as e:` block and before `accel = bus.latest("guidance/accel")`, insert:

```python
        # Target-loss disarm latch: debounce tracker_health==LOST -> disarm.
        # Independent of the staleness watchdog above; overrides state when engaged.
        est = bus.latest("target/estimate")
        health = est.tracker_health if est is not None else None
        latched = latch.update(now_ns, armed, health)
        effective_armed = armed and not latched
        if latched:
            state = FailsafeState.DISARMED
        if latched and not latched_prev:
            log.warning("control: TARGET-LOSS DISARM latched — health LOST > %d ms",
                        fcfg.lost_hold_ms)
        elif latched_prev and not latched:
            log.info("control: target-loss disarm cleared (operator re-arm)")
        latched_prev = latched
        bus.publish("failsafe/disarm", FailsafeCmd(now_ns, latched))
```

(`FailsafeState` is already imported in this file; `now_ns` is already computed at the top of the loop.)

- [ ] **Step 5: Gate throttle + attitude on `effective_armed`**

Change:

```python
        # Choose throttle: armed + fire active → throttle_hold; else 0
        thr = gcfg.throttle_hold if (armed and fire_active) else 0.0

        # Choose attitude: only apply guidance when armed, no failsafe, and accel present
        if armed and fault is None and accel is not None:
```

to:

```python
        # Choose throttle: effective-armed + fire active → throttle_hold; else 0
        thr = gcfg.throttle_hold if (effective_armed and fire_active) else 0.0

        # Choose attitude: only when effective-armed, no failsafe, and accel present
        if effective_armed and fault is None and accel is not None:
```

- [ ] **Step 6: Surface the latch in the health report**

Change:

```python
        if i % _HEALTH_EVERY == 0:
            proc_state = ProcessState.FAILSAFE if in_failsafe else ProcessState.OK
            detail = str(fault) if fault is not None else ""
            bus.publish(
                "system/health",
                HealthReport(monotonic_ns(), "control", proc_state, detail),
            )
            trace.state(monotonic_ns(), armed=armed, fire_active=fire_active,
                        in_failsafe=in_failsafe, fault=detail, throttle=thr)
```

to:

```python
        if i % _HEALTH_EVERY == 0:
            proc_state = ProcessState.FAILSAFE if (in_failsafe or latched) else ProcessState.OK
            detail = str(fault) if fault is not None else (
                "target-loss disarm" if latched else "")
            bus.publish(
                "system/health",
                HealthReport(monotonic_ns(), "control", proc_state, detail),
            )
            trace.state(monotonic_ns(), armed=armed, fire_active=fire_active,
                        in_failsafe=in_failsafe, disarm_latched=latched,
                        fault=detail, throttle=thr)
```

- [ ] **Step 7: Sanity-check the module parses (on the Pi/Linux)**

Run: `python -c "import quadguide.control.worker"`
Expected: no output, exit 0. (On Windows this may fail at `fcntl` via the bus import — that's expected; run this step on the Pi or Linux CI.)

- [ ] **Step 8: Re-run the unit suite that CAN run here**

Run: `python -m pytest tests/unit/test_failsafe_latch.py tests/unit/test_messages.py tests/unit/test_config.py -k "failsafe or FailsafeCmd or cfg_failsafe" -v`
Expected: PASS (no regressions in the pure units).

- [ ] **Step 9: Commit**

```bash
git add src/quadguide/control/worker.py
git commit -m "feat: control worker latches target-loss disarm, gates on effective_armed"
```

---

### Task 5: Arbitrate the disarm in the link worker

**Files:**
- Modify: `src/quadguide/link/worker.py`

**Interfaces:**
- Consumes: `failsafe/disarm` (`FailsafeCmd.disarm`) and `arm/cmd` (`ArmCmd.armed`).
- Produces: feeds `effective = armed and not latched` into the existing `_ArmController.on_arm_state`, so a latched failsafe drives DISARM (retransmit until ACK).

> Verified by the Task 7 bench check (no link-worker harness exists). Pure wiring over the tested `_ArmController`.

- [ ] **Step 1: Read the failsafe latch and arbitrate in `_tx_loop`**

In `src/quadguide/link/worker.py`, inside `_tx_loop`, change:

```python
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
```

to:

```python
        cmd = bus.latest("control/cmd")
        arm_cmd = bus.latest("arm/cmd")
        fs = bus.latest("failsafe/disarm")
        armed = bool(arm_cmd and arm_cmd.armed)
        latched = bool(fs and fs.disarm)
        effective = armed and not latched

        to_send = arm_ctrl.on_arm_state(effective)
        if to_send is not None and state.have_heartbeat:
            await serial.write(encode_arm(mav, to_send,
                                          state.target_system, state.target_component))
            reason = " (target-loss failsafe)" if (not to_send and latched) else ""
            log.info("arm command → %s%s", "ARM" if to_send else "DISARM", reason)

        yaw_hold = latch_yaw(effective, prev_armed, state.last_yaw, yaw_hold)
        prev_armed = effective
```

- [ ] **Step 2: Sanity-check the module parses (on the Pi/Linux)**

Run: `python -c "import quadguide.link.worker"`
Expected: no output, exit 0. (Requires `pymavlink`; run on the Pi or Linux CI.)

- [ ] **Step 3: Commit**

```bash
git add src/quadguide/link/worker.py
git commit -m "feat: link disarms FC on latched failsafe (effective_armed arbitration)"
```

---

### Task 6: Enable in configs

**Files:**
- Modify: `configs/rpi4b.yaml`
- Modify: `configs/rk3588.yaml`
- Test: `tests/unit/test_rpi4b_config.py` (add), `tests/unit/test_rk3588_config.py` (add)

- [ ] **Step 1: Write the failing per-board config tests**

Append to `tests/unit/test_rpi4b_config.py`:

```python
def test_rpi4b_target_loss_disarm_enabled():
    """rpi4b enables the target-loss disarm failsafe: LOST tracker_health disarms
    the FC after lost_hold_ms (bare NanoTrack has no hysteresis → QuadGuide debounces)."""
    from quadguide.core.config import cfg_failsafe
    config = load_config(str(CONFIG), {})
    f = cfg_failsafe(config)
    assert f.disarm_on_lost is True
    assert f.lost_hold_ms == 300
```

Append to `tests/unit/test_rk3588_config.py`:

```python
def test_rk3588_target_loss_disarm_present_disabled():
    """rk3588 ships the failsafe section for parity but leaves it off by default."""
    from quadguide.core.config import cfg_failsafe
    config = load_config(str(CONFIG), {})
    f = cfg_failsafe(config)
    assert f.disarm_on_lost is False
    assert f.lost_hold_ms == 300
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/unit/test_rpi4b_config.py::test_rpi4b_target_loss_disarm_enabled tests/unit/test_rk3588_config.py::test_rk3588_target_loss_disarm_present_disabled -v`
Expected: FAIL — `cfg_failsafe(config).disarm_on_lost` is `False` for rpi4b (no `failsafe:` section yet).

- [ ] **Step 3: Add the `failsafe:` section to `configs/rpi4b.yaml`**

Insert this top-level block immediately after the `watchdog:` section (before `link:`):

```yaml
failsafe:
  # Disarm the FC over MAVLink when the tracker reports LOST (bare NanoTrack conf <
  # score_lost) continuously for lost_hold_ms. tracker_health is already the
  # confidence gate (see tracker.params.score_lost); this only debounces + disarms.
  # Latches until the operator re-arms — cycle the ground arm switch off→on.
  disarm_on_lost: true
  lost_hold_ms: 300         # continuous LOST required before disarm (debounce)
```

Also update the comment previously added under `tracker.params`: change the phrase
`(see failsafe.lost_hold_ms, TBD)` to `(see the failsafe: section below)`.

- [ ] **Step 4: Add the `failsafe:` section to `configs/rk3588.yaml`**

Append a top-level `failsafe:` block (put it after the `watchdog:` section for locality):

```yaml
failsafe:
  # Target-loss disarm (see configs/rpi4b.yaml). Disabled on this build by default;
  # the RK3588 flight config typically runs acquire_track, whose LOST is already
  # heavily debounced. Enable if running bare nanotrack here.
  disarm_on_lost: false
  lost_hold_ms: 300
```

- [ ] **Step 5: Run the per-board config tests to verify they pass**

Run: `python -m pytest tests/unit/test_rpi4b_config.py::test_rpi4b_target_loss_disarm_enabled tests/unit/test_rk3588_config.py::test_rk3588_target_loss_disarm_present_disabled -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add configs/rpi4b.yaml configs/rk3588.yaml tests/unit/test_rpi4b_config.py tests/unit/test_rk3588_config.py
git commit -m "feat: enable target-loss disarm in rpi4b config (rk3588 parity, off)"
```

---

### Task 7: On-device bench verification + docs

**Files:**
- Modify: `ARCHITECTURE.md`

> This is the behavioral end-to-end. It runs on the Pi (or ArduPilot SITL via `serial.mode=tcp`), because the worker stack needs the Linux bus and a MAVLink endpoint. No automated harness exists to script it yet.

- [ ] **Step 1: Document the topic + failsafe in ARCHITECTURE.md**

Add a short subsection near the bus-topics / failsafe discussion:

```markdown
### Target-loss disarm failsafe

When `failsafe.disarm_on_lost` is set, the control worker debounces
`target/estimate.tracker_health == LOST` for `failsafe.lost_hold_ms` and, while
armed, publishes a latching `failsafe/disarm` (FailsafeCmd). The link worker
computes `effective_armed = arm/cmd AND NOT failsafe/disarm` and commands the FC
to DISARM via the existing arm path. The latch is sticky until the operator
re-arms (cycle the ground arm switch off→on). `tracker_health` is already the
NanoTrack confidence gate (`tracker.params.score_lock`/`score_lost`); this
feature only debounces and disarms.
```

- [ ] **Step 2: Bench run — arm, lock, confirm nominal flight is unaffected**

On the Pi (or SITL): start the stack with the rpi4b config, arm from the ground
UI, lock onto a target, and confirm the FC stays armed and tracking while
`tracker_health` is `nominal`/`uncertain`. Watch the link log — no unexpected
`arm command → DISARM`.

Expected: no disarm while a target is held.

- [ ] **Step 3: Bench run — lose the target, confirm disarm**

With the target locked and the FC armed, occlude/remove the target so NanoTrack
confidence collapses (`tracker_health → LOST`). Watch the control and link logs.

Expected: control logs `TARGET-LOSS DISARM latched` ~`lost_hold_ms` after loss;
link logs `arm command → DISARM (target-loss failsafe)`; the FC disarms (verify
via `fc/status` / the ground UI). Motors stop.

- [ ] **Step 4: Bench run — confirm the latch holds and manual re-arm**

Keep the ground arm switch ON and let the target reappear (`tracker_health`
recovers to `nominal`). Confirm the FC stays DISARMED (latch is sticky). Then
cycle the ground arm switch OFF then ON.

Expected: it stays disarmed on tracker recovery alone; re-arm succeeds only after
the off→on switch cycle.

- [ ] **Step 5: Commit the docs**

```bash
git add ARCHITECTURE.md
git commit -m "docs: describe target-loss disarm failsafe in ARCHITECTURE.md"
```

---

## Self-Review

**Spec coverage:**
- Signal = `tracker_health == LOST` → Tasks 3/4. ✓
- New `failsafe:` config (optional, off by default, defaults) → Task 2, enabled Task 6. ✓
- New `failsafe/disarm` topic + `FailsafeCmd` → Task 1. ✓
- `LostDisarmLatch` (pure, testable) → Task 3. ✓
- Control evaluates, applies `effective_armed`, publishes, reports DISARMED → Task 4. ✓
- Link arbitrates via `_ArmController` → Task 5. ✓
- Latch keys on `arm/cmd`, clears on operator disarm → Task 3 logic + Task 7 Step 4. ✓
- Escalation (level then disarm) → inherent (existing watchdog untouched); confirmed in Task 7 Step 3. ✓
- Configs updated (rpi4b enabled, rk3588 parity off) → Task 6. ✓
- Testing split (unit here, bus on Linux, wiring on-device) → Tasks 1–7. ✓
- ARCHITECTURE.md note → Task 7. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code and test step shows full content. (The `configs/rk3588.yaml` insertion point is described as "after `watchdog:`" because top-level YAML sections are order-independent and `cfg_failsafe` reads by key — the block content is complete.)

**Type consistency:** `FailsafeCmd(timestamp_ns, disarm)`, `FMT_FAILSAFE_CMD="!QB"`, topic `"failsafe/disarm"`, `LostDisarmLatch(enabled, hold_ns).update(now_ns, armed, health) -> bool`, `cfg_failsafe(d) -> FailsafeConfig(disarm_on_lost, lost_hold_ms)`, and `effective_armed = armed and not latched` are used identically across Tasks 1–6. ✓

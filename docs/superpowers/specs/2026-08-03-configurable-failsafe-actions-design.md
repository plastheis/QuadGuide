# Configurable Failsafe Actions — Design Spec

Date: 2026-08-03

Supersedes parts of `2026-07-30-target-loss-disarm-failsafe-design.md` (the
target-loss disarm that shipped is generalized here).

---

## Overview

Today the target-loss failsafe has exactly one response — **disarm the FC** —
and the staleness watchdog has exactly one response — **level the aircraft**.
Both are hard-coded. This spec makes each actionable failsafe condition resolve
to a **configurable terminal action**: either **disarm** (cut motors) or
**mode** (command the FC into an ArduPilot flight mode such as `LAND` or
`ALTHOLD` and hand off).

Two conditions get the configurable action:

1. **Target loss** — `target/estimate.tracker_health == LOST` held for a
   debounce window. (Generalizes the shipped `disarm_on_lost`.)
2. **Watchdog staleness** — any watched bus topic stale past its timeout.

Both **latch and hand off**: once tripped, QuadGuide commands the action and
stays hands-off until the operator manually re-engages. No auto-resume.

Two other degraded-state conditions are **out of scope** — they are structurally
the FC's responsibility, not QuadGuide's:

- **Link / serial I/O loss** (link worker `ConnectionError`/`OSError`) — handled
  by the FC's own RC/GCS heartbeat failsafe; QuadGuide cannot send a mode over a
  dead link anyway. It publishes `DEGRADED` and reconnects, as today.
- **Worker process death** — the orchestrator SIGTERMs the whole stack; the FC
  falls back to its own failsafe. As today.

---

## Goals

- Each actionable failsafe condition (target-loss, watchdog) selects, per
  config, a terminal action of `disarm` or `mode: <NAME>`.
- Flight mode is a friendly name, resolved to an ArduCopter `custom_mode` number
  and **validated at config load** against a GPS-denied-safe allowlist.
- Both conditions **latch + hand off** — no auto-resume on recovery. Cleared by
  operator disarm (`arm/cmd` → False), the existing manual re-arm gate.
- Reuse the existing arm path (`link._ArmController`) for `disarm`; add a
  parallel `_ModeController` for `mode` (same edge-triggered,
  retransmit-until-ACK machinery).
- Keep the debounce/latch a pure, action-agnostic class, unit-testable on
  Windows without the Linux bus.
- Keep every bus topic single-writer.
- Preserve today's soft-LEVEL watchdog behavior when the watchdog failsafe is
  **disabled** (backward compat for rk3588 / feature-off).
- Validate the new MAVLink commands (`DO_SET_MODE`, disarm) end-to-end against
  ArduPilot SITL.

## Non-goals

- **Auto-resume.** Once a failsafe latches, QuadGuide stays hands-off until an
  operator re-arm edge. Target reacquisition / telemetry recovery does not
  resume the attack.
- **QuadGuide commanding GUIDED_NOGPS.** The operator restores guided mode on
  the FC before re-engaging; QuadGuide only ever streams `SET_ATTITUDE_TARGET`
  and commands `disarm` / `DO_SET_MODE(<failsafe mode>)`. This keeps the mode
  transition one-directional and minimizes mode-race surface.
- **GPS-dependent failsafe modes** (RTL, LOITER, POSHOLD, AUTO, GUIDED). We fly
  GPS-denied; these are rejected at config load.
- **Link-loss / worker-death actions.** FC's job (see Overview).
- **Per-topic watchdog actions.** All four watched topics share one watchdog
  action + hold. (Considered and dropped for config-surface simplicity.)

---

## Behavior model (unified)

Both conditions resolve to one terminal action:

- **`disarm`** — command the FC to disarm (motors cut). Today's target-loss
  behavior.
- **`mode: <NAME>`** — command the FC into an ArduPilot flight mode via
  `DO_SET_MODE`, **stop streaming `SET_ATTITUDE_TARGET`**, stay armed, hand off.

On the trip edge the condition **latches**. The latch clears only when the
operator commands disarm (`arm/cmd` → False) — the manual re-arm gate. For a
mode handoff in flight the operator flies/lands under FC/RC authority; to resume
autonomy they restore GUIDED_NOGPS on the FC and cycle the arm switch off→on,
after which QuadGuide resumes streaming `SET_ATTITUDE_TARGET` and re-arms.

### Latch keyed on `arm/cmd`, not `fc/status`

Unchanged from the shipped design and now doubly important: the disarm action
drives `fc/status.armed` False, and a mode action changes `fc/status.custom_mode`
— keying the clear condition on either would immediately un-latch. The latch
clears only on the operator's *commanded* disarm intent (`arm/cmd`).

---

## Architecture

### Signal path

```
tracker ──target/estimate(health)─┐
                                  ├─► control ──failsafe/action──► link ──DISARM|DO_SET_MODE──► FC
watchdog topics (staleness) ──────┘   (arbiter)   (action,mode)     (executor)
ground ──arm/cmd──────────────────────┴───────────────────────────────┘
```

- **control** is the safety authority + **arbiter**: it evaluates both condition
  latches, arbitrates the effective action, and publishes it on
  `failsafe/action`.
- **link** is the **executor**: `disarm` drives the existing `_ArmController`;
  `mode` drives a new `_ModeController` and suppresses the attitude stream.

---

## Components

### 1. Config — `failsafe:` restructured into per-condition blocks

`configs/rpi4b.yaml` (enabled, both → LAND):

```yaml
failsafe:
  target_loss:
    enabled: true
    action: mode          # disarm | mode
    mode: LAND            # required when action=mode; validated at load
    hold_ms: 300          # continuous LOST before the latch trips (debounce)
  watchdog:
    enabled: true
    action: mode          # disarm | mode
    mode: LAND
    hold_ms: 200          # continuous staleness before the latch trips
```

`configs/rk3588.yaml`: same structure with `enabled: false` on both (parity /
docs, feature off).

`core/config.py`:

```python
class FailsafeAction(Enum):
    DISARM = "disarm"
    MODE   = "mode"

# ArduCopter flight-mode custom_mode numbers (subset).
ARDUCOPTER_MODES = {
    "STABILIZE": 0, "ALTHOLD": 2, "AUTO": 3, "GUIDED": 4, "LOITER": 5,
    "RTL": 6, "LAND": 9, "POSHOLD": 16, "GUIDED_NOGPS": 20, ...
}
# GPS-denied-safe failsafe targets. Everything else is rejected at load.
FAILSAFE_MODE_ALLOWLIST = {"LAND", "ALTHOLD", "STABILIZE"}

@dataclass(frozen=True)
class ConditionFailsafe:
    enabled: bool
    action: FailsafeAction
    mode: str | None            # friendly name when action=MODE, else None
    custom_mode: int | None     # resolved ArduCopter number when action=MODE
    hold_ms: int

@dataclass(frozen=True)
class FailsafeConfig:
    target_loss: ConditionFailsafe
    watchdog: ConditionFailsafe

def cfg_failsafe(d: dict) -> FailsafeConfig:
    f = d.get("failsafe") or {}
    _reject_legacy_keys(f)      # disarm_on_lost / lost_hold_ms → clear error
    return FailsafeConfig(
        target_loss=_condition(f.get("target_loss"), default_hold_ms=300),
        watchdog=_condition(f.get("watchdog"), default_hold_ms=200),
    )
```

`_condition` resolves `mode` → `custom_mode` via `ARDUCOPTER_MODES` and validates
against `FAILSAFE_MODE_ALLOWLIST`, raising `ValueError` at load on an unknown or
disallowed mode, or on `action: mode` with no `mode` key.

**Fail-fast on legacy keys.** If `disarm_on_lost` or `lost_hold_ms` are present,
`cfg_failsafe` raises a `ValueError` naming the new structure. A silently-ignored
failsafe config is a safety hazard, so we refuse to load rather than run with the
feature unexpectedly off.

`failsafe:` remains an **optional** section (not in `_REQUIRED_SECTIONS`); an
absent section or `enabled: false` sub-block → that condition off.

### 2. Message + topic — `failsafe/action` (renamed from `failsafe/disarm`)

The topic and message added by the shipped target-loss feature are generalized.
Neither is fielded to a mixed fleet, so the rename + wire bump is safe.

`core/messages.py`:

```python
class FailsafeActionWire(IntEnum):   # byte on the wire
    NONE     = 0
    DISARM   = 1
    SET_MODE = 2

FMT_FAILSAFE_CMD = "!QBI"
# Q(8) + action(B=1) + custom_mode(I=4) = 13 bytes
# custom_mode is meaningful only for SET_MODE (0 otherwise).

@dataclass(frozen=True)
class FailsafeCmd:
    timestamp_ns: int
    action: FailsafeActionWire
    custom_mode: int = 0
    # pack/unpack mirror the other byte-enum messages
```

`core/bus.py`: rename the topic key `"failsafe/disarm"` → `"failsafe/action"`.

### 3. Latch — `control/failsafe.py` generalized to `FailsafeLatch`

`LostDisarmLatch` becomes `FailsafeLatch` — a pure, **action-agnostic** debounce/
latch/reset machine. The trip predicate is passed in by the caller (control
worker), so the same class serves both conditions.

```python
class FailsafeLatch:
    """Debounced trip -> latch machine (pure; no bus/clock).

    * Only trips while `armed` (operator's commanded arm intent, arm/cmd).
    * Trips when `tripped` is True continuously for `hold_ns`.
    * Any non-tripped tick resets the debounce.
    * Once latched, stays latched until `armed` goes False (operator disarm).
    * Disabled -> always False.
    """
    def __init__(self, enabled: bool, hold_ns: int) -> None:
        self._enabled = enabled
        self._hold_ns = hold_ns
        self._trip_since: int | None = None
        self._latched = False

    def update(self, now_ns: int, armed: bool, tripped: bool) -> bool:
        if not self._enabled:
            return False
        if not armed:
            self._latched = False
            self._trip_since = None
            return False
        if self._latched:
            return True
        if tripped:
            if self._trip_since is None:
                self._trip_since = now_ns
            elif now_ns - self._trip_since >= self._hold_ns:
                self._latched = True
        else:
            self._trip_since = None
        return self._latched
```

The action attached to each latch comes from config, not the latch. This keeps
the machine identical for `health == LOST` (target-loss) and `any-topic-stale`
(watchdog).

### 4. Control worker changes (`control/worker.py`) — the arbiter

- Build two latches:
  `tl_latch = FailsafeLatch(fcfg.target_loss.enabled, target_loss.hold_ms * 1e6)`,
  `wd_latch = FailsafeLatch(fcfg.watchdog.enabled, watchdog.hold_ms * 1e6)`.
- Each tick:
  - Target-loss trip predicate: `health == TrackerHealth.LOST`.
  - Watchdog trip predicate: **any watched topic stale right now**. `watchdog.py`
    exposes a boolean staleness check (`any_stale()`) alongside the existing
    `check_all()` that raises for the soft-LEVEL path.
  - `tl_latched = tl_latch.update(now, armed, health == LOST)`.
  - `wd_latched = wd_latch.update(now, armed, any_stale)`.
  - **Arbitrate** the effective action (precedence **DISARM > MODE**):
    - If any latched condition's configured action is `disarm` → `DISARM`.
    - Else if any latched → `SET_MODE` with that condition's `custom_mode`; when
      both latch as MODE with different modes, **target-loss wins** (documented;
      moot when both = LAND).
    - Else → `NONE`.
  - Publish `FailsafeCmd(now, action, custom_mode)` on `failsafe/action` every
    tick.
  - `effective_armed = armed and not (tl_latched or wd_latched)`, gating throttle
    and attitude as today (level + zero throttle + slew reset on any latch).
  - State/health: `FailsafeState.DISARMED` for a disarm latch; add
    `FailsafeState.MODE` for a mode latch; detail string carries specifics.
- **Watchdog failsafe disabled** → keep today's `check_all()` → `HealthFault` →
  soft-LEVEL path exactly as-is (backward compat).

### 5. Watchdog escalation detail

The watchdog latch debounces "any topic stale right now" (existing per-topic
timeouts as the staleness test) with `watchdog.hold_ms`. Sustained staleness for
`hold_ms` latches the terminal action; a brief blip does not. During the debounce
window the control loop still commands **level + zero throttle** — it will not
bank on stale data. That is inherent safe handling, **not** a separate
configurable "LEVEL stage" (which was removed in favor of going straight to the
terminal action). This is what prevents a single ~100 ms `guidance/accel` blip
from instantly LANDing / disarming the aircraft.

### 6. Link worker changes (`link/worker.py`) — the executor

Read `failsafe/action`; decode `(action, custom_mode)`:

- **DISARM** → existing path: `effective = armed and NOT disarm`, fed to
  `_ArmController`. Unchanged.
- **SET_MODE** →
  - **suppress the `SET_ATTITUDE_TARGET` stream** (don't fight the FC mode);
  - command `DO_SET_MODE(custom_mode)` via a new **`_ModeController`**
    (edge-triggered, retransmit-until-ACK, mirroring `_ArmController`);
  - stay armed (do not disarm).
- `_rx_loop` routes `COMMAND_ACK` for `MAV_CMD_DO_SET_MODE` to `_ModeController`
  (existing routing for `MAV_CMD_COMPONENT_ARM_DISARM` → `_ArmController` stays).

`link/fc.py`:

```python
def encode_set_mode(mav, custom_mode, target_sys, target_comp) -> bytes:
    """COMMAND_LONG / MAV_CMD_DO_SET_MODE.
    param1 = MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, param2 = custom_mode."""
    msg = mav.command_long_encode(
        target_sys, target_comp,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
        float(custom_mode), 0, 0, 0, 0, 0,
    )
    return msg.pack(mav)
```

`link/mavlink_codec.py`: expose `MAV_CMD_DO_SET_MODE` /
`MAV_MODE_FLAG_CUSTOM_MODE_ENABLED` if a local alias is wanted for consistency
with the other codec consts.

---

## Behavior timeline (armed, firing, target lost, action=mode LAND)

| t | event |
|---|-------|
| 0 | target lost → `tracker_health = LOST` |
| ~100 ms | `guidance/accel` stale → local level (roll/pitch 0), throttle continues |
| `target_loss.hold_ms` (~300 ms) | latch trips → local throttle 0 + `failsafe/action(SET_MODE, LAND)` → link suppresses attitude stream and commands `DO_SET_MODE(LAND)` (retransmit until ACK) → FC enters LAND and descends |

`disarm` variant is identical except the terminal action cuts motors (as today).

## Re-engage procedure

After a trip the FC is in LAND (or disarmed) and `arm/cmd` is still armed, so the
latch holds and QuadGuide stays hands-off. To re-engage: operator commands disarm
(`arm/cmd` → False) to clear the latch, restores GUIDED_NOGPS on the FC, then
re-arms (off→on). QuadGuide resumes streaming `SET_ATTITUDE_TARGET`. No automatic
re-engage on condition recovery.

---

## Files touched

- `core/config.py` — `FailsafeAction`, `ConditionFailsafe`, restructured
  `FailsafeConfig`, `ARDUCOPTER_MODES`, `FAILSAFE_MODE_ALLOWLIST`, `cfg_failsafe`
  with mode resolution + allowlist validation + legacy-key rejection.
- `core/messages.py` — `FailsafeCmd` carries `action` + `custom_mode`;
  `FMT_FAILSAFE_CMD` `!QB` → `!QBI`; `FailsafeActionWire` byte enum.
- `core/bus.py` — rename topic `failsafe/disarm` → `failsafe/action`.
- `control/failsafe.py` — `LostDisarmLatch` → generic `FailsafeLatch`.
- `control/worker.py` — two latches, arbitration + precedence, publish, local
  safing, keep soft-LEVEL when watchdog failsafe off.
- `control/worker.py` — reuse the existing `watchdog.check_all()` fault as the
  watchdog staleness predicate (`stale = fault is not None`); no new
  `any_stale()` method was added to `control/watchdog.py`.
- `link/fc.py` — `encode_set_mode`.
- `link/mavlink_codec.py` — DO_SET_MODE / custom-mode-flag consts (if aliased).
- `link/worker.py` — `_ModeController`, SET_MODE execution, attitude-stream
  suppression, ACK routing.
- `configs/rpi4b.yaml` — restructured `failsafe:`, both → mode LAND.
- `configs/rk3588.yaml` — restructured `failsafe:`, both disabled (parity).
- `docs/ARCHITECTURE.md` — update the failsafe section.

## Testing

### Unit (runs on Windows — pure logic / codec)

- `FailsafeLatch`: no trip before `hold_ms`; trip after continuous `tripped`
  while armed; debounce resets on a non-tripped tick; latch persists through
  recovery; cleared by operator disarm; no-op when disabled or not armed.
- `cfg_failsafe`: parse both conditions; `mode` → `custom_mode` resolution;
  allowlist rejects RTL/LOITER/unknown with `ValueError`; `action: mode` with no
  `mode` errors; legacy `disarm_on_lost`/`lost_hold_ms` keys error; disabled →
  off.
- `FailsafeCmd` pack/unpack round-trip incl. `action` + `custom_mode`.
- `encode_set_mode`: decode the packed bytes with pymavlink; assert
  `MAV_CMD_DO_SET_MODE`, param1 = custom-mode-enabled flag, param2 = custom_mode.
- Control arbitration: DISARM > MODE precedence; both-MODE target-loss
  precedence.
- Link `_ModeController`: emits `DO_SET_MODE` on the trip edge, re-transmits
  until ACK; attitude stream suppressed while a mode failsafe is active.

### SITL HIL (Linux / WSL with ArduPilot SITL, `serial.mode: tcp`)

- New `scripts/test_failsafe_sitl.py` (mirrors `scripts/test_link.py`): connect
  SITL, arm, drive each condition, and **assert via SITL HEARTBEAT that
  `custom_mode` becomes LAND (9)** after a target-loss / watchdog trip, and that
  the `disarm` variant disarms; assert the transition lands within `hold_ms` +
  margin, and that it stays latched until an operator re-arm edge.
- Fill the empty `tests/hil/test_lost_target.py` accordingly.
- SITL runs on Linux; on the Windows dev box that means WSL or running on the
  SBC / Linux CI (the stack needs Linux regardless). `encode_set_mode`
  correctness is covered by the Windows unit test; SITL validates the
  end-to-end command exchange and the resulting FC mode.

## Known limitations

- **Flicker dodge** (inherited): the debounce hard-resets on any non-tripped
  tick, so a pathological fast flicker can avoid the trip. Acceptable for v1.
- **Latch is process-local** (inherited): resets if the control worker restarts,
  but the orchestrator kills the whole stack on any worker exit, so a lone
  control restart cannot happen in practice.
- **No auto-resume / no auto GUIDED_NOGPS restore.** Re-engaging after a mode
  handoff is an explicit operator procedure (see Re-engage). Deliberate — keeps
  the mode transition one-directional and avoids mode-race oscillation.

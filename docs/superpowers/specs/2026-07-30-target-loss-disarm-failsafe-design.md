# Target-Loss Disarm Failsafe — Design Spec

Date: 2026-07-30

---

## Overview

When the tracker loses the target, QuadGuide today only *levels* the aircraft
and keeps the motors at `throttle_hold` if the operator is holding fire — it
never cuts power. This spec adds a config-gated failsafe that **disarms the FC
over MAVLink** when the tracker reports `LOST` continuously for a debounce
window.

The trigger is `target/estimate.tracker_health == LOST`. No new confidence
plumbing is introduced: `tracker_health` is *already* a confidence gate inside
EdgeCV. For bare NanoTrack (`edgecv/trackers/nn/nanotrack.py:_status_from`):
`conf ≥ score_lock` → `LOCKED`, `score_lost ≤ conf < score_lock` → `COASTING`,
and `conf < score_lost` → `LOST`. Those thresholds are set from `tracker.params`
in the QuadGuide YAML (surfaced as `score_lock`/`score_lost` in `rpi4b.yaml`).
QuadGuide adds only a debounce (bare NanoTrack flips `LOST` on a single frame
with no hysteresis) and the disarm wiring.

Net effect: one new optional config section, one new bus topic + 9-byte
message, one new pure/testable latch class, and small additions to the control
and link workers. No existing wire format changes.

---

## Goals

- Disarm the FC over MAVLink when `tracker_health == LOST` persists for
  `lost_hold_ms` while armed.
- Off by default; enabled per-config via a new `failsafe:` section.
- Reuse the existing arm path (`link._ArmController`, edge-triggered,
  retransmit-until-ACK) — no new arm mechanism.
- Keep every topic single-writer: control writes `failsafe/disarm`, ground
  keeps writing `arm/cmd`, link arbitrates.
- Latch until the operator manually re-arms; no silent auto re-arm.
- Keep the debounce/latch logic a pure class, unit-testable on Windows without
  the Linux bus.

## Non-goals

- Gating on the raw `confidence` float. `tracker_health` is the signal; the
  confidence thresholds live in `tracker.params` (`score_lock`/`score_lost`).
- Changing the existing LEVEL/staleness watchdog. It still runs and still
  levels the aircraft ~100 ms into a loss; the disarm escalates on top of it.
- Handling tracker-process death. Any worker exit trips the orchestrator, which
  SIGTERMs the whole stack (`scripts/run.py`); the FC then falls back to its own
  RC/GCS failsafe. Out of scope here.
- Hybrid (`acquire_track`) tuning. The design works for it unchanged (its `LOST`
  is already heavily debounced), but the target platform is rpi4b + bare
  NanoTrack.

---

## Architecture

### Signal path

```
tracker ──target/estimate(health)──► control ──failsafe/disarm──► link ──DISARM──► FC
                                        ▲                          ▲
ground ──arm/cmd──────────────────────  ┴  ─────────────────────  ┘
```

- **control** is the safety authority: it reads `target/estimate.tracker_health`
  and `arm/cmd`, runs the debounce+latch, and publishes the latch on
  `failsafe/disarm`.
- **link** is the actuator: it computes `effective_armed = arm_cmd.armed AND NOT
  failsafe.disarm` and feeds that to the existing `_ArmController`.

### The latch is keyed on `arm/cmd`, not `fc/status`

The clear condition uses the operator's *commanded* arm intent (`arm/cmd`), not
the FC's actual armed state (`fc/status`). This is essential: once the failsafe
disarms the FC, `fc/status.armed` goes False — if the latch cleared on that it
would immediately re-arm. Keying on `arm/cmd` means the latch only clears when
the *operator* commands disarm (switch off), which is the manual re-arm gate.

---

## Components

### 1. Config — new optional `failsafe:` section

`core/config.py`:

```python
@dataclass(frozen=True)
class FailsafeConfig:
    disarm_on_lost: bool = False
    lost_hold_ms: int = 300

def cfg_failsafe(d: dict) -> FailsafeConfig:
    f = d.get("failsafe") or {}
    return FailsafeConfig(
        disarm_on_lost=f.get("disarm_on_lost", False),
        lost_hold_ms=f.get("lost_hold_ms", 300),
    )
```

Read via `d.get("failsafe", …)` with defaults and **not** added to
`_REQUIRED_SECTIONS`, so existing configs (e.g. `rk3588.yaml`) keep loading with
the feature off.

`configs/rpi4b.yaml` (enabled):

```yaml
failsafe:
  disarm_on_lost: true    # disarm the FC when the tracker reports LOST
  lost_hold_ms: 300       # continuous LOST required before disarm (debounce)
```

`configs/rk3588.yaml` gets the same block with `disarm_on_lost: false` for
documentation/parity.

### 2. New message + topic — `failsafe/disarm`

`core/messages.py`:

```python
FMT_FAILSAFE_CMD = "!QB"   # Q(8) + disarm(B=1) = 9 bytes

@dataclass(frozen=True)
class FailsafeCmd:
    timestamp_ns: int
    disarm: bool
    # pack/unpack mirror ArmCmd
```

Add `FailsafeCmd`, `FMT_FAILSAFE_CMD` to `__all__`, a `_ST_FAILSAFE_CMD`
struct, and register in `core/bus.py` `TOPICS`:

```python
"failsafe/disarm": (FailsafeCmd, FMT_FAILSAFE_CMD),
```

### 3. Latch — `control/failsafe.py` (new, pure)

Parallel to `control/watchdog.py`. All state is process-local; no bus, no clock
of its own — the caller passes `now_ns`, so it is deterministic and testable.

```python
class LostDisarmLatch:
    def __init__(self, enabled: bool, hold_ns: int) -> None:
        self._enabled = enabled
        self._hold_ns = hold_ns
        self._lost_since: int | None = None
        self._latched = False

    def update(self, now_ns: int, armed: bool, health) -> bool:
        if not self._enabled:
            return False
        if not armed:                      # operator disarm clears latch + debounce
            self._latched = False
            self._lost_since = None
            return False
        if self._latched:                  # sticky until 'not armed' clears it
            return True
        if health == TrackerHealth.LOST:
            if self._lost_since is None:
                self._lost_since = now_ns
            elif now_ns - self._lost_since >= self._hold_ns:
                self._latched = True
        else:
            self._lost_since = None         # any non-LOST frame resets the debounce
        return self._latched
```

### 4. Control worker changes (`control/worker.py`)

- Build once: `latch = LostDisarmLatch(fcfg.disarm_on_lost, fcfg.lost_hold_ms * 1_000_000)`.
- Each tick:
  - `est = bus.latest("target/estimate")`; `health = est.tracker_health if est else None`.
  - `latched = latch.update(now_ns, armed, health)`.
  - `effective_armed = armed and not latched`.
  - Use `effective_armed` (not `armed`) for **both** the throttle gate
    (`thr = throttle_hold if effective_armed and fire_active else 0.0`) and the
    attitude gate — so on latch the command goes to level + zero throttle
    locally the same tick, before the FC even ACKs the disarm.
  - Publish `FailsafeCmd(now_ns, latched)` on `failsafe/disarm` every tick.
  - On the False→True latch edge: `log.warning`, set
    `state = FailsafeState.DISARMED` (first real use of that enum), report
    `system/health` FAILSAFE with detail.

### 5. Link worker changes (`link/worker.py` `_tx_loop`)

```python
arm_cmd = bus.latest("arm/cmd")
armed   = bool(arm_cmd and arm_cmd.armed)
fs      = bus.latest("failsafe/disarm")
latched = bool(fs and fs.disarm)
effective = armed and not latched

to_send = arm_ctrl.on_arm_state(effective)   # existing edge + retransmit + ACK
```

`effective` also drives the `latch_yaw`/`prev_armed` edge so heading re-latches
on a real re-arm. The disarm log line distinguishes a failsafe disarm from an
operator disarm.

---

## Behavior timeline (armed, firing, target lost)

| t | event |
|---|---|
| 0 | target lost → NanoTrack `conf < score_lost` → `tracker_health = LOST` |
| ~100 ms | `guidance/accel` stale → existing watchdog `LEVEL` → roll/pitch 0 |
| `lost_hold_ms` (~300 ms) | latch trips → local throttle 0 + `failsafe/disarm(True)` → link commands **DISARM** (retransmit until ACK) → FC cuts motors |

A clean escalation: **level first, disarm if the loss persists.**

## Re-arm procedure

After a trip the FC is disarmed but the ground's `arm/cmd` is still `armed=True`,
so `effective_armed` stays False and it stays disarmed. To re-engage, the
operator **cycles the ground arm switch off→on**: `arm/cmd=False` clears the
latch, then re-arm is clean. No automatic re-arm on tracker recovery.

---

## Files touched

- `core/config.py` — `FailsafeConfig` + `cfg_failsafe` (optional section).
- `core/messages.py` — `FailsafeCmd`, `FMT_FAILSAFE_CMD`, struct, `__all__`.
- `core/bus.py` — register `failsafe/disarm` topic + import.
- `control/failsafe.py` — **new** `LostDisarmLatch`.
- `control/worker.py` — build latch, evaluate, apply `effective_armed`, publish.
- `link/worker.py` — read `failsafe/disarm`, arbitrate `effective`, log.
- `configs/rpi4b.yaml` — `failsafe:` section enabled.
- `configs/rk3588.yaml` — `failsafe:` section disabled (parity/docs).

## Testing

- **Unit (runs on Windows — pure logic):** `tests/unit/test_failsafe_latch.py`
  — no trip before `hold_ms`; trip after continuous LOST while armed; debounce
  resets on a non-LOST frame; latch persists through health recovery; cleared by
  operator disarm; no-op when `disarm_on_lost=false` or not armed. Plus a
  `cfg_failsafe` defaults/parse case in `tests/unit/test_config.py`.
- **Link arbitration (unit):** feed `effective = armed AND NOT latched`
  transitions through `_ArmController` and assert a DISARM is emitted on the trip
  edge and re-transmitted until ACK.
- **HIL (runs on the SBC / Linux CI, not Windows):** fill the empty
  `tests/hil/test_lost_target.py` — lose the target mid-track and assert a DISARM
  command reaches the FC within `lost_hold_ms` + margin, and that it stays
  disarmed until an operator re-arm edge.

## Known limitations

- **Flicker dodge.** The debounce hard-resets on any non-LOST frame, so a
  pathological fast flicker (LOST 290 ms → 1 nominal frame → LOST…) can avoid the
  trip. Acceptable for v1: a genuinely lost NanoTrack drifts and stays `LOST`
  rather than flickering to nominal.
- **Latch is process-local.** It resets if the control worker restarts — but the
  orchestrator kills the whole stack on any worker exit, so a lone control
  restart cannot happen in practice.

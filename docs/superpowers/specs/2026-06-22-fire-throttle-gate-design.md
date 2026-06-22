# Fire Button Throttle Gate Design

**Date:** 2026-06-22
**Status:** draft

---

## 1. Scope

Decouple throttle from the arm command by adding a fire button to the ground
station. Currently, arming the quad applies `guidance.throttle_hold` after a 2 s
arm dwell. After this change:

- **Arm** arms the FC and enables attitude guidance — throttle stays at 0.
- **Fire** (toggle, `f` key or button) applies `guidance.throttle_hold`.
- The 2 s arm dwell is removed entirely.

Fire does **not** require a lock-on: pressing fire without a target gives level
attitude at `throttle_hold` (useful for takeoff / climb).

Both the verbose (`index.html`) and minimal (`minimal.html`) UIs get the fire
control.

### Non-goals

- No change to the link worker, guidance worker, tracker worker, or config file.
- No new config keys — reuses `guidance.throttle_hold` (0.4).
- No change to the failsafe path (it already forces throttle to 0).

---

## 2. Architecture

### 2.1 New bus message: `FireCmd`

Same shape as `ArmCmd` — a timestamp and a boolean — 9 bytes on the wire.

```
FMT_FIRE_CMD = "!QB"        # Q(8) + B(1) = 9 bytes
FireCmd(timestamp_ns: int, active: bool)
```

Registered on topic `fire/cmd` in `TOPICS`.

### 2.2 Data flow

```
Ground UI [f] key / button click
    ↓ POST /fire {active: bool}
Ground Server
    ↓ publish FireCmd
fire/cmd bus topic
    ↓ bus.latest()
Control Worker
    thr = throttle_hold if (armed AND fire_active) else 0.0
    ↓ publish ControlCmd {throttle_norm}
control/cmd bus topic
    ↓ bus.latest()
Link Worker (unchanged — reads control/cmd passively)
    ↓ SET_ATTITUDE_TARGET {thrust}
FC
```

### 2.3 Attitude gating (simplified)

Before (with dwell):

```
thr = throttle_hold if (armed and dwell_done) else 0.0
attitude live if armed and dwell_done and no fault and accel present
```

After (no dwell, fire-gated):

```
thr = throttle_hold if (armed and fire_active) else 0.0
attitude live if armed and no fault and accel present
```

Arm enables attitude immediately. Fire independently controls throttle.

### 2.4 Edge cases

| Scenario | Throttle | Roll/Pitch |
|---|---|---|
| Disarmed, fire off | 0 | 0 (level) |
| Disarmed, fire on | 0 (armed gate) | 0 (level) |
| Armed, fire off | 0 | guidance active |
| Armed, fire on | `throttle_hold` | guidance active |
| Armed, fire on, in failsafe | 0 | 0 (level) |
| Armed, fire on, no accel yet | `throttle_hold` | 0 (level) |

The last row is the "fire without lock-on" case — the quad climbs at
`throttle_hold` with level attitude.

### 2.5 Safety: auto-reset fire on disarm

When the operator sends disarm, both UIs also reset fire to off. This prevents
the quad from applying throttle immediately on the next arm if the operator
forgot fire was still toggled on.

---

## 3. Files Changed

| File | Change |
|---|---|
| `core/messages.py` | Add `FireCmd` dataclass, `FMT_FIRE_CMD`, pack/unpack |
| `core/bus.py` | Register `fire/cmd` in `TOPICS` |
| `ground/server.py` | Add `POST /fire` endpoint; add `fire_active` to SSE telemetry |
| `ground/static/index.html` | Fire button + `f` key + fire state indicator in ARM CONTROL section |
| `ground/static/minimal.html` | `f` key toggle + fire state overlay text on video feed |
| `control/worker.py` | Remove arm dwell; read `fire/cmd`; new throttle gate |

**Unchanged:** config, link worker, guidance worker, tracker worker, all tests
(existing messages and formats are untouched).

---

## 4. UI Details

### 4.1 Verbose (`index.html`)

- Fire button added to the ARM CONTROL section, next to ARM/DISARM buttons.
- Button label toggles between "FIRE OFF" and "FIRE ON".
- `f` key toggles fire.
- Fire state indicator text next to the arm state display.
- `sendArm(false)` (disarm) also calls `sendFire(false)`.

### 4.2 Minimal (`minimal.html`)

- `f` key toggles fire (same as verbose).
- Fire state shown as an overlay text on the video feed (like the existing
  "ARMED" overlay), e.g. "FIRE" in orange when active.
- `sendArm(false)` also resets fire state to off.

### 4.3 SSE telemetry addition

The `/telemetry` SSE stream gains one field:

```json
{"fire_active": true, ...}
```

Read from `bus.latest("fire/cmd")` in the SSE loop. Both UIs consume it to
keep fire state in sync with the server (self-healing across page reload).

---

## 5. Testing

- Existing unit and integration tests continue to pass (no wire format changes
  to existing messages).
- New unit test: `FireCmd` pack/unpack round-trip.
- The control worker throttle gate is exercised by existing integration tests
  (`test_control_pipeline.py`) — the fire state defaults to off (no message on
  `fire/cmd`), so throttle stays at 0 and the tests remain valid.
- Manual verification: run the stack with `--no-ground` and publish to
  `fire/cmd` directly; observe `control/cmd.throttle_norm`.

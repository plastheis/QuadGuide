# QuadGuide ⇄ ArduCopter (GUIDED_NOGPS) — flight-controller setup & arm/fire debug

**Date:** 2026-07-06
**Applies to:** QuadGuide `link/` MAVLink2 adapter → ArduCopter (H743) in GUIDED_NOGPS.
**Symptom being solved:** QuadGuide arms in any mode; after arming you can't "fire",
the motors sit at idle and the FC auto-disarms after ~10 s.

> You said you can't SSH into the Pi/FC to iterate. This doc is written to be
> applied **once**: change the parameters, follow the arm/fire procedure, and it
> should work without a debug loop. Everything below is traced to the actual
> QuadGuide source and the current ArduPilot docs (links at the bottom).

---

## 1. TL;DR — what's actually happening

Two independent things are stacking up:

1. **The ~10 s disarm is `DISARM_DELAY` (default 10 s).** ArduCopter auto-disarms
   a vehicle that is armed but has never left the ground. It is a *symptom*, not
   the cause — the real problem is that the quad never spins up and takes off, so
   the idle-disarm timer expires.

2. **The quad never spins up on "fire" for one (or both) of these reasons:**

   - **(A) You are not actually in `GUIDED_NOGPS` at the moment you arm + fire.**
     `SET_ATTITUDE_TARGET` — the only thing QuadGuide sends to move the vehicle —
     is **ignored by ArduPilot in every mode except GUIDED / GUIDED_NOGPS.** In
     STABILIZE the motors are driven by the RC throttle stick, which is at idle,
     so "fire" does nothing and `DISARM_DELAY` disarms you. This is why "quadguide
     arms from any mode" and "mavlink is going through in any flight mode" — both
     are true and both are by design; arming and the message stream are
     mode-independent, but **thrust is not.**

   - **(B) `GUID_OPTIONS` is at its default (0), so the thrust field is
     misinterpreted.** QuadGuide sends `guidance.throttle_hold` (0.6 on your
     `rpi4b.yaml`) *as direct thrust*. With `GUID_OPTIONS=0`, ArduPilot reads the
     thrust field as a **climb rate** where **0.5 = hover, 0 = descend, 1 = climb**.
     So "fire off" (thrust 0) is a *descend* command and "fire on" (0.6) is only a
     weak ~20 %-of-max climb. The design intends **direct thrust**, which requires
     **`GUID_OPTIONS` bit 3 (value 8)**.

**Fix = set the FC parameters in §3 (chiefly `GUID_OPTIONS=8`) + follow the
arm/fire procedure in §4 (be in GUIDED_NOGPS *before* you arm).**

Neither of these is a bug in the QuadGuide MAVLink adapter — the adapter behaves
exactly as its design spec (`docs/superpowers/specs/2026-06-21-mavlink-link-migration-design.md`,
§"FC-Side Parameters") intended. The gaps are (a) FC configuration and (b) the
operator having no on-screen way to see the FC mode (§5).

---

## 2. How QuadGuide drives the FC (verified against source)

```
Ground UI  ── POST /arm  ──▶ arm/cmd ─┐
Ground UI  ── POST /fire ──▶ fire/cmd ┤
                                       ▼
control/worker.py:  thr = throttle_hold if (armed AND fire_active) else 0.0
                    roll/pitch from guidance (only when armed, no failsafe, accel present)
                                       ▼ control/cmd {roll_deg, pitch_deg, throttle_norm}
link/worker.py _tx_loop  @ tx_rate_hz (100 Hz on rpi4b):
   • arm edge      → COMMAND_LONG / MAV_CMD_COMPONENT_ARM_DISARM  (retried until ACK)
   • every tick    → SET_ATTITUDE_TARGET  (type_mask 0x07, quaternion + thrust)
link/worker.py _heartbeat_loop @ 1 Hz → HEARTBEAT (MAV_TYPE_ONBOARD_CONTROLLER)
```

Facts that matter for this bug, from the code:

- **QuadGuide never sends a mode-change command.** There is no `DO_SET_MODE`
  anywhere in `link/`. Mode authority is 100 % on the pilot's RC switch (this is
  the approved design). QuadGuide will happily arm and stream setpoints while the
  FC is in STABILIZE — they just do nothing until the FC is in GUIDED_NOGPS.
- **Arming is not gated on mode.** `_tx_loop` arms on any `arm/cmd` edge as long
  as a HEARTBEAT has been seen (`link/worker.py:161`). Hence "arms from any mode."
- **`type_mask = 0x07` is correct** — ArduPilot's doc says it "should always be
  0x07" (ignore the three body-rate fields; attitude comes from the quaternion).
  This part is right; don't change it.
- **Thrust = `clamp(throttle_norm, 0, 1)` sent as the raw thrust field**
  (`link/fc.py:74`). This is the value that `GUID_OPTIONS` decides how to read.
- **The stream is continuous at 100 Hz** — good; ArduPilot reverts GUIDED
  setpoints if the stream stops, and QuadGuide never stops while connected.

---

## 3. Mission Planner parameters

Set via **CONFIG → Full Parameter List** (type the name in the Search box, edit
Value, then **Write Params**, then reboot the FC). The three **must-change**
rows are also in the loadable file `docs/ardupilot-quadguide.param`
(CONFIG → Full Parameter List → **Load from file**).

### 3.1 Must change — this is the fix

| Parameter | Set to | Why |
|---|---|---|
| **`GUID_OPTIONS`** | **`8`** | Bit 3 = "SetAttitudeTarget interprets Thrust As Thrust". Makes `SET_ATTITUDE_TARGET.thrust` a **direct 0–1 throttle** — what QuadGuide sends. Default `0` treats it as a climb rate (0.5=hover) and the quad won't take off on "fire". *(If you use other GUIDED features, OR the bits together — but 8 alone is correct for QuadGuide.)* |
| **`DISARM_DELAY`** | **`30`** (bench) | Seconds armed-and-idle before auto-disarm. Default `10` = your "disarms after ~10 s". Raise it so you have time between ARM and FIRE. `0` disables auto-disarm entirely (bench only). |
| **`MOT_SPIN_ARM`** | **`0.10`** | Idle prop-spin the moment you ARM so "armed" is visible and ESCs are spooled before FIRE. If yours is `0`, motors won't spin at idle at all. |

### 3.2 Verify (don't blindly load) — depends on your wiring/setup

These are **not** in the loadable file because a wrong value here can break your
*working* link or arming. Check each in the Full Parameter List.

| Parameter | Expected | Notes |
|---|---|---|
| `SERIALx_PROTOCOL` | `2` (MAVLink2) | `x` = the TELEM port the Pi is wired to (**TELEM1 = SERIAL1, TELEM2 = SERIAL2**). MAVLink is already flowing for you, so this is almost certainly set — confirm it's `2` (MAVLink2), not `1` (MAVLink1). |
| `SERIALx_BAUD` | `921` | Must equal `platform.serial.baud`. **Your `rpi4b.yaml` uses `921600`** → set the dropdown to **921**. (`config.yaml` still says 115200 → 57 — make sure the FC matches whichever board file you actually deploy.) A mismatch = no MAVLink at all; since yours works, it already matches. |
| `SERIALx_OPTIONS` | `0` | Unless you wired RTS/CTS flow control (you didn't — Pi GPIO14/15 only). |
| `FLTMODEx` (RC mode switch) | one position = `20` | `20` = GUIDED_NOGPS. **GUIDED_NOGPS is usually NOT in the Mission Planner "Flight Modes" dropdown** — set `FLTMODE1..6 = 20` for the switch position you want via the Full Parameter List. You said you can already select GUIDED_NOGPS, so this is done — but confirm the *exact* switch position and that it stays there through arm+fire. |
| `ARMING_CHECK` | keep as-is | You can already arm, so leave it. If arming is ever refused in GUIDED_NOGPS, it's usually the GPS/EKF check — uncheck "GPS" (bit) rather than setting `0`, so you keep the other prearm safety checks. |
| `FS_GCS_ENABLE` | `0` for bench | QuadGuide's HEARTBEAT uses `system_id = 1`. GCS-failsafe only counts heartbeats whose sysid == `SYSID_MYGCS` (default 255), so QuadGuide's beats normally **don't** feed GCS failsafe. But if you *also* run Mission Planner as a GCS and it drops, `FS_GCS` can fire. Disable on the bench to remove the variable. **Note:** a GCS failsafe while landed disarms *immediately*, not after 10 s — so your 10 s disarm is `DISARM_DELAY`, not this. |
| `MOT_SPIN_MIN` | `0.15` (typ.) | Minimum in-flight throttle. Leave at your tune if the quad already flies manually. |
| `MOT_THST_HOVER` | informational | With direct-thrust (`GUID_OPTIONS=8`) there is **no altitude hold** — thrust is open-loop. Your `throttle_hold` must be near the real hover thrust to hover; above → climb, below → sink. |

### 3.3 Tuning caution on `throttle_hold`

With `GUID_OPTIONS=8`, `guidance.throttle_hold = 0.6` becomes **60 % throttle,
open-loop.** For a 180 g micro that is well above hover and will leap on FIRE.
For the first powered test, drop `guidance.throttle_hold` to **~0.30–0.40** in
`configs/rpi4b.yaml`, confirm a gentle lift, then tune up. (This is a QuadGuide
config change, not an FC parameter.)

---

## 4. Correct arm / fire procedure (bench)

Order matters — the mode must be live *before* you arm, and you must fire before
`DISARM_DELAY` expires.

1. Power up, let the FC boot and QuadGuide connect (link log: `FC HEARTBEAT: sys=… comp=…`).
2. **Put the FC in `GUIDED_NOGPS` first** (RC mode switch, or `mode GUIDED_NOGPS`
   from a GCS). Confirm the FC actually reports it — see §5; on Mission Planner
   the HUD mode text must read **"Guided_NoGPS"**.
3. **ARM** from QuadGuide. Props should idle (`MOT_SPIN_ARM`).
4. **FIRE within `DISARM_DELAY` seconds.** Motors should spin up to
   `throttle_hold`. If you wait too long you'll hit the idle auto-disarm again.
5. To stop: FIRE off (throttle → 0), then DISARM. (QuadGuide auto-resets fire on
   disarm, per the fire-gate design.)

If step 4 still doesn't spin the motors, use the decision table in §6.

---

## 5. Observability gap (why this was invisible) + optional code fix

The deployed Pi runs `ui_mode: minimal` (`rpi4b.yaml`). The minimal kiosk UI
(`ground/static/minimal.html`) **only shows the FC armed state — it never shows
the FC flight mode**, even though the server already publishes it
(`fc_mode` in the `/telemetry` SSE stream, `server.py:208`; source is the FC
HEARTBEAT `custom_mode` via `fc/status`). So on the bench you had **no on-screen
signal that the FC was in the wrong mode** when you armed and fired.

ArduCopter `custom_mode` values you care about: **`0` = STABILIZE, `2` = ALT_HOLD,
`4` = GUIDED, `20` = GUIDED_NOGPS.** If the HUD ever showed anything other than
`20` when you armed, that alone explains the whole symptom.

**Recommended QuadGuide hardening (optional, I can implement on request):**

1. **Show the FC mode as a human-readable name** on both UIs (map `custom_mode`
   → `"GUIDED_NOGPS"` etc.), so the operator can see mode at a glance.
2. **Gate the ARM button on `fc_mode == 20`** (or at least warn loudly when
   arming outside GUIDED_NOGPS). This directly removes the "arms from any mode →
   confusing dead arm" trap. *Note:* this narrows the approved design ("QuadGuide
   never gates on mode locally"), so it's your call — I'd make it a config flag
   (`link.require_guided_nogps_to_arm`, default off) to preserve the current
   behavior unless you opt in.

Neither is required to fly once §3 + §4 are done — they just make the failure
mode visible next time.

---

## 6. If it still won't fire — decision table (keyed on what you can observe)

| Observation | Most likely cause | Action |
|---|---|---|
| Mission Planner HUD mode ≠ "Guided_NoGPS" when you arm | Cause (A): wrong mode | Get into GUIDED_NOGPS first (§4.2); check the RC switch position / `FLTMODEx=20` |
| HUD = Guided_NoGPS, arms, props idle, FIRE gives only a weak climb or a sink | Cause (B): `GUID_OPTIONS≠8` | Set `GUID_OPTIONS=8`, write, reboot |
| Arms, props idle, disarms at ~10 s regardless of FIRE | `DISARM_DELAY` + never took off | Fix (A)/(B) so it actually lifts; raise `DISARM_DELAY` for headroom |
| FIRE seems ignored, but ARM works | Confirm `fire/cmd` reaches control | Watch `control/cmd.throttle_norm` (enable `diag.trace`, or the SSE `fire_active` field); the code path is intact, so this points back to (A)/(B) |
| Mission Planner shows "PreArm" / arming refused | prearm/EKF/GPS check | Adjust `ARMING_CHECK` GPS bit (not `0`); ensure EKF happy without GPS |
| Disarms *instantly* (not 10 s) when a GCS drops | GCS failsafe | `FS_GCS_ENABLE=0` for bench, or set `SYSID_MYGCS` deliberately |

---

## 7. Sources (current ArduPilot docs, fetched 2026-07-06)

- Copter Commands in Guided Mode — SET_ATTITUDE_TARGET thrust / GUID_OPTIONS / type_mask:
  https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html
- Complete Parameter List (GUID_OPTIONS, DISARM_DELAY, MOT_SPIN_*, FS_GCS_*):
  https://ardupilot.org/copter/docs/parameters.html
- GCS Failsafe (SYSID_MYGCS / FS_GCS behaviour):
  https://ardupilot.org/copter/docs/gcs-failsafe.html
- Guided Mode overview (GUIDED_NOGPS "flies as if in AltHold"):
  https://ardupilot.org/copter/docs/ac2_guidedmode.html
- ArduCopter `GUID_OPTIONS` bitmask source of truth:
  https://github.com/ArduPilot/ardupilot/blob/master/ArduCopter/Parameters.cpp

`GUID_OPTIONS` bitmask (current): `0`=Allow Arming from Transmitter, `2`=Ignore
pilot yaw, **`3`=SetAttitudeTarget interprets Thrust As Thrust (value 8)**,
`4`=Do not stabilize PositionXY, `5`=Do not stabilize VelocityXY, `6`=Waypoint
nav for position targets, `7`=Allow weathervaning.

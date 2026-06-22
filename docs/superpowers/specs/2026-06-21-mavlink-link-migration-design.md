# Link Module — MAVLink2 / ArduPilot Migration Design Spec
**Date:** 2026-06-21
**Status:** Approved

---

## Overview

Replaces the CRSF link implementation with **MAVLink2 over UART to an ArduPilot
flight controller** (Holybro/generic **H743**, ArduCopter). The airframe no
longer uses the madflight ESP32 FC and quadguide no longer impersonates an RC
radio. Instead:

- **Telemetry in:** quadguide reads vehicle attitude + body rates and IMU
  accel/gyro from MAVLink telemetry streams.
- **Setpoints out:** quadguide streams `SET_ATTITUDE_TARGET` (roll/pitch
  attitude + thrust) to the FC, which flies them in **GUIDED_NOGPS** mode.
- **Arming:** quadguide arms/disarms over MAVLink (`MAV_CMD_COMPONENT_ARM_DISARM`).
- **Mode authority stays with the human pilot.** A manual RC transmitter, via a
  TX switch, toggles the FC between full manual RC control and GUIDED_NOGPS.
  quadguide **never** sends a mode-change command; the pilot's switch is the
  hardware override path.

The change is **confined to `link/` + config + HIL target**. Every bus
contract (`fc/attitude`, `fc/imu`, `control/cmd`, `arm/cmd`, `system/health`)
and every other worker (camera, tracker, guidance, control, ground) is
**untouched**. `core/messages.py` is unchanged — no wire-format changes.

### Why the other workers don't move

- `control/cmd` already carries `roll_deg, pitch_deg, yaw_rate_dps,
  throttle_norm`. The control worker **already hardcodes `yaw_rate_dps = 0.0`**
  (`control/worker.py:122`), so "hold heading" needs no upstream change.
- `AttitudeState` already has native body-rate fields
  (`roll_rate_rps/pitch_rate_rps/yaw_rate_rps`). Under CRSF these were grafted
  in from the custom `0x80` IMU frame; MAVLink `ATTITUDE` supplies them
  directly, so the graft goes away — same bus payload, cleaner source.
- `IMUFrame` (ax,ay,az,gx,gy,gz, body NED/FRD) maps directly onto a MAVLink
  IMU message.

---

## Integration Approach

**pymavlink as a pure codec; keep the existing transport abstraction.**

Build a `pymavlink.dialects.v20.ardupilotmega.MAVLink` object in **codec mode**
(constructed with `file=None`). It is used only to parse incoming bytes and to
serialize outgoing messages — it does **not** own the connection.

- RX: bytes from `serial.read_stream()` → `mav.parse_char(b)` → message objects.
- TX: `mav.<msg>_encode(...).pack(mav)` → bytes → `serial.write(...)`.

`SerialPort` / `TCPSerialPort`, the reconnect loop, the health loop, and the
`DiagTrace` latency capture are preserved as-is. The HIL/flight transport seam
(`platform.serial.mode: uart | tcp`) is unchanged; for HIL the TCP port targets
**ArduPilot SITL** (`tcp:127.0.0.1:5760`), which speaks MAVLink2 natively.

Rejected alternatives: letting `mavutil.mavlink_connection` own the port
(discards the transport abstraction + reconnect/health machinery and tangles
asyncio with a blocking connection); hand-rolling MAVLink2 framing (MAVLink2
headers + per-message CRC_EXTRA + signing is a large, error-prone reimplementation
of pymavlink).

**New dependency:** `pymavlink>=2.4` added to `requirements.txt`.

---

## MAVLink Message Set

### Inbound (FC → quadguide)

| Message | ID | Mapped to | Notes |
|---|---|---|---|
| `ATTITUDE` | 30 | `fc/attitude` | `roll/pitch/yaw` (rad) + `rollspeed/pitchspeed/yawspeed` (rad/s), body frame — fills `AttitudeState` directly |
| `RAW_IMU` | 27 | `fc/imu` | accel (mG) + gyro (mrad/s), body NED/FRD. Only `RAW_IMU` is requested via `SET_MESSAGE_INTERVAL`; `SCALED_IMU2` is a parse-only fallback used **only** until the first `RAW_IMU` arrives, so `fc/imu` is never double-published |
| `HEARTBEAT` | 0 | link state | `base_mode` armed flag + `custom_mode`; learns `target_system`/`target_component`; logs mode changes (replaces `decode_flight_mode`) |
| `COMMAND_ACK` | 77 | log / retry | acknowledges arm/disarm + `SET_MESSAGE_INTERVAL` |

### Outbound (quadguide → FC)

| Message / Command | ID | Purpose | Rate |
|---|---|---|---|
| `SET_ATTITUDE_TARGET` | 82 | roll/pitch attitude + thrust setpoint | **50 Hz, constant** |
| `MAV_CMD_COMPONENT_ARM_DISARM` (via `COMMAND_LONG`) | 400 | arm / disarm | edge-triggered, retried until ACK |
| `MAV_CMD_SET_MESSAGE_INTERVAL` (via `COMMAND_LONG`) | 511 | request `ATTITUDE` + `RAW_IMU` stream rates | once per connect |
| `HEARTBEAT` | 0 | companion liveness | 1 Hz |

### `SET_ATTITUDE_TARGET` field semantics

- `type_mask = 0x07` — ignore all three body-rate fields. ArduPilot GUIDED_NOGPS
  **ignores the body-rate fields regardless**, so attitude (incl. yaw) must be
  carried entirely by the quaternion.
- `q = euler_to_quaternion(roll_rad, pitch_rad, yaw_hold)` — w,x,y,z order
  (`q[0]=w`). `roll_rad`/`pitch_rad` from `ControlCmd`; `yaw_hold` is the latched
  heading (see Yaw Handling).
- `thrust = clamp(throttle_norm, 0.0, 1.0)` — **direct thrust**, requires
  `GUID_OPTIONS` bit 3 set on the FC (see FC Parameters). Without it ArduPilot
  interprets `thrust` as climb rate (0.5 = hover), which is **not** what
  `throttle_norm` means.
- `body_roll_rate/body_pitch_rate/body_yaw_rate = 0`, `target_system/component`
  from the HEARTBEAT handshake, `time_boot_ms` from `monotonic_ns()`.

Roll/pitch are hard-clamped to `airframe.control_limits.max_roll_deg /
max_pitch_deg` inside the encoder as defense-in-depth (control already saturates
upstream).

---

## Yaw Handling — "Hold Heading"

ArduPilot ignores the `SET_ATTITUDE_TARGET` body-rate fields, so `yaw_rate_dps`
cannot pass through. Since the control worker already emits `yaw_rate_dps = 0`,
quadguide holds a fixed heading:

- On the **disarmed → armed edge**, latch `yaw_hold` from the most recent
  `ATTITUDE.yaw`. If no `ATTITUDE` has arrived yet, latch `0.0` and update on the
  first frame.
- Bake `yaw_hold` into every commanded quaternion until the next arm edge.

This keeps the seeker pointed by roll/pitch (thrust-vector tilt) and lets the FC
hold heading. No upstream change; `ControlCmd` wire format is unchanged.

---

## Arming — Edge-Triggered

`arm/cmd` is read in the TX loop (as today) but translated differently:

- Detect transitions of `arm_cmd.armed` (edge), not per-frame level.
- On an edge, send `COMMAND_LONG / MAV_CMD_COMPONENT_ARM_DISARM` with
  `param1 = 1.0` (arm) or `0.0` (disarm). `param2 = 0` (no force-arm).
- Retry the command (e.g. up to N times at the TX cadence) until a matching
  `COMMAND_ACK` with `MAV_RESULT_ACCEPTED` is seen, then stop.
- quadguide owns arming; **the pilot's TX switch owns the RC↔GUIDED_NOGPS toggle.**
  `SET_ATTITUDE_TARGET` streams continuously and is simply ignored by the FC
  until the pilot selects GUIDED_NOGPS — providing a hardware override.
- Prearm/mode-eligibility checks are enforced FC-side; a rejected arm surfaces as
  a non-accepted `COMMAND_ACK` and is logged. quadguide does not gate arming on
  mode locally.

The control worker's existing 2 s arm-dwell (throttle held at 0 after arming;
`control/worker.py:_ARM_DWELL_NS`) is retained unchanged and gives the FC time to
spin up before thrust is commanded.

---

## Startup Handshake (per connect, inside the reconnect loop)

1. Open transport (`SerialPort` or `TCPSerialPort`).
2. Wait for the first `HEARTBEAT` → record `target_system` / `target_component`.
3. Send `SET_MESSAGE_INTERVAL` for `ATTITUDE` (30) and `RAW_IMU` (27) at the
   configured stream rate (default 50 Hz → interval 20000 µs).
4. Start the three loops: RX, TX (50 Hz), companion HEARTBEAT (1 Hz). Health
   loop runs as today.

If the transport drops, the reconnect loop re-runs the handshake from step 1,
identical to the current CRSF behavior (DEGRADED health + 500 ms retry).

---

## Configuration Changes

### `configs/config.yaml`

```yaml
platform:
  serial:
    mode: uart            # "uart" (real FC) | "tcp" (HIL — MAVLink2 over TCP to SITL)
    port: /dev/ttyAMA0    # UART device path — change per SBC
    baud: 115200          # H743 MAVLink2 baud — TODO confirm actual SERIALn_BAUD (115200 vs 921600)
    tcp_host: "127.0.0.1" # used when mode=tcp — ArduPilot SITL host
    tcp_port: 5760        # used when mode=tcp — SITL MAVLink TCP port
    rx_pin: "GPIO15"      # wiring reference only
    tx_pin: "GPIO14"      # wiring reference only

link:
  tx_rate_hz: 50          # SET_ATTITUDE_TARGET stream rate (Hz) — keep constant
  stream_rate_hz: 50      # requested ATTITUDE + RAW_IMU telemetry rate (Hz)
  system_id: 1            # quadguide MAVLink source system id
  component_id: 191       # MAV_COMP_ID_ONBOARD_COMPUTER
  target_system: 1        # FC system id (overridden by HEARTBEAT if it differs)
  target_component: 1     # MAV_COMP_ID_AUTOPILOT1
  arm_retry_count: 5      # COMMAND_LONG arm/disarm retries before giving up
```

**Removed from `link`:** `channels` (the entire µs/tick calibration table) and
`diff_lowpass_alpha` (body rates are now native, not differentiated).

`baud` is left at a placeholder `115200` with a TODO — the actual H743
`SERIALn_BAUD` is to be confirmed; it is config-only and changes no code.

`target_system`/`target_component` defaults are the ArduPilot norm (1 / 1) and
are corrected at runtime from the first HEARTBEAT.

---

## Files

### Removed

- `link/crsf.py` — CRSF framing/CRC/parser. pymavlink owns framing now.
- `CRSF_PROTOCOL.md` — retired (CRSF no longer used).
- CRSF-specific helpers in `link/fc.py` (`encode_rc`, `ChannelConfig`,
  `channel_config_from_cfg`, `pack_channels`, tick conversions).

### New

#### `link/mavlink_codec.py`
Pure codec + math, no bus or serial dependencies (unit-testable on Windows).

- `make_mav(system_id, component_id) -> MAVLink` — builds the codec-mode MAVLink
  object (`file=None`), sets `srcSystem`/`srcComponent`.
- `euler_to_quaternion(roll, pitch, yaw) -> tuple[float, float, float, float]` —
  returns `(w, x, y, z)`.
- `quaternion_to_yaw(...)` / helpers as needed for tests.
- Message-ID constants and the `type_mask` constant (`0x07`).

#### `link/fc.py` (rewritten — semantic map FC ⇄ bus)
Imports `core/messages.py` + `mavlink_codec.py`.

- `decode_attitude(msg) -> AttitudeState` — `ATTITUDE` → angles + native rates.
- `decode_imu(msg) -> IMUFrame` — `RAW_IMU`/`SCALED_IMU2` → m/s² + rad/s.
- `decode_heartbeat(msg) -> tuple[bool, int]` — `(armed, custom_mode)`.
- `encode_attitude_target(mav, cmd, yaw_hold, target_sys, target_comp, limits) -> bytes`
  — builds `SET_ATTITUDE_TARGET` (type_mask 0x07, quaternion, thrust passthrough,
  roll/pitch clamp). Neutral/level when `cmd is None`.
- `encode_arm(mav, arm, target_sys, target_comp) -> bytes` — `COMMAND_LONG`
  `MAV_CMD_COMPONENT_ARM_DISARM`.
- `encode_set_message_interval(mav, msg_id, rate_hz, target_sys, target_comp) -> bytes`
- `encode_heartbeat(mav) -> bytes` — companion heartbeat
  (`MAV_TYPE_ONBOARD_CONTROLLER`, `MAV_AUTOPILOT_INVALID`).

#### `link/worker.py` (rewritten loops, same skeleton)

```python
async def _rx_loop(serial, mav, state, bus, log):
    async for byte in serial.read_stream():
        msg = mav.parse_char(byte)        # may return None or a message
        if msg is None:
            continue
        t = msg.get_type()
        if t == "ATTITUDE":
            state.last_yaw = msg.yaw
            bus.publish("fc/attitude", decode_attitude(msg))
        elif t == "RAW_IMU":
            state.have_raw_imu = True
            bus.publish("fc/imu", decode_imu(msg))
        elif t == "SCALED_IMU2" and not state.have_raw_imu:  # fallback until RAW_IMU seen
            bus.publish("fc/imu", decode_imu(msg))
        elif t == "HEARTBEAT":
            armed, mode = decode_heartbeat(msg)
            state.update_heartbeat(msg.get_srcSystem(), msg.get_srcComponent(),
                                   armed, mode, log)
        elif t == "COMMAND_ACK":
            state.note_ack(msg, log)

async def _tx_loop(serial, mav, bus, state, cfg, log, trace):
    interval = 1.0 / cfg.tx_rate_hz
    while True:
        cmd     = bus.latest("control/cmd")
        arm_cmd = bus.latest("arm/cmd")
        armed   = bool(arm_cmd and arm_cmd.armed)
        state.on_arm_edge(armed, mav, serial, log)     # edge → COMMAND_LONG arm/disarm
        yaw_hold = state.yaw_hold_for(armed)
        await serial.write(encode_attitude_target(mav, cmd, yaw_hold, ...))
        now = monotonic_ns()
        if cmd is not None and cmd.origin_ns > 0:
            trace.latency(now, cmd.timestamp_ns, cmd.origin_ns)   # actuation latency point
        await asyncio.sleep(interval)

async def _heartbeat_loop(serial, mav):
    while True:
        await serial.write(encode_heartbeat(mav))
        await asyncio.sleep(1.0)
```

`_health_loop` and the outer reconnect/`_serial_factory` structure are kept from
the current `worker.py`. The startup handshake (wait HEARTBEAT → request streams)
runs after `serial.open()` before the loops start.

---

## HIL / SITL

The HIL transport seam is unchanged — `platform.serial.mode: tcp` selects
`TCPSerialPort`, which yields raw bytes that pymavlink parses identically to
UART. The target becomes **ArduPilot SITL**:

```bash
# Linux dev box / SBC
sim_vehicle.py -v ArduCopter -f quad --console
# SITL exposes MAVLink2 on tcp:127.0.0.1:5760
```

```yaml
platform:
  serial:
    mode: tcp
    tcp_host: "127.0.0.1"
    tcp_port: 5760
```

The pilot/mode toggle in SITL is emulated by setting GUIDED_NOGPS from the SITL
console/MAVProxy (`mode GUIDED_NOGPS`) — quadguide still only arms + streams
setpoints. The previous custom CRSF-over-TCP `hil-test` bridge is retired for the
link path; `QUADGUIDE_HIL_INTEGRATION.md` is updated to describe SITL. (The
`NetworkCamera` MJPEG path is independent and unchanged.)

---

## FC-Side Parameters (documentation, not code)

To be set on the H743 via Mission Planner / MAVProxy (recorded here, applied
manually):

| Parameter | Value | Reason |
|---|---|---|
| `SERIALn_PROTOCOL` | `2` | MAVLink2 on the companion UART |
| `SERIALn_BAUD` | TBD (matches `serial.baud`) | confirm 115200 vs 921600 |
| `GUID_OPTIONS` | bit 3 set (e.g. `8`) | `SET_ATTITUDE_TARGET.thrust` = direct thrust 0–1 |
| RC mode switch | one position = GUIDED_NOGPS | pilot override / mode authority |
| `FS_GCS_*` | as desired | optional GCS-heartbeat failsafe if companion HEARTBEAT is treated as GCS |

---

## Failsafe Interplay

- **50 Hz constant TX** is preserved as the core invariant (ARCHITECTURE §2.5,
  §12). ArduPilot reverts GUIDED commands if the stream stops; quadguide never
  stops streaming while connected.
- **Pilot RC switch** is the ultimate override — flipping out of GUIDED_NOGPS
  returns full manual control regardless of quadguide state.
- **Transport loss** → `ConnectionError` → DEGRADED health + 500 ms reconnect,
  identical to today.
- quadguide's own watchdogs (`fc/attitude`, `fc/imu`, `target/estimate`
  staleness) are unchanged and still drive the control worker to level/neutral.

---

## Testing

### Unit (run on Windows dev box — pure functions, no bus/serial)
- `euler_to_quaternion` against known rotations (level, ±roll, ±pitch, yaw).
- `decode_attitude` / `decode_imu` from synthetic pymavlink message objects →
  correct `AttitudeState` / `IMUFrame` (units + frame).
- `encode_attitude_target`: type_mask = 0x07, quaternion matches euler, thrust
  passthrough + clamp, roll/pitch clamp to limits, neutral on `cmd=None`.
- Arm edge logic: emits exactly one `COMMAND_LONG` per transition; retries until
  ACK; no per-frame spam.
- Round-trip: encode a message, parse it back through a second `MAVLink` codec.

### Integration (Linux — SBC or Linux dev box, against SITL)
- Link connects to SITL over TCP; `ATTITUDE`/`RAW_IMU` arrive at ~50 Hz and
  populate `fc/attitude`/`fc/imu`.
- Arm via ground UI → SITL arms; disarm works.
- With SITL in GUIDED_NOGPS, `SET_ATTITUDE_TARGET` tilts the vehicle as commanded.
- Reconnect: kill/restart SITL → DEGRADED then re-handshake.
- Full engagement: lock → track → SITL flies the intercept.

> Per project memory, the bus/runtime are Linux-only (`fcntl`); only the
> pure-codec unit tests run on the Windows dev box.

---

## Out of Scope

- GPS / position-based GUIDED modes (we use GUIDED_NOGPS, attitude+thrust only).
- quadguide-initiated mode changes (pilot owns mode).
- MAVLink signing/encryption.
- GPS, battery, and other ArduPilot telemetry beyond `ATTITUDE` + IMU
  (can be added later as new bus topics without touching this design).

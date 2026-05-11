# Link Module — CRSF Design Spec
**Date:** 2026-05-11  
**Status:** Approved

---

## Overview

Replaces the MSP v2 link implementation with CRSF (Crossfire/ExpressLRS) protocol.
The link worker is the sole process that owns the UART serial port. It maintains a
continuous RC channels uplink to the ESP-FC (required to keep the FC out of failsafe
and to receive attitude telemetry), decodes attitude frames from the FC, derives
approximate body rates via finite-difference, and publishes to the shared memory bus.

---

## Protocol Summary

**Transport:** Full-duplex UART, 420000 baud, 8N1  
**Frame format (all types):**
```
[sync=0xC8][len][type][payload...][CRC8-poly0xD5]
```
- `len` = number of bytes following including type, payload, and CRC
- CRC covers type + payload (excludes sync and len)
- Max frame size: 64 bytes

**Frame types used:**

| Direction | Type | Name | Description |
|---|---|---|---|
| FC → companion | 0x1E | ATTITUDE | pitch/roll/yaw as int16, units = 100 µrad |
| companion → FC | 0x16 | RC_CHANNELS_PACKED | 16 channels × 11 bits packed into 22 bytes |

### Attitude frame payload (0x1E)
```
int16_t pitch   # × 1e-4 → radians
int16_t roll    # × 1e-4 → radians
int16_t yaw     # × 1e-4 → radians
```
Byte order: big-endian. Range: -180° to +180°.

### RC channels packed payload (0x16)
22 bytes containing 16 channels × 11 bits, LSB first.
Channel range: 172 (min/1000µs) – 992 (center/1500µs) – 1811 (max/2000µs).

**Important:** The FC will not emit attitude telemetry until it receives a steady CRSF
uplink from the companion. The TX loop must start immediately on port open, sending
neutral (disarmed) frames at the configured rate, before any control commands arrive.

---

## Configuration Changes

### `configs/config.yaml`

```yaml
platform:
  serial:
    port: /dev/ttyS0      # UART device path — varies by SBC
    baud: 420000          # CRSF standard baud rate
    rx_pin: "GPIO15"      # documentation only — physical wiring reference
    tx_pin: "GPIO14"      # documentation only — physical wiring reference

link:
  tx_rate_hz: 50          # RC channels uplink rate to FC (Hz)
  diff_lowpass_alpha: 0.3 # LP filter alpha for differentiated body rates (0=heavy, 1=none)
```

`rx_pin` / `tx_pin` are string labels stored for hardware wiring reference. The serial
driver uses only `port` and `baud`. Porting to a new companion computer requires
updating `port`, `baud`, and optionally the pin labels — no source changes.

---

## New Bus Topic

| Topic | Type | Producer | Consumers |
|---|---|---|---|
| `arm/cmd` | `ArmCmd` | ground worker | link worker |

### `ArmCmd` message (added to `core/messages.py`)

```python
FMT_ARM_CMD = "!QB"   # timestamp(Q=u64) + armed(B=bool) = 9 bytes

@dataclass(frozen=True)
class ArmCmd:
    timestamp_ns: int
    armed: bool
```

---

## Files

### Removed
- `link/msp.py` — MSP v2 implementation, replaced by CRSF

### New / rewritten

#### `link/crsf.py`
Pure protocol implementation, no bus or serial dependencies.

- `CRC8_POLY = 0xD5` — precomputed 256-entry lookup table
- `crc8(data: bytes) -> int` — CRC over type + payload
- `CRSFFrame` dataclass: `type: int`, `payload: bytes`, `timestamp_ns: int`
- `CRSFParser` — stateful byte-by-byte parser
  - States: `WAIT_SYNC`, `READ_LEN`, `READ_TYPE`, `READ_PAYLOAD`, `READ_CRC`
  - `feed(byte: int) -> CRSFFrame | None`
  - Resets on bad length (< 4 or > 62) or CRC mismatch; logs CRC errors at DEBUG
- `build_frame(type: int, payload: bytes) -> bytes` — builds complete frame with CRC
- `pack_channels(channels: list[int]) -> bytes` — packs 16 × 11-bit values into 22 bytes

Frame type constants:
```python
CRSF_ATTITUDE         = 0x1E
CRSF_RC_CHANNELS      = 0x16
```

#### `link/espfc.py`
ESP-FC-specific encoding/decoding. Imports from `crsf.py` and `core/messages.py`.

- `decode_attitude(frame: CRSFFrame, diff: AttitudeDifferentiator) -> tuple[AttitudeState, IMUFrame]`
  - Unpacks 3 × int16 big-endian from payload, multiplies by 1e-4 for radians
  - Calls `diff.update(roll, pitch, yaw, frame.timestamp_ns)`
  - Returns `AttitudeState` (angles + filtered rates) and `IMUFrame` (gx/gy/gz from diff, ax=ay=az=0.0)
- `encode_rc(cmd: ControlCmd | None, armed: bool) -> bytes`
  - Maps `cmd` roll/pitch/throttle/yaw to CRSF ticks via `us_to_ticks`
  - Falls back to neutral (992) on each axis if `cmd` is None
  - CH5: `1811` if `armed` else `172`
  - CH6–CH16: 992
  - Returns complete CRSF frame bytes via `build_frame(CRSF_RC_CHANNELS, pack_channels(...))`

Helpers:
```python
def us_to_ticks(us: float) -> int:
    return int(clamp((us - 1500) * 8/5 + 992, 172, 1811))

def ticks_to_us(ticks: int) -> float:
    return (ticks - 992) * 5/8 + 1500
```

RC channel mapping (same as previous MSP mapping):
- CH1 = roll, CH2 = pitch, CH3 = throttle, CH4 = yaw, CH5 = arm

#### `link/differentiator.py`
Finite-difference attitude differentiator with per-axis first-order LP filter.

```python
class AttitudeDifferentiator:
    def __init__(self, alpha: float):
        # alpha: LP filter coefficient. 0 = heavily smoothed, 1 = no filtering.
        ...

    def update(self, roll: float, pitch: float, yaw: float, now_ns: int
               ) -> tuple[float, float, float]:
        # Returns (roll_rate_rps, pitch_rate_rps, yaw_rate_rps)
        # On first call: returns (0.0, 0.0, 0.0), stores state.
        # dt computed from actual timestamp delta — rate-agnostic.
        # Yaw wraps at ±π: shortest-path delta used to avoid rate spikes at ±180°.
        ...
```

Filter equation per axis:
```
raw_rate = Δangle / dt
filtered = alpha * raw_rate + (1 - alpha) * prev_filtered
```

#### `link/serial_port.py`
Async serial port wrapper.

- `SerialPort(port: str, baud: int)`
- `async open() -> None` — opens via pyserial, wraps fd in `asyncio.StreamReader`
- `async read_stream() -> AsyncGenerator[int, None]` — yields one byte at a time
- `async write(data: bytes) -> None` — non-blocking write
- `async reconnect_loop() -> None` — on `serial.SerialException`, closes port,
  waits 500ms, retries `open()`; caller's `read_stream()` raises to trigger reconnect
- `close() -> None`
- `is_connected: bool` property

#### `link/worker.py`
Process entry point. Three asyncio coroutines under one event loop.

```python
async def rx_loop(serial, parser, differentiator, bus, log):
    async for byte in serial.read_stream():
        frame = parser.feed(byte)
        if frame is None:
            continue
        if frame.type == CRSF_ATTITUDE:
            att, imu = espfc.decode_attitude(frame, differentiator)
            bus.publish("fc/attitude", att)
            bus.publish("fc/imu", imu)

async def tx_loop(serial, bus, config, log):
    interval = 1.0 / config.link.tx_rate_hz
    while True:
        cmd     = bus.latest("control/cmd")
        arm_cmd = bus.latest("arm/cmd")
        armed   = arm_cmd.armed if arm_cmd else False
        data    = espfc.encode_rc(cmd, armed)
        await serial.write(data)
        await asyncio.sleep(interval)

async def health_loop(bus, log):
    while True:
        bus.publish("system/health", HealthReport("link", ProcessState.OK, ""))
        await asyncio.sleep(0.2)   # 5 Hz

async def _main(config, bus):
    serial       = SerialPort(config.platform.serial.port, config.platform.serial.baud)
    differentiator = AttitudeDifferentiator(config.link.diff_lowpass_alpha)
    parser       = CRSFParser()
    await serial.open()
    await asyncio.gather(
        rx_loop(serial, parser, differentiator, bus, log),
        tx_loop(serial, bus, config, log),
        health_loop(bus, log),
    )

def run(config, bus):
    # SIGTERM handler: cancel tasks, serial.close(), bus.detach()
    asyncio.run(_main(config, bus))
```

On serial disconnect: `rx_loop` catches the exception from `read_stream()`, logs error,
publishes `HealthReport("link", ProcessState.DEGRADED, ...)`, awaits
`serial.reconnect_loop()`, then resumes. TX loop writes will silently fail during
outage (logged at DEBUG) and resume once reconnected.

---

## Ground Station Changes

`ground/server.py` gains one new endpoint:

```
POST /arm   body: {"armed": true|false}
→ bus.publish("arm/cmd", ArmCmd(monotonic_ns(), armed))
```

`ground/static/index.html` gains an ARM / DISARM button in the operator controls area.

---

## Test Scripts

### `scripts/test_link_rx.py`
Opens the serial port, runs `CRSFParser`, prints decoded attitude frames in real time.
No bus, no workers — pure protocol verification.

```
usage:  python scripts/test_link_rx.py --port /dev/ttyS0 [--baud 420000] [--duration 10] [--verbose]

normal: [t= 1.234s] roll=  2.31°  pitch= -1.05°  yaw= 45.20°  rates: p= 0.021 q=-0.008 r= 0.003 rad/s
--verbose also prints:
        raw hex: c8 06 1e 00 12 ff a3 01 4c  CRC OK
        (on mismatch): CRC FAIL expected=0x4c got=0x3f  raw: ...
```

### `scripts/test_link_tx.py`
Sends CRSF RC channel frames at a configurable rate. Use to verify FC receives valid
uplink (attitude telemetry should begin), test arming channel, sweep control axes.

```
usage:  python scripts/test_link_tx.py --port /dev/ttyS0 [--baud 420000] [--rate 50]
                                       [--arm] [--roll 992] [--pitch 992]
                                       [--throttle 172] [--yaw 992]

output: [t= 0.020s] TX: ch1= 992 ch2= 992 ch3= 172 ch4= 992 ch5= 172  DISARMED
        [t= 0.040s] TX: ch1= 992 ch2= 992 ch3= 172 ch4= 992 ch5= 172  DISARMED
```

Both scripts import directly from `link/crsf.py` and `link/espfc.py` — they exercise
the real production code paths, not test-only shims.

---

## IPC Table Additions

Full updated IPC table (additions marked **bold**):

| Topic | Type | Producer | Consumers | Approx rate |
|---|---|---|---|---|
| `fc/attitude` | AttitudeState | link worker | guidance, control (watchdog), ground | 50–100 Hz |
| `fc/imu` | IMUFrame | link worker | ground | 50–100 Hz (derived, not raw sensor) |
| **`arm/cmd`** | **ArmCmd** | **ground worker** | **link worker** | **event-driven** |

`fc/imu` accel fields (ax, ay, az) are always 0.0 until a direct IMU is wired to the
companion computer. Gyro fields (gx, gy, gz) are populated from differentiated attitude.
This preserves the bus contract so the IMUFrame consumer (ground station) requires no
change when a real IMU is added later.

---

## Known Limitations

- Body rates are finite-difference approximations of FC attitude, not raw gyro data.
  Differentiation noise is suppressed by the LP filter but rates will lag true body
  rates and may be noisier at high maneuver rates. This is acceptable until a direct
  IMU connection to the companion computer is implemented.
- Accel fields in `IMUFrame` are always zero under this implementation.
- CRSF does not provide raw IMU (gyro/accel) frames — only Euler attitude.

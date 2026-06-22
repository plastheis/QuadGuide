# QuadGuide HIL Integration

**Goal**: run the full quadguide stack against a dev-machine simulator instead
of flight hardware, toggled entirely from `configs/*.yaml`. The camera source
becomes an HTTP MJPEG reader and the MAVLink2 link becomes a TCP socket; every
worker (tracker, guidance, control, link, ground, bus) is untouched.

HIL is **toggled by two config fields**, flipped together:

| Field | Flight | HIL |
|-------|--------|-----|
| `platform.camera.backend` | `v4l2` / `gstreamer` | `network` |
| `platform.serial.mode`    | `uart`               | `tcp`     |

No separate platform, no separate run command, no code path forks beyond the
two construction seams below.

---

## Overview

```
┌─────────────────────────────────┐         ┌──────────────────────────────────┐
│  SBC (quadguide)                │  HTTP   │  Dev Machine (ArduPilot SITL)    │
│                                 │←────────│                                  │
│  NetworkCamera                  │  MJPEG  │  FrameServer :8090               │
│     ↓                           │         │     ↑                           │
│  TrackerWorker → GuidanceWorker │  TCP    │  sim_vehicle.py (ArduCopter)     │
│     ↓              ↓            │────────→│  tcp:127.0.0.1:5760             │
│  ControlWorker → LinkWorker     │←────────│                                  │
│     ↓              ↓            │MAVLink2 │                                  │
│  TCPSerialPort ←───┘            │         │                                  │
└─────────────────────────────────┘         └──────────────────────────────────┘
```

| Real Hardware | HIL Replacement | Direction | Protocol |
|---------------|----------------|-----------|----------|
| USB/CSI camera | `NetworkCamera` → HTTP MJPEG | Dev → SBC | `GET /camera` → multipart JPEG |
| UART to FC | `TCPSerialPort` → TCP socket | Bidirectional | MAVLink2, same as UART |

---

## How the toggle is wired

quadguide has no `PLATFORMS` factory table; the two resources are constructed
at distinct, existing seams, each keyed off a config string:

- **Camera** — `perception/camera/worker.py:run_from_config` looks the source
  class up in `_SOURCES` by `platform.camera.backend`. HIL adds the
  `"network": NetworkCamera` row.
- **Serial** — `link/worker.py` builds its port inside the reconnect loop via
  `_serial_factory(config)`, which switches on `platform.serial.mode`
  (`uart` → `SerialPort`, `tcp` → `TCPSerialPort`). Both satisfy the same async
  port interface (`open`/`read_stream`/`write`/`close`/`is_connected`), so the
  RX/TX loops are transport-agnostic.

Config carries the extra fields through typed dataclasses
(`core/config.py`): `CameraConfig.url`, and `SerialConfig.{mode,tcp_host,
tcp_port}`. `cfg_platform` reads them with defaults, so existing flight configs
that omit them still load (`mode` defaults to `uart`).

---

## Files Created / Modified

| Action | File | What |
|--------|------|------|
| **Create** | `perception/camera/network_source.py` | `NetworkCamera` (HTTP MJPEG, buffersize=1) |
| **Create** | `link/tcp_serial.py` | `TCPSerialPort` (MAVLink2 over TCP) |
| **Modify** | `perception/camera/worker.py` | register `"network"` in `_SOURCES` |
| **Modify** | `link/worker.py` | `_serial_factory()` picks transport by `serial.mode` |
| **Modify** | `core/config.py` | `CameraConfig.url`; `SerialConfig.mode/tcp_host/tcp_port`; tolerant `cfg_platform` |
| **Modify** | `configs/config.yaml`, `configs/rk3588.yaml` | toggle fields + dev-machine IP/port |

`platform/factory.py` is unused (empty) and is **not** part of this; the spec's
earlier "platform dispatch table" did not exist.

### Notes on the two new classes

- **`NetworkCamera`** receives the `CameraConfig` dataclass (like the other
  sources) and reads `url` via `getattr`. It sets `CAP_PROP_BUFFERSIZE=1` so
  cv2 hands back the newest frame rather than a stale queued one — otherwise the
  buffer inflates the glass→track latency the tracker's new-frame gate is built
  to keep low (ARCHITECTURE §13). The frame timestamp is stamped at SBC-receive
  (`monotonic_ns()`), so `origin_ns` measures arrival-at-SBC, not render time on
  the dev machine — acceptable for HIL, but not the same number §13 documents
  for a real camera.
- **`TCPSerialPort`** raises `ConnectionError` from `read_stream` on disconnect
  (peer close or socket error), matching `SerialPort` — so the link worker
  reports `DEGRADED` health and runs its 500 ms reconnect loop identically for
  both transports. `TCP_NODELAY` is set so MAVLink2 bytes aren't Nagle-buffered.

---

## How to Run a HIL Session

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

### 2. Flip the config to HIL

In `configs/rk3588.yaml` (or whichever you pass to `--config`):

```yaml
platform:
  camera:
    backend: network
    url: "http://<dev-machine-ip>:8090/camera"
  serial:
    mode: tcp
    tcp_host: "<dev-machine-ip>"
    tcp_port: 5760
```

Equivalently, toggle without editing the file:

```bash
python scripts/run.py --config configs/rk3588.yaml \
  --set platform.camera.backend=network \
  --set platform.serial.mode=tcp
```

(IP/port come from the YAML; `--set` only flips the two mode strings.)

### 3. Lock on and arm

- Open quadguide's ground UI at `http://<sbc-ip>:8080`, drag a lock-on bbox
  around the rendered target.
- Arm via the ground UI.
- The loop closes: quadguide tracks the synthetic target → MAVLink2 over TCP →
  SITL simulates the airframe → rendered frames stream back over MJPEG.

---

## Verification Checklist

- [ ] `cfg_platform` loads the HIL config without `KeyError` (port/baud absent OK)
- [ ] `NetworkCamera` opens the MJPEG stream; frames arrive at ~30 Hz
- [ ] `TCPSerialPort` connects to SITL's TCP port (link logs `HIL: MAVLink2 over TCP`)
- [ ] MAVLink2 SET_ATTITUDE_TARGET flows SBC → SITL
- [ ] MAVLink2 ATTITUDE + RAW_IMU telemetry flows SITL → SBC
- [ ] On SITL restart, link reports DEGRADED then reconnects
- [ ] TrackerWorker initializes and tracks the target
- [ ] Full engagement: lock → track → intercept (or scenario completion)

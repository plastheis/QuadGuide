# SBC Deployment: systemd Autostart + WiFi AP Design

**Date:** 2026-06-26
**Status:** draft
**Target board:** Radxa ROCK 5C (RK3588S2), Radxa OS (Debian/Ubuntu, NetworkManager)

---

## 1. Scope

Make quadguide run on boot on the SBC, and add a self-hosted WiFi access point
for the ground station — replacing the need to reach the ground webserver only
over the HaLow ethernet bridge.

Delivered as a single idempotent script, `scripts/install_sbc.sh`, run once as
root on the board. It:

1. Installs and enables a **single** systemd service that supervises the whole
   stack on boot.
2. Creates a boot-persistent **WiFi AP** (`drone` / WPA2 `drone123`) on the
   onboard radio, serving the ground UI at `http://10.42.0.1:8080`.

Plus supporting repo changes: a service unit template, a corrected
`configs/rk3588.yaml` for real flight (CSI camera + UART link), deletion of the
stale per-worker unit stubs, and an ARCHITECTURE.md fix.

The spec also carries two **on-SBC bring-up checklists** (§7) that cannot be
executed or verified from the development machine (Windows): enabling the UART
overlay, and bringing up the unsupported OV9281 sensor.

### Non-goals

- No application/source code changes. The webserver already binds `0.0.0.0`,
  so it serves on the AP interface with no edit (`ground/worker.py:13`).
- The WiFi AP does **not** replace HaLow — they coexist on separate interfaces.
- No internet sharing / NAT through the AP (ground clients only need to reach
  the SBC itself).
- HIL workflows still work by flipping the documented toggles in
  `configs/rk3588.yaml` back to `tcp` / `raw_tcp` — defaults change to flight,
  the toggle comments stay.

---

## 2. Background: why one service, not six

ARCHITECTURE.md §8 describes six systemd units (`qg-camera`, `qg-tracker`, …),
"each a thin invocation of the matching worker." **This is not viable with the
current code** and the seven `systemd/*.service` / `quadguide.target` files are
empty stubs.

`core/bus.py:Bus.__init__` builds every topic's `multiprocessing.Lock()`,
`multiprocessing.Value`, and an anonymous `os.pipe()` pair, all created **once
in the parent and inherited across `fork()`** (the class docstring states this
explicitly). Those locks and pipe fds cannot be shared between independent
systemd services — only the named `SharedMemory` could. Splitting workers into
separate units would require rewriting the bus onto named shm + a socket-based
wakeup mechanism: a large refactor, out of scope.

`scripts/run.py` already is the correct supervisor: it builds the bus +
frame buffer in the parent, forks all workers, and on any worker exit sends
SIGTERM to all, waits a 5 s grace, SIGKILLs stragglers, then unlinks shm. So
systemd's only job is to keep that one parent process alive.

**Decision:** one service, `quadguide.service`, runs `scripts/run.py`. Delete
the stub units. Fix §8.

---

## 3. Component: systemd service

### 3.1 Unit template — `systemd/quadguide.service`

Shipped in-repo with `@PLACEHOLDER@` tokens that the installer substitutes, so
no path is hardcoded to one checkout location.

```ini
[Unit]
Description=quadguide flight guidance stack
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=@REPO_DIR@
ExecStartPre=/bin/mkdir -p @LOG_DIR@
ExecStart=@PYTHON@ scripts/run.py --config @CONFIG@
Restart=always
RestartSec=3
KillMode=mixed
TimeoutStopSec=10
LimitRTPRIO=99
LimitMEMLOCK=infinity
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Rationale, line by line:

- **`After/Wants=network.target`** — soft ordering only. The stack does not
  hard-depend on the network (the webserver binds `0.0.0.0` regardless, and the
  AP is brought up independently by NetworkManager autoconnect). Deliberately
  **not** `network-online.target` so flight isn't blocked waiting on DHCP.
- **`User=root`** — chosen in brainstorming. Gives the control worker's
  `SCHED_FIFO` + `sched_setaffinity` (needs `CAP_SYS_NICE`), `/dev/video*`,
  `/dev/ttyS6`, and `/var/log/quadguide` with zero extra group/cap wiring.
- **`ExecStart=@PYTHON@ scripts/run.py --config @CONFIG@`** — `@PYTHON@` is the
  venv interpreter (`@REPO_DIR@/.venv/bin/python`), so no `source activate`
  needed; `@CONFIG@` defaults to `configs/rk3588.yaml`.
- **`Restart=always RestartSec=3`** — a flight computer should keep trying to
  recover. No `StartLimit*` cap: we never want systemd to permanently give up
  mid-mission. Trade-off in §3.3.
- **`KillMode=mixed TimeoutStopSec=10`** — on stop, SIGTERM goes to the main
  process (`run.py`) only, letting **its** ordered `_shutdown()` run (SIGTERM
  children → 5 s grace → SIGKILL → unlink shm). After 10 s systemd SIGKILLs the
  whole cgroup as a backstop. (`run.py`'s own grace is 5 s, comfortably inside
  10 s.)
- **`LimitRTPRIO=99`** — without this the control worker's
  `sched_setscheduler(SCHED_FIFO, prio=80)` (config `control_fifo_prio`) can be
  denied by the default rlimit. **`LimitMEMLOCK=infinity`** — headroom for any
  mlock the RT path uses; harmless otherwise.

### 3.2 Installer responsibilities (systemd portion)

`scripts/install_sbc.sh`, run as root from the repo root:

1. Refuse to run if `EUID != 0`.
2. Resolve `REPO_DIR` from the script's own location (`realpath`), and
   `PYTHON=$REPO_DIR/.venv/bin/python` (error if absent).
3. `LOG_DIR` from config (`/var/log/quadguide`); `CONFIG` = arg 1 or default
   `configs/rk3588.yaml`.
4. Render the template to `/etc/systemd/system/quadguide.service` by
   substituting the four tokens (`sed`).
5. `mkdir -p "$LOG_DIR"`.
6. `systemctl daemon-reload` then `systemctl enable --now quadguide.service`.

Idempotent: re-running overwrites the unit and re-applies cleanly.

### 3.3 Crash-loop behavior (accepted)

If the active config can't initialize a resource (FC unpowered, OV9281 not yet
brought up, UART overlay not enabled), a worker raises, `run.py` exits non-zero,
and systemd restarts every 3 s indefinitely. This is the intended
"keep-trying-to-recover" posture for an airframe. It is visible and diagnosable:

```
systemctl status quadguide
journalctl -u quadguide -f
```

The operator brings the missing resource online and the next restart succeeds —
no manual `systemctl start` needed.

---

## 4. Component: WiFi access point

### 4.1 Hardware & driver reality

The ROCK 5C's onboard wireless is an **AIC8800D80** (AICSemi). Radxa OS ships
the AIC8800 driver, and Radxa's official Rock 5C docs document AP mode **through
NetworkManager** (mode "Access Point" + IPv4 "Shared"). So no hostapd or driver
build is required — `nmcli` is the supported path.

Caveats from Radxa's docs, both handled in the profile below:
- **WPA3/SAE can fail to connect on this chip** → force `wpa-psk` (WPA2).
- **5 GHz needs an explicit channel** → we use 2.4 GHz (`band bg`), which also
  gives better range for a ground link. (5 GHz is a future tunable, not now.)

### 4.2 NetworkManager profile — `qg-ap`

Created by the installer (idempotent: delete any existing `qg-ap` first):

```sh
nmcli connection delete qg-ap 2>/dev/null || true
nmcli connection add type wifi ifname "$WIFI_IFACE" con-name qg-ap \
    autoconnect yes ssid "drone"
nmcli connection modify qg-ap \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    ipv4.method shared \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "drone123"
nmcli connection up qg-ap
```

- **`802-11-wireless.mode ap`** + **`ipv4.method shared`** → NetworkManager runs
  its built-in dnsmasq (DHCP + DNS) and assigns the SBC the gateway
  **`10.42.0.1/24`** (NM's default shared subnet; clients get `10.42.0.x`).
- **`autoconnect yes`** → the AP comes up on every boot **without** a separate
  systemd unit. NM owns its lifecycle.
- **`$WIFI_IFACE`** resolved by the installer (default `wlan0`; confirmed via
  `nmcli device status | grep wifi`).
- **No `iptables MASQUERADE`** — Radxa's doc adds it only for internet sharing
  to an egress NIC, which we don't need.

### 4.3 Coexistence with HaLow

HaLow is an **ethernet** bridge on a separate interface (e.g. `end0`); the AP is
the WiFi radio. No AP/station radio conflict. The ground server binds
`0.0.0.0:8080`, so it is reachable simultaneously at:

- `http://10.42.0.1:8080` over the `drone` WiFi AP, and
- the HaLow bridge IP over ethernet (unchanged).

### 4.4 Installer responsibilities (AP portion)

1. Verify NetworkManager is the active network manager (`nmcli` present and
   `nmcli -t -f RUNNING general` is `running`); abort with guidance if not.
2. Resolve `$WIFI_IFACE`.
3. Apply the `qg-ap` profile above.
4. Print: SSID `drone`, PSK `drone123`, URL `http://10.42.0.1:8080`.

---

## 5. Component: `configs/rk3588.yaml` flight edits

The current file is HIL-pointed (`serial.mode: tcp`, dev-machine SITL/camera
hosts). For boot autostart it must be self-contained flight. Edits:

```yaml
platform:
  camera:
    backend: gstreamer        # was v4l2; CSI via GStreamer → CSICamera
    # OV9281 (mono global shutter, 1280x800). Capture FULL sensor to preserve
    # the 79° FoV, then downscale for tracker latency. VERIFY node + format on
    # the SBC (see spec §7.2): node is typically /dev/video11; mono output may
    # be GRAY8/NV12 depending on whether rkcif (raw) or rkisp (ISP) is used.
    pipeline: "v4l2src device=/dev/video11 io-mode=4 ! video/x-raw,format=NV12,width=1280,height=800,framerate=60/1 ! videoscale ! video/x-raw,width=640,height=400 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false"
    width: 640                # downscaled frame the stack processes
    height: 400               # 16:10 (OV9281 native aspect) — full FoV preserved
    fps: 60
  serial:
    mode: uart                # was tcp (HIL)
    port: /dev/ttyS6          # UART6-M1 (overlay "Enable UART6-M1"); see §7.1

guidance:
  fov_horizontal_rad: 1.379   # 79° HORIZONTAL. If 79° is the lens DIAGONAL spec,
                              # horizontal at 16:10 ≈ 67° → set 1.17 instead.
```

The HIL toggle comments already in the file are preserved, so reverting to HIL
remains a one-edit operation.

**FoV assumption:** `79°` is taken as the horizontal field of view (the config
field is `fov_horizontal_rad`, consumed by `guidance/pure_pursuit.py` to turn a
normalized centroid into a LOS angle). The downscale preserves FoV because the
full sensor array is captured before `videoscale`; cropping would not.

**OpenCV/GStreamer dependency:** `CSICamera` opens
`cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)` (`sources.py:89`). The pip
`opencv-python` wheel is built **without** GStreamer and will silently fail to
open. The SBC venv must use an OpenCV with GStreamer (distro `python3-opencv`,
or a custom build). Verification step in §7.2.

---

## 6. Repo changes summary

| Change | File(s) |
| --- | --- |
| Add installer | `scripts/install_sbc.sh` (new, executable) |
| Add unit template | `systemd/quadguide.service` (new, with `@TOKENS@`) |
| Delete stale stubs | `systemd/qg-camera.service`, `qg-control.service`, `qg-ground.service`, `qg-guidance.service`, `qg-link.service`, `qg-tracker.service`, `quadguide.target` |
| Flight config | `configs/rk3588.yaml` (camera, serial, guidance.fov) |
| Doc fix | `ARCHITECTURE.md` §8 → single-service model |

`scripts/deploy.py` (empty) is left untouched — out of scope.

---

## 7. On-SBC bring-up checklists (cannot be run from dev machine)

These are operator steps performed on the board, captured here so the
implementation plan can turn them into a runbook. They are **not** automated by
`install_sbc.sh` because they involve hardware enumeration and reboots.

### 7.1 UART6 to the flight controller

Authoritative Rock 5C 40-pin mapping (Radxa hardware-interface doc):

| Signal | Pin | GPIO | Connect to FC |
| --- | --- | --- | --- |
| `UART6_TX_M1` (SBC→FC) | **19** | GPIO1_A1 | FC **RX** |
| `UART6_RX_M1` (SBC←FC) | **21** | GPIO1_A0 | FC **TX** |
| GND | **20** | — | FC GND |

Steps:
1. `rsetup` → Overlays → enable **"Enable UART6-M1"** → reboot.
2. Confirm `/dev/ttyS6` exists. (Frees the port from the default `ttyFIQ0` mux.)
3. Wire FC ↔ SBC per the table — cross TX/RX, common GND, **do not connect 5 V**.
4. On the FC (ArduPilot): set the matching `SERIALn_PROTOCOL=2` (MAVLink2) and
   `SERIALn_BAUD=115` (matches config `baud: 115200`). Bump both to `921` if the
   100 Hz SET_ATTITUDE_TARGET stream + telemetry saturates 115200.
5. Also required by the link design: `GUID_OPTIONS` bit 3 (direct thrust) — see
   ARCHITECTURE §6.4 / §12.

Alternative UART if pins 19/21 are occupied: **UART4-M2** → TX **pin 7**, RX
**pin 29**, node `/dev/ttyS4` (set `serial.port: /dev/ttyS4`); GND is on pin 9
or 25 (less adjacent than UART6's pin 20).

### 7.2 OV9281 sensor bring-up (unsupported sensor)

Radxa officially supports only OV5647, IMX415, IMX219 on the Rock 5C — there is
**no stock `rsetup` overlay for the OV9281**. Bring-up is a custom-overlay task.

1. **Driver.** Confirm the kernel has the OV9281 module:
   `modinfo ov9281` (Rockchip BSP kernels ship `drivers/media/i2c/ov9281.c`,
   `CONFIG_VIDEO_OV9281`). If absent, enable/build the module for the running
   kernel.
2. **Custom overlay.** Copy the board's shipped OV5647 overlay
   (the "OKDO 5MP" / `…rpi-camera-v1p3-ov5647…` `.dts`, which already targets the
   Rock 5C CSI connector, I2C bus, and the rkcif/rkisp graph) as the template,
   then change the sensor node:
   - `compatible = "ovti,ov9281";`
   - I2C `reg = <0x60>;` (OV9281 default address — verify with `i2cdetect`),
   - `data-lanes = <1 2>;` (OV9281 is 2-lane),
   - mono media-bus format `Y10` (`MEDIA_BUS_FMT_Y10_1X10`),
   - keep `clock-frequency = <24000000>` and the existing
     `csi2_dphy`/endpoint linkage.

   Build with the kernel overlay toolchain and load it (or place under the
   `/boot` overlay dir and enable). Reboot.
3. **Enumerate.** `v4l2-ctl --list-devices` and `media-ctl -p` (or
   `media-ctl -p -d /dev/mediaN`) to find the capture video node (Rock 5C
   convention: **`/dev/video11`**) and the `ov9281` subdev.
4. **Set the pipeline format** along the graph (sensor → csi2 → isp/cif → node),
   e.g.
   `media-ctl --set-v4l2 '"ov9281 …":0[fmt:Y10_1X10/1280x800]'`, propagating to
   the capture node. Confirm with `v4l2-ctl -d /dev/video11 --all` and a test
   grab.
5. **Match the GStreamer pipeline** in `configs/rk3588.yaml` to the actual node
   and output format. If the node delivers mono `GRAY8` (raw via rkcif) rather
   than `NV12` (via ISP), change `format=NV12` → `format=GRAY8` in the caps.
6. **OpenCV+GStreamer** (§5): ensure the venv's `cv2` reports GStreamer support
   (`python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer`
   → `YES`). If not, install the distro `python3-opencv` into the environment.

Until §7.2 completes, the camera worker will fail to open and the service will
crash-loop (§3.3) — expected.

---

## 8. Verification

On the SBC after `install_sbc.sh` and the §7 bring-up:

- `systemctl is-enabled quadguide` → `enabled`; `systemctl status quadguide` →
  `active (running)`; `journalctl -u quadguide` shows
  `started 6 workers: [...]`.
- Reboot; confirm the service comes back `active` and the `drone` SSID is
  broadcasting (`nmcli device wifi list` from a phone, or `nmcli connection show
  --active` on the SBC shows `qg-ap`).
- A laptop joins `drone` (PSK `drone123`), gets a `10.42.0.x` lease, and loads
  `http://10.42.0.1:8080`.
- The HaLow ethernet path to the same UI still works (coexistence).
- `ls /dev/ttyS6` present; FC HEARTBEAT seen in the link worker log.

Items that can be checked now on the dev machine (mock/no hardware): the
installer's argument parsing / token substitution renders a valid unit
(`systemd-analyze verify` on the rendered file in a Linux container if
available), and `configs/rk3588.yaml` still loads via `core/config.py`
(`load_config`) without raising.

---

## 9. Open items / future

- 5 GHz AP (needs explicit channel for AIC8800) — deferred; 2.4 GHz now.
- A dedicated `configs/flight.yaml` separate from `rk3588.yaml` — not done; per
  the brainstorming decision we edit `rk3588.yaml` in place and keep HIL toggles.
- Baud escalation to 921600 if 115200 saturates at the 100 Hz attitude stream.

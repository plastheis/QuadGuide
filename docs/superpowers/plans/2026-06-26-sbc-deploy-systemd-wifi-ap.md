# SBC Deployment: systemd Autostart + WiFi AP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the quadguide stack start on boot on the Radxa ROCK 5C and serve the ground UI over a self-hosted WiFi access point, installed by one idempotent script.

**Architecture:** A single `quadguide.service` runs `scripts/run.py` (the existing fork-and-supervise orchestrator) — the per-worker unit split in ARCHITECTURE §8 is not viable because the bus's locks/pipes are fork-inherited. `scripts/install_sbc.sh` renders + enables that unit and creates a boot-persistent NetworkManager WiFi AP (`drone` / WPA2). `configs/rk3588.yaml` is switched from HIL to real flight (CSI OV9281 + UART link). On-SBC hardware bring-up (UART overlay, unsupported OV9281 sensor, OpenCV-GStreamer) is captured as an operator runbook, not automated.

**Tech Stack:** bash, systemd, NetworkManager (`nmcli`), GStreamer/V4L2, Python 3 (config loader), pytest.

## Global Constraints

- **Dev box is Windows; it cannot run the stack, systemd, or nmcli.** Only config-load and static-artifact tests run here (the runtime needs Linux `fcntl`). systemd/AP/OV9281 verification happens on the SBC.
- **Target:** Radxa ROCK 5C (RK3588S2), Radxa OS (Debian/Ubuntu) with NetworkManager running.
- **Service runs as `root`.** Single unit only — `core/bus.py` builds `multiprocessing.Lock`/`Value`/anonymous `os.pipe()` once in the parent and inherits them across `fork()`; workers cannot be separate units.
- **No application/source code changes.** Webserver already binds `0.0.0.0:8080` (`ground/worker.py:13`).
- **systemd unit policy (verbatim):** `Type=simple`, `User=root`, `Restart=always`, `RestartSec=3`, `KillMode=mixed`, `TimeoutStopSec=10`, `LimitRTPRIO=99`, `LimitMEMLOCK=infinity`, `After=network.target`, `WantedBy=multi-user.target`, `Environment=PYTHONUNBUFFERED=1`.
- **WiFi AP (verbatim):** NM profile `qg-ap`; SSID `drone`; WPA2 `wifi-sec.key-mgmt wpa-psk` PSK `drone123`; `802-11-wireless.mode ap`; `band bg` (2.4 GHz); `ipv4.method shared`; `autoconnect yes`; gateway `10.42.0.1`; no NAT. Coexists with the HaLow ethernet bridge.
- **Flight config (verbatim):** `camera.backend: gstreamer`; capture full 1280×800, downscale 640×400 @ 60 fps; `serial.mode: uart`; `serial.port: /dev/ttyS6` (UART6-M1); `guidance.fov_horizontal_rad: 1.379` (79° horizontal).
- Spec: `docs/superpowers/specs/2026-06-26-sbc-deploy-systemd-wifi-ap-design.md`.

**Before Task 1, create a working branch (we are on `main`):**
```bash
git checkout -b feat/sbc-deploy-systemd-wifi-ap
```

## File Structure

| File | Responsibility |
| --- | --- |
| `configs/rk3588.yaml` (modify) | Flight defaults: CSI OV9281 camera, UART link, 79° FoV |
| `tests/unit/test_rk3588_config.py` (create) | Guards the committed flight config loads + values |
| `systemd/quadguide.service` (create) | Single-service unit template with `@TOKEN@` placeholders |
| `tests/unit/test_systemd_unit.py` (create) | Guards the unit's sections/directives/tokens |
| `scripts/install_sbc.sh` (create) | Idempotent root installer: unit + WiFi AP |
| `systemd/qg-*.service`, `quadguide.target` (delete) | Stale empty stubs of the non-viable 6-unit model |
| `ARCHITECTURE.md` §8 (modify) | Correct startup model to single service |
| `docs/sbc-setup.md` (create) | Operator runbook for on-SBC bring-up (UART, OV9281, verify) |

---

### Task 1: Flight config + load guard

**Files:**
- Modify: `configs/rk3588.yaml` (camera block, serial block, `guidance.fov_horizontal_rad`)
- Test: `tests/unit/test_rk3588_config.py`

**Interfaces:**
- Consumes: `quadguide.core.config.load_config(path, overrides) -> dict`, `cfg_platform(dict) -> PlatformConfig` (existing).
- Produces: a committed `configs/rk3588.yaml` whose `cfg_platform` yields `camera.backend=="gstreamer"`, `serial.mode=="uart"`, `serial.port=="/dev/ttyS6"`; `config["guidance"]["fov_horizontal_rad"]==1.379`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_rk3588_config.py`:
```python
from pathlib import Path

from quadguide.core.config import load_config, cfg_platform

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "rk3588.yaml"


def test_rk3588_is_flight_default():
    """The committed rk3588.yaml is the flight default the boot service uses.
    HIL (tcp / raw_tcp) is a local, uncommitted toggle — see the file's comments.
    """
    config = load_config(str(CONFIG), {})
    pcfg = cfg_platform(config)

    # CSI OV9281 via GStreamer, full sensor downscaled to 640x400.
    assert pcfg.camera.backend == "gstreamer"
    assert "/dev/video11" in pcfg.camera.pipeline
    assert "format=BGR" in pcfg.camera.pipeline
    assert (pcfg.camera.width, pcfg.camera.height, pcfg.camera.fps) == (640, 400, 60)

    # MAVLink over the real UART (UART6-M1).
    assert pcfg.serial.mode == "uart"
    assert pcfg.serial.port == "/dev/ttyS6"

    # 79 degrees horizontal field of view.
    assert abs(config["guidance"]["fov_horizontal_rad"] - 1.379) < 1e-6
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_rk3588_config.py -v`
Expected: FAIL — current config has `backend: v4l2`, `serial.mode: tcp` (HIL), `fov_horizontal_rad: 0.972`.

- [ ] **Step 3: Edit `configs/rk3588.yaml`**

In `platform.camera`, replace `backend`, `pipeline`, `width`, `height`, `fps` with:
```yaml
    backend: gstreamer       # CSI camera via GStreamer → CSICamera (was v4l2)
    # OV9281 (mono global shutter, native 1280x800). Capture the FULL sensor to
    # preserve the 79° FoV, then downscale for tracker latency. VERIFY node +
    # format on the SBC (see docs/sbc-setup.md): node is typically /dev/video11;
    # mono output may be NV12 (rkisp) or GRAY8 (rkcif raw) — adjust caps to match.
    pipeline: "v4l2src device=/dev/video11 io-mode=4 ! video/x-raw,format=NV12,width=1280,height=800,framerate=60/1 ! videoscale ! video/x-raw,width=640,height=400 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false"
    width: 640               # downscaled frame the stack processes
    height: 400              # 16:10 (OV9281 native aspect) — full FoV preserved
    fps: 60
```
Keep the existing `url`, `raw_tcp_host`, `raw_tcp_port` lines and their HIL comments below (so reverting to HIL stays a one-edit toggle).

In `platform.serial`, change the two lines:
```yaml
    mode: uart            # was tcp (HIL); real MAVLink2 over UART to the FC
    port: /dev/ttyS6      # UART6-M1 (enable overlay "Enable UART6-M1"); see docs/sbc-setup.md
```
Leave `tcp_host` / `tcp_port` and their HIL comments in place.

In `guidance`, change the FoV line:
```yaml
  fov_horizontal_rad: 1.379   # 79° HORIZONTAL (OV9281 + lens). If 79° is the lens
                              # DIAGONAL spec, horizontal at 16:10 ≈ 67° → use 1.17.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/unit/test_rk3588_config.py -v`
Expected: PASS.

- [ ] **Step 5: Run the unit suite for regressions**

Run: `python -m pytest tests/unit -q`
Expected: PASS (no other config-dependent test regressed). On Windows, Linux-only tests may skip — that is expected.

- [ ] **Step 6: Commit**

```bash
git add configs/rk3588.yaml tests/unit/test_rk3588_config.py
git commit -m "config: switch rk3588.yaml to flight (OV9281 CSI + UART link)"
```

---

### Task 2: systemd unit template

**Files:**
- Create: `systemd/quadguide.service`
- Test: `tests/unit/test_systemd_unit.py`

**Interfaces:**
- Produces: `systemd/quadguide.service` containing `@REPO_DIR@`, `@PYTHON@`, `@CONFIG@`, `@LOG_DIR@` tokens (substituted by `install_sbc.sh` in Task 3) and the Global-Constraints unit directives.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_systemd_unit.py`:
```python
import configparser
from pathlib import Path

UNIT = Path(__file__).resolve().parents[2] / "systemd" / "quadguide.service"


def _parse():
    cp = configparser.ConfigParser()
    cp.optionxform = str  # preserve directive case: ExecStart, not execstart
    cp.read(UNIT)
    return cp


def test_unit_sections_and_directives():
    cp = _parse()
    assert {"Unit", "Service", "Install"} <= set(cp.sections())

    svc = cp["Service"]
    assert svc["Type"] == "simple"
    assert svc["User"] == "root"
    assert "scripts/run.py" in svc["ExecStart"]
    assert svc["Restart"] == "always"
    assert svc["RestartSec"] == "3"
    assert svc["KillMode"] == "mixed"
    assert svc["TimeoutStopSec"] == "10"
    assert svc["LimitRTPRIO"] == "99"

    assert cp["Unit"]["After"] == "network.target"
    assert cp["Install"]["WantedBy"] == "multi-user.target"


def test_unit_keeps_installer_tokens():
    text = UNIT.read_text()
    for tok in ("@REPO_DIR@", "@PYTHON@", "@CONFIG@", "@LOG_DIR@"):
        assert tok in text, f"missing installer token {tok}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_systemd_unit.py -v`
Expected: FAIL — `systemd/quadguide.service` is currently an empty stub (no sections).

- [ ] **Step 3: Create `systemd/quadguide.service`**

Overwrite the empty stub with:
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

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/unit/test_systemd_unit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add systemd/quadguide.service tests/unit/test_systemd_unit.py
git commit -m "systemd: add single quadguide.service unit template"
```

---

### Task 3: Installer script

**Files:**
- Create: `scripts/install_sbc.sh` (executable)

**Interfaces:**
- Consumes: `systemd/quadguide.service` template (Task 2); `configs/rk3588.yaml` (Task 1).
- Produces: `/etc/systemd/system/quadguide.service` (rendered) + enabled service + `qg-ap` NM profile, on the SBC. `--dry-run` prints both artifacts and touches nothing privileged (the only check runnable on this Windows box).

- [ ] **Step 1: Create `scripts/install_sbc.sh`**

```bash
#!/usr/bin/env bash
# install_sbc.sh — set up quadguide systemd autostart + WiFi AP on a Radxa ROCK 5C.
#
# Run once as root from the repo:
#     sudo ./scripts/install_sbc.sh [CONFIG_PATH]
#
# Preview without root / without changing anything (renders unit + nmcli plan):
#     ./scripts/install_sbc.sh --dry-run [CONFIG_PATH]
#
# Env overrides: AP_SSID (default drone), AP_PSK (default drone123),
#                WIFI_IFACE (default: first nmcli wifi device), LOG_DIR (/var/log/quadguide)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
UNIT_TEMPLATE="$REPO_DIR/systemd/quadguide.service"
UNIT_DEST="/etc/systemd/system/quadguide.service"

DRY_RUN=0
CONFIG="configs/rk3588.yaml"
for arg in "$@"; do
    case "$arg" in
        --dry-run|--render-only) DRY_RUN=1 ;;
        -h|--help) grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --*) echo "unknown flag: $arg" >&2; exit 2 ;;
        *) CONFIG="$arg" ;;
    esac
done

AP_SSID="${AP_SSID:-drone}"
AP_PSK="${AP_PSK:-drone123}"
LOG_DIR="${LOG_DIR:-/var/log/quadguide}"

render_unit() {
    [ -f "$UNIT_TEMPLATE" ] || { echo "missing template: $UNIT_TEMPLATE" >&2; exit 1; }
    sed \
        -e "s|@REPO_DIR@|$REPO_DIR|g" \
        -e "s|@PYTHON@|$PYTHON|g" \
        -e "s|@CONFIG@|$CONFIG|g" \
        -e "s|@LOG_DIR@|$LOG_DIR|g" \
        "$UNIT_TEMPLATE"
}

ap_iface() {
    if [ -n "${WIFI_IFACE:-}" ]; then echo "$WIFI_IFACE"; return 0; fi
    nmcli -t -f DEVICE,TYPE device status 2>/dev/null | awk -F: '$2=="wifi"{print $1; exit}'
}

if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== rendered $UNIT_DEST ==="
    render_unit
    echo
    echo "=== WiFi AP plan ==="
    echo "iface : $(ap_iface || true)"
    echo "ssid  : $AP_SSID"
    echo "psk   : $AP_PSK  (wpa-psk / 2.4GHz / ipv4 shared / autoconnect)"
    echo "url   : http://10.42.0.1:8080"
    exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "must run as root (use sudo). Preview with: $0 --dry-run" >&2
    exit 1
fi
[ -x "$PYTHON" ] || { echo "venv python missing/not executable: $PYTHON" >&2; exit 1; }

echo "[1/2] installing $UNIT_DEST"
mkdir -p "$LOG_DIR"
render_unit > "$UNIT_DEST"
systemctl daemon-reload
systemctl enable --now quadguide.service

echo "[2/2] configuring WiFi AP '$AP_SSID'"
command -v nmcli >/dev/null || { echo "nmcli not found — NetworkManager required" >&2; exit 1; }
[ "$(nmcli -t -f RUNNING general 2>/dev/null)" = "running" ] \
    || { echo "NetworkManager is not running" >&2; exit 1; }
IFACE="$(ap_iface)"
[ -n "$IFACE" ] || { echo "no wifi device found (see: nmcli device status)" >&2; exit 1; }
nmcli connection delete qg-ap >/dev/null 2>&1 || true
nmcli connection add type wifi ifname "$IFACE" con-name qg-ap autoconnect yes ssid "$AP_SSID"
nmcli connection modify qg-ap \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    ipv4.method shared \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$AP_PSK"
nmcli connection up qg-ap

cat <<EOF

Done.
  Service  : quadguide.service (enabled; follow with: journalctl -u quadguide -f)
  WiFi AP  : SSID '$AP_SSID'  PSK '$AP_PSK'  (2.4GHz, iface $IFACE)
  Ground UI: http://10.42.0.1:8080  (also reachable over the HaLow bridge)
EOF
```

Then make it executable:
```bash
chmod +x scripts/install_sbc.sh
```

- [ ] **Step 2: Verify the dry-run renders cleanly (runnable on this Windows box via Git Bash)**

Run: `bash scripts/install_sbc.sh --dry-run configs/rk3588.yaml`
Expected: prints `[Unit]` / `[Service]` / `[Install]` with `@REPO_DIR@`/`@PYTHON@`/`@CONFIG@`/`@LOG_DIR@` replaced by real values, then the AP plan showing `ssid : drone` and `psk : drone123`.

- [ ] **Step 3: Assert no tokens leaked into the rendered unit**

Run: `bash scripts/install_sbc.sh --dry-run configs/rk3588.yaml | grep -c '@[A-Z_]*@' || true`
Expected: `0` (every token substituted).

- [ ] **Step 4: Static-lint if available (best effort)**

Run: `command -v shellcheck >/dev/null && shellcheck scripts/install_sbc.sh || echo "shellcheck not installed — skipping"`
Expected: no warnings, or the skip message.

- [ ] **Step 5: Commit**

```bash
git add scripts/install_sbc.sh
git commit -m "scripts: add install_sbc.sh (systemd + WiFi AP installer)"
```

---

### Task 4: Remove stale stubs + fix ARCHITECTURE §8

**Files:**
- Delete: `systemd/qg-camera.service`, `systemd/qg-control.service`, `systemd/qg-ground.service`, `systemd/qg-guidance.service`, `systemd/qg-link.service`, `systemd/qg-tracker.service`, `systemd/quadguide.target`
- Modify: `ARCHITECTURE.md` (§8 Startup, production)

**Interfaces:** none (cleanup + docs).

- [ ] **Step 1: Delete the empty stub units**

```bash
git rm systemd/qg-camera.service systemd/qg-control.service systemd/qg-ground.service \
       systemd/qg-guidance.service systemd/qg-link.service systemd/qg-tracker.service \
       systemd/quadguide.target
```

- [ ] **Step 2: Rewrite ARCHITECTURE §8 production startup**

Replace this block:
```markdown
### Startup (systemd, production)

`quadguide.target` Wants/After: `qg-camera`, `qg-tracker`, `qg-link`,
`qg-guidance`, `qg-control`, `qg-ground`. Each unit is a thin invocation of
the matching worker entry point. The tracker unit has `After=qg-camera`.
```
with:
```markdown
### Startup (systemd, production)

A **single** unit, `quadguide.service`, runs `scripts/run.py` — the same
orchestrator used in development. systemd keeps that one parent alive; the
parent forks and supervises all six workers and owns `Bus`/`FrameBuffer`
lifecycle. A per-worker unit split is **not** possible: the bus's
`multiprocessing.Lock`/`Value` and anonymous `os.pipe()` wakeups are created
once in the parent and inherited across `fork()` (`core/bus.py`), so the
workers must share one parent process. Install with `scripts/install_sbc.sh`
(which also sets up the WiFi AP). Unit policy: `Restart=always` (keep
recovering in flight), `KillMode=mixed` (lets `run.py`'s ordered shutdown run),
`LimitRTPRIO=99` (control worker SCHED_FIFO), runs as `root`. See the operator
runbook `docs/sbc-setup.md`.
```

- [ ] **Step 3: Verify the cleanup**

Run: `ls systemd`
Expected: only `quadguide.service`.

Run: `grep -n "quadguide.target" ARCHITECTURE.md`
Expected: no output (the stale reference is gone).

- [ ] **Step 4: Commit**

```bash
git add -A systemd ARCHITECTURE.md
git commit -m "systemd: drop non-viable per-worker stubs; fix ARCHITECTURE §8"
```

---

### Task 5: On-SBC operator runbook

**Files:**
- Create: `docs/sbc-setup.md`

**Interfaces:** none (operator documentation distilled from spec §7–§8).

- [ ] **Step 1: Create `docs/sbc-setup.md`**

```markdown
# ROCK 5C field setup runbook

Steps performed **on the board** (Radxa OS). The install script automates the
systemd service and WiFi AP; the hardware bring-up below is manual (needs
device enumeration + reboots). Rationale lives in the design spec
`docs/superpowers/specs/2026-06-26-sbc-deploy-systemd-wifi-ap-design.md`.

## 1. One-time install (service + WiFi AP)

```bash
cd /home/radxa/quadguide
sudo ./scripts/install_sbc.sh                 # uses configs/rk3588.yaml
# preview only, no changes:  ./scripts/install_sbc.sh --dry-run
```
After this: `quadguide.service` is enabled on boot, and SSID `drone`
(WPA2 `drone123`, 2.4 GHz) serves the UI at `http://10.42.0.1:8080`. The HaLow
ethernet path still works.

## 2. UART6 to the flight controller

| Signal | Pin | Connect to FC |
| --- | --- | --- |
| UART6_TX_M1 (SBC→FC) | 19 | FC RX |
| UART6_RX_M1 (SBC←FC) | 21 | FC TX |
| GND | 20 | FC GND |

1. `rsetup` → Overlays → enable **"Enable UART6-M1"** → reboot.
2. Confirm `ls /dev/ttyS6`.
3. Wire per table — cross TX/RX, common GND, **do not connect 5 V**.
4. On the FC (ArduPilot): set the matching `SERIALn_PROTOCOL=2` (MAVLink2),
   `SERIALn_BAUD=115`, and `GUID_OPTIONS` bit 3 (direct thrust). Bump baud to
   `921` if 115200 saturates at the 100 Hz attitude stream.

Alternative if pins 19/21 are taken: UART4-M2 → TX pin 7, RX pin 29,
node `/dev/ttyS4` (set `serial.port: /dev/ttyS4`).

## 3. OV9281 camera bring-up (sensor not officially supported)

Radxa supports only OV5647/IMX415/IMX219 on the 5C, so the OV9281 needs a
custom overlay.

1. **Driver:** `modinfo ov9281` (Rockchip BSP ships `CONFIG_VIDEO_OV9281`). Build
   the module if absent.
2. **Overlay:** copy the board's shipped OV5647 ("OKDO 5MP") overlay `.dts` as a
   template and change the sensor node: `compatible = "ovti,ov9281";`,
   I2C `reg = <0x60>;` (verify with `i2cdetect`), `data-lanes = <1 2>;`, mono
   `Y10` mediabus format; keep `clock-frequency = <24000000>` and the existing
   `csi2_dphy`/endpoint linkage. Build, install under the boot overlay dir,
   enable, reboot.
3. **Enumerate:** `v4l2-ctl --list-devices` and `media-ctl -p` to find the
   capture node (typically `/dev/video11`) and the `ov9281` subdev.
4. **Set formats** along the graph, e.g.
   `media-ctl --set-v4l2 '"ov9281 ...":0[fmt:Y10_1X10/1280x800]'`, then
   `v4l2-ctl -d /dev/video11 --all` and a test grab.
5. **Match the pipeline** in `configs/rk3588.yaml` to the real node/format. If
   the node delivers mono `GRAY8` (rkcif raw) rather than `NV12` (ISP), change
   `format=NV12` → `format=GRAY8` in the caps.
6. **OpenCV+GStreamer:**
   `python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer`
   must show `YES`. If not, install the distro `python3-opencv` (the pip wheel
   has no GStreamer).

Until step 3 succeeds, the camera worker fails to open and the service
crash-loops every 3 s — expected.

## 4. Verify

```bash
systemctl is-enabled quadguide          # -> enabled
systemctl status quadguide              # -> active (running)
journalctl -u quadguide -f              # -> "started 6 workers: [...]"
nmcli connection show --active          # -> qg-ap present
```
Reboot and re-check; from a laptop join `drone`, get a `10.42.0.x` lease, open
`http://10.42.0.1:8080`. Confirm the HaLow path still loads the same UI.
```

- [ ] **Step 2: Verify the doc**

Run: `python -c "import pathlib,sys; sys.exit(0 if pathlib.Path('docs/sbc-setup.md').read_text().strip() else 1)"`
Expected: exit 0 (file written, non-empty). Read it once to confirm the pin table and command blocks render.

- [ ] **Step 3: Commit**

```bash
git add docs/sbc-setup.md
git commit -m "docs: add ROCK 5C field setup runbook"
```

---

## Final verification (after all tasks)

On this Windows box (what is possible here):
- `python -m pytest tests/unit/test_rk3588_config.py tests/unit/test_systemd_unit.py -v` → PASS.
- `bash scripts/install_sbc.sh --dry-run configs/rk3588.yaml` → renders unit with no `@TOKEN@` left and the `drone`/`drone123` AP plan.
- `ls systemd` → only `quadguide.service`.

On the SBC (out of scope for this box — see `docs/sbc-setup.md` §4):
- `sudo ./scripts/install_sbc.sh`, reboot, confirm `quadguide.service` active and `drone` AP serving `http://10.42.0.1:8080`.
```

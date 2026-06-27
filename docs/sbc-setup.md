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

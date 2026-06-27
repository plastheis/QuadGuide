# ROCK 5C field setup runbook

Steps performed **on the board** (Radxa OS), at the local console (HDMI +
keyboard). Rationale lives in the design spec
`docs/superpowers/specs/2026-06-26-sbc-deploy-systemd-wifi-ap-design.md`.

## 0. One-command install (recommended) — needs wired ethernet

Fresh image, first boot. Plug the board into **wired ethernet** (for apt/pip/git),
open a terminal, and run ONE line. It clones QuadGuide + EdgeCV, builds the venv +
deps, compiles/enables the OV9281 overlay, and installs the systemd service + WiFi
AP. Idempotent — safe to re-run.

```bash
sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/plastheis/QuadGuide/main/scripts/firstboot_install.sh)"
```
(If EdgeCV lives at a different URL: `GIT_OWNER=youruser sudo -E bash -c "$(curl …)"`.)

Then `sudo reboot` (loads the camera overlay). The two things the script can't do
for you — the OV9281 **kernel driver** (§3a) and EdgeCV **models** (§3b) — are
printed at the end and detailed below. The service autostarts every boot and the
`drone` AP comes up, so you never need SSH or a console again after this.

## 1. (manual alternative) service + WiFi AP only

If the repos are already on the board and you only want the service + AP:
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

Radxa supports only OV5647/IMX415/IMX219 on the 5C. These steps were derived by
reading the actual ROCK 5C boot SD card (Radxa OS, kernel `6.1.43-15-rk2312`):
its base device tree, the shipped OV5647 overlay (template), and the kernel
config. Two independent things are missing and **both** are required — a driver
*and* an overlay.

### 3a. Driver — REQUIRED, and not shipped

The shipped kernel config has **`# CONFIG_VIDEO_OV9281 is not set`** (only
`CONFIG_VIDEO_OV5647=y`). There is no `ov9281` driver built-in or as a module, so
applying the overlay alone creates an I2C node nothing binds to. The card *does*
ship kernel headers (`/usr/src/linux-headers-6.1.43-15-rk2312`), so build the
driver out-of-tree:

```bash
# get the Rockchip BSP ov9281.c matching this kernel line (rockchip-linux/kernel,
# rk-6.1 branch: drivers/media/i2c/ov9281.c), then in a dir with it + a Makefile
# containing `obj-m += ov9281.o`:
make -C /usr/src/linux-headers-6.1.43-15-rk2312 M=$PWD modules
sudo cp ov9281.ko /usr/lib/modules/6.1.43-15-rk2312/updates/
sudo depmod -a && sudo modprobe ov9281
echo ov9281 | sudo tee /etc/modules-load.d/ov9281.conf   # load at boot
```
(Alternative: rebuild the kernel with `CONFIG_VIDEO_OV9281=m`.)

### 3b. Overlay — ready in the repo

[`overlays/rock-5c-ov9281.dts`](../overlays/rock-5c-ov9281.dts) is built verbatim
from the card's working **Rock 5D OV5647 overlay** (the 5D is RK3588S2, identical
CSI wiring to the 5C), with the OV9281 sensor swapped in. Every base-DT label it
references was confirmed present in the real `rk3588s-rock-5c.dtb`. Confirmed
wiring: **I2C bus `i2c3`**, **pwdn `GPIO1_D3` (active-low)**, `csi2_dphy0` →
`mipi2_csi2` → `rkcif`. Build and enable:

```bash
dtc -@ -I dts -O dtb -o rock-5c-ov9281.dtbo overlays/rock-5c-ov9281.dts
sudo cp rock-5c-ov9281.dtbo /boot/dtbo/
# enable it: append `rock-5c-ov9281` to the overlays line in /boot/uEnv.txt
# (this image boots via U-Boot extlinux + uEnv.txt), or use `rsetup` → Overlays.
sudo reboot
```
Still confirm on your hardware (can't be read from the card): the module's xtal
(overlay assumes 24 MHz; some OV9281 modules are 27 MHz) and its I2C address
(assumes `0x60`).

### 3c. Enumerate + format

```bash
dmesg | grep -iE 'ov9281|rkcif|csi2'      # probe / lane / link-freq
i2cdetect -y 3                             # bus i2c3 → device at 0x60 ("UU" = bound)
v4l2-ctl --list-devices                    # find the rkcif "stream_cif_mipi_id0" node
media-ctl -p                               # confirm ov9281 → dphy0 → csi2 → cif graph
media-ctl --set-v4l2 '"ov9281 3-0060":0[fmt:Y10_1X10/1280x800]'
v4l2-ctl -d /dev/videoN --all              # the rkcif node; then a test grab
```
This is a **mono** sensor: capture from the rkcif (VICAP) node, not the ISP's
`/dev/video11`.

### 3d. Point QuadGuide at it

Set the `configs/rk3588.yaml` pipeline to the real node/format. Mono frames come
out `GRAY8`/`Y10`, not `NV12`, so the caps differ from the placeholder:
```yaml
pipeline: "v4l2src device=/dev/videoN io-mode=4 ! video/x-raw,format=GRAY8,width=1280,height=800,framerate=60/1 ! videoscale ! video/x-raw,width=640,height=400 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false"
```
Finally, OpenCV must have GStreamer:
`python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer`
must show `YES` — if not, install the distro `python3-opencv` (the pip wheel has
none).

Until 3a + 3b are both done, the camera worker fails to open and the service
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

# OV9281 CSI camera bring-up — context for on-board debugging

> **Purpose:** hand this to a Claude Code session running **on the ROCK 5C itself**
> (via SSH/console) to finish getting the OV9281 MIPI-CSI camera working with
> QuadGuide. Everything below is verified on the actual board unless marked a
> hypothesis. The camera is the ONLY unresolved item — the rest of the stack runs.

## Goal

Get the **OmniVision OV9281** (1 MP mono **global-shutter**, RAW10/Y10) MIPI-CSI
sensor capturing into QuadGuide's camera worker, which opens it via a GStreamer
pipeline (`cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)` in
`src/quadguide/perception/camera/sources.py:CSICamera`). The global-shutter mono
sensor is required (fast-target tracking) — a USB rolling-shutter cam is not an
acceptable substitute.

## Environment (verified)

- Board: **Radxa ROCK 5C** (RK3588S2). Root compatible: `radxa,rock-5c","rockchip,rk3588`.
- OS: Radxa OS Debian 12, kernel **`6.1.84-8-rk2410`** (aarch64). Boots via U-Boot
  extlinux + `/boot/uEnv.txt` (overlays enabled by name on the `overlays=` line).
- Kernel headers present: `/usr/src/linux-headers-6.1.84-8-rk2410`.
  `build-essential`, `dtc`, `v4l-utils`, `i2c-tools` installed. (`modinfo` is at
  `/sbin/modinfo`.)
- QuadGuide: `~/quadguide` (github.com/plastheis/QuadGuide, `main`), venv at
  `~/quadguide/.venv` (`--system-site-packages`, `pip install -e .` done).
  EdgeCV at `~/EdgeCV` with `.rknn` models under `~/EdgeCV/models` (via git-lfs).
- The full stack RUNS: with the camera `pipeline:` set to a `videotestsrc`, all six
  workers come up and the HUD is live at `http://10.42.0.1:8080`. Link worker opens
  the real UART `/dev/ttyS6` (UART6-M1 overlay enabled). **Only the real CSI camera
  is unresolved.**

## Kernel camera facts (verified)

- `CONFIG_VIDEO_OV9281` is **not set** in this kernel. The OV9281 driver present is
  an **out-of-tree module** (`ov9281.ko`, prints `driver version: 00.01.05`,
  "loading out-of-tree module taints kernel"). It is vermagic-compatible (it loads).
- Capture pipeline drivers ARE built in: `CONFIG_VIDEO_ROCKCHIP_CIF=y`,
  `CONFIG_VIDEO_ROCKCHIP_ISP=y`.
- Media devices: `/dev/media0` = **rkcif** (VICAP, the raw/mono path),
  `/dev/media1` = rkisp (ISP, Bayer — wrong for a mono sensor).
- The mono capture node is **`/dev/video0` = `stream_cif_mipi_id0`** (rkcif raw).
  NOT `/dev/video11` (that's the ISP main path; it errors "Internal data stream
  error" on this mono sensor).

## The sensor and overlay (verified)

- Sensor is on **I²C bus 3, address 0x60**. Probes successfully:
  `ov9281 3-0060: Detected OV009281 sensor`. Non-fatal probe warnings: "Failed to
  get reset-gpios", "no pinctrl", and avdd/dovdd/dvdd "using dummy regulator".
- Overlay source: **`~/quadguide/overlays/rock-5c-ov9281.dts`**, compiled to
  `/boot/dtbo/rock-5c-ov9281.dtbo`, enabled in `/boot/uEnv.txt`.
- The overlay was derived **fragment-for-fragment from the rk2410 image's own
  `rock-5a-okdo-5mp-camera.dtbo`** (its metadata lists `radxa,rock-5c`; the ROCK 5D
  and OKDO-5MP use the same RK3588S2 CSI wiring). Only the sensor node was swapped
  to the OV9281 (`0x60`, `ovti,ov9281`, 24 MHz `clock-frequency`,
  `clock-names="xvclk"`, pwdn `GPIO1_D3` active-low). `link-frequencies` and a
  sensor `pinctrl-0` were deliberately removed (an earlier version had them and the
  sensor still didn't bind; they are not in the reference).
- Wiring path: sensor → `csi2_dphy0` → `mipi2_csi2` → `rkcif`(+ `rkisp0` sditf).
  Base-DT labels used (i2c3, csi2_dphy0, csi2_dphy0_hw, mipi2_csi2, rkcif,
  rkcif_mipi_lvds2, rkcif_mipi_lvds2_sditf, rkcif_mmu, rkisp0, rkisp0_vir0,
  isp0_mmu, pinctrl, gpio1, pcfg_pull_up) — all confirmed present in the 5C base DT.

## THE PROBLEM (precisely diagnosed)

The sensor subdev **registers but never binds** to the Rockchip CSI pipeline:

- `sudo cat /sys/kernel/debug/v4l2-async/pending_async_subdevices` lists
  **`m00_b_ov9281 3-0060`** AND `rockchip-csi2-dphy0`, `rockchip-mipi-csi2`,
  `rkcif-mipi-lvds2`, `rkisp0-vir0`. The entire pipeline is stuck pending — the
  `csi2_dphy0` notifier never matches the sensor, and on Rockchip that one
  un-bind cascades so nothing binds.
- Consequence: `rkcif_update_sensor_info: ... get remote terminal sensor failed
  -19` for every stream; `ov9281` is **not** a media entity in `media-ctl -d
  /dev/media0 -p`; only 3 subdevs exist (`/dev/v4l-subdev0/1/2` = csi2/dphy/mipi,
  no sensor); opening `/dev/video0` returns `ENODEV`.

**The overlay wiring is PROVEN CORRECT — this is NOT a DT problem.** From the live
merged tree (`dtc -I fs -O dts /proc/device-tree`):
- sensor endpoint: `phandle = 0x3de`, `remote-endpoint = 0x3dd`, `data-lanes = <1 2>`.
- DPHY input endpoint (the node at `phandle 0x3dd`):
  `endpoint@2 { data-lanes = <1 2>; remote-endpoint = <0x3de>; reg = <2>; }`.
- i.e. the sensor↔DPHY link is **bidirectional and lane-matched**. Correct.

**Conclusion:** the out-of-tree `ov9281.ko` v00.01.05 registers a subdev whose
fwnode the Rockchip `csi2_dphy0` async notifier will not match. This is a
**driver-side async/fwnode issue**, not overlay, not hardware (sensor is detected),
not capture-node/format.

## What has been tried

1. Overlay v1 (from the older rk2312 OV5647 template) — had `link-frequencies` +
   sensor `pinctrl-0`. Sensor detected, did not register cleanly.
2. Overlay v2 (current, exact rk2410 `rock-5a-okdo-5mp` reference, sensor swapped,
   no link-frequencies, no pinctrl-0) — sensor detected AND registers, but pending
   (doesn't bind). Bidirectional DT link confirmed correct (above).
3. **Rebuilt `ov9281.c` from `radxa/kernel` branch `linux-6.1-stan-rkr4.1`** against
   the on-board headers — compiled cleanly to `~/ov9281-rebuild/ov9281.ko`,
   installed to `/lib/modules/6.1.84-8-rk2410/updates/ov9281.ko`, rebooted.
   **RESULT: still NOT bound, still pending.** `modinfo` confirms the loaded
   module is the rebuilt one (`filename: .../updates/ov9281.ko`,
   `srcversion: 73219C0BBCA2D95A1A018C4`) but it still prints
   `driver version: 00.01.05` — i.e. the `rkr4.1` source IS the same 00.01.05 that
   was already there. **Rebuilding the same source is a dead end.**
4. **Compared `ov9281.c` vs in-tree `ov5647.c` registration** (ov5647 binds on this
   kernel; CONFIG_VIDEO_OV5647=y). They are **IDENTICAL** in the registration tail:
   both set `pad.flags = MEDIA_PAD_FL_SOURCE`, `entity.function =
   MEDIA_ENT_F_CAM_SENSOR`, `media_entity_pads_init(&sd->entity, 1, &pad)`, then
   `v4l2_async_register_subdev_sensor(sd)`; fwnode is set implicitly by
   `v4l2_i2c_subdev_init` (the i2c client node). So the binding MECHANISM is not the
   differentiator — the bug is subtler than the registration call.

## ROOT CAUSE (identified) — load-order race: the sensor is a late module

The boot timeline is the smoking gun:

```
13.103  csi2-dphy0: Fixed dependency cycle(s) with /i2c@feab0000/ov9281@60
13.109  rockchip-mipi-csi2: Async registered subdev
13.177  rkcif-mipi-lvds2:  Async subdev notifier COMPLETED
13.237  rkisp0-vir0:       Async subdev notifier COMPLETED
15.507  ov9281: loading out-of-tree module taints kernel
15.517  ov9281 3-0060: Detected OV009281 sensor
```

The built-in Rockchip camera drivers (csi2-dphy / mipi-csi2 / rkcif / rkisp) set up
and **complete** their async binding at ~13.2 s. The OV9281 driver is a **module
that loads ~2.3 s later** (15.5 s). By then the pipeline notifiers have finalized
without it — and fw_devlink's "Fixed dependency cycle(s)" broke the dphy↔sensor
devlink — so the sensor registers into a pipeline that is no longer waiting for it,
and sits in `pending_async_subdevices` permanently.

**This fully explains why the driver source being byte-identical to the working
`ov5647.c` doesn't matter:** `ov5647` is built **in-tree** (`CONFIG_VIDEO_OV5647=y`)
and probes during kernel init *before* the pipeline notifiers complete; `ov9281`
is `=n` in config and only present as a late-loading out-of-tree `.ko`. Same code,
wrong load time.

### THE FIX (highest confidence): build OV9281 in-tree

Build the kernel with `CONFIG_VIDEO_OV9281=y` (or `=m` AND guaranteed to load before
the camera drivers' notifiers complete) so the sensor probes during kernel init like
ov5647. Path: the Radxa BSP kernel build —
```bash
# get the kernel source matching 6.1.84-8-rk2410 (radxa/kernel), enable the symbol,
# build + install the kernel (rsetup has a kernel-build helper, or use radxa-pkg bsp):
#   in the kernel config: CONFIG_VIDEO_OV9281=y
#   (it already exists in drivers/media/i2c/ov9281.c + Kconfig/Makefile)
# then build Image + modules + dtbs, install, reboot.
```
After that, ov9281 is a built-in driver and binds exactly like the in-tree ov5647.

### Confirm the diagnosis first (no rebuild needed): rebind the DPHY late

Proves the race — re-bind the DPHY *after* the sensor module is already loaded; if it
then binds, load-order is confirmed:
```bash
DPHY=$(ls /sys/bus/platform/devices/ | grep -iE 'csi2-dphy0|\.dphy' | head -1)
DRV=$(basename "$(readlink /sys/bus/platform/devices/$DPHY/driver)")
echo "$DPHY" | sudo tee /sys/bus/platform/drivers/$DRV/unbind
echo "$DPHY" | sudo tee /sys/bus/platform/drivers/$DRV/bind
sleep 1; media-ctl -d /dev/media0 -p | grep -i ov9281 && echo BOUND || echo "not bound"
```
If BOUND → it is purely load-order; the in-tree build is the durable fix. A
userspace-only workaround (no kernel rebuild) is a boot service that, after the
ov9281 module is loaded, unbinds+rebinds the csi2-dphy0 (and possibly mipi-csi2 /
rkcif) so their notifiers re-run with the sensor present. Less clean than in-tree
but avoids a kernel build — viable if a kernel rebuild is impractical.

### Lower-probability avenues (only if the rebind test does NOT bind)
- Early module load via initramfs (`/etc/initramfs-tools/modules` + `update-initramfs
  -u`) — but built-in drivers probe before initramfs, so this may still be too late;
  the rebind test is the better signal.
- Forum tweaks (https://forum.radxa.com/t/support-for-ov9281-camera/26386): real
  avdd/dovdd/dvdd regulators + `pwdn-gpios` ACTIVE_HIGH. Module-specific; our sensor
  already detects as active-low, so treat carefully. Affects streaming more than
  binding.
- `func v4l2_async_match_notify +p` via `/sys/kernel/debug/dynamic_debug/control`
  (note: the glob `*v4l2-async*` was rejected — use the `func` form or
  `file v4l2-async.c +p`) to log the exact match decision.

## If it still doesn't bind — hypotheses to pursue (for on-board Claude)

1. **Driver async API.** Read `~/ov9281-rebuild/ov9281.c` probe path: does it call
   `v4l2_async_register_subdev_sensor()` (modern, walks OF graph) or the old
   `v4l2_async_register_subdev()`? Compare against the in-tree `ov5647.c`
   (`CONFIG_VIDEO_OV5647=y`, which DOES bind on this kernel) — diff how each sets
   `sd->fwnode` / registers. The rkr4.1 source may itself be the buggy 00.01.05; if
   so, try a newer/different `radxa/kernel` tag's `ov9281.c`, or backport the
   ov5647 registration pattern.
2. **In-tree vs module.** Some Rockchip sensor drivers bind only when built in-tree
   (`CONFIG_VIDEO_OV9281=m/y` via the BSP `make kernel`), because of how the
   Rockchip async notifier helpers are wired. A full kernel rebuild with
   `CONFIG_VIDEO_OV9281=y` is the heavy-but-reliable fallback (Radxa `rsetup` /
   `bsp` build).
3. **Forum-reported tweaks** (https://forum.radxa.com/t/support-for-ov9281-camera/26386):
   a working OV9281-on-Rockchip setup used **pwdn-gpios ACTIVE_HIGH** (overlay
   currently active-low) and **real avdd/dovdd/dvdd regulators** (overlay currently
   lets them fall back to dummy). NOTE: our sensor is already *detected* (so it is
   powered and pwdn currently works for detection) — flipping polarity risks
   breaking detection; treat as a hypothesis to test carefully, not a sure fix.
   Supplies/GPIO usually affect power-on/streaming, not async fwnode matching, so
   these are lower-probability for THIS (binding) symptom.
4. **Notifier ownership/timing.** dphy probes ~13 s, the sensor module loads ~15 s
   (it's a late-loaded module). Async notifiers should handle late registration, but
   verify: try forcing the module to load earlier (`/etc/modules-load.d/ov9281.conf`
   + initramfs) and re-check. Low probability but cheap.

## Once the sensor BINDS — finish the camera

1. Find/confirm the capture node + mono format:
   ```bash
   media-ctl -d /dev/media0 -p | grep -A3 ov9281
   media-ctl -d /dev/media0 --set-v4l2 '"m00_b_ov9281 3-0060":0[fmt:Y10_1X10/1280x800]'
   v4l2-ctl -d /dev/video0 --set-fmt-video=width=1280,height=800,pixelformat=GREY \
            --stream-mmap --stream-count=5
   gst-launch-1.0 v4l2src device=/dev/video0 num-buffers=10 ! fakesink -v 2>&1 \
     | grep -iE 'caps|GRAY|Y10|format'
   ```
2. Set `~/quadguide/configs/rk3588.yaml` `platform.camera.pipeline` to the rkcif
   node with the real mono caps (GRAY8/GRAY16/Y10, NOT NV12), e.g.:
   ```
   pipeline: "v4l2src device=/dev/video0 io-mode=4 ! video/x-raw,format=GRAY8,width=1280,height=800,framerate=60/1 ! videoscale ! video/x-raw,width=640,height=400 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false"
   ```
   (Restore the real backend: `backend: gstreamer`, width 640/height 400/fps 60.)
3. Confirm OpenCV has GStreamer: `~/quadguide/.venv/bin/python -c "import cv2;
   print(cv2.getBuildInformation())" | grep -i gstreamer` → must say YES. The venv
   uses `--system-site-packages` so it sees apt's `python3-opencv` (GStreamer-enabled).
4. `sudo systemctl restart quadguide`; verify on the HUD at `http://10.42.0.1:8080`.

## Key paths / commands cheat-sheet

| Thing | Path / command |
| --- | --- |
| Overlay source | `~/quadguide/overlays/rock-5c-ov9281.dts` |
| Compiled overlay | `/boot/dtbo/rock-5c-ov9281.dtbo` (enable in `/boot/uEnv.txt`) |
| Rebuilt driver | `~/ov9281-rebuild/ov9281.ko` |
| Rockchip ov9281.c source | `https://raw.githubusercontent.com/radxa/kernel/linux-6.1-stan-rkr4.1/drivers/media/i2c/ov9281.c` |
| Headers | `/usr/src/linux-headers-6.1.84-8-rk2410` |
| Pending async list | `sudo cat /sys/kernel/debug/v4l2-async/pending_async_subdevices` |
| Media graph | `media-ctl -d /dev/media0 -p` |
| Live DT | `sudo dtc -I fs -O dts /proc/device-tree > /tmp/live.dts` |
| Probe log | `sudo dmesg \| grep -iE 'ov9281\|csi2\|dphy\|rkcif'` |
| QuadGuide config | `~/quadguide/configs/rk3588.yaml` |
| Service | `systemctl status quadguide` ; `journalctl -u quadguide -f` |
| Run foreground | `cd ~/quadguide && sudo .venv/bin/python scripts/run.py --config configs/rk3588.yaml` |

## Definition of done

`media-ctl -d /dev/media0 -p | grep ov9281` shows the sensor as a bound media
entity, `/dev/video0` streams mono frames, and `quadguide.service` runs with the
real CSI pipeline (HUD shows live camera, not the test pattern).

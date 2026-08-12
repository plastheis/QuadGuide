# libcamera IPA tuning files

`ov9281_mono_quadguide.json` is the OV9281's stock Raspberry Pi tuning file with a
**`highlight` constraint list added**. Everything else is byte-identical to
`/usr/share/libcamera/ipa/rpi/vc4/ov9281_mono.json` (libcamera 0.7.1), so a diff
against that file shows the whole change.

## Why a copy exists

libcamera's AGC can only select metering / exposure / constraint modes that the
sensor's tuning file defines, and the stock OV9281 file defines exactly one of each
kind that matters here:

| | stock ov9281_mono.json |
| --- | --- |
| `metering_modes` | `centre-weighted` |
| `exposure_modes` | `normal`, `short`, `long` |
| `constraint_modes` | `normal` |

So `platform.camera.ae_constraint_mode: highlight` is unselectable against the stock
file. Worse, it fails *quietly*: an undefined mode logs `WARN RPiAgc No constraint
list highlight` and AE carries on with the default, so the setting looks applied but
does nothing.

## What was added

A `highlight` list holding the brightest 2% of the frame (the 0.98–1.0 quantile)
between `LOWER` 0.2 and `UPPER` 0.3. That is the behaviour QuadGuide wants against
sky: a blown-out background hides a target flying against it, and plain `normal` AE
will happily average the sky into clipping.

`y_target` bounds are **linear luminance, before the contrast curve**, so they read
far darker than the number suggests. Measured on the real sensor, 98th percentile of
the delivered 8-bit frame at EV 0:

| `UPPER` | delivered p98 |
| --- | --- |
| *(no constraint)* | 215/255 |
| 0.50 | 206/255 |
| 0.35 | 178/255 |
| **0.30** | **165/255** ← current |
| 0.25 | 149/255 |

### Why the first version of this list did nothing

It shipped as this sensor's own `normal` constraint (`LOWER` 0.4) plus `imx296_mono.json`'s
ceiling (`UPPER` 0.8), on the theory that `highlight` = `normal` + a ceiling. Both halves
of that are traps:

* **`UPPER` 0.8 lands at ~242/255 — above where AE already settles (215/255).** A
  constraint only binds when it is *tighter* than the unconstrained result, so an 0.8
  ceiling never fires. Selecting `highlight` vs `normal` was measurably a no-op, which
  is indistinguishable from the mode not being applied at all.
* **`LOWER` 0.4 is actively harmful on the exact scene this is for.** Constraints run in
  **list order**, each able to raise (`LOWER`) or lower (`UPPER`) the running gain. On a
  flat sky the top 2% *is* the whole frame, so a `LOWER` bound drags the entire image up
  to its target — 0.4 pins the sky at ~198/255 no matter what ceiling follows. Hence 0.2:
  enough to keep a genuinely dark scene off the floor, too low to blow out a flat sky.

Do not reach for `exposure_value` to fix sky exposure. EV scales the base `y_target` *and*
every constraint target alike, so it darkens the target silhouette by exactly as much as
the sky and buys no contrast.

## How it is loaded

`platform.camera.tuning_file` in `configs/rpi4b.yaml`. `CSICamera` resolves it
against the repo root and exports `LIBCAMERA_RPI_TUNING_FILE` before opening the
pipeline. The var name is `RPI`, **not** `RPI_VC4`: the VC4 spelling is accepted
without complaint and the stock file loads anyway.

## Verifying a change

A constraint only moves AE when the top 2% would otherwise exceed the bound, so
eyeballing a preview proves nothing — measure. Open the pipeline, discard ~50 frames
so AE settles, then compare the 98th percentile of the grey channel across
`ae_constraint_mode: normal` and `highlight`:

```python
import numpy as np, cv2, os
os.environ["LIBCAMERA_RPI_TUNING_FILE"] = "<abs path to this file>"
pipe = ("libcamerasrc ae-enable=true ae-exposure-mode=short "
        "ae-metering-mode=centre-weighted ae-constraint-mode=highlight "
        "! video/x-raw,width=640,height=400,framerate=60/1 "
        "! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false")
cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
for _ in range(50): cap.read()                       # let AE settle
g = [np.percentile(cap.read()[1][:, :, 0], 98) for _ in range(15)]
print("p98 =", np.mean(g))
```

`highlight` must come out **below** `normal`. If the two match, the `UPPER` bound is
sitting above where AE already lands and is not binding — lower it until the numbers
separate (see the table above). Point the camera at the brightest thing available;
against a dim indoor wall neither bound engages and both modes read the same for a
legitimate reason.

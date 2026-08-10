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

A `highlight` list derived from `imx296_mono.json` — the other mono global-shutter
sensor Raspberry Pi ships with a highlight list. It is this sensor's own `normal`
constraint (`LOWER` 0.4) plus that reference's ceiling (`UPPER` 0.8) on the same
0.98–1.0 quantile, so `highlight` is exactly `normal` **plus a ceiling on the
brightest 2% of the frame**. That is the behaviour QuadGuide wants against sky: a
blown-out background hides a target flying against it, and plain `normal` AE will
happily average the sky into clipping.

`y_target` bounds are **linear luminance, before the contrast curve**. Through this
tuning's gamma, `UPPER` 0.8 lands at ~242/255 in the delivered frame — just below
clipping. Use 0.6 (~226/255) for more headroom.

## How it is loaded

`platform.camera.tuning_file` in `configs/rpi4b.yaml`. `CSICamera` resolves it
against the repo root and exports `LIBCAMERA_RPI_TUNING_FILE` before opening the
pipeline. The var name is `RPI`, **not** `RPI_VC4`: the VC4 spelling is accepted
without complaint and the stock file loads anyway.

## Verifying a change

The constraint only moves AE when the top 2% would otherwise exceed the bound, so on
an ordinary indoor scene it is a no-op and proves nothing. To see it engage, copy the
file, tighten `UPPER` to ~0.35, and compare `ae_constraint_mode: normal` against
`highlight` on that copy — `highlight` should drop the sensor's exposure while
`normal` leaves it alone.

# QuadGuide

SBC-resident flight guidance stack for a manual lock-on target-tracking
quadcopter. Runs on a companion computer (RK3588 target) on the airframe:
camera → object tracker → proportional-navigation / pure-pursuit guidance →
roll/pitch attitude setpoints to an ArduPilot FC (H743) over UART (MAVLink2). See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design.

## Running

```bash
# full flight stack (all workers)
python scripts/run.py --config configs/rk3588.yaml

# camera + tracker + ground HUD only (no FC needed) — tracker tuning / bench
python scripts/dev_ground_perception.py --config configs/rk3588.yaml
```

Ground HUD: `http://<sbc-ip>:8080`. Override any config value with
`--set key.path=value` (e.g. `--set guidance.pure_pursuit.K=15`).

## Latency diagnostics

End-to-end latency telemetry and a post-run analysis tool. Full detail in
[`ARCHITECTURE.md` §13](ARCHITECTURE.md#13-latency-model--diagnostics).

**Capture a trace.** Add `--log` to either launcher to record a per-process
diagnostic trace (raw latency timestamps + worker state) for offline analysis:

```bash
python scripts/dev_ground_perception.py --config configs/rk3588.yaml --log
# ...lock on a target in the HUD, let it run, then Ctrl-C (graceful flush)
```

The trace dir is printed at startup (`{logging.dir}/trace/<ts>`, or
`./quadguide-trace/<ts>` if that isn't writable). It is off by default and meant
for bench/diagnosis runs, not production flight.

**Analyze it.**

```bash
# ingest a --log dump: per-stage + cumulative (glass→here) latency,
# distribution, and a spectral view that pins any rhythmic beat
python scripts/diagnose_latency.py trace ./quadguide-trace/<ts>

# no hardware: reproduce the free-running-loop sawtooth and the gated fix
python scripts/diagnose_latency.py sim
```

Latency lineage rides the message chain via `origin_ns` (the frame capture
timestamp), so any stage — and the trace — can report the true glass→actuation
age. The HUD's latency field shows the cumulative glass→control age.

## Tracking library (edgecv)

`quadguide.perception.edgecv_adapter` wires the trackers in `src/edgecv/`
(formerly the standalone `edgecv` package, merged into this repo 2026-08-17)
into the perception worker. See
[`docs/architecture-edgecv.md`](docs/architecture-edgecv.md) for the
library's own design, and [`ARCHITECTURE.md` §11](ARCHITECTURE.md#11-adding-a-new-tracker)
for how it's wired in.

### Install

The tracking library's backends are optional extras on the `quadguide`
package itself — there's no separate install:

```bash
pip install -e .            # core: numpy CF runtime, fusion abstractions, mock backend
pip install -e ".[onnx]"    # ONNXRuntime CPU/dev backend
pip install -e ".[rknn]"    # registers the RKNN backend (see device note below)
pip install -e ".[test]"    # test + lint tooling
```

### RKNN on-device note

`rknn-toolkit-lite2` is **not on PyPI** and is **installed manually on the
device** (Rockchip release archive). The `[rknn]` extra only registers the
backend adapter; it does not and cannot pull the runtime.

### Models & conversion

Trackers never load a weight file directly — they load a **manifest**
(`src/edgecv/models/manifests/*.yaml`) that maps one logical model to
per-backend artifacts plus its preprocessing/IO spec. Weight blobs live in
`models/` at the repo root and are committed directly (real files, not
LFS/pointers — see `.gitignore`'s negations).

Conversion runs offline on x86 via one manifest-driven dispatcher
(`tools/convert.py`), run from the repo root:

```bash
pip install -e ".[dev]"        # torch + onnx for export/validation

# PyTorch checkpoint -> ONNX (writes the manifest's resolved artifact path under models/)
python tools/convert.py --model siamfc_generic --checkpoint models/siamfc_alexnet_e50.pth

# ...and chain to an RK3588 INT8 RKNN (needs rknn-toolkit2 + calibration images)
python tools/convert.py --model siamfc_generic --checkpoint models/siamfc_alexnet_e50.pth \
    --rknn --calib calib/
```

Full mechanics, the add-a-tracker recipe, the preprocessing contract, and
INT8 calibration notes are in [`tools/CONVERSION.md`](tools/CONVERSION.md).

## Tests

```bash
python -m pytest
```

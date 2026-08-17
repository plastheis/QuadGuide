# Model conversion (host-only)

Host-only tooling (ARCHITECTURE.md §11), not a runtime dependency. Conversion runs
offline on x86; the device only ever runs the lite runtime. One dispatcher converts any
registered model, driven by that model's manifest.

## Install

```bash
pip install -e .[dev]      # torch, onnx (+ onnxruntime from [test] for parity checks)
```

`rknn-toolkit2` (for the RKNN step) is **not on PyPI** — install it on an x86 host from
Rockchip's release wheels.

## Where models go

- Weight blobs live in `models/` at the repo root, **gitignored**, never committed.
- A manifest's `artifacts.<backend>.path` is relative and resolves against
  `$EDGECV_MODEL_DIR` (default `models/`). The converter writes the artifact to that same
  resolved path, so the tracker loads exactly what you produced.

## Quick start

```bash
# PyTorch checkpoint -> ONNX (writes models/siamfc_generic.onnx)
python tools/convert.py --model siamfc_generic --checkpoint models/siamfc_alexnet_e50.pth

# ...and chain to an RK3588 INT8 RKNN (needs rknn-toolkit2 + calibration images)
python tools/convert.py --model siamfc_generic --checkpoint models/siamfc_alexnet_e50.pth \
    --rknn --calib calib/

# YOLO26n: Ultralytics checkpoint -> one-to-many ONNX (writes models/yolo26n.onnx)
python tools/convert.py --model yolo26n --checkpoint models/yolo26n.pt

# ...and chain to an RK3588 INT8 RKNN (needs rknn-toolkit2 + calibration images)
python tools/convert.py --model yolo26n --checkpoint models/yolo26n.pt --rknn --calib calib/

# NanoTrack V3: PyTorch checkpoint -> ONNX (writes models/nanotrack.onnx)
python tools/convert.py --model nanotrack --checkpoint models/nanotrackv3.pth

# ...and chain to an RK3588 INT8 RKNN (needs rknn-toolkit2 + calibration images)
python tools/convert.py --model nanotrack --checkpoint models/nanotrackv3.pth --rknn --calib calib/
```

## How it works

Conversion is three stages; only the first is model-specific:

1. **Load checkpoint → `nn.Module`** — per-model *adapter*.
2. **Export module → ONNX** — generic *harness*: `torch.onnx.export` → `onnx.checker` →
   torch-vs-onnxruntime parity (`max|Δ| < 1e-3`).
3. **ONNX → RKNN** — generic, optional.

The **manifest** (`edgecv/models/manifests/<name>.yaml`) is the single source of truth for
input/output names, shapes, and artifact paths — the same manifest the runtime backend
uses, so the converter and the tracker can never disagree on I/O.

- `tools/convert.py` — CLI dispatcher.
- `tools/convert_lib/registry.py` — the `Adapter` registry.
- `tools/convert_lib/harness.py` — export + `onnx.checker` + parity.
- `tools/convert_lib/rknn.py` — generic ONNX → RKNN (deferred `rknn-toolkit2` import).
- `tools/convert_lib/adapters/<name>.py` — one per model.

The dispatcher loads the manifest, looks up the adapter, builds the module, and the harness
exports to `resolve_artifact_path(manifest.artifacts.onnx.path)`. With `--rknn` it chains
stage 3 to the manifest's rknn artifact path. The preprocessing contract (RGB/`[0,255]`/BGR
for SiamFC, etc.) lives in the manifest and is consumed by the tracker, not the converter.

## Adding a new tracker

1. Add a manifest at `edgecv/models/manifests/<name>.yaml` with the model's `io`
   (input/output names, shapes, dtypes) and `artifacts` (onnx + rknn paths). **Input order
   in `io.inputs` MUST match your module's `forward()` argument order** — the harness feeds
   inputs positionally to torch and by name to onnxruntime.
2. Add `tools/convert_lib/adapters/<name>.py`:

   ```python
   from convert_lib.registry import Adapter, register

   def build(checkpoint: str):
       # instantiate the architecture, load_state_dict(strict=True), .eval()
       return module

   register(Adapter(name="<name>", build=build))   # dynamic_axes=... if variable dims
   ```

   Vendor the architecture (as `adapters/siamfc.py` does) or import it from an installed
   package. `strict=True` makes a key/shape mismatch fail loudly.
3. Import it in `tools/convert_lib/adapters/__init__.py` so it self-registers.
4. Run: `python tools/convert.py --model <name> --checkpoint <pth>`

### Variant: upstream already exports ONNX (e.g. YOLO / ultralytics)

Some model families ship their own exporter, so there is no torch `nn.Module` to own.
These register an `export` hook instead of `build` (see `adapters/yolo.py`); the
dispatcher calls it to write the ONNX directly, runs `onnx.checker`, and skips the
torch parity harness. YOLO26 uses the **one-to-many head** (`nms=False`): the NMS-free
end-to-end head is unavailable on RKNN, and both backends must emit the same
`(1, 4+nc, N)` tensor for the `yolov8` decoder.

**Device-path numerics caveat (RKNN, untested in CI):** the tracker's `to_input`
applies `scale=1/255` on the host, and `rknn_convert` configures `mean=0, std=1`
(raw-pixel passthrough). If on-device validation shows a mismatch, set the RKNN
`std_values` to `255` (let the NPU divide) **or** feed raw pixels and drop the host
scale — keep it consistent with `to_input`.

**NanoTrack RKNN/parity caveat (untested in CI):** the DepthwiseBAN head uses a
**data-dependent conv kernel** (`xcorr_depthwise`: the exemplar feature is the conv
weight) and a **matmul** (`xcorr_pixelwise`). Both export to ONNX and pass torch-vs-
onnxruntime parity (legacy exporter, `dynamo=False`), but RKNN operator support for a
dynamic-weight grouped conv is untested on-device. Validate manually; if unsupported,
fall back to a fixed-template (two-graph) export. `to_input` feeds raw `[0,255]`
(`scale=1.0`), so configure `rknn_convert` with `mean=0, std=1` to match.

## Notes

- `tools/` is not an installed package; `tools/convert.py` and `tools/onnx_to_rknn.py`
  insert their own directory onto `sys.path` so `import convert_lib` works when run as
  scripts. Tests do the same via `tests/conftest.py`.
- Dynamic dims: a `-1` in a manifest input shape is exported with a nominal size; declare
  the axis in the adapter's `dynamic_axes` to keep it dynamic in the ONNX graph.

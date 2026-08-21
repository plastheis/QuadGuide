# Model conversion framework (agnostic-ish converter) — design

> **Status:** approved design (2026-06-05). Generalises the single-purpose
> `tools/siamfc_to_onnx.py` into a small **manifest-driven conversion framework** so new
> NN trackers and YOLO detectors can be converted to ONNX (and chained to RKNN) by adding
> a ~15–20 line adapter rather than a new bespoke script. The primary documentation
> deliverable is a rewritten `tools/CONVERSION.md`.

## 1. Context and motivation

`tools/siamfc_to_onnx.py` (merged 2026-06-05) converts one specific model: it hardcodes the
huanglianghua AlexNetV1 backbone, the SiamFC xcorr head, and the I/O (RGB, 127/255,
`score_map`). It cannot generalise — a bare PyTorch `state_dict` requires the exact
`nn.Module` to load, and the head/IO are SiamFC-specific. The user plans to add more NN
trackers and YOLO detectors and wants a low-friction conversion path.

A truly "any weights → any format" converter is impossible: loading a `state_dict` always
needs architecture knowledge. But the conversion splits into three stages, and only the
first is model-specific:

| Stage | Model-specific? | Notes |
|---|---|---|
| 1. Load checkpoint → `nn.Module` | **Yes** | needs the class; *unless* the artifact is TorchScript/ONNX |
| 2. Export module → ONNX | No | `torch.onnx.export(module, example_inputs, names…)` is generic |
| 3. ONNX → RKNN | No | `onnx_to_rknn.py` is already model-agnostic |

So the "agnostic converter" is really **a shared harness for stages 2–3 + a thin per-model
adapter for stage 1**, with the existing **manifest** supplying I/O so the converter and the
runtime backend never drift.

## 2. Goal

`python tools/convert.py --model <name> --checkpoint <pth> [--rknn --calib <dir>]` converts
any registered model, reading I/O from that model's manifest and writing the artifact to the
exact path the tracker will load from. Adding a new tracker = one adapter file. The mechanics
and the "add a new tracker" recipe are documented in `tools/CONVERSION.md`.

## 3. Decisions (settled with the user)

- **CLI shape:** a single dispatcher `tools/convert.py --model <name>` (registry-driven), not
  per-model scripts.
- **Refactor scope:** migrate the existing SiamFC converter into the first adapter now (one
  consistent code path).
- **Second adapter:** framework + SiamFC only. The YOLO/ultralytics adapter is **documented**
  in `CONVERSION.md` as a worked pattern but **not implemented** (no YOLO weights to test).
- **Formats:** ONNX (always) and RKNN (optional chain). No other targets.

## 4. Architecture

Three stages; only the adapter is per-model. The manifest (loaded via edgecv's own
`load_manifest`) is the single source of truth for input/output names, shapes, and the
artifact paths — the same manifest the runtime backend uses.

### 4.1 File structure

```
tools/
  convert.py                 # CLI dispatcher (argparse -> convert_lib.run)
  convert_lib/
    __init__.py              # exposes run(), registry
    harness.py               # generic: export_onnx + onnx.checker + parity check
    registry.py              # Adapter dataclass + register()/get()/registered_names()
    rknn.py                  # generic onnx -> rknn (moved from onnx_to_rknn.py)
    adapters/
      __init__.py            # imports each adapter module so it self-registers
      siamfc.py              # AlexNetV1/head/Net (moved verbatim) + build() + register
  onnx_to_rknn.py            # thin CLI shim over convert_lib.rknn (ONNX-from-elsewhere)
  CONVERSION.md              # documentation deliverable (rewritten)
  track_webcam.py            # unchanged
```

`convert_lib/` is a real package (multiple files import each other). `convert.py` inserts its
own directory onto `sys.path` before `import convert_lib`, so `python tools/convert.py …`
works from the repo root. Tests put `tools/` on `sys.path` via a one-line `tests/conftest.py`.

### 4.2 Registry and the Adapter contract

The entire per-model contract:

```python
# convert_lib/registry.py
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class Adapter:
    name: str                                  # manifest model name, e.g. "siamfc_generic"
    build: Callable[[str], Any]                # checkpoint path -> loaded .eval() nn.Module
    dynamic_axes: dict | None = None           # optional; for variable dims (e.g. YOLO det count)

_REGISTRY: dict[str, Adapter] = {}

def register(adapter: Adapter) -> None: ...    # _REGISTRY[adapter.name] = adapter
def get(name: str) -> Adapter: ...             # KeyError -> caller lists registered_names()
def registered_names() -> list[str]: ...
```

`registry.py` imports nothing heavy (no torch), so register/get is unit-testable with a dummy
adapter. Adapters self-register when `convert_lib/adapters/__init__.py` imports them.

### 4.3 Harness (generic, model-independent)

```python
# convert_lib/harness.py
def export_and_validate(module, example_inputs, in_names, out_names, out_path, *,
                        opset=13, dynamic_axes=None, tol=1e-3) -> float:
    # 1. torch.onnx.export(module, example_inputs, out_path, input_names=in_names,
    #    output_names=out_names, opset_version=opset, dynamic_axes=dynamic_axes,
    #    do_constant_folding=True)
    # 2. onnx.checker.check_model(out_path)
    # 3. parity: random inputs of the example shapes through module (torch.no_grad)
    #    vs onnxruntime; diff = max|ref - got|; raise SystemExit if diff > tol; return diff
```

No hardcoded names/shapes — everything comes from the manifest via the dispatcher.

### 4.4 Dispatcher (`convert_lib.run`)

```python
# convert_lib/__init__.py (run) — pseudocode
def run(model, checkpoint, out=None, *, rknn=False, target="rk3588", calib=None):
    mf = load_manifest(MANIFESTS_DIR / f"{model}.yaml")          # edgecv loader
    adapter = registry.get(model)                                # -> lists names on miss
    module = adapter.build(checkpoint)                           # strict=True load
    in_names  = [i["name"] for i in mf.inputs]
    out_names = [o["name"] for o in mf.outputs]
    example   = tuple(_zeros(_concrete(i["shape"])) for i in mf.inputs)   # -1 -> nominal dim
    onnx_out  = out or resolve_artifact_path(mf.artifacts["onnx"]["path"])  # edgecv path resolver
    export_and_validate(module, example, in_names, out_names, onnx_out,
                        dynamic_axes=adapter.dynamic_axes)
    if rknn:
        rk_out = resolve_artifact_path(mf.artifacts["rknn"]["path"])
        rknn_convert(onnx_out, rk_out, target, calib, in_names)
    return onnx_out
```

Reusing `edgecv.models.manifest.load_manifest` and `edgecv.models.paths.resolve_artifact_path`
(host-side, dev-only imports) guarantees the converter writes ONNX to **exactly** where the
backend will look, and that I/O matches the runtime. `_concrete` replaces a `-1` dim with a
nominal example size; the adapter's `dynamic_axes` marks those axes dynamic in the export.

### 4.5 SiamFC adapter

```python
# convert_lib/adapters/siamfc.py
class AlexNetV1(nn.Module): ...      # moved verbatim from siamfc_to_onnx.py
class SiamFCHead(nn.Module): ...     # F.conv2d(x, z) * out_scale (batch-1 xcorr)
class Net(nn.Module): ...            # backbone(z), backbone(x), head

def build(checkpoint: str):
    sd = torch.load(checkpoint, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net = Net()
    net.load_state_dict(sd, strict=True)   # fails loudly on key/shape mismatch
    net.eval()
    return net

register(Adapter(name="siamfc_generic", build=build))   # no dynamic_axes (all static)
```

### 4.6 RKNN module + shim

`convert_lib/rknn.py` holds `rknn_convert(onnx_path, out_path, target, calib_dir, input_names)`
— the body of today's `onnx_to_rknn.convert` (deferred `rknn-toolkit2` import, dataset-file
generation, INT8 when `calib_dir` is set). `tools/onnx_to_rknn.py` becomes a thin argparse CLI
that imports and calls it, preserving the standalone ONNX-from-elsewhere path (e.g. an
ultralytics-exported YOLO ONNX).

## 5. Error handling

- Unknown `--model` → error listing `registry.registered_names()`.
- No manifest file for the model → clear "no manifest at <path>".
- `strict=True` mismatch → torch `RuntimeError` (missing/unexpected keys) propagates, prefixed
  with the model name.
- Dynamic dim (`-1`) in a manifest shape → harness uses a nominal example dim; adapter
  `dynamic_axes` marks it dynamic in the export.
- Parity over `tol` → `SystemExit` with the measured diff (current behaviour).
- Missing `rknn-toolkit2` when `--rknn` → the existing actionable install hint.

## 6. Migration of existing files

- `tools/siamfc_to_onnx.py` → **deleted**; `AlexNetV1`/`SiamFCHead`/`Net` move verbatim into
  `convert_lib/adapters/siamfc.py`. The `onnx.checker` + parity logic moves into the harness.
- `tests/test_convert_siamfc.py` → rewritten to drive `convert_lib.run("siamfc_generic", …)`
  and the adapter's `build()` (random-weight round-trip + parity), torch-guarded.
- `tools/onnx_to_rknn.py` → body moves to `convert_lib/rknn.py`; file becomes a thin shim.
  `tests/test_onnx_to_rknn_scaffold.py` → repointed (assert the shim and `convert_lib.rknn`
  expose `convert`/`rknn_convert` and import without `rknn-toolkit2`).
- No `edgecv/` runtime changes.

## 7. Testing strategy

- **registry** (`tests/test_convert_registry.py`, no torch): `register`/`get`/`registered_names`
  with a dummy adapter; `get` on an unknown name raises and the dispatcher surfaces the list.
- **harness** (`tests/test_convert_harness.py`, torch-guarded): export+checker+parity on a tiny
  dummy `nn.Module` with an inline 2-in/1-out shape set; assert parity passes and a
  deliberately mismatched module would exceed `tol`.
- **siamfc adapter / dispatcher** (`tests/test_convert_siamfc.py`, torch-guarded): build a
  random-weight `Net`, save a `.pth`, `run("siamfc_generic", ckpt, out=tmp)`, assert the ONNX
  loads under onnxruntime and parity `< 1e-3`; assert exemplar/search/score_map names+shapes
  match the manifest.
- **rknn scaffold** (`tests/test_onnx_to_rknn_scaffold.py`): imports without `rknn-toolkit2`.
- `tests/conftest.py` adds `tools/` to `sys.path` once.
- Full suite stays green on x86 without torch (torch-dependent tests skip cleanly).

## 8. Dependencies

No new dependencies: `torch`, `onnx` (`[dev]`) and `onnxruntime` (`[test]`) already cover it.
`rknn-toolkit2` remains a documented manual host install.

## 9. CONVERSION.md (primary documentation deliverable)

Rewritten into three parts:

1. **Pipeline & where models go** — keep current content (host-only; `models/` gitignored;
   `$EDGECV_MODEL_DIR`; manifest indirection).
2. **How the framework works** — manifest-driven I/O; the harness (export → checker → parity);
   the registry; the `convert.py` dispatcher; the closed loop that writes ONNX to the
   backend's resolved path; the optional `--rknn` chain.
3. **How to add a new tracker** — a concrete checklist:
   1. Ensure a manifest exists at `edgecv/models/manifests/<name>.yaml` with the model's
      `io` and `artifacts`.
   2. Add `tools/convert_lib/adapters/<name>.py` with a `build(checkpoint) -> nn.Module`
      (vendor or import the architecture; load `state_dict` `strict=True`) and
      `register(Adapter(name="<name>", build=build, dynamic_axes=…))`.
   3. Import it in `convert_lib/adapters/__init__.py`.
   4. Run `python tools/convert.py --model <name> --checkpoint <pth>`.
   Plus a documented (un-built) **YOLO/ultralytics adapter example** showing the
   "upstream already exports ONNX" variant: a `build`-free path where the adapter shells out
   to `ultralytics`' export and the artifact is handed straight to stage 3 — recorded so the
   non-state-dict pattern is on file.

## 10. Out of scope (YAGNI)

- Implementing the YOLO adapter (documented only).
- Entry-point/plugin auto-discovery of adapters (explicit imports in `adapters/__init__.py`
  suffice).
- Targets other than ONNX/RKNN.
- Any `edgecv/` runtime change.

## 11. Risks / notes

- **Module-as-graph-input export** (the SiamFC xcorr head feeds an activation as the conv
  weight) already works at opset 13 and is guarded by the parity check; unchanged by this
  refactor.
- **`tools/` importability.** The `sys.path` insertion (CLI) + `conftest.py` (tests) is the
  one non-obvious mechanic; documented in CONVERSION.md and covered by tests.
- **Manifest as contract.** If a future model's manifest omits `io` or `artifacts`, the
  dispatcher fails early with a clear message rather than exporting a mismatched graph.
```

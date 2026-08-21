# Conversion Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalise the single-purpose SiamFC converter into a manifest-driven `tools/convert.py` dispatcher where adding a new tracker means writing a ~20-line adapter.

**Architecture:** A `tools/convert_lib/` package: a torch-free **registry** of per-model adapters, a generic **harness** (`torch.onnx.export` → `onnx.checker` → torch-vs-onnxruntime parity), a generic **rknn** module, and one **adapter** per model. A `run()` dispatcher reads I/O and artifact paths from the model's existing manifest (via edgecv's own loader + path resolver), so the converter and runtime backend never drift. SiamFC becomes the first adapter; the YOLO/ultralytics path is documented, not built.

**Tech Stack:** Python 3.10+, PyTorch + ONNX (`[dev]`), onnxruntime (`[test]`), pytest. All host-only; not in the wheel.

**Spec:** `docs/superpowers/specs/2026-06-05-conversion-framework-design.md`

---

## File Structure

- `tests/conftest.py` — **create**: put `tools/` on `sys.path` so `import convert_lib` works in tests.
- `tools/convert_lib/__init__.py` — **create**: package surface (`registry` re-exports; `run()` added in Task 4).
- `tools/convert_lib/registry.py` — **create**: `Adapter` dataclass + `register`/`get`/`registered_names` (torch-free).
- `tools/convert_lib/harness.py` — **create**: `export_and_validate(...)` (export + checker + parity).
- `tools/convert_lib/rknn.py` — **create**: `rknn_convert(...)` (moved from `onnx_to_rknn.py`).
- `tools/convert_lib/adapters/__init__.py` — **create**: imports each adapter so it self-registers.
- `tools/convert_lib/adapters/siamfc.py` — **create**: `AlexNetV1`/`SiamFCHead`/`Net` (moved) + `build()` + register.
- `tools/convert.py` — **create**: CLI dispatcher.
- `tools/onnx_to_rknn.py` — **modify**: becomes a thin shim over `convert_lib.rknn`.
- `tools/siamfc_to_onnx.py` — **delete**.
- `tools/CONVERSION.md` — **modify**: rewrite (pipeline / how it works / add a tracker).
- `tests/test_convert_registry.py` — **create** (torch-free).
- `tests/test_convert_harness.py` — **create** (torch-guarded).
- `tests/test_convert_siamfc.py` — **modify**: rewrite to drive `run()` + the adapter.
- `tests/test_onnx_to_rknn_scaffold.py` — **modify**: repoint to `convert_lib.rknn` + the shim.

---

## Task 1: Test seam + registry

**Files:**
- Create: `tests/conftest.py`, `tools/convert_lib/__init__.py`, `tools/convert_lib/registry.py`, `tests/test_convert_registry.py`

- [ ] **Step 1: Create the test seam**

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

# tools/ is not an installed package; put it on sys.path so `import convert_lib` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_convert_registry.py`:

```python
import pytest

from convert_lib.registry import Adapter, get, register, registered_names


def test_register_and_get():
    a = Adapter(name="dummy_reg", build=lambda ckpt: None)
    register(a)
    assert get("dummy_reg") is a
    assert "dummy_reg" in registered_names()


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get("definitely_not_registered")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_convert_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'convert_lib'`

- [ ] **Step 4: Write the package surface**

Create `tools/convert_lib/__init__.py`:

```python
"""Host-only model conversion framework (ARCHITECTURE.md §11). See tools/CONVERSION.md.

convert_lib is imported by tools/convert.py (which puts tools/ on sys.path). The registry
is torch-free; the harness, adapters, and rknn helpers import torch / rknn-toolkit2 lazily,
so importing the registry never pulls heavy deps.
"""

from __future__ import annotations

from .registry import Adapter, get, register, registered_names

__all__ = ["Adapter", "get", "register", "registered_names"]
```

Create `tools/convert_lib/registry.py`:

```python
"""Adapter registry (torch-free). Each model contributes one Adapter; adapters
self-register on import (see adapters/__init__.py)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Adapter:
    name: str                              # manifest model name, e.g. "siamfc_generic"
    build: Callable[[str], Any]            # checkpoint path -> loaded .eval() nn.Module
    dynamic_axes: dict | None = None       # optional; variable dims (e.g. YOLO det count)


_REGISTRY: dict[str, Adapter] = {}


def register(adapter: Adapter) -> None:
    _REGISTRY[adapter.name] = adapter


def get(name: str) -> Adapter:
    return _REGISTRY[name]


def registered_names() -> list[str]:
    return sorted(_REGISTRY)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_convert_registry.py -v`
Expected: PASS (2 passed) — no torch required.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tools/convert_lib/__init__.py tools/convert_lib/registry.py tests/test_convert_registry.py
git commit -m "feat(convert): adapter registry + tools sys.path test seam

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Generic export/validate/parity harness

**Files:**
- Create: `tools/convert_lib/harness.py`, `tests/test_convert_harness.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_convert_harness.py`:

```python
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import torch.nn as nn  # noqa: E402

from convert_lib.harness import export_and_validate  # noqa: E402


class _TwoIn(nn.Module):
    def forward(self, a, b):
        return (a.mean() + b.mean()).reshape(1, 1, 1, 1)


def test_export_and_validate_parity(tmp_path):
    m = _TwoIn().eval()
    ex = (torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    out = tmp_path / "m.onnx"
    diff = export_and_validate(m, ex, ["a", "b"], ["y"], str(out))
    assert out.exists()
    assert diff < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_convert_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'convert_lib.harness'` (or SKIPPED if torch is absent — that is acceptable, but install `.[dev]` to actually exercise it where possible).

- [ ] **Step 3: Write the harness**

Create `tools/convert_lib/harness.py`:

```python
"""Generic ONNX export + validation (ARCHITECTURE.md §11). Model-independent:
all input/output names and shapes come from the caller (the dispatcher reads them
from the manifest). Validates single-output graphs by torch-vs-onnxruntime parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def export_and_validate(module: Any, example_inputs: Sequence[Any],
                        in_names: Sequence[str], out_names: Sequence[str],
                        out_path: str, *, opset: int = 13,
                        dynamic_axes: dict | None = None, tol: float = 1e-3) -> float:
    """Export `module` to ONNX at `out_path`, run onnx.checker, then assert
    torch-vs-onnxruntime parity on random inputs of the example shapes. Returns the
    max abs diff; raises SystemExit if it exceeds `tol`. `in_names` order must match
    `module.forward` positional order (torch is positional; onnxruntime keys by name)."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module, tuple(example_inputs), out_path,
        input_names=list(in_names), output_names=list(out_names),
        opset_version=opset, dynamic_axes=dynamic_axes, do_constant_folding=True)
    onnx.checker.check_model(out_path)

    rng = np.random.default_rng(0)
    feeds = {n: rng.standard_normal(tuple(t.shape)).astype(np.float32)
             for n, t in zip(in_names, example_inputs, strict=True)}
    with torch.no_grad():
        ref = module(*[torch.from_numpy(feeds[n]) for n in in_names])
    if isinstance(ref, (tuple, list)):
        ref = ref[0]
    ref = ref.numpy()
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(list(out_names), feeds)[0]
    diff = float(np.max(np.abs(ref - got)))
    if diff > tol:
        raise SystemExit(f"parity check FAILED: max|delta|={diff:.2e} > {tol:.0e}")
    return diff
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_convert_harness.py -v`
Expected: PASS (1 passed) when torch is installed; SKIPPED otherwise.

- [ ] **Step 5: Lint**

Run: `.venv/bin/ruff check tools/convert_lib/harness.py tests/test_convert_harness.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tools/convert_lib/harness.py tests/test_convert_harness.py
git commit -m "feat(convert): generic export+checker+parity harness

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: RKNN module + thin shim

Move the existing `onnx_to_rknn.convert` body into `convert_lib/rknn.py` as `rknn_convert`, and reduce `tools/onnx_to_rknn.py` to a CLI shim. Repoint its test.

**Files:**
- Create: `tools/convert_lib/rknn.py`
- Modify: `tools/onnx_to_rknn.py`, `tests/test_onnx_to_rknn_scaffold.py`

- [ ] **Step 1: Rewrite the scaffold test (failing)**

Replace the entire contents of `tests/test_onnx_to_rknn_scaffold.py` with:

```python
import importlib.util
from pathlib import Path

from convert_lib.rknn import rknn_convert


def test_rknn_convert_importable_without_toolkit():
    # the rknn-toolkit2 import is deferred, so importing the function never needs it
    assert callable(rknn_convert)


def test_shim_exposes_main():
    path = Path(__file__).resolve().parent.parent / "tools" / "onnx_to_rknn.py"
    spec = importlib.util.spec_from_file_location("onnx_to_rknn", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_onnx_to_rknn_scaffold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'convert_lib.rknn'`

- [ ] **Step 3: Create the rknn module**

Create `tools/convert_lib/rknn.py`:

```python
"""Generic ONNX -> RKNN conversion (ARCHITECTURE.md §11). Host-only, x86. The
rknn-toolkit2 import is deferred so this module imports anywhere; conversion only
runs where the toolkit is installed. Not exercised in CI.

rknn-toolkit2 is not on PyPI cleanly — install it on an x86 host from Rockchip's
release wheels. INT8 quantisation needs a folder of representative calibration images."""

from __future__ import annotations

_INSTALL_HINT = (
    "rknn-toolkit2 is not importable. Install it on an x86 host from Rockchip's "
    "release wheels (it is not on PyPI). This tool runs offline; the device only "
    "runs the lite runtime (ARCHITECTURE.md §11, §12)."
)


def _import_rknn():
    from rknn.api import RKNN  # type: ignore

    return RKNN


def _write_dataset_file(calibration_dir: str) -> str:
    """RKNN's build() wants a text file listing one calibration image per line."""
    from pathlib import Path

    imgs = sorted(
        str(p) for p in Path(calibration_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not imgs:
        raise SystemExit(f"no calibration images found in {calibration_dir!r}")
    listing = Path(calibration_dir) / "dataset.txt"
    listing.write_text("\n".join(imgs) + "\n")
    return str(listing)


def rknn_convert(onnx_path: str, out_path: str, target: str,
                 calibration_dir: str | None, input_names: list[str]) -> str:
    try:
        RKNN = _import_rknn()
    except Exception as e:  # pragma: no cover - depends on host toolkit
        raise RuntimeError(_INSTALL_HINT) from e

    quantize = calibration_dir is not None
    rknn = RKNN(verbose=True)
    # mean/std [0,255] passthrough: weights consume raw pixels (scale handled in the
    # tracker preprocessing, not the model). Adjust if a future model normalises.
    rknn.config(mean_values=[[0, 0, 0]] * len(input_names),
                std_values=[[1, 1, 1]] * len(input_names),
                target_platform=target)
    if rknn.load_onnx(model=onnx_path, inputs=input_names) != 0:
        raise RuntimeError(f"load_onnx failed for {onnx_path!r}")
    dataset = _write_dataset_file(calibration_dir) if quantize else None
    if rknn.build(do_quantization=quantize, dataset=dataset) != 0:
        raise RuntimeError("rknn build failed")
    if rknn.export_rknn(out_path) != 0:
        raise RuntimeError(f"export_rknn failed for {out_path!r}")
    rknn.release()
    print(f"exported {out_path} (target={target}, quantized={quantize})")
    return out_path
```

- [ ] **Step 4: Reduce onnx_to_rknn.py to a shim**

Replace the entire contents of `tools/onnx_to_rknn.py` with:

```python
"""ONNX -> RKNN CLI (host-only). Thin wrapper over convert_lib.rknn for converting an
ONNX produced elsewhere (e.g. an ultralytics export). See tools/CONVERSION.md.

Usage (on a host with rknn-toolkit2):
    python tools/onnx_to_rknn.py --onnx models/m.onnx --out models/m.rk3588.rknn \
        --target rk3588 --calibration-dir calib/ --inputs exemplar search
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_lib.rknn import rknn_convert  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX -> RKNN (host-only)")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--calibration-dir", default=None,
                    help="folder of representative images; enables INT8 quantisation")
    ap.add_argument("--inputs", nargs="+", default=["exemplar", "search"],
                    help="model input names (order must match the ONNX graph)")
    args = ap.parse_args()
    rknn_convert(args.onnx, args.out, args.target, args.calibration_dir, args.inputs)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_onnx_to_rknn_scaffold.py -v`
Expected: PASS (2 passed) — imports succeed without `rknn-toolkit2`.

- [ ] **Step 6: Lint**

Run: `.venv/bin/ruff check tools/convert_lib/rknn.py tools/onnx_to_rknn.py tests/test_onnx_to_rknn_scaffold.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add tools/convert_lib/rknn.py tools/onnx_to_rknn.py tests/test_onnx_to_rknn_scaffold.py
git commit -m "refactor(convert): move onnx->rknn into convert_lib.rknn; onnx_to_rknn is a shim

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: SiamFC adapter + dispatcher + CLI (and delete old script)

Move the SiamFC module into the first adapter, add the `run()` dispatcher and `convert.py` CLI, delete `siamfc_to_onnx.py`, and rewrite its test to drive `run()`. Keeps the suite green in one commit.

**Files:**
- Create: `tools/convert_lib/adapters/__init__.py`, `tools/convert_lib/adapters/siamfc.py`, `tools/convert.py`
- Modify: `tools/convert_lib/__init__.py`, `tests/test_convert_siamfc.py`
- Delete: `tools/siamfc_to_onnx.py`

- [ ] **Step 1: Rewrite the converter test (failing)**

Replace the entire contents of `tests/test_convert_siamfc.py` with:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from convert_lib import run  # noqa: E402
from convert_lib.adapters.siamfc import Net, build  # noqa: E402


def _save_random_ckpt(tmp_path):
    ckpt = tmp_path / "fake.pth"
    torch.save(Net().state_dict(), ckpt)
    return ckpt


def test_build_loads_strict_and_shapes(tmp_path):
    net = build(str(_save_random_ckpt(tmp_path)))
    z = torch.zeros(1, 3, 127, 127)
    x = torch.zeros(1, 3, 255, 255)
    with torch.no_grad():
        out = net(z, x)
    assert tuple(out.shape) == (1, 1, 17, 17)


def test_run_roundtrip_parity(tmp_path):
    ckpt = _save_random_ckpt(tmp_path)
    out = tmp_path / "siamfc.onnx"
    run("siamfc_generic", str(ckpt), str(out))
    assert out.exists()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    z = rng.standard_normal((1, 3, 127, 127)).astype(np.float32)
    x = rng.standard_normal((1, 3, 255, 255)).astype(np.float32)
    got = sess.run(["score_map"], {"exemplar": z, "search": x})[0]
    assert got.shape == (1, 1, 17, 17)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_convert_siamfc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'convert_lib.adapters'` (or SKIPPED without torch).

- [ ] **Step 3: Create the SiamFC adapter**

Create `tools/convert_lib/adapters/siamfc.py`:

```python
"""SiamFC adapter: vendored huanglianghua/siamfc-pytorch AlexNetV1 backbone + batch-1
cross-correlation head, registered for the `siamfc_generic` manifest. The module is
self-contained so the original repo need not be installed."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from convert_lib.registry import Adapter, register


def _bn(c: int) -> nn.BatchNorm2d:
    # huanglianghua uses eps=1e-6, momentum=0.05; eval-time eps affects numerics.
    return nn.BatchNorm2d(c, eps=1e-6, momentum=0.05)


class AlexNetV1(nn.Module):
    """Backbone matching siamfc-pytorch state_dict keys (backbone.conv1..conv5).
    Conv layers keep their default bias (the checkpoint stores conv biases)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2), _bn(96), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2), _bn(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1), _bn(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2), _bn(384), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class SiamFCHead(nn.Module):
    """Batch-1 cross-correlation: conv2d(search_feat, exemplar_feat) * out_scale.
    Numerically identical to the repo's _fast_xcorr for batch size 1."""

    def __init__(self, out_scale: float = 0.001) -> None:
        super().__init__()
        self.out_scale = out_scale

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, z) * self.out_scale


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = AlexNetV1()
        self.head = SiamFCHead()

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # forward arg order (z=exemplar, x=search) MUST match manifest io.inputs order.
        return self.head(self.backbone(z), self.backbone(x))


def build(checkpoint: str) -> Net:
    """Load a checkpoint state_dict into Net (strict=True) and return it in eval mode."""
    sd = torch.load(checkpoint, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:   # tolerate wrapped checkpoints
        sd = sd["state_dict"]
    net = Net()
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


register(Adapter(name="siamfc_generic", build=build))
```

- [ ] **Step 4: Create the adapters package init**

Create `tools/convert_lib/adapters/__init__.py`:

```python
"""Importing this package registers every adapter. Add new adapters here."""

from __future__ import annotations

from . import siamfc  # noqa: F401  (import side effect: registers the adapter)
```

- [ ] **Step 5: Add the dispatcher to the package surface**

Replace the entire contents of `tools/convert_lib/__init__.py` with:

```python
"""Host-only model conversion framework (ARCHITECTURE.md §11). See tools/CONVERSION.md.

convert_lib is imported by tools/convert.py (which puts tools/ on sys.path). The registry
is torch-free; the harness, adapters, and rknn helpers import torch / rknn-toolkit2 lazily,
so importing the registry never pulls heavy deps."""

from __future__ import annotations

from pathlib import Path

from . import registry
from .registry import Adapter, get, register, registered_names

__all__ = ["Adapter", "get", "register", "registered_names", "run"]

_MANIFESTS = Path(__file__).resolve().parents[2] / "edgecv" / "models" / "manifests"
_NOMINAL_DIM = 1


def _concrete(shape) -> tuple[int, ...]:
    """Replace dynamic (-1) dims with a nominal size for the export example input."""
    return tuple(d if isinstance(d, int) and d > 0 else _NOMINAL_DIM for d in shape)


def run(model: str, checkpoint: str, out: str | None = None, *,
        rknn: bool = False, target: str = "rk3588", calib: str | None = None) -> str:
    """Convert `checkpoint` for `model` to ONNX (and optionally RKNN), driven by the
    model's manifest. Writes ONNX to `out` or to the manifest's resolved artifact path."""
    import torch

    from edgecv.models.manifest import load_manifest
    from edgecv.models.paths import resolve_artifact_path

    from . import adapters  # noqa: F401  (registers adapters; imports torch)
    from .harness import export_and_validate

    mf_path = _MANIFESTS / f"{model}.yaml"
    if not mf_path.exists():
        raise SystemExit(f"no manifest at {mf_path}")
    mf = load_manifest(mf_path)
    try:
        adapter = registry.get(model)
    except KeyError:
        raise SystemExit(
            f"no adapter registered for {model!r}; registered: {registry.registered_names()}"
        ) from None
    try:
        module = adapter.build(checkpoint)
    except RuntimeError as e:
        raise SystemExit(f"failed to load checkpoint for {model!r}: {e}") from e

    in_names = [i["name"] for i in mf.inputs]
    out_names = [o["name"] for o in mf.outputs]
    example = tuple(torch.zeros(_concrete(i["shape"])) for i in mf.inputs)
    onnx_out = out or resolve_artifact_path(mf.artifacts["onnx"]["path"])
    diff = export_and_validate(module, example, in_names, out_names, onnx_out,
                               dynamic_axes=adapter.dynamic_axes)
    print(f"exported {onnx_out}  (parity max|delta|={diff:.2e})")

    if rknn:
        from .rknn import rknn_convert
        rk_out = resolve_artifact_path(mf.artifacts["rknn"]["path"])
        rknn_convert(onnx_out, rk_out, target, calib, in_names)
    return onnx_out
```

- [ ] **Step 6: Create the CLI dispatcher**

Create `tools/convert.py`:

```python
"""CLI dispatcher for model conversion (ARCHITECTURE.md §11). Host-only.

Adds tools/ to sys.path so `import convert_lib` resolves when run as a script:
    python tools/convert.py --model siamfc_generic --checkpoint models/siamfc.pth
    python tools/convert.py --model siamfc_generic --checkpoint models/siamfc.pth --rknn --calib calib/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_lib import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a tracker checkpoint to ONNX (+ optional RKNN)")
    ap.add_argument("--model", required=True, help="manifest model name, e.g. siamfc_generic")
    ap.add_argument("--checkpoint", required=True, help="path to the .pth state_dict")
    ap.add_argument("--out", default=None,
                    help="ONNX output path (default: the manifest's resolved artifact path)")
    ap.add_argument("--rknn", action="store_true", help="also convert the ONNX to RKNN")
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--calib", default=None, help="calibration image dir for INT8 RKNN")
    args = ap.parse_args()
    run(args.model, args.checkpoint, args.out,
        rknn=args.rknn, target=args.target, calib=args.calib)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Delete the old single-purpose script**

```bash
git rm tools/siamfc_to_onnx.py
```

- [ ] **Step 8: Run the converter test**

Run: `.venv/bin/pytest tests/test_convert_siamfc.py -v`
Expected: PASS (2 passed) when torch installed; SKIPPED otherwise.

- [ ] **Step 9: Run the full suite + lint**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check tools/ tests/`
Expected: full suite passes (converter/harness tests skip without torch); ruff clean.

- [ ] **Step 10: Commit**

```bash
git add tools/convert_lib/adapters/__init__.py tools/convert_lib/adapters/siamfc.py tools/convert.py tools/convert_lib/__init__.py tests/test_convert_siamfc.py
git rm tools/siamfc_to_onnx.py
git commit -m "feat(convert): siamfc adapter + manifest-driven dispatcher + convert.py CLI

Replaces tools/siamfc_to_onnx.py with the first adapter on the new framework.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Rewrite CONVERSION.md

**Files:**
- Modify: `tools/CONVERSION.md`

- [ ] **Step 1: Replace the doc**

Replace the entire contents of `tools/CONVERSION.md` with:

```markdown
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

# convert an ONNX produced elsewhere (e.g. ultralytics) straight to RKNN
python tools/onnx_to_rknn.py --onnx models/yolo.onnx --out models/yolo.rk3588.rknn \
    --target rk3588 --calibration-dir calib/ --inputs images
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

Some model families ship their own exporter, so you don't need a torch `nn.Module` adapter
at all — export the ONNX with the upstream tool, then convert straight to RKNN:

```bash
yolo export model=yolo.pt format=onnx        # ultralytics writes yolo.onnx
python tools/onnx_to_rknn.py --onnx yolo.onnx --out models/yolo_generic.rk3588.rknn \
    --target rk3588 --calibration-dir calib/ --inputs images
```

(Folding this behind `tools/convert.py` — an adapter that shells out to the upstream
exporter in place of `build()` and skips the torch export — is a possible future addition,
not implemented yet.)

## Notes

- `tools/` is not an installed package; `tools/convert.py` and `tools/onnx_to_rknn.py`
  insert their own directory onto `sys.path` so `import convert_lib` works when run as
  scripts. Tests do the same via `tests/conftest.py`.
- Dynamic dims: a `-1` in a manifest input shape is exported with a nominal size; declare
  the axis in the adapter's `dynamic_axes` to keep it dynamic in the ONNX graph.
```

- [ ] **Step 2: Commit**

```bash
git add tools/CONVERSION.md
git commit -m "docs(convert): document the conversion framework and how to add a tracker

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Final verification

- [ ] **Step 1: Full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (torch-dependent converter/harness/siamfc tests skip without torch; registry + rknn scaffold tests run).

- [ ] **Step 2: Lint + types**

Run: `.venv/bin/ruff check . && .venv/bin/mypy edgecv`
Expected: ruff clean; mypy success (`tools/` is host-only and not in the `mypy edgecv` target — unchanged from before).

- [ ] **Step 3: Confirm the old script is gone and the new entry point exists**

Run: `test ! -f tools/siamfc_to_onnx.py && test -f tools/convert.py && echo "OK: migrated"`
Expected: prints `OK: migrated`

- [ ] **Step 4: Manual smoke test (documented, optional — needs torch + a real checkpoint)**

```bash
pip install -e .[dev]
python tools/convert.py --model siamfc_generic --checkpoint models/siamfc_alexnet_e50.pth
```

Expected: exports `models/siamfc_generic.onnx` with a passing parity check, printed as
`exported models/siamfc_generic.onnx (parity max|delta|=…)`.

---

## Self-Review Notes (addressed)

- **Spec coverage:** §4.1 file structure → Tasks 1–4; §4.2 registry → Task 1; §4.3 harness →
  Task 2; §4.4 dispatcher → Task 4; §4.5 siamfc adapter → Task 4; §4.6 rknn module + shim →
  Task 3; §5 error handling → `run()` in Task 4 (unknown model, missing manifest, strict
  load); §6 migration (delete siamfc_to_onnx, rewrite tests) → Tasks 3–4; §7 testing →
  Tasks 1–4 + Task 6; §8 deps (none new) → n/a; §9 CONVERSION.md → Task 5; §10 YAGNI (YOLO
  documented only) → Task 5. All covered.
- **Green at each commit:** Task 3 repoints the rknn test in the same commit that moves the
  code; Task 4 deletes `siamfc_to_onnx.py` and rewrites its test in the same commit.
- **Type/name consistency:** `Adapter(name, build, dynamic_axes)`, `register/get/registered_names`,
  `export_and_validate(module, example_inputs, in_names, out_names, out_path, *, opset,
  dynamic_axes, tol)`, `rknn_convert(onnx_path, out_path, target, calibration_dir, input_names)`,
  `run(model, checkpoint, out=None, *, rknn, target, calib)`, and `build(checkpoint)` /
  `Net` are used consistently across tasks and tests.
- **Torch-free seam preserved:** `convert_lib/__init__` and `registry`/`rknn` import no torch;
  adapters/harness are imported only inside `run()` or torch-guarded tests, so the registry
  and rknn-scaffold tests run on x86 without torch.
```

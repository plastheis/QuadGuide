# SiamFC PyTorch → ONNX Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert a `huanglianghua/siamfc-pytorch` AlexNetV1 checkpoint to ONNX and fix the manifest/preprocessing/path wiring so `SiamFC(manifest=..., backend="onnx")` produces valid score maps.

**Architecture:** Host-only conversion tool in `tools/` vendors a self-contained AlexNetV1 backbone + batch-1 xcorr head, loads the checkpoint `strict=True`, and exports a single two-input graph `(exemplar[1,3,127,127], search[1,3,255,255]) → score_map[1,1,17,17]` with a torch-vs-onnxruntime parity self-check. Alongside: correct the SiamFC manifest to RGB / raw `[0,255]`, wire `manifest.preprocessing → SiamFC` for `color`/`scale`, and add `$EDGECV_MODEL_DIR`-based artifact-path resolution. RKNN conversion is scaffolded (written + documented, not run).

**Tech Stack:** Python 3.10+, numpy, PyYAML, PyTorch (host/dev only), ONNX, onnxruntime, pytest.

**Spec:** `docs/superpowers/specs/2026-06-05-siamfc-onnx-conversion-design.md`

---

## File Structure

- `edgecv/models/paths.py` — **create**: `resolve_artifact_path()` (artifact path resolution).
- `edgecv/backends/onnx/__init__.py` — **modify**: resolve artifact path before opening session.
- `edgecv/backends/rknn/__init__.py` — **modify**: resolve artifact path before `load_rknn`.
- `edgecv/trackers/nn/base.py` — **modify**: `UNSET` sentinel, `resolve_pp()`, `manifest_preprocessing()`, store `self._preprocessing` on `NNTracker`.
- `edgecv/trackers/nn/siamfc.py` — **modify**: resolve `color`/`scale` via precedence; thread `scale` into `to_input`.
- `edgecv/models/manifests/siamfc_generic.yaml` — **modify**: RGB 3-channel I/O, `color: rgb`, `scale: 1.0`.
- `tests/_nn_stubs.py` — **modify**: `siam_io` → 3-channel.
- `tests/_onnx_synth.py` — **modify**: `build_siamfc_onnx` → 3-channel inputs, channel-reduced `[1,1,17,17]` output.
- `tests/test_siamfc.py` — **modify**: exemplar-shape assertion → `(1,3,127,127)`.
- `tests/test_manifests_nn.py` — **modify**: `color` assertion → `"rgb"`.
- `tests/test_paths.py` — **create**: path-resolution unit tests.
- `tests/test_pp_precedence.py` — **create**: precedence + SiamFC default tests.
- `tools/siamfc_to_onnx.py` — **create**: converter (vendored net + export + parity).
- `tools/onnx_to_rknn.py` — **create**: RKNN scaffold (not run).
- `tools/CONVERSION.md` — **create**: docs.
- `tests/test_convert_siamfc.py` — **create**: converter test (torch-guarded).
- `tests/test_onnx_to_rknn_scaffold.py` — **create**: scaffold import/guard test.
- `pyproject.toml` — **modify**: populate `[dev]` extra.

---

## Task 1: Artifact path resolution

**Files:**
- Create: `edgecv/models/paths.py`
- Create: `tests/test_paths.py`
- Modify: `edgecv/backends/onnx/__init__.py`
- Modify: `edgecv/backends/rknn/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_paths.py`:

```python
import os

from edgecv.models.paths import resolve_artifact_path


def test_absolute_path_passes_through(tmp_path):
    p = tmp_path / "model.onnx"
    assert resolve_artifact_path(str(p)) == str(p)


def test_relative_resolves_against_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EDGECV_MODEL_DIR", str(tmp_path))
    assert resolve_artifact_path("siamfc_generic.onnx") == str(tmp_path / "siamfc_generic.onnx")


def test_relative_default_is_models_dir(monkeypatch):
    monkeypatch.delenv("EDGECV_MODEL_DIR", raising=False)
    assert resolve_artifact_path("siamfc_generic.onnx") == os.path.join("models", "siamfc_generic.onnx")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edgecv.models.paths'`

- [ ] **Step 3: Write minimal implementation**

Create `edgecv/models/paths.py`:

```python
"""Artifact path resolution (ARCHITECTURE.md §10.1, §11).

Manifests carry relative artifact paths (e.g. ``siamfc_generic.onnx``). Model
blobs are host-only and gitignored, living under a models directory rather than
in the package. Backends resolve a relative path against ``$EDGECV_MODEL_DIR``
(default ``models``); absolute paths pass through unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_artifact_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    base = Path(os.environ.get("EDGECV_MODEL_DIR", "models"))
    return str(base / p)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_paths.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Wire the onnx backend**

In `edgecv/backends/onnx/__init__.py`, add the import after the existing imports (below line 9):

```python
from edgecv.models.paths import resolve_artifact_path
```

Replace the session construction (currently lines 75-77):

```python
        session = ort.InferenceSession(
            artifact["path"], providers=["CPUExecutionProvider"]
        )
```

with:

```python
        session = ort.InferenceSession(
            resolve_artifact_path(artifact["path"]), providers=["CPUExecutionProvider"]
        )
```

- [ ] **Step 6: Wire the rknn backend**

In `edgecv/backends/rknn/__init__.py`, add after the existing imports (below line 16):

```python
from edgecv.models.paths import resolve_artifact_path
```

Replace the load line (currently line 89):

```python
        if rknn.load_rknn(artifact["path"]) != 0:
```

with:

```python
        if rknn.load_rknn(resolve_artifact_path(artifact["path"])) != 0:
```

- [ ] **Step 7: Run the affected suites to verify nothing broke**

Run: `pytest tests/test_paths.py tests/test_onnx_backend.py tests/test_nn_onnx.py tests/test_registry.py -v`
Expected: PASS (the onnx tests inject absolute `tmp_path` artifact paths, which pass through unchanged).

- [ ] **Step 8: Commit**

```bash
git add edgecv/models/paths.py tests/test_paths.py edgecv/backends/onnx/__init__.py edgecv/backends/rknn/__init__.py
git commit -m "feat(models): resolve relative artifact paths via EDGECV_MODEL_DIR

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Preprocessing precedence helper + NNTracker storage

Adds the precedence machinery and stores `self._preprocessing` on `NNTracker`. No tracker *behaviour* changes yet (SiamFC starts consuming it in Task 3). `resolve_model` is left untouched so `YoloDetector`/`YoloTracker` are unaffected.

**Files:**
- Modify: `edgecv/trackers/nn/base.py`
- Create: `tests/test_pp_precedence.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pp_precedence.py`:

```python
from edgecv.trackers.nn.base import UNSET, manifest_preprocessing, resolve_pp


def test_explicit_kwarg_wins_over_manifest_and_default():
    assert resolve_pp("rgb", {"color": "gray"}, "color", "bgr") == "rgb"


def test_manifest_wins_over_default_when_unset():
    assert resolve_pp(UNSET, {"color": "gray"}, "color", "rgb") == "gray"


def test_default_when_unset_and_absent_from_manifest():
    assert resolve_pp(UNSET, {}, "color", "rgb") == "rgb"


def test_manifest_preprocessing_none_is_empty():
    assert manifest_preprocessing(None) == {}


def test_manifest_preprocessing_reads_yaml():
    pp = manifest_preprocessing("edgecv/models/manifests/yolo_generic.yaml")
    assert pp["output_format"] == "yolov5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pp_precedence.py -v`
Expected: FAIL with `ImportError: cannot import name 'UNSET'`

- [ ] **Step 3: Write the implementation**

In `edgecv/trackers/nn/base.py`, add module-level definitions after the imports (below the `load_manifest` import, before `@dataclass class Template`):

```python
UNSET = object()  # sentinel: "__init__ kwarg not explicitly passed"


def resolve_pp(value, manifest_pp: dict, key: str, default):
    """Precedence: explicit kwarg > manifest preprocessing > hardcoded default
    (ARCHITECTURE.md §10.1; nn-trackers design §7)."""
    if value is not UNSET:
        return value
    if key in manifest_pp:
        return manifest_pp[key]
    return default


def manifest_preprocessing(
    manifest: ModelManifest | str | Path | None,
) -> dict:
    """The preprocessing dict for a manifest (path, object, or None)."""
    if manifest is None:
        return {}
    mf = manifest if isinstance(manifest, ModelManifest) else load_manifest(manifest)
    return dict(mf.preprocessing)
```

Then, in `NNTracker.__init__`, store the preprocessing dict. Replace the current body (lines 55-60):

```python
    def __init__(self, manifest: ModelManifest | str | Path | None = None, *,
                 backend: str = "auto", model: Model | None = None) -> None:
        self._model: Model = resolve_model(manifest, backend, model)
        self._status: TrackStatus = TrackStatus.INITIALIZING
        self._seq: int = 0
        self._closed: bool = False
```

with:

```python
    def __init__(self, manifest: ModelManifest | str | Path | None = None, *,
                 backend: str = "auto", model: Model | None = None) -> None:
        self._preprocessing: dict = manifest_preprocessing(manifest)
        self._model: Model = resolve_model(manifest, backend, model)
        self._status: TrackStatus = TrackStatus.INITIALIZING
        self._seq: int = 0
        self._closed: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pp_precedence.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Run NN suites to verify base change is non-breaking**

Run: `pytest tests/test_nn_base.py tests/test_siamfc.py tests/test_yolo.py -v`
Expected: PASS (storing `self._preprocessing` is additive; SiamFC/Yolo behaviour unchanged this task).

- [ ] **Step 6: Commit**

```bash
git add edgecv/trackers/nn/base.py tests/test_pp_precedence.py
git commit -m "feat(nn): preprocessing precedence helper + NNTracker stores manifest preprocessing

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Switch SiamFC + manifest + fixtures to RGB and wire color/scale

The correctness change: real weights are RGB and trained on raw `[0,255]`. Update manifest, fixtures, and `SiamFC` together so the suite stays green.

**Files:**
- Modify: `edgecv/models/manifests/siamfc_generic.yaml`
- Modify: `edgecv/trackers/nn/siamfc.py`
- Modify: `tests/_nn_stubs.py`
- Modify: `tests/_onnx_synth.py`
- Modify: `tests/test_siamfc.py`
- Modify: `tests/test_manifests_nn.py`

- [ ] **Step 1: Update the manifest**

Replace `edgecv/models/manifests/siamfc_generic.yaml` entirely with:

```yaml
name: siamfc_generic
task: sot_template_matching
# Weights: huanglianghua/siamfc-pytorch AlexNetV1. RGB 3-channel, trained on raw
# [0,255] pixels (no /255, no mean/std), cv2/BGR channel order. The caller feeds
# BGR frames (e.g. tools/track_webcam.py via cv2); to_input preserves channel order.
preprocessing:
  color: rgb
  scale: 1.0
  exemplar: 127
  search: 255
  context: 0.5
  total_stride: 8
  response_up: 16
  scale_num: 3
  scale_step: 1.0375
  scale_penalty: 0.9745
  scale_lr: 0.59
  window_influence: 0.176
io:
  inputs:
    - { name: exemplar, shape: [1, 3, 127, 127], dtype: float32 }
    - { name: search,   shape: [1, 3, 255, 255], dtype: float32 }
  outputs:
    - { name: score_map, shape: [1, 1, 17, 17], dtype: float32 }
artifacts:
  onnx: { path: siamfc_generic.onnx }
  rknn: { path: siamfc_generic.rk3588.rknn, quant: int8 }
```

- [ ] **Step 2: Update the SiamFC tracker to resolve color/scale**

In `edgecv/trackers/nn/siamfc.py`, update the import (line 15):

```python
from edgecv.trackers.nn.base import NNTracker, Template
```

to:

```python
from edgecv.trackers.nn.base import UNSET, NNTracker, Template, resolve_pp
```

Change the `__init__` signature `color` parameter (line 31) from `color="gray"` to `color=UNSET`, and add a `scale=UNSET` parameter immediately after it. The signature lines 27-31 become:

```python
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 exemplar_size=127, search_size=255, context=0.5,
                 total_stride=8, response_up=16, scale_num=3, scale_step=1.0375,
                 scale_penalty=0.9745, scale_lr=0.59, window_influence=0.176,
                 color=UNSET, scale=UNSET, score_lock=8.0, score_lost=4.0) -> None:
```

Replace the color assignment (line 43, `self._color = color`) with both resolved attributes:

```python
        self._color = resolve_pp(color, self._preprocessing, "color", "rgb")
        self._scale = resolve_pp(scale, self._preprocessing, "scale", 1.0)
```

Thread `scale` into the two `to_input` calls. In `init` (line 77), replace:

```python
        z = to_input(patch, spec_z, color=self._color)
```

with:

```python
        z = to_input(patch, spec_z, color=self._color, scale=self._scale)
```

In `update` (line 107), replace:

```python
            x = to_input(patch, spec_x, color=self._color)
```

with:

```python
            x = to_input(patch, spec_x, color=self._color, scale=self._scale)
```

- [ ] **Step 3: Update the stub io spec to 3-channel**

In `tests/_nn_stubs.py`, replace the `siam_io` input specs (lines 38-39):

```python
        inputs=(TensorSpec("exemplar", (1, 1, 127, 127), "float32"),
                TensorSpec("search", (1, 1, 255, 255), "float32")),
```

with:

```python
        inputs=(TensorSpec("exemplar", (1, 3, 127, 127), "float32"),
                TensorSpec("search", (1, 3, 255, 255), "float32")),
```

- [ ] **Step 4: Update the synthetic onnx builder to 3-channel**

In `tests/_onnx_synth.py`, replace the whole `build_siamfc_onnx` function (lines 11-25) with:

```python
def build_siamfc_onnx(path: str, score_size: int = 17) -> None:
    ex = helper.make_tensor_value_info("exemplar", TensorProto.FLOAT, [1, 3, 127, 127])
    se = helper.make_tensor_value_info("search", TensorProto.FLOAT, [1, 3, 255, 255])
    sc = helper.make_tensor_value_info(
        "score_map", TensorProto.FLOAT, [1, 1, score_size, score_size]
    )
    # AveragePool(255, k=15, s=15) -> [1,3,17,17]; mean over channels -> [1,1,17,17];
    # add scalar mean(exemplar) so both inputs are consumed.
    pool = helper.make_node("AveragePool", ["search"], ["pooled"],
                            kernel_shape=[15, 15], strides=[15, 15])
    redc = helper.make_node("ReduceMean", ["pooled"], ["pooled_c"],
                            axes=[1], keepdims=1)            # [1,1,17,17]
    rm = helper.make_node("ReduceMean", ["exemplar"], ["ex_mean"], keepdims=0)  # scalar
    add = helper.make_node("Add", ["pooled_c", "ex_mean"], ["score_map"])
    graph = helper.make_graph([pool, redc, rm, add], "siamfc_stub", [ex, se], [sc])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
```

- [ ] **Step 5: Update the exemplar-shape assertion**

In `tests/test_siamfc.py`, in `test_init_builds_127_exemplar_template`, replace:

```python
    assert z.shape == (1, 1, 127, 127)
```

with:

```python
    assert z.shape == (1, 3, 127, 127)
```

- [ ] **Step 6: Update the manifest color assertion**

In `tests/test_manifests_nn.py`, in `test_siamfc_manifest_loads`, replace:

```python
    assert m.preprocessing["color"] == "gray"
```

with:

```python
    assert m.preprocessing["color"] == "rgb"
```

- [ ] **Step 7: Run the full NN + manifest suite**

Run: `pytest tests/test_siamfc.py tests/test_nn_onnx.py tests/test_manifests_nn.py tests/test_nn_base.py tests/test_yolo.py tests/test_nn_preprocess.py -v`
Expected: PASS. (Scripted-model tests ignore inputs, so `scale=1.0` is numerically irrelevant there; the onnx integration test now drives a 3-channel synthetic model.)

- [ ] **Step 8: Run the whole suite**

Run: `pytest -q`
Expected: PASS (full suite green).

- [ ] **Step 9: Commit**

```bash
git add edgecv/models/manifests/siamfc_generic.yaml edgecv/trackers/nn/siamfc.py tests/_nn_stubs.py tests/_onnx_synth.py tests/test_siamfc.py tests/test_manifests_nn.py
git commit -m "fix(siamfc): RGB 3-channel + raw [0,255] preprocessing for real weights

Manifest, fixtures, and SiamFC now match huanglianghua/siamfc-pytorch (RGB, no
/255). color/scale resolve via manifest-preprocessing precedence.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Populate the `[dev]` extra

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit the dev extra**

In `pyproject.toml`, replace the line:

```toml
dev = []
```

with:

```toml
# Host-only conversion tooling (ARCHITECTURE.md §11). Not a runtime dependency.
dev = ["torch>=2.0", "onnx>=1.15"]
```

- [ ] **Step 2: Verify the file still parses**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add torch/onnx to [dev] extra for host conversion tooling

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: SiamFC → ONNX converter tool

**Files:**
- Create: `tools/siamfc_to_onnx.py`
- Create: `tests/test_convert_siamfc.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_convert_siamfc.py`:

```python
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

_PATH = Path(__file__).resolve().parent.parent / "tools" / "siamfc_to_onnx.py"
_spec = importlib.util.spec_from_file_location("siamfc_to_onnx", _PATH)
conv = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(conv)


def test_build_net_loads_strict_and_runs():
    net = conv.build_net()
    z = torch.zeros(1, 3, 127, 127)
    x = torch.zeros(1, 3, 255, 255)
    with torch.no_grad():
        out = net(z, x)
    assert tuple(out.shape) == (1, 1, 17, 17)


def test_roundtrip_checkpoint_to_onnx_parity(tmp_path):
    net = conv.build_net()                       # random init
    ckpt = tmp_path / "fake.pth"
    torch.save(net.state_dict(), ckpt)
    out = tmp_path / "siamfc.onnx"
    conv.convert(str(ckpt), str(out))            # loads strict=True, exports, parity-checks
    assert out.exists()

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    z = rng.standard_normal((1, 3, 127, 127)).astype(np.float32)
    x = rng.standard_normal((1, 3, 255, 255)).astype(np.float32)
    got = sess.run(["score_map"], {"exemplar": z, "search": x})[0]
    with torch.no_grad():
        ref = conv.build_net(torch.load(ckpt))(
            torch.from_numpy(z), torch.from_numpy(x)).numpy()
    assert got.shape == (1, 1, 17, 17)
    assert float(np.max(np.abs(ref - got))) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convert_siamfc.py -v`
Expected: FAIL with `FileNotFoundError`/load error for `tools/siamfc_to_onnx.py` (module does not exist yet). If torch is not installed, the test is SKIPPED — install with `pip install -e .[dev]` to exercise it.

- [ ] **Step 3: Write the converter**

Create `tools/siamfc_to_onnx.py`:

```python
"""Convert a huanglianghua/siamfc-pytorch AlexNetV1 checkpoint to ONNX.

Host-only tooling (ARCHITECTURE.md §11); NOT a runtime dependency. Requires the
[dev] extra (torch, onnx) and onnxruntime (in [test]) for the parity check.
Vendors a self-contained AlexNetV1 backbone + batch-1 cross-correlation head so
the original repo need not be installed.

Usage:
    python tools/siamfc_to_onnx.py \
        --checkpoint models/siamfc_alexnet_e50.pth \
        --out models/siamfc_generic.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Numerically identical to the repo's _fast_xcorr for batch size 1 (the export
    and the edgecv tracker both run one exemplar against one search per call)."""

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
        return self.head(self.backbone(z), self.backbone(x))


def build_net(state_dict: dict | None = None) -> Net:
    """Build the net; load a checkpoint state_dict strict=True when provided."""
    net = Net()
    if state_dict is not None:
        net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net


def export_onnx(net: Net, out_path: str, opset: int = 13) -> None:
    z = torch.zeros(1, 3, 127, 127)
    x = torch.zeros(1, 3, 255, 255)
    torch.onnx.export(
        net, (z, x), out_path,
        input_names=["exemplar", "search"], output_names=["score_map"],
        opset_version=opset, do_constant_folding=True)


def _parity_check(net: Net, out_path: str, tol: float = 1e-3) -> float:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    z = rng.standard_normal((1, 3, 127, 127)).astype(np.float32)
    x = rng.standard_normal((1, 3, 255, 255)).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.from_numpy(z), torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(["score_map"], {"exemplar": z, "search": x})[0]
    diff = float(np.max(np.abs(ref - got)))
    if diff > tol:
        raise SystemExit(f"parity check FAILED: max|delta|={diff:.2e} > {tol:.0e}")
    return diff


def convert(checkpoint: str, out: str) -> str:
    sd = torch.load(checkpoint, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:   # tolerate wrapped checkpoints
        sd = sd["state_dict"]
    net = build_net(sd)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    export_onnx(net, out)
    diff = _parity_check(net, out)
    print(f"exported {out}  (parity max|delta|={diff:.2e})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="SiamFC PyTorch -> ONNX")
    ap.add_argument("--checkpoint", required=True, help="path to the .pth state_dict")
    ap.add_argument("--out", default="models/siamfc_generic.onnx")
    args = ap.parse_args()
    convert(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_convert_siamfc.py -v`
Expected: PASS (2 passed) when torch is installed; SKIPPED otherwise.

> **Implementation note:** before converting the *real* checkpoint, confirm the vendored
> module's keys match it: `python -c "import torch; print(list(torch.load('models/siamfc_alexnet_e50.pth', map_location='cpu').keys())[:6])"`.
> Keys should be `backbone.conv1.0.weight`, `backbone.conv1.0.bias`, `backbone.conv1.1.weight`, …
> If the checkpoint is wrapped (`{"state_dict": ...}` or `{"model": ...}`), `convert` already
> unwraps a `state_dict` key; extend the unwrap if it uses a different key. A `strict=True`
> mismatch raises a clear `RuntimeError` listing missing/unexpected keys.

- [ ] **Step 5: Commit**

```bash
git add tools/siamfc_to_onnx.py tests/test_convert_siamfc.py
git commit -m "feat(tools): SiamFC PyTorch->ONNX converter with parity self-check

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: RKNN conversion scaffold (not run)

**Files:**
- Create: `tools/onnx_to_rknn.py`
- Create: `tests/test_onnx_to_rknn_scaffold.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_onnx_to_rknn_scaffold.py`:

```python
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "tools" / "onnx_to_rknn.py"
_spec = importlib.util.spec_from_file_location("onnx_to_rknn", _PATH)
rk = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rk)


def test_module_exposes_main_and_convert():
    # Scaffold imports without rknn-toolkit2 present (guarded import).
    assert hasattr(rk, "main")
    assert hasattr(rk, "convert")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_onnx_to_rknn_scaffold.py -v`
Expected: FAIL with `FileNotFoundError` (module does not exist yet).

- [ ] **Step 3: Write the scaffold**

Create `tools/onnx_to_rknn.py`:

```python
"""ONNX -> RKNN conversion (ARCHITECTURE.md §11). Host-only, x86. SCAFFOLD: the
rknn-toolkit2 import is deferred so this module imports anywhere; conversion only
runs where the toolkit is installed. Not exercised in CI.

rknn-toolkit2 is not on PyPI cleanly — install it on an x86 host from Rockchip's
release wheels. INT8 quantisation needs a folder of representative calibration
images (frames resembling deployment input).

Usage (on a host with rknn-toolkit2):
    python tools/onnx_to_rknn.py \
        --onnx models/siamfc_generic.onnx \
        --out models/siamfc_generic.rk3588.rknn \
        --target rk3588 \
        --calibration-dir calib/ \
        --inputs exemplar search
"""

from __future__ import annotations

import argparse

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


def convert(onnx_path: str, out_path: str, target: str,
            calibration_dir: str | None, input_names: list[str]) -> str:
    try:
        RKNN = _import_rknn()
    except Exception as e:  # pragma: no cover - depends on host toolkit
        raise RuntimeError(_INSTALL_HINT) from e

    quantize = calibration_dir is not None
    rknn = RKNN(verbose=True)
    # mean/std [0,255] passthrough: these weights consume raw pixels (scale handled
    # in the tracker preprocessing, not the model). Adjust if a future model normalises.
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


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX -> RKNN (host-only scaffold)")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--calibration-dir", default=None,
                    help="folder of representative images; enables INT8 quantisation")
    ap.add_argument("--inputs", nargs="+", default=["exemplar", "search"],
                    help="model input names (order must match the ONNX graph)")
    args = ap.parse_args()
    convert(args.onnx, args.out, args.target, args.calibration_dir, args.inputs)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_onnx_to_rknn_scaffold.py -v`
Expected: PASS (1 passed) — the module imports without `rknn-toolkit2` because the import is deferred inside `convert`.

- [ ] **Step 5: Commit**

```bash
git add tools/onnx_to_rknn.py tests/test_onnx_to_rknn_scaffold.py
git commit -m "feat(tools): ONNX->RKNN conversion scaffold (deferred toolkit import)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Conversion docs

**Files:**
- Create: `tools/CONVERSION.md`

- [ ] **Step 1: Write the doc**

Create `tools/CONVERSION.md`:

```markdown
# Model conversion (host-only)

These tools are **host-only** and not runtime dependencies (ARCHITECTURE.md §11).
Conversion runs offline on x86; the device only ever runs the lite runtime.

## Install

```bash
pip install -e .[dev]      # torch, onnx  (+ onnxruntime from [test] for parity checks)
```

`rknn-toolkit2` (for the RKNN step) is **not on PyPI** — install it on an x86 host
from Rockchip's release wheels.

## SiamFC: PyTorch → ONNX

Weights: `huanglianghua/siamfc-pytorch` AlexNetV1 (e.g. `siamfc_alexnet_e50.pth`).
Place the checkpoint under `models/` (gitignored) and run:

```bash
python tools/siamfc_to_onnx.py \
    --checkpoint models/siamfc_alexnet_e50.pth \
    --out models/siamfc_generic.onnx
```

The tool vendors a self-contained AlexNetV1 backbone + batch-1 cross-correlation
head, loads the checkpoint with `strict=True` (a key mismatch fails loudly), exports
the single two-input graph `(exemplar[1,3,127,127], search[1,3,255,255]) →
score_map[1,1,17,17]`, and runs a torch-vs-onnxruntime parity check (`max|Δ| < 1e-3`).

**Preprocessing contract (matches training):** RGB 3-channel, **raw `[0,255]`** pixels
(no `/255`, no mean/std), **BGR** channel order (cv2 convention). This is encoded in
`edgecv/models/manifests/siamfc_generic.yaml` (`color: rgb`, `scale: 1.0`) and consumed
by `SiamFC` via manifest-preprocessing precedence. The caller feeds BGR frames; e.g.
`tools/track_webcam.py` reads frames with cv2 (already BGR).

## ONNX → RKNN (scaffold)

Run on an x86 host with `rknn-toolkit2` installed. INT8 quantisation needs a folder
of representative calibration images:

```bash
python tools/onnx_to_rknn.py \
    --onnx models/siamfc_generic.onnx \
    --out models/siamfc_generic.rk3588.rknn \
    --target rk3588 \
    --calibration-dir calib/ \
    --inputs exemplar search
```

INT8 quantisation noise in NPU-derived score maps is largely self-correcting once the
tracker runs (PSR gate / appearance robustness); start from `siamfc_generic.onnx` for
dev/CI on x86, add the RKNN artifact for on-device deployment.

## Running the artifact / paths

Backends resolve a relative `artifacts.<backend>.path` against `$EDGECV_MODEL_DIR`
(default `models/`); absolute paths pass through. So from the repo root:

```bash
python tools/track_webcam.py --tracker siamfc   # finds models/siamfc_generic.onnx
```

(Model blobs — `.pth`, `.onnx`, `.rknn` — are gitignored and never committed.)
```

- [ ] **Step 2: Commit**

```bash
git add tools/CONVERSION.md
git commit -m "docs(tools): conversion guide (PyTorch->ONNX, ONNX->RKNN, paths)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Final verification

- [ ] **Step 1: Full suite**

Run: `pytest -q`
Expected: PASS (with torch installed, the converter test runs; without it, it skips).

- [ ] **Step 2: Lint + types**

Run: `ruff check . && mypy edgecv`
Expected: clean. (Match the repo's existing ruff/mypy config in `pyproject.toml`.)

- [ ] **Step 3: Manual end-to-end smoke test (documented, optional)**

With the real checkpoint placed in `models/`:

```bash
pip install -e .[dev]
python tools/siamfc_to_onnx.py --checkpoint models/siamfc_alexnet_e50.pth --out models/siamfc_generic.onnx
python tools/track_webcam.py --tracker siamfc
```

Expected: the tool exports with a passing parity check, and the webcam tracker locks
onto a selected target. (Real-weight tracking quality is verified by eye here; CI covers
export fidelity + the synthetic-onnx integration path.)

---

## Self-Review Notes (addressed)

- **Spec coverage:** §4.1 converter → Task 5; §4.2 RKNN scaffold → Task 6; §4.3 docs → Task 7;
  §4.4 manifest → Task 3; §4.5 precedence wiring → Tasks 2-3; §4.6 path resolution → Task 1;
  §5 tests → Tasks 1-3, 5-6 + Task 8 smoke test; §6 deps → Task 4. All covered.
- **Channel-order:** documented (manifest comment, CONVERSION.md), not enforced in code — matches
  spec §7 (out of scope to add a swap).
- **Type/name consistency:** `resolve_pp`, `manifest_preprocessing`, `UNSET`, `resolve_artifact_path`,
  `build_net`, `convert`, `export_onnx` used consistently across tasks and tests.
```

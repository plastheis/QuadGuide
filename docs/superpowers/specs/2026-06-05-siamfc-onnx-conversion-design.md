# SiamFC PyTorch → ONNX conversion + preprocessing correctness

> **Status:** approved design (2026-06-05). Scope: host-side tooling to convert a
> `huanglianghua/siamfc-pytorch` AlexNetV1 checkpoint to ONNX, plus the manifest /
> preprocessing / path-resolution fixes needed for the converted weights to produce
> *valid* score maps through the existing `SiamFC` tracker.

## 1. Context and problem

The repo ships a `SiamFC` NN tracker (`edgecv/trackers/nn/siamfc.py`) that drives a
**single two-input ONNX graph** `(exemplar 127, search 255) → score_map 17×17`, calling
it 3× per frame for multi-scale search (ARCHITECTURE.md §6.2). No real weights exist yet:
the manifest `edgecv/models/manifests/siamfc_generic.yaml` and the test fixtures were
written against synthetic, **grayscale 1-channel** placeholders.

The user has generic SiamFC weights in PyTorch format from
[`huanglianghua/siamfc-pytorch`](https://github.com/huanglianghua/siamfc-pytorch)
(confirmed). That implementation is:

- **AlexNetV1 backbone, RGB 3-channel** (not grayscale).
- Trained on **raw `[0,255]` pixels** — its `ToTensor` does *no* `/255`, no mean/std.
- Uses a fast cross-correlation head with `out_scale = 0.001`.
- Trained with OpenCV (`cv2`) crops, i.e. **BGR** channel order.
- Hyperparameters (`scale_step 1.0375`, `scale_penalty 0.9745`, `scale_lr 0.59`,
  `window_influence 0.176`, `response_up 16`, `scale_num 3`, `total_stride 8`,
  `exemplar 127`, `search 255`) — these already match the `SiamFC.__init__` defaults and
  the manifest, so they need no change.

Two things in the current code are therefore **wrong for these weights** and would make
the tracker feed garbage to the network:

1. The manifest declares `color: gray` and 1-channel I/O (`[1,1,127,127]`).
2. `to_input` defaults to `scale = 1/255`, and `SiamFC` defaults to `color="gray"`. The
   tracker also **ignores `manifest.preprocessing` entirely** today
   (the known precedence gap — see ARCHITECTURE §10.1 and the project memory note), so the
   manifest cannot currently correct these defaults.

A third latent issue: the onnx backend loads `artifacts.onnx.path` (a bare filename) via
`ort.InferenceSession(path)`, which only resolves if CWD happens to be `models/`.

## 2. Goal

`SiamFC(manifest=".../siamfc_generic.yaml", backend="onnx")` loads the converted weights
and produces **valid** score maps end-to-end. Achieving this requires the conversion tool
*and* the manifest/preprocessing/path corrections below.

## 3. Decisions (settled with the user)

- **Target format:** ONNX now; **scaffold** the RKNN path (write the script + docs, do not
  run it, do not require `rknn-toolkit2`).
- **Graph shape:** **single two-input graph** `forward(exemplar, search) → score_map`,
  re-embedding the exemplar each call. Matches the current tracker exactly; no structural
  tracker change. (Wasteful re-embed of the 127 exemplar 3×/frame is accepted; split
  embed/correlate is explicitly deferred.)
- **Manifest-precedence wiring:** in scope. Closing the `color`/`scale` half of the known
  precedence gap is what makes the converted weights correct end-to-end.
- **Artifact placement:** repo-root `models/` (already gitignored, alongside the `.pth`).
  No binaries committed.

## 4. Components

### 4.1 `tools/siamfc_to_onnx.py` (PyTorch → ONNX)

CLI:

```
python tools/siamfc_to_onnx.py \
    --checkpoint models/siamfc_alexnet_e50.pth \
    --out models/siamfc_generic.onnx
```

- **Vendors a self-contained AlexNetV1 backbone + xcorr head** (~40 lines), so the tool
  does not depend on the original repo being installed. The module definition must match
  the checkpoint's layer naming exactly (`backbone.conv1.0.weight`, grouped convs in
  conv2/conv4/conv5, BatchNorm, `out_scale=0.001` head). State dict is loaded with
  **`strict=True`** so a key mismatch fails loudly rather than silently dropping layers.
  *(Implementation note: confirm exact keys by printing `state_dict().keys()` against the
  real checkpoint before finalising the vendored module.)*
- Exports the single two-input graph with **static** shapes:
  `exemplar [1,3,127,127]`, `search [1,3,255,255]` → `score_map [1,1,17,17]`,
  input names `exemplar`/`search`, output name `score_map` (matching the manifest io spec).
- **Parity self-check:** after export, run torch eval vs `onnxruntime` on a random input;
  assert `max|Δ| < 1e-3`. Verifies export fidelity independent of preprocessing semantics.
- Writes to `--out` (default under `models/`) and prints the resolved path.

### 4.2 `tools/onnx_to_rknn.py` (RKNN scaffold — not run)

- Generic ONNX → RKNN conversion via `rknn-toolkit2`, parameterised by `--onnx`,
  `--out`, `--target rk3588`, `--calibration-dir` (folder of representative images for
  INT8). Import of `rknn` is guarded with a clear "install rknn-toolkit2 on an x86 host"
  error. Written and documented; **not executed** in this work.

### 4.3 `tools/CONVERSION.md`

Documents: required deps (`pip install -e .[dev]` for torch/onnx; manual `rknn-toolkit2`
for the RKNN step), exact commands, the **BGR + `[0,255]`** input expectation for these
weights, calibration-data guidance for INT8, and how to smoke-test via `track_webcam.py`.

### 4.4 Manifest correction (`edgecv/models/manifests/siamfc_generic.yaml`)

- I/O → **3-channel**: `exemplar [1,3,127,127]`, `search [1,3,255,255]`
  (`score_map [1,1,17,17]` unchanged).
- `preprocessing.color: rgb` (3-channel passthrough; no graying).
- `preprocessing.scale: 1.0` (raw `[0,255]`; the repo applies no `/255`, no mean/std).
- Comment documenting the **BGR / cv2** channel-order expectation.

### 4.5 Preprocessing precedence wiring

- Add a small precedence helper `resolve_pp(kwarg, manifest_value, default)` (or
  equivalent), with a sentinel so an explicitly passed `__init__` kwarg still wins:
  **explicit kwarg > `manifest.preprocessing[key]` > hardcoded default** (spec intent from
  the nn-trackers design §7).
- `SiamFC` resolves `color` and `scale` through it. `scale` is threaded into the
  `to_input(...)` calls in `init` and `update` (today they use the `1/255` default).
- Scope limited to `color` and `scale` (the only params that differ from defaults for
  these weights). Other hyperparameters already coincide with the manifest and are left
  as-is. The `manifest` (or its `preprocessing` dict) must reach `SiamFC.__init__`; when a
  bare `model=` is injected with no manifest (tests), resolution falls back to kwargs and
  defaults.

### 4.6 Artifact path resolution

- Add a shared `resolve_artifact_path(path) -> str`, used by both onnx and rknn backends:
  absolute paths pass through; relative paths resolve against **`$EDGECV_MODEL_DIR`**
  (default `./models`). Backends call it before opening the artifact.
- The converter writes into the same dir and prints the resolved location. Tests that
  inject `model=` bypass this path entirely and are unaffected.

## 5. Test plan

- **Fixture correction:** `tests/_nn_stubs.py` `siam_io` → 3-channel `TensorSpec`s;
  `tests/test_siamfc.py` exemplar-shape assertion → `(1,3,127,127)`.
- **Converter test** `tests/test_convert_siamfc.py`, guarded by
  `pytest.importorskip("torch")`: build a tiny random-weight AlexNetV1+head, save a fake
  `.pth`, run the converter, assert the ONNX loads under `onnxruntime` and parity holds.
  Skips cleanly where torch is absent (host-only tooling, ARCHITECTURE §11).
- **Precedence unit test:** kwarg > manifest > default for `color`/`scale`.
- **Path-resolution unit test:** absolute pass-through; relative resolves against
  `$EDGECV_MODEL_DIR`.
- **Manual end-to-end smoke test:** documented, via `tools/track_webcam.py` against the
  converted `models/siamfc_generic.onnx` (real-weight tracking can't be asserted in CI
  without a labelled clip; parity + the synthetic-onnx integration tests cover the rest).

## 6. Dependencies

- Populate the empty `[dev]` extra in `pyproject.toml`: `torch`, `onnx` (export + checker).
  `onnxruntime` is already pulled by `[test]`.
- RKNN's `rknn-toolkit2` stays a **documented manual host install** (not on PyPI cleanly),
  consistent with the existing `[rknn]` extra convention.

## 7. Out of scope (YAGNI)

- Split embed/correlate two-graph design (single-graph chosen).
- Actually running ONNX→RKNN or requiring `rknn-toolkit2` now.
- Wiring *every* preprocessing param through precedence — only `color`/`scale` differ for
  these weights; the rest already match.
- Adding an RGB↔BGR swap in `to_input` (documented expectation instead; caller feeds BGR).
- Committing any model binaries (`.pth`/`.onnx`/`.rknn` are gitignored, host-only).

## 8. Risks / notes

- **Dynamic-weight Conv export.** The xcorr head is `F.conv2d(search_feat, exemplar_feat)`
  with the kernel coming from an activation, not a parameter. ONNX `Conv` permits a
  non-constant `W` input and `onnxruntime` supports it; the parity self-check is the guard
  if a given opset misbehaves (bump opset if needed).
- **Channel order.** Faithful reproduction needs BGR `[0,255]`. `to_input` preserves caller
  channel order, so correctness depends on the caller (e.g. `track_webcam` via cv2 already
  yields BGR). Documented, not enforced in code.
- **Checkpoint key names.** The vendored module must match the real `state_dict` keys for
  `strict=True` load; verify against the actual file during implementation.

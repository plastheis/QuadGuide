"""Generic ONNX export + validation (ARCHITECTURE.md §11). Model-independent:
all input/output names and shapes come from the caller (the dispatcher reads them
from the manifest). Validates single-output graphs by torch-vs-onnxruntime parity."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


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
    # dynamo=False forces the legacy TorchScript exporter: it emits a standard Conv with
    # the exemplar embedding as a graph input (what the runtime + parity check expect) and
    # needs no onnxscript. The torch>=2.9 default (dynamo=True) traces via torch.export,
    # which decomposes the dynamic-weight xcorr differently and pulls onnxscript.
    torch.onnx.export(
        module, tuple(example_inputs), out_path,
        input_names=list(in_names), output_names=list(out_names),
        opset_version=opset, dynamic_axes=dynamic_axes, dynamo=False)
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

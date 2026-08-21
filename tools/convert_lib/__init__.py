"""Host-only model conversion framework (ARCHITECTURE.md §11). See tools/CONVERSION.md.

convert_lib is imported by tools/convert.py (which puts tools/ on sys.path). The registry
is torch-free; the harness, adapters, and rknn helpers import torch / rknn-toolkit2 lazily,
so importing the registry never pulls heavy deps.
"""

from __future__ import annotations

from pathlib import Path

from . import registry
from .registry import Adapter, get, register, registered_names

__all__ = ["Adapter", "get", "register", "registered_names", "run"]

# Host-only tooling: assumes tools/convert_lib/ lives two levels below the repo root.
_MANIFESTS = Path(__file__).resolve().parents[2] / "edgecv" / "models" / "manifests"
_NOMINAL_DIM = 1


def _concrete(shape) -> tuple[int, ...]:
    """Replace dynamic (-1) dims with a nominal size for the export example input."""
    return tuple(d if isinstance(d, int) and d > 0 else _NOMINAL_DIM for d in shape)


def _artifact_path(mf, model: str, backend: str) -> str:
    """The manifest's relative artifact path for `backend`, or a clean operator error."""
    artifact = mf.artifacts.get(backend)
    if not artifact or "path" not in artifact:
        raise SystemExit(f"manifest {model!r} has no {backend!r} artifact path")
    return artifact["path"]


def run(model: str, checkpoint: str, out: str | None = None, *,
        rknn: bool = False, target: str = "rk3588", calib: str | None = None) -> str:
    """Convert `checkpoint` for `model` to ONNX (and optionally RKNN), driven by the
    model's manifest. Writes ONNX to `out` or to the manifest's resolved artifact path.
    Two paths: a torch adapter (build + parity harness) or an upstream-exporter adapter
    (adapter.export writes the ONNX directly; we only run onnx.checker)."""
    from edgecv.models.manifest import load_manifest
    from edgecv.models.paths import resolve_artifact_path

    from . import adapters  # noqa: F401  (registers adapters)

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

    in_names = [i["name"] for i in mf.inputs]
    out_names = [o["name"] for o in mf.outputs]
    onnx_out = out or resolve_artifact_path(_artifact_path(mf, model, "onnx"))

    if adapter.export is not None:
        adapter.export(checkpoint, onnx_out, mf)
        import onnx
        onnx.checker.check_model(onnx_out)
        print(f"exported {onnx_out}  (via upstream exporter)")
    else:
        import torch

        from .harness import export_and_validate
        try:
            module = adapter.build(checkpoint)
        except (RuntimeError, OSError) as e:   # strict-load mismatch, missing/corrupt file
            raise SystemExit(f"failed to load checkpoint for {model!r}: {e}") from e
        example = tuple(torch.zeros(_concrete(i["shape"])) for i in mf.inputs)
        diff = export_and_validate(module, example, in_names, out_names, onnx_out,
                                   dynamic_axes=adapter.dynamic_axes)
        print(f"exported {onnx_out}  (parity max|delta|={diff:.2e})")

    if rknn:
        from .rknn import rknn_convert
        rk_out = resolve_artifact_path(_artifact_path(mf, model, "rknn"))
        rknn_convert(onnx_out, rk_out, target, calib, in_names)
    return onnx_out

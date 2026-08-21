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
    # Don't pass `inputs` here: rknn-toolkit2 (>=2.x) requires `input_size_list`
    # alongside `inputs` and otherwise raises "If 'inputs' set, the
    # 'input_size_list' should be set also!". `inputs` is only needed to crop the
    # graph, which this generic converter never does — so let load_onnx read the
    # input names/shapes from the ONNX itself. `input_names` is still used above to
    # size the per-input mean/std lists.
    if rknn.load_onnx(model=onnx_path) != 0:
        raise RuntimeError(f"load_onnx failed for {onnx_path!r}")
    dataset = _write_dataset_file(calibration_dir) if quantize else None
    if rknn.build(do_quantization=quantize, dataset=dataset) != 0:
        raise RuntimeError("rknn build failed")
    if rknn.export_rknn(out_path) != 0:
        raise RuntimeError(f"export_rknn failed for {out_path!r}")
    rknn.release()
    print(f"exported {out_path} (target={target}, quantized={quantize})")
    return out_path

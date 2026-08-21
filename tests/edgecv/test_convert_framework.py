from pathlib import Path

import convert_lib
import pytest
from convert_lib import registry, run
from convert_lib.registry import Adapter

_MANIFEST = (
    "name: fakeyolo\ntask: detection\n"
    "io:\n"
    "  inputs: [{name: images, shape: [1, 3, 8, 8], dtype: float32}]\n"
    "  outputs: [{name: output0, shape: [1, 84, -1], dtype: float32}]\n"
    "artifacts:\n"
    "  onnx: {path: fakeyolo.onnx}\n"
    "  rknn: {path: fakeyolo.rk3588.rknn}\n"
)


def _setup(tmp_path, monkeypatch):
    pytest.importorskip("torch")   # run() imports convert_lib.adapters -> siamfc (torch)
    onnx = pytest.importorskip("onnx")
    (tmp_path / "fakeyolo.yaml").write_text(_MANIFEST)
    monkeypatch.setattr(convert_lib, "_MANIFESTS", tmp_path)
    monkeypatch.setenv("EDGECV_MODEL_DIR", str(tmp_path))   # artifact paths resolve here
    monkeypatch.setattr(onnx.checker, "check_model", lambda p: None)


def test_export_hook_runs_and_skips_torch_harness(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    seen = {}

    def fake_export(checkpoint, onnx_out, manifest):
        seen["args"] = (checkpoint, onnx_out)
        seen["imgsz"] = manifest.inputs[0]["shape"][-1]
        Path(onnx_out).write_bytes(b"onnx")
        return onnx_out

    registry.register(Adapter(name="fakeyolo", export=fake_export))
    out = run("fakeyolo", "ckpt.pt")
    assert seen["args"][0] == "ckpt.pt"
    assert seen["imgsz"] == 8
    assert Path(out).exists() and out.endswith("fakeyolo.onnx")


def test_export_hook_chains_rknn(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    import convert_lib.rknn as rknnmod

    rk = {}
    monkeypatch.setattr(rknnmod, "rknn_convert",
                        lambda onnx_p, out_p, target, calib, names: rk.update(
                            out=out_p, names=names) or out_p)
    registry.register(Adapter(
        name="fakeyolo",
        export=lambda c, o, m: Path(o).write_bytes(b"onnx") or o))
    run("fakeyolo", "ckpt.pt", rknn=True)
    assert rk["names"] == ["images"]
    assert rk["out"].endswith("fakeyolo.rk3588.rknn")

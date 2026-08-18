import sys
import types

import pytest
from convert_lib import registry


def test_yolo_adapters_registered_with_export_hook():
    pytest.importorskip("torch")   # importing convert_lib.adapters pulls siamfc (torch)
    import convert_lib.adapters  # noqa: F401  (registers all adapters)
    n, s = registry.get("yolo26n"), registry.get("yolo26s")
    assert n.export is not None and n.build is None
    assert s.export is not None and s.build is None


def test_export_invokes_ultralytics_one_to_many(tmp_path, monkeypatch):
    pytest.importorskip("torch")   # convert_lib.adapters.yolo pulls the package (siamfc/torch)
    from edgecv.models.manifest import load_manifest

    produced = tmp_path / "src" / "yolo26n.onnx"
    produced.parent.mkdir()
    produced.write_bytes(b"onnx")
    captured = {}

    class FakeYOLO:
        def __init__(self, ckpt):
            captured["ckpt"] = ckpt

        def export(self, **kw):
            captured["kw"] = kw
            return str(produced)

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    from convert_lib.adapters.yolo import _export
    mf = load_manifest("src/edgecv/models/manifests/yolo26n.yaml")
    dest = tmp_path / "dest" / "yolo26n.onnx"
    out = _export("models/yolo26n.pt", str(dest), mf)

    assert captured["ckpt"] == "models/yolo26n.pt"
    assert captured["kw"]["nms"] is False          # one-to-many head (design §3)
    assert captured["kw"]["imgsz"] == 640
    assert out == str(dest)
    assert dest.exists()                            # moved into place
    assert not produced.exists()                    # source consumed

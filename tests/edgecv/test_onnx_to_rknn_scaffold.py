import importlib.util
from pathlib import Path

from convert_lib.rknn import rknn_convert


def test_rknn_convert_importable_without_toolkit():
    # the rknn-toolkit2 import is deferred, so importing the function never needs it
    assert callable(rknn_convert)


def test_shim_exposes_main():
    path = Path(__file__).resolve().parents[2] / "tools" / "onnx_to_rknn.py"
    spec = importlib.util.spec_from_file_location("onnx_to_rknn", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")

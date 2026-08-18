import importlib.util

import numpy as np
import pytest

from edgecv.backends.rknn import RknnBackend

ort = pytest.importorskip("onnxruntime")

# The two "without_runtime" tests below assert the off-device behaviour (x86/CI
# with no rknn-toolkit-lite2). On a real Rockchip device the runtime IS present,
# so those assertions are inverted — skip them there rather than report a
# spurious failure.
_rknn_runtime_present = importlib.util.find_spec("rknnlite") is not None
_skip_on_device = pytest.mark.skipif(
    _rknn_runtime_present,
    reason="rknn-toolkit-lite2 is installed (on-device); these assert the off-device path",
)


def _make_identity_onnx(path):
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    import onnx
    onnx.save(model, str(path))


def test_onnx_backend_loads_and_infers(tmp_path):
    pytest.importorskip("onnx")
    from edgecv.backends.onnx import OnnxBackend
    from edgecv.models.manifest import ModelManifest

    onnx_path = tmp_path / "id.onnx"
    _make_identity_onnx(onnx_path)
    man = ModelManifest(
        name="id", task="test", preprocessing={},
        inputs=[{"name": "x", "shape": [1, 3], "dtype": "float32"}],
        outputs=[{"name": "y", "shape": [1, 3], "dtype": "float32"}],
        artifacts={"onnx": {"path": str(onnx_path)}},
    )
    model = OnnxBackend().load(man)
    out = model.infer({"x": np.array([[1.0, 2.0, 3.0]], np.float32)})
    np.testing.assert_allclose(out["y"], [[1.0, 2.0, 3.0]])
    assert model.io_spec.inputs[0].name == "x"
    model.close()


@_skip_on_device
def test_rknn_backend_reports_unavailable_without_runtime():
    # On x86/CI there is no rknnlite; the adapter must say so, not crash.
    assert RknnBackend().is_available() is False


@_skip_on_device
def test_rknn_load_raises_clear_error_without_runtime():
    from edgecv.models.manifest import ModelManifest

    man = ModelManifest(name="m", task="t", preprocessing={},
                        inputs=[], outputs=[], artifacts={"rknn": {"path": "x.rknn"}})
    with pytest.raises(RuntimeError, match="rknn-toolkit-lite2"):
        RknnBackend().load(man)

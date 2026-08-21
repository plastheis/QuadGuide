import numpy as np

from edgecv.backends.mock import MockBackend
from edgecv.models.manifest import ModelManifest


def _manifest():
    return ModelManifest(
        name="t",
        task="sot_template_matching",
        preprocessing={},
        inputs=[{"name": "x", "shape": [1, 3, 8, 8], "dtype": "float32"}],
        outputs=[{"name": "y", "shape": [1, 1, 4, 4], "dtype": "float32"}],
        artifacts={"mock": {}},
    )


def test_mock_backend_is_always_available():
    assert MockBackend().is_available() is True


def test_mock_model_io_spec_from_manifest():
    model = MockBackend().load(_manifest())
    spec = model.io_spec
    assert spec.inputs[0].name == "x"
    assert spec.outputs[0].shape == (1, 1, 4, 4)


def test_mock_infer_returns_zeros_of_declared_shape():
    model = MockBackend().load(_manifest())
    out = model.infer({"x": np.zeros((1, 3, 8, 8), np.float32)})
    assert set(out) == {"y"}
    assert out["y"].shape == (1, 1, 4, 4)
    assert out["y"].dtype == np.float32


def test_mock_async_handle_matches_sync():
    model = MockBackend().load(_manifest())
    out = model.infer_async({"x": np.zeros((1, 3, 8, 8), np.float32)}).wait()
    assert out["y"].shape == (1, 1, 4, 4)

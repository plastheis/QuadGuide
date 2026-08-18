import numpy as np
import pytest

from edgecv.backends.base import IOSpec, TensorSpec
from edgecv.trackers.nn.base import NNTracker, Template
from tests.edgecv._nn_stubs import ScriptedModel


def _stub():
    io = IOSpec(outputs=(TensorSpec("y", (1, 1), "float32"),))
    return ScriptedModel(io, [{"y": np.zeros((1, 1), np.float32)}])


def test_template_dataclass_holds_arrays():
    from edgecv.core.bbox import BoundingBox
    t = Template(arrays={"exemplar": np.zeros((1,))}, bbox=BoundingBox(0, 0, 0.1, 0.1), meta={})
    assert "exemplar" in t.arrays


def test_model_injection_bypasses_backend():
    m = _stub()
    trk = NNTracker(model=m)        # no manifest, no backend
    assert trk._model is m


def test_close_is_idempotent_and_closes_model():
    m = _stub()
    trk = NNTracker(model=m)
    trk.close()
    trk.close()
    assert getattr(m, "closed", False) is True


def test_mock_backend_resolves_via_manifest():
    trk = NNTracker("src/edgecv/models/manifests/yolo26n.yaml", backend="mock")
    assert trk._model is not None
    trk.close()


def test_auto_with_no_real_backend_raises_never_mock(monkeypatch):
    import edgecv.trackers.nn.base as base
    monkeypatch.setattr(base, "available_backends", lambda: ["mock"])
    with pytest.raises(RuntimeError, match="no inference backend"):
        NNTracker("src/edgecv/models/manifests/yolo26n.yaml", backend="auto")


def test_rknn_unavailable_off_device_is_clean():
    from edgecv.backends.registry import get_backend
    be = get_backend("rknn")
    if be.is_available():            # on a real device
        pytest.skip("rknn runtime present; load tested on-device manually")
    assert be.is_available() is False

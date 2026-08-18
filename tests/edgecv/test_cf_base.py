import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.trackers.cf.base import CorrelationFilterTracker, EvalResult, FilterState


def test_filter_state_and_eval_result_construct():
    fs = FilterState(arrays={"H": np.zeros((3, 3), np.complex64)},
                     bbox=BoundingBox(0, 0, 0.1, 0.1), meta={"feature": "raw"})
    assert "H" in fs.arrays
    er = EvalResult(bbox=fs.bbox, response_map=np.zeros((3, 3)), psr=5.0)
    assert er.psr == 5.0


def test_cf_tracker_is_abstract():
    with pytest.raises(TypeError):
        CorrelationFilterTracker()


def test_contract_requires_pure_ops_and_state_access():
    # A subclass missing build_filter must not be instantiable.
    class Incomplete(CorrelationFilterTracker):
        def init(self, frame, bbox): ...
        def update(self, frame): ...
        @property
        def status(self): ...
        def name(self): return "x"
        # missing build_filter/evaluate/get_filter/set_filter/response_map/psr
    with pytest.raises(TypeError):
        Incomplete()


def test_fully_implemented_subclass_instantiates():
    class Ok(CorrelationFilterTracker):
        def init(self, frame, bbox): self._fs = FilterState({}, bbox, {})
        def update(self, frame): ...
        @property
        def status(self): return None
        def name(self): return "Ok"
        def build_filter(self, frame, bbox): return FilterState({}, bbox, {})
        def evaluate(self, frame, state):
            return EvalResult(state.bbox, np.zeros((2, 2)), 1.0)
        def get_filter(self): return self._fs
        def set_filter(self, state, search_box=None): self._fs = state
        @property
        def response_map(self): return np.zeros((2, 2))
        @property
        def psr(self): return 1.0
    t = Ok()
    t.init(np.zeros((4, 4), np.uint8), BoundingBox(0, 0, 0.5, 0.5))
    assert t.name() == "Ok"
    assert t.get_filter().bbox.w == 0.5

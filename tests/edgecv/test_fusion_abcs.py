import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.fusion.policy import DetectorOutput, FusionDecision, FusionPolicy
from edgecv.fusion.predict import MotionPredictor
from edgecv.trackers.cf.base import EvalResult


def test_fusion_policy_is_abstract():
    with pytest.raises(TypeError):
        FusionPolicy()


def test_motion_predictor_is_abstract():
    with pytest.raises(TypeError):
        MotionPredictor()


def test_detector_output_and_decision_construct():
    do = DetectorOutput(boxes=np.zeros((1, 4), np.float32),
                        scores=np.array([0.5], np.float32))
    assert do.boxes.shape == (1, 4)
    dec = FusionDecision(take_candidate=True)
    assert dec.take_candidate is True


def test_concrete_policy_can_be_implemented():
    class KeepIncumbent(FusionPolicy):
        def fuse(self, incumbent, candidate, detector_out):
            return FusionDecision(take_candidate=False)

    er = EvalResult(bbox=BoundingBox(0, 0, 0.1, 0.1),
                    response_map=np.zeros((2, 2)), psr=3.0)
    assert KeepIncumbent().fuse(er, None, None).take_candidate is False


def test_concrete_predictor_can_be_implemented():
    class Hold(MotionPredictor):
        def predict(self, history, dt):
            return history[-1][1]

    bb = BoundingBox(0.2, 0.2, 0.1, 0.1)
    assert Hold().predict([(0.0, bb)], dt=0.03) is bb

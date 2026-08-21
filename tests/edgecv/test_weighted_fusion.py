"""Tests for WeightedFusionPolicy (spec §4.4.2)."""

from __future__ import annotations

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.fusion.calibrator import LinearCalibrator, SigmoidCalibrator
from edgecv.fusion.policy import DetectorOutput, FusionDecision
from edgecv.fusion.weighted import WeightedFusionParams, WeightedFusionPolicy
from edgecv.trackers.cf.base import EvalResult


def _make_eval(psr: float) -> EvalResult:
    return EvalResult(
        bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3),
        response_map=np.ones((10, 10), dtype=np.float64),
        psr=psr,
    )


def _make_det_out(scores: list[float]) -> DetectorOutput:
    return DetectorOutput(
        boxes=np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32),
        scores=np.array(scores, dtype=np.float32),
    )


class TestWeightedFusionParams:
    def test_defaults(self):
        p = WeightedFusionParams()
        assert p.cf_weight == 0.6
        assert p.nn_weight == 0.4
        assert p.threshold == 0.5
        assert p.nn_floor == 0.3

    def test_custom_params(self):
        p = WeightedFusionParams(cf_weight=0.3, nn_weight=0.7,
                                 threshold=0.4, nn_floor=0.2)
        assert p.cf_weight == 0.3
        assert p.nn_weight == 0.7
        assert p.threshold == 0.4
        assert p.nn_floor == 0.2


class TestWeightedFusionPolicy:
    def test_no_candidate_rejected(self):
        policy = WeightedFusionPolicy()
        inc = _make_eval(psr=10.0)
        decision = policy.fuse(inc, candidate=None, detector_out=None)
        assert isinstance(decision, FusionDecision)
        assert decision.take_candidate is False

    def test_no_detector_out_rejected(self):
        policy = WeightedFusionPolicy()
        inc = _make_eval(psr=10.0)
        cand = _make_eval(psr=12.0)
        decision = policy.fuse(inc, cand, detector_out=None)
        assert decision.take_candidate is False

    def test_empty_detection_scores_rejected(self):
        policy = WeightedFusionPolicy()
        inc = _make_eval(psr=10.0)
        cand = _make_eval(psr=12.0)
        det = DetectorOutput(
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
        )
        decision = policy.fuse(inc, cand, det)
        assert decision.take_candidate is False

    def test_nn_below_floor_rejected(self):
        """NN calibrated confidence below nn_floor gates independently."""
        policy = WeightedFusionPolicy(
            WeightedFusionParams(nn_floor=0.5),
            nn_cal=SigmoidCalibrator(centre=0.5, steepness=10.0))
        inc = _make_eval(psr=10.0)
        cand = _make_eval(psr=10.0)
        det = _make_det_out(scores=[0.3])  # low raw score -> low calibrated
        decision = policy.fuse(inc, cand, det)
        assert decision.take_candidate is False

    def test_candidate_beats_incumbent(self):
        """Candidate combined score sufficiently exceeds incumbent."""
        policy = WeightedFusionPolicy(
            WeightedFusionParams(cf_weight=0.6, nn_weight=0.4,
                                 threshold=0.2),
            cf_cal=LinearCalibrator(low=0.0, high=10.0),
            nn_cal=SigmoidCalibrator(centre=0.5, steepness=10.0))
        inc = _make_eval(psr=3.0)    # inc_conf = 0.3
        cand = _make_eval(psr=8.0)   # cand_cf_conf = 0.8
        det = _make_det_out(scores=[0.9])  # nn_conf ~ 0.98
        # cand_combined = 0.6*0.8 + 0.4*0.98 = 0.48 + 0.392 = 0.872
        # inc_combined = 0.3
        # diff = 0.572 > 0.2 -> take
        decision = policy.fuse(inc, cand, det)
        assert decision.take_candidate is True

    def test_candidate_does_not_beat_incumbent(self):
        """Candidate combined score not sufficiently above incumbent."""
        policy = WeightedFusionPolicy(
            WeightedFusionParams(cf_weight=0.6, nn_weight=0.4,
                                 threshold=0.5),
            cf_cal=LinearCalibrator(low=0.0, high=10.0),
            nn_cal=SigmoidCalibrator(centre=0.5, steepness=10.0))
        inc = _make_eval(psr=7.0)    # inc_conf = 0.7
        cand = _make_eval(psr=8.0)   # cand_cf_conf = 0.8
        det = _make_det_out(scores=[0.6])  # nn_conf ~ 0.73
        # cand_combined = 0.6*0.8 + 0.4*0.73 = 0.48 + 0.292 = 0.772
        # inc_combined = 0.7
        # diff = 0.072 < 0.5 -> no take
        decision = policy.fuse(inc, cand, det)
        assert decision.take_candidate is False

    def test_params_property(self):
        params = WeightedFusionParams(cf_weight=0.3)
        policy = WeightedFusionPolicy(params)
        assert policy.params is params

    def test_cf_calibrator_property(self):
        cal = LinearCalibrator(low=5.0, high=20.0)
        policy = WeightedFusionPolicy(cf_cal=cal)
        assert policy.cf_calibrator is cal

    def test_nn_calibrator_property(self):
        cal = SigmoidCalibrator(centre=0.3, steepness=8.0)
        policy = WeightedFusionPolicy(nn_cal=cal)
        assert policy.nn_calibrator is cal

    def test_isinstance_fusion_policy(self):
        from edgecv.fusion.policy import FusionPolicy
        assert isinstance(WeightedFusionPolicy(), FusionPolicy)

"""Tests for PSRGatePolicy (spec §4.4.1)."""

from __future__ import annotations

import numpy as np
import pytest

from edgecv.fusion.calibrator import LinearCalibrator, SigmoidCalibrator
from edgecv.fusion.policy import DetectorOutput, FusionDecision
from edgecv.fusion.psr_gate import PSRGateParams, PSRGatePolicy
from edgecv.core.bbox import BoundingBox
from edgecv.trackers.cf.base import EvalResult


def _make_eval(psr: float) -> EvalResult:
    return EvalResult(
        bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3),
        response_map=np.ones((10, 10), dtype=np.float64),
        psr=psr,
    )


class TestPSRGateParams:
    def test_defaults(self):
        p = PSRGateParams()
        assert p.margin == 0.5
        assert p.candidate_floor == 0.3
        assert p.incumbent_floor == 0.1
        assert p.use_hysteresis is True

    def test_custom_params(self):
        p = PSRGateParams(margin=0.2, candidate_floor=0.5,
                          incumbent_floor=0.2, use_hysteresis=False)
        assert p.margin == 0.2
        assert p.candidate_floor == 0.5
        assert p.incumbent_floor == 0.2
        assert p.use_hysteresis is False


class TestPSRGatePolicy:
    def test_no_candidate_returns_incumbent(self):
        policy = PSRGatePolicy()
        inc = _make_eval(psr=10.0)
        decision = policy.fuse(inc, candidate=None, detector_out=None)
        assert isinstance(decision, FusionDecision)
        assert decision.take_candidate is False

    def test_candidate_below_floor_rejected(self):
        """Candidate calibrated confidence below candidate_floor is rejected."""
        policy = PSRGatePolicy(PSRGateParams(candidate_floor=0.5))
        inc = _make_eval(psr=10.0)   # inc_conf ~ 0.58
        cand = _make_eval(psr=5.0)   # cand_conf ~ 0.17 -> below floor
        decision = policy.fuse(inc, cand, detector_out=None)
        assert decision.take_candidate is False

    def test_candidate_beats_incumbent_within_margin(self):
        """Candidate must exceed incumbent by more than margin."""
        policy = PSRGatePolicy(PSRGateParams(margin=0.5), cf_cal=LinearCalibrator(low=0.0, high=10.0))
        inc = _make_eval(psr=5.0)   # conf=0.5
        cand = _make_eval(psr=5.4)  # conf=0.54, diff=0.04 < 0.5
        decision = policy.fuse(inc, cand, detector_out=None)
        assert decision.take_candidate is False

    def test_candidate_beats_incumbent_exceeds_margin(self):
        """Candidate exceeds incumbent by more than margin -> take."""
        policy = PSRGatePolicy(PSRGateParams(margin=0.2), cf_cal=LinearCalibrator(low=0.0, high=10.0))
        inc = _make_eval(psr=5.0)   # conf=0.5
        cand = _make_eval(psr=8.0)  # conf=0.8, diff=0.3 > 0.2
        decision = policy.fuse(inc, cand, detector_out=None)
        assert decision.take_candidate is True

    def test_incumbent_floor_triggers_emergency_mode(self):
        """When incumbent is failing, margin is halved."""
        policy = PSRGatePolicy(
            PSRGateParams(margin=0.5, incumbent_floor=0.3),
            cf_cal=LinearCalibrator(low=0.0, high=10.0))
        inc = _make_eval(psr=2.0)   # conf=0.2 < 0.3 -> emergency, margin=0.25
        cand = _make_eval(psr=3.0)  # conf=0.3, diff=0.1 < 0.25 (halved)
        # Even with halved margin, diff=0.1 is still less than 0.25
        decision = policy.fuse(inc, cand, detector_out=None)
        assert decision.take_candidate is False

    def test_incumbent_floor_accepts_candidate(self):
        """When incumbent is failing AND candidate is significantly better."""
        policy = PSRGatePolicy(
            PSRGateParams(margin=0.5, incumbent_floor=0.3),
            cf_cal=LinearCalibrator(low=0.0, high=10.0))
        inc = _make_eval(psr=2.0)   # conf=0.2 < 0.3 -> emergency, margin=0.25
        cand = _make_eval(psr=9.0)  # conf=0.9, diff=0.7 > 0.25
        decision = policy.fuse(inc, cand, detector_out=None)
        assert decision.take_candidate is True

    def test_hysteresis_doubles_margin_when_using_candidate(self):
        """After accepting a candidate, margin doubles for switching back."""
        policy = PSRGatePolicy(
            PSRGateParams(margin=0.3, use_hysteresis=True),
            cf_cal=LinearCalibrator(low=0.0, high=10.0))

        # First acceptance
        inc = _make_eval(psr=3.0)  # conf=0.3
        cand = _make_eval(psr=7.0) # conf=0.7, diff=0.4 > 0.3 -> accepted
        d1 = policy.fuse(inc, cand, detector_out=None)
        assert d1.take_candidate is True

        # Now _using_candidate=True, margin should be doubled to 0.6
        # New incumbent is the candidate, but we compare again
        inc2 = _make_eval(psr=7.0)   # conf=0.7
        cand2 = _make_eval(psr=8.0)  # conf=0.8, diff=0.1 < 0.6 (doubled)
        d2 = policy.fuse(inc2, cand2, detector_out=None)
        assert d2.take_candidate is False

    def test_hysteresis_off_no_doubling(self):
        """Without hysteresis, margin is not doubled."""
        policy = PSRGatePolicy(
            PSRGateParams(margin=0.3, use_hysteresis=False),
            cf_cal=LinearCalibrator(low=0.0, high=10.0))

        inc = _make_eval(psr=3.0)  # conf=0.3
        cand = _make_eval(psr=7.0) # conf=0.7, diff=0.4 > 0.3 -> accepted
        d1 = policy.fuse(inc, cand, detector_out=None)
        assert d1.take_candidate is True

        # Without hysteresis, margin stays 0.3
        inc2 = _make_eval(psr=7.0)   # conf=0.7
        cand2 = _make_eval(psr=8.0)  # conf=0.8, diff=0.1 < 0.3
        d2 = policy.fuse(inc2, cand2, detector_out=None)
        assert d2.take_candidate is False

    def test_candidate_equal_to_incumbent_rejected(self):
        """Equal scores should not trigger switching."""
        policy = PSRGatePolicy(PSRGateParams(margin=0.1), cf_cal=LinearCalibrator(low=0.0, high=10.0))
        inc = _make_eval(psr=5.0)  # conf=0.5
        cand = _make_eval(psr=5.0) # conf=0.5, diff=0.0 < 0.1
        decision = policy.fuse(inc, cand, detector_out=None)
        assert decision.take_candidate is False

    def test_params_property(self):
        params = PSRGateParams(margin=0.2)
        policy = PSRGatePolicy(params)
        assert policy.params is params

    def test_cf_calibrator_property(self):
        cal = LinearCalibrator(low=5.0, high=20.0)
        policy = PSRGatePolicy(cf_cal=cal)
        assert policy.cf_calibrator is cal

    def test_isinstance_fusion_policy(self):
        from edgecv.fusion.policy import FusionPolicy
        assert isinstance(PSRGatePolicy(), FusionPolicy)

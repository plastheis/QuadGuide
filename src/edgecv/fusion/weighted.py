"""Weighted fusion policy (MAFiD spec §4.4.2).

Combines calibrated CF confidence and calibrated NN confidence into a single
decision score. Useful when the NN detector has strong priors about target
identity (e.g., class-specific YOLO, or re-identification models).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edgecv.fusion.calibrator import LinearCalibrator, ScoreCalibrator, SigmoidCalibrator
from edgecv.fusion.policy import DetectorOutput, FusionDecision, FusionPolicy
from edgecv.trackers.cf.base import EvalResult


@dataclass
class WeightedFusionParams:
    """Tunable weights for blended CF+NN fusion."""

    cf_weight: float = 0.6     # weight of calibrated CF PSR in decision
    nn_weight: float = 0.4     # weight of calibrated NN score in decision
    threshold: float = 0.5     # combined score must exceed this to take candidate
    nn_floor: float = 0.3      # NN calibrated confidence floor (independent gate)


class WeightedFusionPolicy(FusionPolicy):
    """Cross-source fusion combining calibrated CF and NN scores."""

    def __init__(self,
                 params: WeightedFusionParams | None = None,
                 cf_cal: ScoreCalibrator | None = None,
                 nn_cal: ScoreCalibrator | None = None) -> None:
        self._params = params if params is not None else WeightedFusionParams()
        self._cf_cal = cf_cal if cf_cal is not None else LinearCalibrator(low=3.0, high=15.0)
        self._nn_cal = (nn_cal if nn_cal is not None
                        else SigmoidCalibrator(centre=0.4, steepness=12.0))

    @property
    def params(self) -> WeightedFusionParams:
        return self._params

    @property
    def cf_calibrator(self) -> ScoreCalibrator:
        return self._cf_cal

    @property
    def nn_calibrator(self) -> ScoreCalibrator:
        return self._nn_cal

    def fuse(self,
             incumbent: EvalResult,
             candidate: EvalResult | None,
             detector_out: DetectorOutput | None) -> FusionDecision:
        if candidate is None or detector_out is None:
            return FusionDecision(take_candidate=False)

        inc_conf = self._cf_cal.calibrate(incumbent.psr)
        cand_cf_conf = self._cf_cal.calibrate(candidate.psr)

        # Highest NN score across all detections
        nn_scores = detector_out.scores
        if len(nn_scores) == 0:
            return FusionDecision(take_candidate=False)
        nn_conf = self._nn_cal.calibrate(float(np.max(nn_scores)))

        if nn_conf < self._params.nn_floor:
            return FusionDecision(take_candidate=False)

        cand_combined = (self._params.cf_weight * cand_cf_conf +
                         self._params.nn_weight * nn_conf)
        inc_combined = inc_conf  # incumbent has no NN score

        take = (cand_combined - inc_combined) > self._params.threshold
        return FusionDecision(take_candidate=take)

"""PSR gate fusion policy (ARCHITECTURE.md §8).

Reference confidence-gate policy: compares two CF filters evaluated on the same
frame by the same engine and selects between incumbent and candidate. Ships as a
generic fusion abstraction for hybrids to build on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from edgecv.fusion.calibrator import LinearCalibrator, ScoreCalibrator
from edgecv.fusion.policy import DetectorOutput, FusionDecision, FusionPolicy
from edgecv.trackers.cf.base import EvalResult


@dataclass
class PSRGateParams:
    """All parameters tunable at construction time and runtime-inspectable."""

    margin: float = 0.5
        # Candidate calibrated confidence must exceed incumbent by this much.
        # Prevents rapid flapping between filters on noisy frames.
    candidate_floor: float = 0.3
        # Candidate calibrated confidence must be above this, even if it beats
        # the incumbent. Rejects filters built from low-quality detections.
    incumbent_floor: float = 0.1
        # If the incumbent's calibrated confidence is below this, the track is
        # already failing -- accept the candidate more eagerly (margin is halved).
    use_hysteresis: bool = True
        # When True, once the candidate is accepted, the margin for switching
        # BACK to a future incumbent is doubled (prevents oscillation).


class PSRGatePolicy(FusionPolicy):
    """PSR-gate fusion: compare two CF filters evaluated on the same frame.

    This is the MAFiD paper's core fusion method (Figure 5 comparison).
    """

    def __init__(self,
                 params: PSRGateParams | None = None,
                 cf_cal: ScoreCalibrator | None = None) -> None:
        self._params = params if params is not None else PSRGateParams()
        self._cf_cal = cf_cal if cf_cal is not None else LinearCalibrator(low=3.0, high=15.0)
        self._using_candidate = False  # hysteresis state

    @property
    def params(self) -> PSRGateParams:
        return self._params

    @property
    def cf_calibrator(self) -> ScoreCalibrator:
        return self._cf_cal

    def fuse(self,
             incumbent: EvalResult,
             candidate: EvalResult | None,
             detector_out: DetectorOutput | None) -> FusionDecision:
        if candidate is None:
            return FusionDecision(take_candidate=False)

        inc_conf = self._cf_cal.calibrate(incumbent.psr)
        cand_conf = self._cf_cal.calibrate(candidate.psr)

        margin = self._params.margin
        if self._params.use_hysteresis and self._using_candidate:
            margin *= 2.0  # harder to switch back

        # Emergency mode: incumbent is failing, be more permissive
        if inc_conf < self._params.incumbent_floor:
            margin /= 2.0

        if cand_conf < self._params.candidate_floor:
            return FusionDecision(take_candidate=False)

        take = (cand_conf - inc_conf) > margin
        if take:
            self._using_candidate = True
        return FusionDecision(take_candidate=take)

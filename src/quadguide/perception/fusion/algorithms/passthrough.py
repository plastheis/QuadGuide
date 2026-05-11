from __future__ import annotations

from quadguide.core.messages import ActiveTracker, TargetEstimate, TrackerEstimate, TrackerHealth

from ._helpers import passthrough_result
from .base import BaseFusion


class PassthroughFusion(BaseFusion):
    """Forward one estimate unchanged; no blending.

    Uses whichever tracker is designated fast by cfg.fast_tracker.  If that
    tracker has no estimate, falls back to the other one.  Intended for
    single-tracker development setups or as a diagnostic mode.
    """

    def fuse(
        self,
        ccv: TrackerEstimate | None,
        ncv: TrackerEstimate | None,
        cfg,
    ) -> TargetEstimate | None:
        fast_is_ccv = cfg.fast_tracker == "ccv"
        fast, slow = (ccv, ncv) if fast_is_ccv else (ncv, ccv)
        fast_label = ActiveTracker.CCV if fast_is_ccv else ActiveTracker.NCV
        slow_label = ActiveTracker.NCV if fast_is_ccv else ActiveTracker.CCV

        if fast is not None:
            return passthrough_result(fast, fast_label)
        if slow is not None:
            return passthrough_result(slow, slow_label)
        return None

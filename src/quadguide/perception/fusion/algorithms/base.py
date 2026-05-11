from __future__ import annotations
from abc import ABC, abstractmethod

from quadguide.core.messages import TargetEstimate, TrackerEstimate


class BaseFusion(ABC):
    @abstractmethod
    def fuse(
        self,
        ccv: TrackerEstimate | None,
        ncv: TrackerEstimate | None,
        cfg,
    ) -> TargetEstimate | None: ...

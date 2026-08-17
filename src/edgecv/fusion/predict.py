"""Motion predictor abstraction (ARCHITECTURE.md §9). Supplies the search window
for set_filter, bridging detection latency at edge frame rates. The
constant-velocity default lands in a later build."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from edgecv.core.bbox import BoundingBox


class MotionPredictor(ABC):
    @abstractmethod
    def predict(self,
                history: Sequence[tuple[float, BoundingBox]],
                dt: float) -> BoundingBox:
        """Predict the box dt seconds after the last history sample."""

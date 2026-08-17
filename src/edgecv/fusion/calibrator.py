"""Score calibrators (MAFiD spec §4.3).

Map raw tracker/detector scores (CF PSR ~2-50, NN confidence 0-1) to a common
normalised 0-1 confidence scale for fusion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class ScoreCalibrator(ABC):
    """Map a raw tracker/detector score to a normalised 0-1 confidence."""

    @abstractmethod
    def calibrate(self, raw: float) -> float:
        """Return 0-1, where 1 = maximum confidence."""

    @property
    @abstractmethod
    def params(self) -> dict:
        """Expose tunable parameters for introspection / tuning scripts."""


@dataclass
class LinearCalibrator(ScoreCalibrator):
    """Linear mapping: [low, high] -> [0, 1], clamped.

    Suitable for CF PSR where a known 'good' range exists per tracker
    (e.g., MOSSE: locked ~7-50, lost ~2-5).
    """

    low: float = 3.0      # raw score that maps to 0
    high: float = 15.0    # raw score that maps to 1

    def calibrate(self, raw: float) -> float:
        return float(max(0.0, min(1.0, (raw - self.low) / max(self.high - self.low, 1e-9))))

    @property
    def params(self) -> dict:
        return {"low": self.low, "high": self.high}


@dataclass
class SigmoidCalibrator(ScoreCalibrator):
    """Sigmoid mapping with configurable centre and steepness.

    Suitable for NN scores where the useful range is concentrated (e.g., YOLO
    confidence is rarely above 0.9, and 0.3-0.6 is the decision boundary).
    """

    centre: float = 0.5        # raw score at which confidence = 0.5
    steepness: float = 10.0    # higher = sharper transition

    def calibrate(self, raw: float) -> float:
        return float(1.0 / (1.0 + np.exp(-self.steepness * (raw - self.centre))))

    @property
    def params(self) -> dict:
        return {"centre": self.centre, "steepness": self.steepness}

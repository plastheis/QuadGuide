"""Fusion abstractions (ARCHITECTURE.md §8). The library ships the abstractions
hybrids need, not specific hybrid trackers. The reference PSR-gate policy lands in
a later build."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from edgecv.trackers.cf.base import EvalResult


@dataclass
class DetectorOutput:
    boxes: np.ndarray       # (N, 4), normalised
    scores: np.ndarray      # (N,)
    meta: dict | None = None


@dataclass
class FusionDecision:
    take_candidate: bool


class FusionPolicy(ABC):
    @abstractmethod
    def fuse(self,
             incumbent: EvalResult,
             candidate: EvalResult | None,
             detector_out: DetectorOutput | None) -> FusionDecision: ...

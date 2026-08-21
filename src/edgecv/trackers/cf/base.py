"""Correlation-filter base contract (ARCHITECTURE.md §6.1, §14.5).

Every CF tracker subclasses CorrelationFilterTracker and implements both the
online (mutating) loop AND the pure ops. build_filter/evaluate MUST NOT mutate
self: a worker builds a FilterState in one process and the caller evaluates
incumbent vs candidate on the current frame in another. This purity is what makes
build-elsewhere / evaluate-here / swap safe across processes."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.core.tracker import Tracker


@dataclass
class FilterState:
    """Transferable CF model state (ARCHITECTURE.md §5/§6.1)."""

    arrays: dict[str, np.ndarray]   # e.g. {"H": ..., "A": ..., "B": ...}, arbitrary shapes
    bbox: BoundingBox               # ROI the filter was built for
    meta: dict                      # feature type, window params, scale/aspect, abi tag


@dataclass
class EvalResult:
    bbox: BoundingBox
    response_map: np.ndarray
    psr: float


class CorrelationFilterTracker(Tracker):
    # --- pure ops: MUST NOT mutate self ---
    @abstractmethod
    def build_filter(self, frame: np.ndarray, bbox: BoundingBox) -> FilterState: ...

    @abstractmethod
    def evaluate(self, frame: np.ndarray, state: FilterState) -> EvalResult: ...

    # --- state access ---
    @abstractmethod
    def get_filter(self) -> FilterState: ...

    @abstractmethod
    def set_filter(self, state: FilterState,
                   search_box: BoundingBox | None = None) -> None: ...

    @property
    @abstractmethod
    def response_map(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def psr(self) -> float: ...

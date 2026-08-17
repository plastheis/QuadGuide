"""Hybrid tracker scaffolding.

Reusable building blocks for composing CF + NN/detector hybrids across processes
(detector adapters, the detector worker, and filter-state serialisation). The
concrete MAFiD reference hybrids have been removed; these pieces remain for a
future hybrid to build on.
"""

from edgecv.trackers.hybrid.acquire_kalman_track import AcquireKalmanTrack
from edgecv.trackers.hybrid.acquire_track import AcquireTrack, State
from edgecv.trackers.hybrid.detector_adapter import (
    NanoTrackDetectorAdapter,
    NNDetectorAdapter,
    YoloDetectorAdapter,
)
from edgecv.trackers.hybrid.verified_acquire_track import VerifiedAcquireTrack

__all__ = [
    "AcquireTrack",
    "AcquireKalmanTrack",
    "VerifiedAcquireTrack",
    "State",
    "NNDetectorAdapter",
    "NanoTrackDetectorAdapter",
    "YoloDetectorAdapter",
]

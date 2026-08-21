"""edgecv — single-object visual trackers for real-time edge deployment."""

__version__ = "0.0.1"

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.core.tracker import Tracker

__all__ = [
    "BoundingBox",
    "PixelBox",
    "TrackResult",
    "TrackStatus",
    "Tracker",
    "__version__",
]

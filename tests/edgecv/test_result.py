from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackResult, TrackStatus


def test_status_values_match_contract():
    assert TrackStatus.INITIALIZING == 0
    assert TrackStatus.LOCKED == 1
    assert TrackStatus.COASTING == 2
    assert TrackStatus.LOST == 3


def test_track_result_allows_none_bbox_and_confidence():
    r = TrackResult(bbox=None, confidence=None, status=TrackStatus.LOST,
                    timestamp=1.5, seq=7)
    assert r.bbox is None
    assert r.confidence is None
    assert r.seq == 7


def test_track_result_carries_bbox_and_score():
    bb = BoundingBox(0.1, 0.1, 0.2, 0.2)
    r = TrackResult(bbox=bb, confidence=12.3, status=TrackStatus.LOCKED,
                    timestamp=2.0, seq=42)
    assert r.bbox is bb
    assert r.confidence == 12.3
    assert r.status is TrackStatus.LOCKED

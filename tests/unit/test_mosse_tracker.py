import numpy as np
import pytest
from quadguide.core.messages import TrackerHealth, BoundingBox
from quadguide.perception.mosse.tracker import MOSSETracker


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestMOSSETrackerNoLock:
    def test_update_before_init_returns_no_lock(self, blank_frame):
        tracker = MOSSETracker()
        est = tracker.update(blank_frame)
        assert est.tracker_health == TrackerHealth.NO_LOCK

    def test_update_before_init_confidence_zero(self, blank_frame):
        tracker = MOSSETracker()
        est = tracker.update(blank_frame)
        assert est.confidence == 0.0


class TestMOSSETrackerInit:
    def test_init_sets_initialized(self, blank_frame):
        tracker = MOSSETracker()
        tracker.init(blank_frame, BoundingBox(0.2, 0.2, 0.3, 0.3))
        assert tracker._initialized is True

    def test_update_after_init_not_no_lock(self, blank_frame):
        tracker = MOSSETracker()
        tracker.init(blank_frame, BoundingBox(0.2, 0.2, 0.3, 0.3))
        est = tracker.update(blank_frame)
        assert est.tracker_health != TrackerHealth.NO_LOCK

    def test_close_does_not_raise(self):
        MOSSETracker().close()

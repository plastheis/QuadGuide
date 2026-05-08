import numpy as np
import pytest
from quadguide.core.config import KCFConfig
from quadguide.core.messages import TrackerHealth, BoundingBox
from quadguide.perception.kcf.tracker import KCFTracker


@pytest.fixture
def cfg():
    return KCFConfig(detect_thresh=0.5, sigma=0.2, lambda_=0.0001)


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestKCFTrackerNoLock:
    def test_update_before_init_returns_no_lock(self, cfg, blank_frame):
        tracker = KCFTracker(cfg)
        est = tracker.update(blank_frame)
        assert est.tracker_health == TrackerHealth.NO_LOCK

    def test_update_before_init_confidence_is_zero(self, cfg, blank_frame):
        tracker = KCFTracker(cfg)
        est = tracker.update(blank_frame)
        assert est.confidence == 0.0


class TestKCFTrackerInit:
    def test_init_sets_initialized_flag(self, cfg, blank_frame):
        tracker = KCFTracker(cfg)
        bbox = BoundingBox(0.2, 0.2, 0.3, 0.3)
        tracker.init(blank_frame, bbox)
        assert tracker._initialized is True

    def test_update_after_init_returns_tracker_estimate(self, cfg, blank_frame):
        from quadguide.core.messages import TrackerEstimate
        tracker = KCFTracker(cfg)
        bbox = BoundingBox(0.2, 0.2, 0.3, 0.3)
        tracker.init(blank_frame, bbox)
        est = tracker.update(blank_frame)
        assert isinstance(est, TrackerEstimate)

    def test_update_after_init_health_is_not_no_lock(self, cfg, blank_frame):
        tracker = KCFTracker(cfg)
        bbox = BoundingBox(0.2, 0.2, 0.3, 0.3)
        tracker.init(blank_frame, bbox)
        est = tracker.update(blank_frame)
        assert est.tracker_health != TrackerHealth.NO_LOCK

    def test_close_does_not_raise(self, cfg):
        KCFTracker(cfg).close()

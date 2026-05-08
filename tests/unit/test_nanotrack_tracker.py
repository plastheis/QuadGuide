import numpy as np
import pytest
from quadguide.core.config import NanotrackConfig
from quadguide.core.messages import TrackerHealth, BoundingBox, TrackerEstimate
from quadguide.perception.nanotrack.tracker import NanoTracker


class _MockRuntime:
    def load(self, path: str):
        return {"path": path}

    def infer(self, model, inputs: dict) -> dict:
        if "input" in inputs:
            return {"features": np.zeros((1, 256, 6, 6), dtype=np.float32)}
        # head call: inputs has "z" and "x"
        return {
            "score": np.zeros((1, 1, 25, 25), dtype=np.float32),
            "bbox":  np.zeros((1, 4, 25, 25), dtype=np.float32),
        }

    def close(self) -> None:
        pass


@pytest.fixture
def cfg():
    return NanotrackConfig(exemplar_sz=127, instance_sz=255, score_threshold=0.7)


@pytest.fixture
def tracker(cfg):
    runtime = _MockRuntime()
    return NanoTracker(runtime, runtime.load("backbone.onnx"), runtime.load("head.onnx"), cfg)


@pytest.fixture
def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestNanoTrackerNoLock:
    def test_update_before_init_returns_no_lock(self, tracker, frame):
        est = tracker.update(frame)
        assert est.tracker_health == TrackerHealth.NO_LOCK

    def test_update_before_init_confidence_zero(self, tracker, frame):
        est = tracker.update(frame)
        assert est.confidence == 0.0


class TestNanoTrackerAfterInit:
    def test_update_after_init_returns_tracker_estimate(self, tracker, frame):
        tracker.init(frame, BoundingBox(0.3, 0.3, 0.2, 0.2))
        est = tracker.update(frame)
        assert isinstance(est, TrackerEstimate)

    def test_update_after_init_not_no_lock(self, tracker, frame):
        tracker.init(frame, BoundingBox(0.3, 0.3, 0.2, 0.2))
        est = tracker.update(frame)
        assert est.tracker_health != TrackerHealth.NO_LOCK

    def test_reinit_works_after_first_init(self, tracker, frame):
        tracker.init(frame, BoundingBox(0.1, 0.1, 0.2, 0.2))
        tracker.init(frame, BoundingBox(0.5, 0.5, 0.2, 0.2))
        est = tracker.update(frame)
        assert est.tracker_health != TrackerHealth.NO_LOCK

    def test_close_does_not_raise(self, tracker):
        tracker.close()

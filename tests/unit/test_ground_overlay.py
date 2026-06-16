import cv2
import numpy as np
import pytest

from quadguide.core.messages import (
    BoundingBox, TrackerEstimate, TrackerHealth,
)
from quadguide.ground.overlay import draw_overlay


def _black_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _estimate(health: TrackerHealth) -> TrackerEstimate:
    return TrackerEstimate(
        timestamp_ns=0,
        bbox=BoundingBox(0.25, 0.25, 0.5, 0.5),
        confidence=0.9,
        tracker_health=health,
    )


def _plain_jpeg(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def _is_jpeg(data: bytes) -> bool:
    return data[:2] == b"\xff\xd8"


def test_none_estimate_returns_jpeg():
    assert _is_jpeg(draw_overlay(_black_frame(), None))


def test_no_lock_returns_jpeg():
    assert _is_jpeg(draw_overlay(_black_frame(), _estimate(TrackerHealth.NO_LOCK)))


def test_lost_returns_jpeg():
    assert _is_jpeg(draw_overlay(_black_frame(), _estimate(TrackerHealth.LOST)))


def test_nominal_returns_jpeg():
    assert _is_jpeg(draw_overlay(_black_frame(), _estimate(TrackerHealth.NOMINAL)))


def test_uncertain_returns_jpeg():
    assert _is_jpeg(draw_overlay(_black_frame(), _estimate(TrackerHealth.UNCERTAIN)))


def test_no_lock_matches_plain_encode():
    frame = _black_frame()
    assert draw_overlay(frame, _estimate(TrackerHealth.NO_LOCK)) == _plain_jpeg(frame)


def test_lost_matches_plain_encode():
    frame = _black_frame()
    assert draw_overlay(frame, _estimate(TrackerHealth.LOST)) == _plain_jpeg(frame)


def test_none_matches_plain_encode():
    frame = _black_frame()
    assert draw_overlay(frame, None) == _plain_jpeg(frame)


def test_nominal_modifies_frame():
    frame = _black_frame()
    assert draw_overlay(frame, _estimate(TrackerHealth.NOMINAL)) != _plain_jpeg(frame)


def test_uncertain_modifies_frame():
    frame = _black_frame()
    assert draw_overlay(frame, _estimate(TrackerHealth.UNCERTAIN)) != _plain_jpeg(frame)


def test_acquiring_modifies_frame():
    frame = _black_frame()
    assert draw_overlay(frame, _estimate(TrackerHealth.ACQUIRING)) != _plain_jpeg(frame)


def test_does_not_mutate_input_frame():
    frame = _black_frame()
    original = frame.copy()
    draw_overlay(frame, _estimate(TrackerHealth.NOMINAL))
    assert np.array_equal(frame, original)

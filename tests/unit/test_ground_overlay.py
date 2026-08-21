import cv2
import numpy as np
import pytest

from quadguide.core.messages import (
    BoundingBox, TrackerEstimate, TrackerHealth,
)
from quadguide.ground.overlay import acquire_crop_from_config, draw_overlay, to_display_bgr


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


# ── acquire-crop guideline ───────────────────────────────────────────────────

def test_acquire_guide_drawn_without_estimate():
    # Faint crop square is drawn even with no tracking box (pre-lock acquire view).
    frame = _black_frame()
    assert draw_overlay(frame, None, acquire_crop=0.7) != _plain_jpeg(frame)


def test_acquire_guide_drawn_with_lost_estimate():
    # Drawn in every state, including NO_LOCK/LOST where no tracking box appears.
    frame = _black_frame()
    assert draw_overlay(frame, _estimate(TrackerHealth.LOST), acquire_crop=0.7) \
        != _plain_jpeg(frame)


def test_acquire_guide_none_is_plain():
    frame = _black_frame()
    assert draw_overlay(frame, None, acquire_crop=None) == _plain_jpeg(frame)


def test_acquire_guide_does_not_mutate_input_frame():
    frame = _black_frame()
    original = frame.copy()
    draw_overlay(frame, None, acquire_crop=0.7)
    assert np.array_equal(frame, original)


def test_acquire_crop_from_config_for_acquire_tracker():
    cfg = {"tracker": {"params": {"tracker": "acquire_track", "acquire_crop": 0.7}}}
    assert acquire_crop_from_config(cfg) == pytest.approx(0.7)


def test_acquire_crop_from_config_default_when_unset():
    cfg = {"tracker": {"params": {"tracker": "verified_acquire_track"}}}
    assert acquire_crop_from_config(cfg) == pytest.approx(0.5)


def test_acquire_crop_from_config_none_for_other_trackers():
    assert acquire_crop_from_config({"tracker": {"params": {"tracker": "mosse"}}}) is None
    assert acquire_crop_from_config(None) is None
    assert acquire_crop_from_config({}) is None


# ── show_bbox (shared HUD toggle) ────────────────────────────────────────────

def test_show_bbox_false_suppresses_tracking_box():
    frame = _black_frame()
    assert draw_overlay(frame, _estimate(TrackerHealth.NOMINAL), show_bbox=False) \
        == _plain_jpeg(frame)


def test_show_bbox_false_suppresses_acquire_guide():
    frame = _black_frame()
    assert draw_overlay(frame, None, acquire_crop=0.7, show_bbox=False) \
        == _plain_jpeg(frame)


def test_show_bbox_defaults_to_drawing():
    frame = _black_frame()
    assert draw_overlay(frame, _estimate(TrackerHealth.NOMINAL)) != _plain_jpeg(frame)


# ── to_display_bgr adapter ──────────────────────────────────────────────────

def test_to_display_bgr_converts_mono16():
    frame = np.full((8, 10), 800, dtype=np.uint16)
    out = to_display_bgr(frame, tonemap_mode="percentile")
    assert out.dtype == np.uint8
    assert out.shape == (8, 10, 3)


def test_to_display_bgr_passthrough_for_bgr8():
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    out = to_display_bgr(frame)
    assert out is frame          # legacy path: no copy, no conversion

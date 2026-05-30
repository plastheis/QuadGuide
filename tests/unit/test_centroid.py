import pytest

from quadguide.core.messages import BoundingBox
from quadguide.guidance._centroid import bbox_centroid_norm


def test_centered_bbox_is_origin():
    cx, cy = bbox_centroid_norm(BoundingBox(0.45, 0.45, 0.1, 0.1))
    assert cx == pytest.approx(0.0)
    assert cy == pytest.approx(0.0)


def test_top_left_bbox_is_negative():
    cx, cy = bbox_centroid_norm(BoundingBox(0.0, 0.0, 0.1, 0.1))
    assert cx == pytest.approx(-0.9)
    assert cy == pytest.approx(-0.9)


def test_bottom_right_bbox_is_positive():
    cx, cy = bbox_centroid_norm(BoundingBox(0.9, 0.9, 0.1, 0.1))
    assert cx == pytest.approx(0.9)
    assert cy == pytest.approx(0.9)


def test_x_only_offset():
    cx, cy = bbox_centroid_norm(BoundingBox(0.95, 0.45, 0.1, 0.1))
    assert cx == pytest.approx(1.0)
    assert cy == pytest.approx(0.0)

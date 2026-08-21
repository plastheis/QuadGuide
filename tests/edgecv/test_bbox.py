# tests/test_bbox.py
import math

import pytest

from edgecv.core.bbox import BoundingBox, PixelBox


def test_to_pixels_scales_by_dimensions():
    bb = BoundingBox(x=0.1, y=0.2, w=0.5, h=0.25)
    px = bb.to_pixels(width=200, height=100)
    assert px == PixelBox(x=20.0, y=20.0, w=100.0, h=25.0)


def test_from_pixels_is_inverse_of_to_pixels():
    bb = BoundingBox(x=0.3, y=0.4, w=0.2, h=0.1)
    px = bb.to_pixels(640, 480)
    back = BoundingBox.from_pixels(px, 640, 480)
    assert math.isclose(back.x, bb.x) and math.isclose(back.y, bb.y)
    assert math.isclose(back.w, bb.w) and math.isclose(back.h, bb.h)


def test_negative_dimension_rejected():
    with pytest.raises(ValueError):
        BoundingBox(x=0.0, y=0.0, w=-0.1, h=0.5)


def test_clamp_keeps_box_inside_unit_square():
    bb = BoundingBox(x=0.8, y=0.8, w=0.5, h=0.5).clamp()
    assert bb.x + bb.w <= 1.0 + 1e-9
    assert bb.y + bb.h <= 1.0 + 1e-9


def test_pixelbox_center():
    px = PixelBox(x=10.0, y=20.0, w=4.0, h=6.0)
    assert px.center == (12.0, 23.0)

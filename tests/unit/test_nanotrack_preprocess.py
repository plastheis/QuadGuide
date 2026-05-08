import numpy as np
import pytest
from quadguide.core.messages import BoundingBox
from quadguide.perception.nanotrack.preprocess import (
    get_exemplar_crop, get_search_crop, normalise,
)


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def bbox():
    return BoundingBox(x=0.3, y=0.3, w=0.2, h=0.2)


class TestGetExemplarCrop:
    def test_output_shape_is_exemplar_sz(self, frame, bbox):
        crop = get_exemplar_crop(frame, bbox, exemplar_sz=127)
        assert crop.shape == (127, 127, 3)

    def test_output_dtype_uint8(self, frame, bbox):
        crop = get_exemplar_crop(frame, bbox, exemplar_sz=127)
        assert crop.dtype == np.uint8


class TestGetSearchCrop:
    def test_output_shape_is_instance_sz(self, frame, bbox):
        crop = get_search_crop(frame, bbox, scale=2.0, instance_sz=255)
        assert crop.shape == (255, 255, 3)

    def test_output_dtype_uint8(self, frame, bbox):
        crop = get_search_crop(frame, bbox, scale=2.0, instance_sz=255)
        assert crop.dtype == np.uint8


class TestNormalise:
    def test_output_shape_is_1_3_H_W(self, frame, bbox):
        crop = get_exemplar_crop(frame, bbox, exemplar_sz=127)
        norm = normalise(crop)
        assert norm.shape == (1, 3, 127, 127)

    def test_output_dtype_float32(self, frame, bbox):
        crop = get_exemplar_crop(frame, bbox, exemplar_sz=127)
        norm = normalise(crop)
        assert norm.dtype == np.float32

    def test_mean_approximately_zero(self, frame, bbox):
        # After ImageNet normalisation the mean of a random image ≈ 0
        crop = get_exemplar_crop(frame, bbox, exemplar_sz=127)
        norm = normalise(crop)
        assert abs(float(norm.mean())) < 1.5

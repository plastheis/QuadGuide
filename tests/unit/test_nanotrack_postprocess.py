import numpy as np
import pytest
from quadguide.perception.nanotrack.postprocess import decode_response


class TestDecodeResponse:
    def test_returns_four_coord_tuple_and_float(self):
        score_map = np.zeros((1, 1, 25, 25), dtype=np.float32)
        bbox_map  = np.zeros((1, 4, 25, 25), dtype=np.float32)
        coords, conf = decode_response(score_map, bbox_map, stride=8, instance_sz=255)
        assert len(coords) == 4
        assert isinstance(conf, float)

    def test_confidence_in_zero_one(self):
        score_map = np.zeros((1, 1, 25, 25), dtype=np.float32)
        bbox_map  = np.zeros((1, 4, 25, 25), dtype=np.float32)
        _, conf = decode_response(score_map, bbox_map, stride=8, instance_sz=255)
        assert 0.0 <= conf <= 1.0

    def test_peak_at_known_location_recovers_roughly_correct_center(self):
        score_map = np.zeros((1, 1, 25, 25), dtype=np.float32)
        score_map[0, 0, 12, 12] = 10.0   # strong peak at centre cell (12,12)
        bbox_map  = np.zeros((1, 4, 25, 25), dtype=np.float32)
        # ltrb all zero → decoded bbox is a degenerate point at the peak cell centre
        (cx_n, cy_n, w_n, h_n), conf = decode_response(score_map, bbox_map, stride=8, instance_sz=255)
        # Peak at cell (12,12): px centre = (12+0.5)*8 = 100.
        # Normalised: 100/255 ≈ 0.392
        assert abs(cx_n - 100 / 255) < 0.02
        assert abs(cy_n - 100 / 255) < 0.02

    def test_high_score_gives_high_confidence(self):
        score_map = np.full((1, 1, 25, 25), 10.0, dtype=np.float32)
        bbox_map  = np.zeros((1, 4, 25, 25), dtype=np.float32)
        _, conf = decode_response(score_map, bbox_map, stride=8, instance_sz=255)
        assert conf > 0.9

    def test_zero_score_gives_moderate_confidence(self):
        score_map = np.zeros((1, 1, 25, 25), dtype=np.float32)
        bbox_map  = np.zeros((1, 4, 25, 25), dtype=np.float32)
        _, conf = decode_response(score_map, bbox_map, stride=8, instance_sz=255)
        # sigmoid(0) = 0.5
        assert abs(conf - 0.5) < 0.01

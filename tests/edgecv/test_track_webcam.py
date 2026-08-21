import importlib.util
from pathlib import Path

import pytest

from edgecv.core.result import TrackStatus

_PATH = Path(__file__).resolve().parents[2] / "tools" / "track_webcam.py"
_spec = importlib.util.spec_from_file_location("track_webcam", _PATH)
tw = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(tw)


def test_clamp_box_size_within_bounds_unchanged():
    assert tw.clamp_box_size(96, 480, 640) == 96


def test_clamp_box_size_floors_to_min():
    assert tw.clamp_box_size(8, 480, 640) == tw.MIN_BOX_PX


def test_clamp_box_size_caps_to_smaller_frame_dim():
    assert tw.clamp_box_size(500, 480, 640) == 480


def test_clamp_box_size_tiny_frame_returns_frame_dim():
    # frame smaller than MIN_BOX_PX -> return the frame dimension
    assert tw.clamp_box_size(96, 10, 20) == 10


def test_centered_square_is_centered():
    pix = tw.centered_square(480, 640, 96)
    assert pix.w == 96 and pix.h == 96
    assert pix.x == (640 - 96) / 2.0
    assert pix.y == (480 - 96) / 2.0
    assert pix.center == (320.0, 240.0)


def test_centered_square_clamps_size():
    pix = tw.centered_square(480, 640, 5000)
    assert pix.w == 480 and pix.h == 480


@pytest.mark.parametrize(
    "status,expected",
    [
        (TrackStatus.LOCKED, tw.ORANGE),
        (TrackStatus.INITIALIZING, tw.ORANGE),
        (TrackStatus.COASTING, tw.YELLOW),
        (TrackStatus.LOST, tw.RED),
    ],
)
def test_status_color(status, expected):
    assert tw.status_color(status) == expected


def test_tracker_names_include_nn():
    assert set(tw.TRACKERS) >= {"mosse", "siamfc", "yolo"}


def test_build_tracker_mosse_needs_no_model():
    t = tw.build_tracker("mosse")
    assert type(t).__name__ == "Mosse"
    t.close()


@pytest.mark.parametrize("name", ["siamfc", "yolo"])
def test_build_tracker_missing_model_errors_clearly(name):
    with pytest.raises(FileNotFoundError, match="ONNX model"):
        tw.build_tracker(name, model_path="/no/such/model.onnx")

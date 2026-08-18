import numpy as np
import pytest

from edgecv.backends.base import TensorSpec
from edgecv.trackers.nn.preprocess import (
    class_agnostic_nms,
    crop_with_context,
    letterbox,
    resize_bilinear,
    to_input,
)


def test_resize_bilinear_identity():
    img = np.random.default_rng(0).standard_normal((8, 8, 1)).astype(np.float32)
    out = resize_bilinear(img, (8, 8))
    np.testing.assert_allclose(out, img, atol=1e-5)


def test_resize_bilinear_changes_shape_keeps_channels():
    img = np.zeros((4, 4, 3), np.float32)
    out = resize_bilinear(img, (8, 16))
    assert out.shape == (8, 16, 3)


def test_resize_bilinear_cv2_matches_numpy(monkeypatch):
    """cv2 fast path and numpy reference agree (same half-pixel grid)."""
    import edgecv.trackers.nn.preprocess as pp
    if pp._cv2 is None:
        pytest.skip("cv2 not installed")
    img = (np.random.default_rng(1).random((37, 53, 3)) * 255).astype(np.float32)
    cv2_out = pp.resize_bilinear(img, (64, 96))           # uses cv2
    monkeypatch.setattr(pp, "_cv2", None)
    numpy_out = pp.resize_bilinear(img, (64, 96))         # forces numpy
    assert cv2_out.shape == numpy_out.shape == (64, 96, 3)
    # Same sampling formula → close; allow small interpolation-weight differences.
    assert np.abs(cv2_out - numpy_out).mean() < 1.0


def test_crop_with_context_centre_and_shape():
    frame = np.zeros((100, 120, 3), np.uint8)
    frame[48:52, 58:62] = 200  # bright square at centre (~ (60, 50))
    patch, xf = crop_with_context(frame, (60.0, 50.0), (20.0, 20.0), (40, 40))
    assert patch.shape == (40, 40, 3)
    py, px = np.unravel_index(int(patch[..., 0].argmax()), patch.shape[:2])
    fx, fy = xf.to_frame((px, py))
    assert fx == pytest.approx(60.0, abs=2.0)
    assert fy == pytest.approx(50.0, abs=2.0)


def test_crop_with_context_edge_replicates_off_frame():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch, _ = crop_with_context(frame, (0.0, 0.0), (6.0, 6.0), (12, 12))
    assert patch.shape == (12, 12)
    assert np.isfinite(patch).all()


def test_crop_xform_to_frame_roundtrip():
    _, xf = crop_with_context(np.zeros((50, 50)), (25.0, 30.0), (10.0, 20.0), (32, 64))
    fx, fy = xf.to_frame((32.0 - 0.5, 16.0 - 0.5))  # (ow/2-0.5, oh/2-0.5)
    assert fx == pytest.approx(25.0, abs=1e-6)
    assert fy == pytest.approx(30.0, abs=1e-6)


# ── cv2 fast-path parity with the numpy reference (spec 2026-06-14) ───────────
from edgecv.trackers.nn import preprocess as _pp  # noqa: E402

_CV2_CASES = [
    ((60.0, 50.0), (40.0, 40.0), (48, 48)),     # centred, upscaled
    ((30.0, 80.0), (50.0, 30.0), (32, 64)),     # off-centre, non-square aspect
    ((5.0, 5.0), (40.0, 40.0), (40, 40)),       # window overruns the top-left border
    ((118.0, 95.0), (30.0, 30.0), (24, 24)),    # window overruns the bottom-right
]


def _gradient_frame():
    h, w = 100, 120
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    chans = [xx * 2.0, yy * 2.0, (xx + yy)]
    return np.clip(np.stack(chans, axis=-1), 0, 255).astype(np.uint8)


@pytest.mark.skipif(_pp._cv2 is None, reason="cv2 not installed")
@pytest.mark.parametrize("center,size,out", _CV2_CASES)
def test_crop_resize_cv2_matches_numpy(center, size, out):
    frame = _gradient_frame()
    fast = _pp._crop_resize(frame, center, size, out)
    ref = _pp._crop_resize_numpy(frame, center, size, out)
    assert fast.shape == ref.shape
    assert fast.dtype == np.float32
    # interpolation rounding only — both share the same half-pixel grid.
    assert np.max(np.abs(fast - ref)) <= 1.0


@pytest.mark.skipif(_pp._cv2 is None, reason="cv2 not installed")
def test_crop_resize_cv2_grayscale_parity():
    frame = _gradient_frame()[..., 0]  # 2-D
    fast = _pp._crop_resize(frame, (40.0, 40.0), (30.0, 30.0), (32, 32))
    ref = _pp._crop_resize_numpy(frame, (40.0, 40.0), (30.0, 30.0), (32, 32))
    assert fast.shape == (32, 32) == ref.shape
    assert np.max(np.abs(fast - ref)) <= 1.0


@pytest.mark.skipif(_pp._cv2 is None, reason="cv2 not installed")
def test_crop_with_context_inversion_invariant_across_backends():
    """CropXform inversion must be identical whichever sampler produced the patch."""
    frame = _gradient_frame()
    center, size, out = (60.0, 50.0), (40.0, 40.0), (48, 48)
    _, xf_fast = crop_with_context(frame, center, size, out)
    # Same xform regardless of backend — pin the spec invariant on a few pixels.
    for ox, oy in [(0.0, 0.0), (23.5, 23.5), (47.0, 12.0)]:
        fx = (center[0] - size[1] / 2.0) + (ox + 0.5) / out[1] * size[1]
        fy = (center[1] - size[0] / 2.0) + (oy + 0.5) / out[0] * size[0]
        assert xf_fast.to_frame((ox, oy)) == pytest.approx((fx, fy), abs=1e-6)


def test_crop_with_context_numpy_fallback(monkeypatch):
    """With cv2 forced absent, crop_with_context still returns a float32 patch."""
    monkeypatch.setattr(_pp, "_cv2", None)
    frame = _gradient_frame()
    patch, xf = crop_with_context(frame, (60.0, 50.0), (40.0, 40.0), (40, 40))
    assert patch.shape == (40, 40, 3)
    assert patch.dtype == np.float32
    assert np.isfinite(patch).all()


def test_letterbox_preserves_aspect_and_pads():
    img = np.zeros((50, 100, 3), np.uint8)  # 2:1 wide
    out, xf = letterbox(img, (64, 64), pad_value=114)
    assert out.shape == (64, 64, 3)
    # 100->64 sets scale 0.64; height 50*0.64=32, padded symmetrically in 64
    assert xf.scale == pytest.approx(0.64)
    assert xf.pad[1] == pytest.approx((64 - 32) / 2.0)  # vertical pad


def test_letterbox_inverts_box():
    img = np.zeros((50, 100, 3), np.uint8)
    _, xf = letterbox(img, (64, 64))
    # a box covering the whole original maps from the unpadded letterbox region
    px, py = xf.pad
    s = xf.scale
    x1, y1, x2, y2 = xf.to_orig_xyxy((px, py, px + 100 * s, py + 50 * s))
    assert (x1, y1, x2, y2) == pytest.approx((0.0, 0.0, 100.0, 50.0), abs=1e-4)


def test_to_input_gray_layout_and_scale():
    patch = np.full((8, 8, 3), 255, np.uint8)
    spec = TensorSpec(name="exemplar", shape=(1, 1, 8, 8), dtype="float32")
    arr = to_input(patch, spec, color="gray", scale=1 / 255)
    assert arr.shape == (1, 1, 8, 8)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, 1.0, atol=1e-4)


def test_to_input_int8_quant():
    patch = np.zeros((4, 4, 3), np.uint8)
    spec = TensorSpec(name="x", shape=(1, 3, 4, 4), dtype="int8",
                      quant={"scale": 0.5, "zero_point": -3})
    arr = to_input(patch, spec, color="rgb", scale=1 / 255)
    assert arr.dtype == np.int8
    # value 0.0 -> round(0/0.5) + (-3) = -3
    assert int(arr.flat[0]) == -3


def test_points_grid_shape_and_centre():
    from edgecv.trackers.nn.preprocess import points_grid
    pts = points_grid(stride=16, size=15)
    assert pts.shape == (2, 15 * 15)
    centre = (15 // 2) * 15 + (15 // 2)        # row-major index of the middle cell
    assert pts[0, centre] == 0.0               # x at centre is 0
    assert pts[1, centre] == 0.0               # y at centre is 0


def test_points_grid_spacing_and_row_major():
    from edgecv.trackers.nn.preprocess import points_grid
    pts = points_grid(stride=16, size=15)
    # index = row*size + col ; x varies with col, y varies with row.
    assert pts[0, 0] == -(15 // 2) * 16        # top-left x = ori
    assert pts[1, 0] == -(15 // 2) * 16        # top-left y = ori
    assert pts[0, 1] - pts[0, 0] == 16         # next column is +stride in x
    assert pts[1, 0] == pts[1, 14]             # whole first row shares one y
    assert pts[1, 15] - pts[1, 0] == 16        # next row is +stride in y


def test_class_agnostic_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = class_agnostic_nms(boxes, scores, iou_thresh=0.5)
    assert set(keep.tolist()) == {0, 2}  # box 1 overlaps the higher-scored box 0

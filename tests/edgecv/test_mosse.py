import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.trackers.cf.mosse import (
    Mosse,
    _bilinear_sample,
    _crop_patch,
    _preprocess,
    _rand_warp,
    _subpixel_peak,
)


def test_crop_patch_fully_outside_frame_returns_filled_patch():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch = _crop_patch(frame, center=(-100.0, -100.0), size=(6, 6))
    assert patch.shape == (6, 6)
    # window is entirely off the top-left; edge policy fills from the nearest pixel
    assert np.all(patch == frame[0, 0])


def test_crop_patch_fully_inside_keeps_shape():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch = _crop_patch(frame, center=(5.0, 5.0), size=(6, 6))
    assert patch.shape == (6, 6)


def test_crop_patch_edge_pads_when_window_crosses_border():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch = _crop_patch(frame, center=(1.0, 1.0), size=(6, 6))
    assert patch.shape == (6, 6)
    # top-left is outside the frame; edge mode replicates frame[0, 0] == 0
    assert patch[0, 0] == frame[0, 0]
    assert patch[3, 3] == frame[1, 1]   # interior pixel maps to the correct frame location


def test_crop_patch_preserves_channels():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    patch = _crop_patch(frame, center=(5.0, 5.0), size=(6, 6))
    assert patch.shape == (6, 6, 3)


def test_bilinear_sample_identity_grid_returns_same_image():
    img = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
    ys, xs = np.indices((8, 8)).astype(np.float32)
    out = _bilinear_sample(img, xs, ys)
    np.testing.assert_allclose(out, img, atol=1e-5)


def test_rand_warp_preserves_shape_and_constant_image():
    rng = np.random.default_rng(1)
    const = np.full((16, 16), 5.0, np.float32)
    out = _rand_warp(const, rng)
    assert out.shape == (16, 16)
    np.testing.assert_allclose(out, 5.0, atol=1e-4)


def test_rand_warp_is_seed_deterministic():
    patch = np.random.default_rng(2).standard_normal((16, 16)).astype(np.float32)
    a = _rand_warp(patch, np.random.default_rng(7))
    b = _rand_warp(patch, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_preprocess_constant_patch_is_all_zero():
    # z-score of a constant has zero std -> zero; windowing keeps it zero.
    patch = np.full((16, 16, 3), 100, np.uint8)
    window = np.ones((16, 16), np.float32)
    out = _preprocess(patch, window)
    assert out.shape == (16, 16)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 0.0, atol=1e-5)


def test_subpixel_peak_interpolates_fractional_offset():
    r = np.zeros((5, 5), np.float32)
    r[2, 1], r[2, 2], r[2, 3] = 2.0, 4.0, 3.0  # peak at (2,2), skewed toward +x
    py, px = _subpixel_peak(r)
    assert py == pytest.approx(2.0, abs=1e-6)
    assert px == pytest.approx(2.0 + 0.5 * (2.0 - 3.0) / (2.0 - 8.0 + 3.0), abs=1e-6)


def _blob_frame(h=120, w=160, cx=80.0, cy=60.0, blob_sigma=6.0):
    ys, xs = np.indices((h, w)).astype(np.float32)
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * blob_sigma**2))
    img = (g * 255.0).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def _box_at(cx, cy, w_img, h_img, bw=40, bh=40):
    return BoundingBox(
        x=(cx - bw / 2) / w_img, y=(cy - bh / 2) / h_img, w=bw / w_img, h=bh / h_img
    )


def test_build_filter_produces_complex64_AB_of_template_size():
    frame = _blob_frame()
    t = Mosse(n_warps=2)
    state = t.build_filter(frame, _box_at(80, 60, 160, 120))
    th, tw = state.meta["template_size"]
    assert (th, tw) == (64, 64)
    assert state.arrays["A"].dtype == np.complex64
    assert state.arrays["B"].dtype == np.complex64
    assert state.arrays["A"].shape == (64, 64)
    assert state.meta["abi"] == "mosse-1"


def test_build_filter_is_pure_and_seed_deterministic():
    frame = _blob_frame()
    t = Mosse(n_warps=4, rng_seed=3)
    box = _box_at(80, 60, 160, 120)
    s1 = t.build_filter(frame, box)
    s2 = t.build_filter(frame, box)
    np.testing.assert_array_equal(s1.arrays["A"], s2.arrays["A"])
    np.testing.assert_array_equal(s1.arrays["B"], s2.arrays["B"])


def test_name_is_mosse():
    assert Mosse().name() == "MOSSE"


def test_evaluate_is_pure_and_returns_centered_peak_on_build_frame():
    frame = _blob_frame(cx=80.0, cy=60.0)
    t = Mosse(n_warps=2)
    box = _box_at(80, 60, 160, 120)
    state = t.build_filter(frame, box)
    a_before = state.arrays["A"].copy()
    b_before = state.arrays["B"].copy()

    er = t.evaluate(frame, state)

    # purity: evaluate must not mutate the state it was given
    np.testing.assert_array_equal(state.arrays["A"], a_before)
    np.testing.assert_array_equal(state.arrays["B"], b_before)
    th, tw = state.meta["template_size"]
    assert er.response_map.shape == (th, tw)
    assert np.isfinite(er.psr)
    # matched filter on its own build frame -> peak ~ centre -> box centre ~ unchanged
    cx, cy = er.bbox.to_pixels(160, 120).center
    assert cx == pytest.approx(80.0, abs=1.5)
    assert cy == pytest.approx(60.0, abs=1.5)


def test_init_builds_filter_and_locks():
    frame = _blob_frame()
    t = Mosse(n_warps=2)
    t.init(frame, _box_at(80, 60, 160, 120))
    assert t.status == TrackStatus.LOCKED
    assert t.get_filter().arrays["A"].shape == (64, 64)


def test_set_filter_with_search_box_moves_crop_center():
    frame = _blob_frame()
    t = Mosse(n_warps=2)
    t.init(frame, _box_at(80, 60, 160, 120))
    state = t.get_filter()
    search = _box_at(100, 70, 160, 120)
    t.set_filter(state, search_box=search)
    cx, cy = t.get_filter().bbox.to_pixels(160, 120).center
    assert cx == pytest.approx(100.0, abs=0.5)
    assert cy == pytest.approx(70.0, abs=0.5)


def test_get_set_round_trip_preserves_evaluation():
    frame = _blob_frame()
    t = Mosse(n_warps=2)
    t.init(frame, _box_at(80, 60, 160, 120))
    er1 = t.evaluate(frame, t.get_filter())
    t.set_filter(t.get_filter())
    er2 = t.evaluate(frame, t.get_filter())
    np.testing.assert_allclose(
        er1.bbox.to_pixels(160, 120).center, er2.bbox.to_pixels(160, 120).center, atol=1e-6)


def test_update_tracks_translating_blob():
    t = Mosse(n_warps=4)
    t.init(_blob_frame(cx=80.0, cy=60.0), _box_at(80, 60, 160, 120))
    true_cx, true_cy = 80.0, 60.0
    for _ in range(5):
        true_cx += 3.0
        true_cy += 2.0
        res = t.update(_blob_frame(cx=true_cx, cy=true_cy))
    cx, cy = res.bbox.to_pixels(160, 120).center
    assert cx == pytest.approx(true_cx, abs=3.0)
    assert cy == pytest.approx(true_cy, abs=3.0)
    assert res.status == TrackStatus.LOCKED
    assert res.seq == 5


def test_update_resolves_subpixel_shift():
    t = Mosse(n_warps=4)
    t.init(_blob_frame(cx=80.0, cy=60.0), _box_at(80, 60, 160, 120))
    res = t.update(_blob_frame(cx=80.5, cy=60.0))
    cx, _ = res.bbox.to_pixels(160, 120).center
    assert 80.2 < cx < 80.8   # fractional, not snapped to an integer pixel


def test_update_on_noise_reports_lost_and_freezes_filter():
    t = Mosse(n_warps=4)
    t.init(_blob_frame(cx=80.0, cy=60.0), _box_at(80, 60, 160, 120))
    a_before = t.get_filter().arrays["A"].copy()
    noise = (np.random.default_rng(0).integers(0, 256, (120, 160, 3))).astype(np.uint8)
    res = t.update(noise)
    assert res.status == TrackStatus.LOST
    np.testing.assert_array_equal(t.get_filter().arrays["A"], a_before)


# --- Output-box clamping policy (deliberate "won't clamp") --------------------
# A target that is partially off-frame has coordinates legitimately outside [0,1].
# We report them truthfully (so the motion predictor, ARCHITECTURE.md §9, sees a
# real velocity) and we never shrink the box to fit (MOSSE is no-scale: output w,h
# must equal the init box). These two tests fence that decision: they pass today
# and would FAIL if anyone clamped the output (e.g. via BoundingBox.clamp(), which
# both pins coords into [0,1] and shrinks w,h). Untrustworthiness is signalled by
# `status`, not by mangling geometry — see test_update_on_noise_* above.


def test_update_reports_unclamped_offframe_box_at_left_edge():
    box = _box_at(8, 60, 160, 120)               # centre at x=8px -> normalised x < 0
    t = Mosse(n_warps=2)
    t.init(_blob_frame(cx=8.0, cy=60.0), box)
    res = t.update(_blob_frame(cx=8.0, cy=60.0))
    assert res.bbox.x < 0.0                        # truthful, not clamped to 0
    assert res.bbox.w == pytest.approx(box.w)      # size preserved (no shrink)
    assert res.bbox.h == pytest.approx(box.h)


def test_update_preserves_box_size_past_right_edge():
    box = _box_at(152, 60, 160, 120)             # x + w > 1 (extends past right edge)
    t = Mosse(n_warps=2)
    t.init(_blob_frame(cx=152.0, cy=60.0), box)
    res = t.update(_blob_frame(cx=152.0, cy=60.0))
    assert res.bbox.x + res.bbox.w > 1.0           # truthful overflow, not clamped
    assert res.bbox.w == pytest.approx(box.w)      # clamp() would shrink w to 1 - x
    assert res.bbox.h == pytest.approx(box.h)

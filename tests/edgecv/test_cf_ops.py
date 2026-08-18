"""Tests for the CF shared-ops layer (ARCHITECTURE.md §6.1).

Each op has a numpy reference implementation that always works; optimized
backends (scipy/pyfftw for FFT, numba for features) are optional and selected
behind a stable signature. These tests pin the reference behaviour and the
backend-dispatch contract.
"""

import numpy as np
import pytest

from edgecv.trackers.cf.ops import (
    colornames,
    cos_window,
    extract_hog,
    extract_raw,
    feature_backends,
    fft2,
    fft_backends,
    fft_size,
    gaussian2d_labels,
    ifft2,
    psr,
    set_fft_backend,
)

# --- FFT backend dispatch ---------------------------------------------------


def test_numpy_fft_backend_always_available():
    assert "numpy" in fft_backends()


def test_set_unknown_fft_backend_raises():
    with pytest.raises(ValueError):
        set_fft_backend("nonsense")


def test_fft2_ifft2_roundtrip_2d():
    set_fft_backend("numpy")
    x = np.random.default_rng(0).standard_normal((16, 24))
    back = ifft2(fft2(x))
    np.testing.assert_allclose(back.real, x, atol=1e-9)


def test_fft2_operates_over_spatial_axes_of_3d_input():
    # Channels (axis 2) must be transformed independently over (H, W).
    set_fft_backend("numpy")
    x = np.random.default_rng(1).standard_normal((8, 8, 3))
    out = fft2(x)
    assert out.shape == (8, 8, 3)
    for c in range(3):
        np.testing.assert_allclose(out[..., c], np.fft.fft2(x[..., c]), atol=1e-9)


# --- cosine (Hann) window ---------------------------------------------------


def test_cos_window_shape_and_range():
    w = cos_window((10, 20))
    assert w.shape == (10, 20)
    assert w.dtype == np.float32
    assert w.min() >= 0.0 and w.max() <= 1.0 + 1e-6


def test_cos_window_is_symmetric_and_tapers_to_edges():
    w = cos_window((16, 16))
    np.testing.assert_allclose(w, w[::-1, :], atol=1e-6)
    np.testing.assert_allclose(w, w[:, ::-1], atol=1e-6)
    assert w[0, 0] < 1e-6                      # corner tapered to ~0
    assert w[8, 8] > w[0, 0]                    # centre weighted up


# --- PSR --------------------------------------------------------------------


def test_psr_flat_response_is_near_zero():
    flat = np.full((30, 30), 3.0)
    assert psr(flat) < 1.0


def test_psr_sharp_peak_is_high():
    r = np.zeros((40, 40))
    r[20, 20] = 50.0
    assert psr(r) > 10.0


def test_psr_invariant_to_additive_offset():
    rng = np.random.default_rng(2)
    r = rng.standard_normal((40, 40))
    r[20, 20] += 30.0
    assert psr(r) == pytest.approx(psr(r + 100.0), rel=1e-4)


def test_psr_invariant_to_positive_scale():
    rng = np.random.default_rng(3)
    r = rng.standard_normal((40, 40))
    r[20, 20] += 30.0
    assert psr(r) == pytest.approx(psr(r * 7.0), rel=1e-4)


# --- raw feature ------------------------------------------------------------


def test_extract_raw_normalises_to_channelled_float():
    patch = (np.random.default_rng(4).integers(0, 256, (12, 12, 3))).astype(np.uint8)
    out = extract_raw(patch)
    assert out.shape == (12, 12, 1)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


# --- HOG feature ------------------------------------------------------------


def test_extract_hog_shape_and_dtype():
    patch = (np.random.default_rng(5).integers(0, 256, (32, 24, 3))).astype(np.uint8)
    out = extract_hog(patch, cell_size=4, n_bins=9)
    assert out.shape == (8, 6, 9)
    assert out.dtype == np.float32
    assert (out >= 0).all()


def test_extract_hog_uniform_image_has_no_gradient_energy():
    patch = np.full((16, 16, 3), 128, np.uint8)
    out = extract_hog(patch, cell_size=4, n_bins=9)
    assert out.max() < 1e-3


def test_extract_hog_is_deterministic():
    patch = (np.random.default_rng(6).integers(0, 256, (16, 16, 3))).astype(np.uint8)
    np.testing.assert_array_equal(extract_hog(patch), extract_hog(patch))


def test_feature_backends_includes_numpy():
    assert "numpy" in feature_backends()


# --- color names ------------------------------------------------------------


def test_colornames_index_quantisation():
    # Synthetic table maps each quantised index to itself so we can verify the
    # RGB->index formula: idx = (R>>3) + 32*(G>>3) + 1024*(B>>3).
    w2c = np.arange(32768, dtype=np.float32)[:, None]
    patch = np.array([[[8, 16, 24]]], np.uint8)        # 1x1 RGB pixel
    out = colornames(patch, w2c)
    expected_idx = 1 + 32 * 2 + 1024 * 3
    assert out.shape == (1, 1, 1)
    assert out[0, 0, 0] == expected_idx


def test_colornames_output_shape_matches_table_width():
    w2c = np.random.default_rng(7).standard_normal((32768, 11)).astype(np.float32)
    patch = (np.random.default_rng(8).integers(0, 256, (5, 7, 3))).astype(np.uint8)
    out = colornames(patch, w2c)
    assert out.shape == (5, 7, 11)


def test_colornames_rejects_non_rgb():
    with pytest.raises(ValueError):
        colornames(np.zeros((4, 4), np.uint8), np.zeros((32768, 11), np.float32))


# --- fft_size ---------------------------------------------------------------


def test_fft_size_is_next_power_of_two():
    assert fft_size(1) == 1
    assert fft_size(2) == 2
    assert fft_size(3) == 4
    assert fft_size(64) == 64
    assert fft_size(65) == 128


def test_fft_size_is_monotonic_and_at_least_input():
    prev = 0
    for n in range(1, 130):
        s = fft_size(n)
        assert s >= n
        assert s >= prev
        prev = s


# --- gaussian2d_labels -------------------------------------------------------


def test_gaussian2d_labels_peaks_at_center():
    g = gaussian2d_labels((16, 24), sigma=2.0)
    assert g.shape == (16, 24)
    assert g.dtype == np.float32
    assert np.unravel_index(int(np.argmax(g)), g.shape) == (8, 12)
    assert g[8, 12] == pytest.approx(1.0)


def test_gaussian2d_labels_values_in_unit_interval():
    g = gaussian2d_labels((16, 16), sigma=2.0)
    assert g.min() > 0.0
    assert g.max() <= 1.0


def test_gaussian2d_labels_wider_sigma_has_more_support():
    narrow = gaussian2d_labels((32, 32), sigma=1.0)
    wide = gaussian2d_labels((32, 32), sigma=4.0)
    assert (wide > 0.5).sum() > (narrow > 0.5).sum()

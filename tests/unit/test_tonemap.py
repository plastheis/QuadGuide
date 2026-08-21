import numpy as np
from quadguide.ground.overlay import tonemap


def _sky_with_target(h=32, w=32):
    """Bright flat sky (~900/1023) with a small dark target patch (~150)."""
    frame = np.full((h, w), 900, dtype=np.uint16)
    frame[14:18, 14:18] = 150
    return frame


def test_tonemap_outputs_uint8_full_range():
    out = tonemap(_sky_with_target(), mode="percentile")
    assert out.dtype == np.uint8
    assert out.shape == (32, 32)
    assert out.min() < 40 and out.max() > 215     # stretch actually uses the range


def test_linear_map_is_shift_right_2():
    frame = np.array([[0, 4, 1020, 1023]], dtype=np.uint16)
    out = tonemap(frame, mode="linear")
    np.testing.assert_array_equal(out, (frame >> 2).astype(np.uint8))


def test_percentile_separates_target_from_sky_better_than_linear():
    frame = _sky_with_target()
    lin = tonemap(frame, mode="linear")
    pct = tonemap(frame, mode="percentile")
    sky = (slice(0, 4), slice(0, 4))
    tgt = (slice(14, 18), slice(14, 18))
    # Percentile stretch widens the sky/target 8-bit separation vs the naive >>2.
    assert (int(pct[sky].mean()) - int(pct[tgt].mean())) > \
           (int(lin[sky].mean()) - int(lin[tgt].mean()))


def test_gamma_mode_is_monotonic_uint8():
    frame = (np.arange(1024, dtype=np.uint16)).reshape(32, 32)
    out = tonemap(frame, mode="gamma")
    assert out.dtype == np.uint8
    assert out.flatten()[0] <= out.flatten()[-1]   # monotonic overall

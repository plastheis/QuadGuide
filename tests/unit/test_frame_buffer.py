import numpy as np
from quadguide.core.frame_buffer import FrameBuffer


def test_mono16_roundtrip_shape_dtype_and_values():
    fb = FrameBuffer(4, 3, channels=1, dtype="uint16")
    frame = (np.arange(12, dtype=np.uint16).reshape(3, 4) * 100)  # values > 255
    fb.write_frame(frame, timestamp_ns=42)
    out, ts = fb.read_latest()
    assert out.dtype == np.uint16
    assert out.shape == (3, 4)          # mono → (H, W), no channel axis
    assert ts == 42
    np.testing.assert_array_equal(out, frame)


def test_mono16_preserves_sub_8bit_low_bits():
    fb = FrameBuffer(2, 2, channels=1, dtype="uint16")
    frame = np.array([[1023, 512], [3, 300]], dtype=np.uint16)  # true 10-bit values
    fb.write_frame(frame)
    out, _ = fb.read_latest()
    np.testing.assert_array_equal(out, frame)   # >8-bit values survive intact


def test_uint8_bgr_backcompat_unchanged():
    fb = FrameBuffer(4, 3)  # defaults: channels=3, dtype=uint8
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    frame[1, 2] = (10, 20, 30)
    fb.write_frame(frame)
    out, _ = fb.read_latest()
    assert out.dtype == np.uint8 and out.shape == (3, 4, 3)
    np.testing.assert_array_equal(out, frame)

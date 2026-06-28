import numpy as np
import pytest
from quadguide.perception.camera.sources import (
    CameraSource, USBCamera, CSICamera, CSIY10Camera, VirtualCamera,
    unpack_raw10_to_gray8,
)


def _pack_raw10(pixels10: np.ndarray, stride: int) -> bytes:
    """Pack a (H, W) 10-bit array into MIPI RAW10 with row padding to ``stride``.

    Inverse of unpack_raw10_to_gray8's bit layout: 4 px / 5 bytes (4 high-8-bit
    bytes + 1 LSB byte ``p3 p2 p1 p0`` high→low), each row padded to ``stride``.
    """
    h, w = pixels10.shape
    out = bytearray()
    for r in range(h):
        row = bytearray()
        for c in range(0, w, 4):
            p = pixels10[r, c:c + 4]
            lo = (p[0] & 3) | ((p[1] & 3) << 2) | ((p[2] & 3) << 4) | ((p[3] & 3) << 6)
            row += bytes([p[0] >> 2, p[1] >> 2, p[2] >> 2, p[3] >> 2, lo])
        row += bytes(stride - len(row))   # pad
        out += row
    return bytes(out)


class TestCameraSourceABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            CameraSource()

    def test_usb_camera_is_camera_source(self):
        assert issubclass(USBCamera, CameraSource)

    def test_csi_camera_is_camera_source(self):
        assert issubclass(CSICamera, CameraSource)

    def test_csi_y10_camera_is_camera_source(self):
        assert issubclass(CSIY10Camera, CameraSource)


class TestUnpackRaw10:
    def test_recovers_known_pixels_with_padded_stride(self):
        # 8x2 frame, 10 data bytes/row + 4 pad => stride 14
        px10 = np.array([[0, 1023, 4, 8, 512, 100, 1020, 3],
                         [16, 32, 64, 128, 256, 511, 768, 1019]], dtype=np.uint16)
        stride = 8 * 5 // 4 + 4
        gray = unpack_raw10_to_gray8(_pack_raw10(px10, stride), 8, 2, stride)
        assert gray.shape == (2, 8)
        assert gray.dtype == np.uint8
        np.testing.assert_array_equal(gray, (px10 >> 2).astype(np.uint8))

    def test_full_10bit_range_maps_to_full_8bit(self):
        px10 = np.array([[0, 1023, 1023, 0]], dtype=np.uint16)
        stride = 4 * 5 // 4
        gray = unpack_raw10_to_gray8(_pack_raw10(px10, stride), 4, 1, stride)
        assert int(gray.min()) == 0 and int(gray.max()) == 255

    def test_virtual_camera_is_camera_source(self):
        assert issubclass(VirtualCamera, CameraSource)


class TestVirtualCameraStub:
    def test_open_raises_not_implemented(self):
        cam = VirtualCamera()
        with pytest.raises(NotImplementedError):
            cam.open()

    def test_read_raises_not_implemented(self):
        cam = VirtualCamera()
        with pytest.raises(NotImplementedError):
            cam.read()

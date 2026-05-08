import numpy as np
import pytest
from quadguide.perception.camera.sources import CameraSource, USBCamera, CSICamera, VirtualCamera


class TestCameraSourceABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            CameraSource()

    def test_usb_camera_is_camera_source(self):
        assert issubclass(USBCamera, CameraSource)

    def test_csi_camera_is_camera_source(self):
        assert issubclass(CSICamera, CameraSource)

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

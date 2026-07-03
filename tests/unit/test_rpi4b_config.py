from pathlib import Path

from quadguide.core.config import load_config, cfg_platform, cfg_tracker

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "rpi4b.yaml"


def test_rpi4b_is_usb_camera_flight_default():
    """The committed rpi4b.yaml is the RPi 4B flight default: a USB UVC camera on
    the V4L2 backend, the EdgeCV NanoTrack tracker on the CPU (ONNX), and MAVLink
    over the GPIO UART. HIL (tcp / network) is a local, uncommitted toggle.
    """
    config = load_config(str(CONFIG), {})
    pcfg = cfg_platform(config)

    # USB UVC via V4L2 (USBCamera opens /dev/video0); width/height/fps drive it.
    assert pcfg.camera.backend == "v4l2"
    assert (pcfg.camera.width, pcfg.camera.height, pcfg.camera.fps) == (1280, 720, 30)

    # MAVLink over the GPIO UART (PL011 → /dev/serial0).
    assert pcfg.serial.mode == "uart"
    assert pcfg.serial.port == "/dev/serial0"

    # EdgeCV tracker forced onto the CPU ONNX runtime (no NPU on the RPi 4B).
    tcfg = cfg_tracker(config)
    assert tcfg.import_spec == "quadguide.perception.edgecv_adapter:EdgeCVTracker"
    assert tcfg.params["backend"] == "onnx"
    assert tcfg.params["tracker"] == "nanotrack"

    # Placeholder horizontal field of view (~60°) — set to the real lens spec.
    assert abs(config["guidance"]["fov_horizontal_rad"] - 1.05) < 1e-6

from pathlib import Path

from quadguide.core.config import load_config, cfg_platform

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "rk3588.yaml"


def test_rk3588_is_flight_default():
    """The committed rk3588.yaml is the flight default the boot service uses.
    HIL (tcp / raw_tcp) is a local, uncommitted toggle — see the file's comments.
    """
    config = load_config(str(CONFIG), {})
    pcfg = cfg_platform(config)

    # CSI OV9281 via GStreamer, full sensor downscaled to 640x400.
    assert pcfg.camera.backend == "gstreamer"
    assert "/dev/video11" in pcfg.camera.pipeline
    assert "format=BGR" in pcfg.camera.pipeline
    assert (pcfg.camera.width, pcfg.camera.height, pcfg.camera.fps) == (640, 400, 60)

    # MAVLink over the real UART (UART6-M1).
    assert pcfg.serial.mode == "uart"
    assert pcfg.serial.port == "/dev/ttyS6"

    # 79 degrees horizontal field of view.
    assert abs(config["guidance"]["fov_horizontal_rad"] - 1.379) < 1e-6


def test_rk3588_target_loss_disarm_present_disabled():
    """rk3588 ships the failsafe section for parity but leaves it off by default."""
    from quadguide.core.config import cfg_failsafe
    config = load_config(str(CONFIG), {})
    f = cfg_failsafe(config)
    assert f.disarm_on_lost is False
    assert f.lost_hold_ms == 300

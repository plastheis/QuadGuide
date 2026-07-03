from __future__ import annotations
import signal
import time

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import HealthReport, ProcessState
from quadguide.perception.camera.sources import (
    CameraSource, USBCamera, CSICamera, CSIY10Camera, VirtualCamera,
)
from quadguide.perception.camera.network_source import NetworkCamera
from quadguide.perception.camera.raw_frame_source import RawFrameCamera

__all__ = ["run", "run_from_config"]

_SOURCES = {
    "v4l2": USBCamera,
    "gstreamer": CSICamera,
    "csi": CSIY10Camera,         # OV9281 mono Y10 via direct V4L2 (ROCK 5C / rkcif)
    "virtual": VirtualCamera,
    "network": NetworkCamera,    # HIL: HTTP MJPEG from the dev machine
    "raw_tcp": RawFrameCamera,   # HIL: raw BGR frames over TCP (low-latency)
}
_HEALTH_EVERY = 60  # publish health every N frames


def run(source: CameraSource, frame_buffer: FrameBuffer, bus: Bus,
        config: dict | None = None) -> None:
    """Camera worker process entry point.

    Opens source, writes frames into frame_buffer, publishes system/health.
    Runs until SIGTERM sets the stop flag.
    """
    log = setup_logging("camera", config or {})
    stop = False

    def _on_sigterm(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    source.open()
    log.info("camera: source opened")
    i = 0
    try:
        while not stop:
            try:
                frame, ts = source.read()
            except RuntimeError:
                # A read can fail because SIGTERM interrupted a blocking capture
                # mid-shutdown — that's expected teardown, not a fault.
                if stop:
                    break
                raise
            frame_buffer.write_frame(frame, ts)
            i += 1
            if i % _HEALTH_EVERY == 0:
                bus.publish(
                    "system/health",
                    HealthReport(monotonic_ns(), "camera", ProcessState.OK, ""),
                )
    except Exception as exc:
        log.error(f"camera: fatal error: {exc}")
        try:
            bus.publish(
                "system/health",
                HealthReport(monotonic_ns(), "camera", ProcessState.FAILSAFE, str(exc)),
            )
        except Exception:
            pass
        raise
    finally:
        source.close()
        bus.detach()
        log.info("camera: stopped")


def run_from_config(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    """Construct a CameraSource from config and call run()."""
    from quadguide.core.config import cfg_platform
    pcfg = cfg_platform(config)
    source_cls = _SOURCES.get(pcfg.camera.backend)
    if source_cls is None:
        raise ValueError(
            f"Unknown camera backend {pcfg.camera.backend!r}. "
            f"Valid values: {sorted(_SOURCES)}"
        )
    run(source_cls(pcfg.camera), frame_buffer, bus, config)

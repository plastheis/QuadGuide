from __future__ import annotations

import uvicorn

from quadguide.core.bus import Bus
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.ground.server import create_app


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    port = config.get("ground", {}).get("port", 8080)
    app = create_app(bus, frame_buffer)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    bus.detach()

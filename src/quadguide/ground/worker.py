from __future__ import annotations

import uvicorn

from quadguide.core.bus import Bus
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.ground.server import create_app


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    port = config.get("ground", {}).get("port", 8080)
    log = setup_logging("ground", config)
    level = config.get("logging", {}).get("level", "INFO").lower()

    app = create_app(bus, frame_buffer, config)
    log.info("ground server listening on 0.0.0.0:%d", port)
    # log_config=None keeps uvicorn from replacing the root handlers installed by
    # setup_logging, so its own logs land in the same file/journal as everyone
    # else's. Access logging is per-request and follows the configured level.
    uvicorn.run(
        app, host="0.0.0.0", port=port,
        log_level=level, log_config=None, access_log=(level == "debug"),
    )
    bus.detach()

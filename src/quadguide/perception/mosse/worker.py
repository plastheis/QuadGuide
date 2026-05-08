from quadguide.core.bus import Bus
from quadguide.core.config import cfg_platform
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.perception.ccv_tracker_worker import CCVTrackerWorker
from quadguide.perception.mosse.tracker import MOSSETracker


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    pcfg    = cfg_platform(config)
    tracker = MOSSETracker()
    worker  = CCVTrackerWorker(
        tracker, bus, frame_buffer,
        cpu_core=pcfg.realtime.kcf_cpu_core,
        config=config,
    )
    worker.run()

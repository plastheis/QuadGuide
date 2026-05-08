from quadguide.core.bus import Bus
from quadguide.core.config import cfg_tracker, cfg_platform
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.perception.ccv_tracker_worker import CCVTrackerWorker
from quadguide.perception.kcf.tracker import KCFTracker


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    tcfg    = cfg_tracker(config)
    pcfg    = cfg_platform(config)
    tracker = KCFTracker(tcfg.kcf)
    worker  = CCVTrackerWorker(
        tracker, bus, frame_buffer,
        cpu_core=pcfg.realtime.kcf_cpu_core,
        config=config,
    )
    worker.run()

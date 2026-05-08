from quadguide.core.bus import Bus
from quadguide.core.config import cfg_tracker, cfg_platform
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.inference.factory import get_runtime
from quadguide.perception.ncv_tracker_worker import NCVTrackerWorker
from quadguide.perception.nanotrack.tracker import NanoTracker


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    tcfg     = cfg_tracker(config)
    pcfg     = cfg_platform(config)
    runtime  = get_runtime(config)
    backbone = runtime.load(pcfg.inference.backbone)
    head     = runtime.load(pcfg.inference.head)
    tracker  = NanoTracker(runtime, backbone, head, tcfg.nanotrack)
    worker   = NCVTrackerWorker(tracker, bus, frame_buffer, config=config)
    worker.run()

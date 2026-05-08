from __future__ import annotations

__all__ = ["CCV_TRACKERS", "NCV_TRACKERS", "get_ccv_tracker", "get_ncv_tracker"]

# To add a new CCV tracker: implement the class, add one entry here.
CCV_TRACKERS: dict[str, type] = {}
NCV_TRACKERS: dict[str, type] = {}


def _register_ccv() -> None:
    from quadguide.perception.kcf.tracker import KCFTracker
    from quadguide.perception.mosse.tracker import MOSSETracker
    CCV_TRACKERS["kcf"]   = KCFTracker
    CCV_TRACKERS["mosse"] = MOSSETracker


def _register_ncv() -> None:
    from quadguide.perception.nanotrack.tracker import NanoTracker
    NCV_TRACKERS["nanotrack"] = NanoTracker


_register_ccv()
_register_ncv()


def get_ccv_tracker(config: dict):
    """Return a constructed CCV tracker instance selected by config.tracker.ccv."""
    from quadguide.core.config import cfg_tracker
    tcfg = cfg_tracker(config)
    name = tcfg.ccv
    try:
        cls = CCV_TRACKERS[name]
    except KeyError:
        raise KeyError(f"Unknown ccv tracker {name!r}. Valid: {sorted(CCV_TRACKERS)}")
    if name == "kcf":
        return cls(tcfg.kcf)
    if name == "mosse":
        return cls()
    return cls()


def get_ncv_tracker(config: dict, runtime):
    """Return a constructed NCV tracker instance with a loaded runtime."""
    from quadguide.core.config import cfg_tracker, cfg_platform
    tcfg = cfg_tracker(config)
    pcfg = cfg_platform(config)
    name = tcfg.ncv
    try:
        cls = NCV_TRACKERS[name]
    except KeyError:
        raise KeyError(f"Unknown ncv tracker {name!r}. Valid: {sorted(NCV_TRACKERS)}")
    if name == "nanotrack":
        backbone = runtime.load(pcfg.inference.backbone)
        head     = runtime.load(pcfg.inference.head)
        return cls(runtime, backbone, head, tcfg.nanotrack)
    return cls()

from __future__ import annotations
from typing import Any, Callable

__all__ = ["CCV_TRACKERS", "NCV_TRACKERS", "get_ccv_tracker", "get_ncv_tracker"]


def _build_ccv_registry() -> dict[str, Callable]:
    from quadguide.perception.kcf.tracker import KCFTracker
    from quadguide.perception.mosse.tracker import MOSSETracker
    return {
        "kcf":   lambda tcfg: KCFTracker(tcfg.kcf),
        "mosse": lambda tcfg: MOSSETracker(),
    }


def _build_ncv_registry() -> dict[str, Callable]:
    from quadguide.perception.nanotrack.tracker import NanoTracker
    return {
        "nanotrack": lambda tcfg, pcfg, runtime: NanoTracker(
            runtime,
            runtime.load(pcfg.inference.backbone),
            runtime.load(pcfg.inference.head),
            tcfg.nanotrack,
        ),
    }


CCV_TRACKERS: dict[str, Callable] = _build_ccv_registry()
NCV_TRACKERS: dict[str, Callable] = _build_ncv_registry()


def get_ccv_tracker(config: dict) -> Any:
    """Return a constructed CCV tracker selected by config.tracker.ccv."""
    from quadguide.core.config import cfg_tracker
    tcfg = cfg_tracker(config)
    name = tcfg.ccv
    try:
        return CCV_TRACKERS[name](tcfg)
    except KeyError:
        raise KeyError(f"Unknown ccv tracker {name!r}. Valid: {sorted(CCV_TRACKERS)}")


def get_ncv_tracker(config: dict, runtime: Any) -> Any:
    """Return a constructed NCV tracker with a loaded runtime."""
    from quadguide.core.config import cfg_tracker, cfg_platform
    tcfg = cfg_tracker(config)
    pcfg = cfg_platform(config)
    name = tcfg.ncv
    try:
        return NCV_TRACKERS[name](tcfg, pcfg, runtime)
    except KeyError:
        raise KeyError(f"Unknown ncv tracker {name!r}. Valid: {sorted(NCV_TRACKERS)}")

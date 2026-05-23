from __future__ import annotations

from quadguide.core.config import GuidanceConfig
from quadguide.guidance.base import GuidanceMethod
from quadguide.guidance.pronav import ProNavGuidance
from quadguide.guidance.pure_pursuit import PurePursuitGuidance


def _make_pronav(gcfg: GuidanceConfig, aspect: float) -> GuidanceMethod:
    if gcfg.pronav is None:
        raise KeyError("guidance.method=pronav but guidance.pronav block missing")
    return ProNavGuidance(gcfg.pronav, gcfg.fov_horizontal_rad, aspect)


def _make_pure_pursuit(gcfg: GuidanceConfig, aspect: float) -> GuidanceMethod:
    if gcfg.pure_pursuit is None:
        raise KeyError("guidance.method=pure_pursuit but guidance.pure_pursuit block missing")
    return PurePursuitGuidance(gcfg.pure_pursuit, gcfg.fov_horizontal_rad, aspect)


METHODS = {
    "pronav":       _make_pronav,
    "pure_pursuit": _make_pure_pursuit,
}


def get_guidance(gcfg: GuidanceConfig, aspect: float) -> GuidanceMethod:
    try:
        builder = METHODS[gcfg.method]
    except KeyError:
        raise KeyError(
            f"unknown guidance.method {gcfg.method!r}; expected one of {sorted(METHODS)}"
        ) from None
    return builder(gcfg, aspect)

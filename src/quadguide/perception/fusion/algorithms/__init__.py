from __future__ import annotations

from .base import BaseFusion
from .confidence_weighted import ConfidenceWeightedFusion
from .iou_gated import IoUGatedFusion
from .passthrough import PassthroughFusion

__all__ = ["BaseFusion", "build_fusion_algorithm"]

_REGISTRY: dict[str, type[BaseFusion]] = {
    "confidence_weighted": ConfidenceWeightedFusion,
    "iou_gated": IoUGatedFusion,
    "passthrough": PassthroughFusion,
}


def build_fusion_algorithm(name: str) -> BaseFusion:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown fusion algorithm: {name!r}. Known: {sorted(_REGISTRY)}"
        )
    return cls()

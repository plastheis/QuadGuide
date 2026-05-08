from __future__ import annotations
from typing import Any, Protocol
import numpy as np

__all__ = ["NPURuntime"]


class NPURuntime(Protocol):
    """Structural protocol for all inference runtimes.

    All NanoTrack inference calls go through this interface. No tracker file
    imports onnxruntime, rknn, or any backend directly.
    """

    def load(self, path: str) -> Any:
        """Load a model file and return a model handle."""
        ...

    def infer(
        self, model: Any, inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Run inference. Returns output tensors keyed by name (ONNX) or
        positional index as string ("0", "1", ...) for RKNN."""
        ...

    def close(self) -> None:
        """Release any hardware resources held by this runtime."""
        ...

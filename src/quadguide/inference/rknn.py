from __future__ import annotations
from typing import Any
import numpy as np

__all__ = ["RKNNRuntime"]


class RKNNRuntime:
    """RKNN Lite runtime for RK3576/RK3588 NPU.

    On device: uses rknnlite (lightweight, no model conversion capability).
    On x86 sim: falls back to rknn-toolkit2's RKNN class for simulation.
    RKNN outputs are positional; keys in the returned dict are "0", "1", ...
    NanoTrack tracker accesses outputs via list(outputs.values()) in model order.
    """

    def load(self, path: str) -> Any:
        try:
            from rknnlite.api import RKNNLite
            rknn = RKNNLite()
        except ImportError:
            from rknn.api import RKNN
            rknn = RKNN()
        ret = rknn.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f"RKNNRuntime: load_rknn({path!r}) failed with code {ret}")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNNRuntime: init_runtime() failed with code {ret}")
        return rknn

    def infer(self, model: Any, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        input_list = list(inputs.values())
        outputs = model.inference(inputs=input_list)
        return {str(i): out for i, out in enumerate(outputs)}

    def close(self) -> None:
        pass  # RKNN handle released by GC; explicit release via model.release() if needed

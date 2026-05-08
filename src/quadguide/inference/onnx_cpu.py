from __future__ import annotations
from typing import Any
import numpy as np

__all__ = ["OnnxCPURuntime"]


class OnnxCPURuntime:
    """ONNX Runtime with CPU execution. Universal fallback on any platform."""

    def load(self, path: str) -> Any:
        import onnxruntime as ort
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def infer(self, model: Any, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        output_names = [o.name for o in model.get_outputs()]
        results = model.run(output_names, inputs)
        return dict(zip(output_names, results))

    def close(self) -> None:
        pass  # onnxruntime sessions are garbage-collected

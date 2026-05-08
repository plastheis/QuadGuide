from __future__ import annotations
from typing import Any
import numpy as np

__all__ = ["OnnxCUDARuntime"]


class OnnxCUDARuntime:
    """ONNX Runtime with CUDA execution. Requires onnxruntime-gpu and a CUDA GPU."""

    def load(self, path: str) -> Any:
        import onnxruntime as ort
        return ort.InferenceSession(
            path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def infer(self, model: Any, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        output_names = [o.name for o in model.get_outputs()]
        results = model.run(output_names, inputs)
        return dict(zip(output_names, results))

    def close(self) -> None:
        pass

from __future__ import annotations
from quadguide.inference.onnx_cpu import OnnxCPURuntime
from quadguide.inference.onnx_cuda import OnnxCUDARuntime
from quadguide.inference.rknn import RKNNRuntime

__all__ = ["RUNTIMES", "get_runtime"]

RUNTIMES = {
    "cpu":  OnnxCPURuntime,
    "cuda": OnnxCUDARuntime,
    "rknn": RKNNRuntime,
}


def get_runtime(config: dict):
    """Return a runtime instance selected by config["platform"]["inference"]["device"].

    Raises KeyError for unknown device strings — that is a configuration error.
    """
    device = config["platform"]["inference"]["device"]
    try:
        cls = RUNTIMES[device]
    except KeyError:
        raise KeyError(
            f"Unknown inference device {device!r}. Valid options: {sorted(RUNTIMES)}"
        )
    return cls()

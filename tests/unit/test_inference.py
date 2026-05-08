import pathlib
import pytest
import numpy as np
from quadguide.core.config import load_config
from quadguide.inference.base import NPURuntime
from quadguide.inference.onnx_cpu import OnnxCPURuntime
from quadguide.inference.factory import get_runtime, RUNTIMES

CONFIG_PATH = str(pathlib.Path(__file__).parents[2] / "configs" / "config.yaml")


class TestNPURuntimeProtocol:
    def test_onnx_cpu_has_load(self):
        assert hasattr(OnnxCPURuntime(), "load")

    def test_onnx_cpu_has_infer(self):
        assert hasattr(OnnxCPURuntime(), "infer")

    def test_onnx_cpu_has_close(self):
        assert hasattr(OnnxCPURuntime(), "close")


class TestInferenceFactory:
    def test_runtimes_dict_has_expected_keys(self):
        assert "cpu" in RUNTIMES
        assert "cuda" in RUNTIMES
        assert "rknn" in RUNTIMES

    def test_get_runtime_cpu_returns_onnx_cpu(self):
        config = load_config(CONFIG_PATH, {"platform.inference.device": "cpu"})
        runtime = get_runtime(config)
        assert isinstance(runtime, OnnxCPURuntime)

    def test_get_runtime_unknown_raises(self):
        config = load_config(CONFIG_PATH, {"platform.inference.device": "unknown"})
        with pytest.raises(KeyError):
            get_runtime(config)

    def test_close_is_no_op_for_cpu(self):
        runtime = OnnxCPURuntime()
        runtime.close()  # must not raise

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import torch.nn as nn  # noqa: E402
from convert_lib.harness import export_and_validate  # noqa: E402


class _TwoIn(nn.Module):
    def forward(self, a, b):
        return (a.mean() + b.mean()).reshape(1, 1, 1, 1)


def test_export_and_validate_parity(tmp_path):
    m = _TwoIn().eval()
    ex = (torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    out = tmp_path / "m.onnx"
    diff = export_and_validate(m, ex, ["a", "b"], ["y"], str(out))
    assert out.exists()
    assert diff < 1e-3

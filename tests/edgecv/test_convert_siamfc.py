import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from convert_lib import run  # noqa: E402
from convert_lib.adapters.siamfc import Net, build  # noqa: E402


def _save_random_ckpt(tmp_path):
    ckpt = tmp_path / "fake.pth"
    torch.save(Net().state_dict(), ckpt)
    return ckpt


def test_build_loads_strict_and_shapes(tmp_path):
    net = build(str(_save_random_ckpt(tmp_path)))
    z = torch.zeros(1, 3, 127, 127)
    x = torch.zeros(1, 3, 255, 255)
    with torch.no_grad():
        out = net(z, x)
    assert tuple(out.shape) == (1, 1, 17, 17)


def test_run_roundtrip_parity(tmp_path):
    # run() invokes the harness, which raises SystemExit unless torch-vs-onnxruntime
    # parity holds (max|delta| < 1e-3); reaching the shape assertion means parity passed.
    ckpt = _save_random_ckpt(tmp_path)
    out = tmp_path / "siamfc.onnx"
    run("siamfc_generic", str(ckpt), str(out))
    assert out.exists()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    z = rng.standard_normal((1, 3, 127, 127)).astype(np.float32)
    x = rng.standard_normal((1, 3, 255, 255)).astype(np.float32)
    got = sess.run(["score_map"], {"exemplar": z, "search": x})[0]
    assert got.shape == (1, 1, 17, 17)

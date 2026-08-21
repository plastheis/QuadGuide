import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from convert_lib import run  # noqa: E402
from convert_lib.adapters.nanotrack import Net, build  # noqa: E402


def _save_random_ckpt(tmp_path):
    ckpt = tmp_path / "fake.pth"
    torch.save(Net().state_dict(), ckpt)
    return ckpt


def test_build_loads_strict_and_shapes(tmp_path):
    net = build(str(_save_random_ckpt(tmp_path)))
    z = torch.zeros(1, 3, 127, 127)
    x = torch.zeros(1, 3, 255, 255)
    with torch.no_grad():
        cls, loc = net(z, x)
    assert cls.shape[:2] == (1, 2)
    assert loc.shape[:2] == (1, 4)
    assert cls.shape[2:] == loc.shape[2:]        # same S×S grid


def test_run_roundtrip_parity(tmp_path):
    # run() invokes the harness, which raises SystemExit unless torch-vs-onnxruntime
    # parity holds on the first output (cls); reaching the assertions means it passed.
    ckpt = _save_random_ckpt(tmp_path)
    out = tmp_path / "nanotrack.onnx"
    run("nanotrack", str(ckpt), str(out))
    assert out.exists()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    z = rng.standard_normal((1, 3, 127, 127)).astype(np.float32)
    x = rng.standard_normal((1, 3, 255, 255)).astype(np.float32)
    cls, loc = sess.run(["cls", "loc"], {"exemplar": z, "search": x})
    assert cls.shape[:2] == (1, 2)
    assert loc.shape[:2] == (1, 4)
    assert cls.shape[2:] == loc.shape[2:]

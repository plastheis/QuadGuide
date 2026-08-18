import os

from edgecv.models.paths import (
    apply_rknn_target,
    resolve_artifact_path,
    rknn_target,
)


def test_absolute_path_passes_through(tmp_path):
    p = tmp_path / "model.onnx"
    assert resolve_artifact_path(str(p)) == str(p)


def test_relative_resolves_against_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EDGECV_MODEL_DIR", str(tmp_path))
    assert resolve_artifact_path("siamfc_generic.onnx") == str(tmp_path / "siamfc_generic.onnx")


def test_relative_default_is_models_dir(monkeypatch):
    monkeypatch.delenv("EDGECV_MODEL_DIR", raising=False)
    expected = os.path.join("models", "siamfc_generic.onnx")
    assert resolve_artifact_path("siamfc_generic.onnx") == expected


def test_rknn_target_env_override_wins(monkeypatch):
    monkeypatch.setenv("EDGECV_RKNN_TARGET", "rk3566")
    assert rknn_target() == "rk3566"


def test_apply_rknn_target_fills_token(monkeypatch):
    monkeypatch.setenv("EDGECV_RKNN_TARGET", "rk3566")
    assert apply_rknn_target("nanotrack_quant_{target}/head.rknn") == (
        "nanotrack_quant_rk3566/head.rknn"
    )


def test_apply_rknn_target_noop_without_token(monkeypatch):
    monkeypatch.setenv("EDGECV_RKNN_TARGET", "rk3566")
    # onnx / target-agnostic paths carry no token and pass through unchanged.
    assert apply_rknn_target("nanotrack_head.onnx") == "nanotrack_head.onnx"


def test_rknn_target_default_when_unset_and_no_device_tree(monkeypatch):
    monkeypatch.delenv("EDGECV_RKNN_TARGET", raising=False)
    # Force the device-tree read to fail so we exercise the default fallback
    # deterministically (CI/host has no /proc/device-tree/compatible anyway).
    import edgecv.models.paths as paths

    real_read = paths.Path.read_bytes

    def boom(self, *a, **k):
        if str(self) == "/proc/device-tree/compatible":
            raise OSError("no device tree")
        return real_read(self, *a, **k)

    monkeypatch.setattr(paths.Path, "read_bytes", boom)
    assert rknn_target() == "rk3588"

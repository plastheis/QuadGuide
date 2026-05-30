import pytest

from quadguide.perception.tracker_worker import load_tracker


def _cfg(import_spec: str, params: dict | None = None) -> dict:
    return {"tracker": {"import": import_spec, "params": params or {}}}


class TestLoadTrackerErrors:
    def test_missing_colon_raises_value_error(self):
        with pytest.raises(ValueError, match="module:Class"):
            load_tracker(_cfg("kcf"))

    def test_empty_module_raises_value_error(self):
        with pytest.raises(ValueError, match="module:Class"):
            load_tracker(_cfg(":Whatever"))

    def test_empty_class_raises_value_error(self):
        with pytest.raises(ValueError, match="module:Class"):
            load_tracker(_cfg("somepkg:"))

    def test_missing_module_raises_import_error(self):
        with pytest.raises(ImportError):
            load_tracker(_cfg("nonexistent_pkg_xyz:Whatever"))


class TestLoadTrackerExternal:
    def test_external_class_constructed_with_params(self, tmp_path, monkeypatch):
        pkg = tmp_path / "stub_tracker_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "tracker.py").write_text(
            "class StubTracker:\n"
            "    def __init__(self, **kw): self.kw = kw\n"
            "    def name(self): return 'stub'\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        cfg = _cfg("stub_tracker_pkg.tracker:StubTracker", {"foo": 1, "bar": "x"})
        tracker = load_tracker(cfg)
        assert tracker.name() == "stub"
        assert tracker.kw == {"foo": 1, "bar": "x"}

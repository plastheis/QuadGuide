import pytest

from edgecv.backends.registry import (
    BackendNotFoundError,
    available_backends,
    get_backend,
    list_backends,
)


def test_list_backends_includes_builtins():
    names = list_backends()
    assert {"mock", "onnx", "rknn"} <= set(names)


def test_get_backend_returns_singleton_like_instance():
    b1 = get_backend("mock")
    b2 = get_backend("mock")
    assert b1.name == "mock"
    assert b1 is b2  # cached


def test_unknown_backend_raises():
    with pytest.raises(BackendNotFoundError):
        get_backend("does-not-exist")


def test_available_backends_subset_of_all():
    assert set(available_backends()) <= set(list_backends())
    assert "mock" in available_backends()  # mock is always available

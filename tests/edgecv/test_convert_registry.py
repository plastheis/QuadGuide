import pytest
from convert_lib.registry import Adapter, get, register, registered_names


def test_register_and_get():
    a = Adapter(name="dummy_reg", build=lambda ckpt: None)
    register(a)
    assert get("dummy_reg") is a
    assert "dummy_reg" in registered_names()


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get("definitely_not_registered")

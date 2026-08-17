"""Lazy, entry-point-driven backend registry (ARCHITECTURE.md §10).

Backends register under the `edgecv.backends` entry-point group and are imported
lazily, so a missing vendor runtime only errors when that backend is used. A
built-in fallback map covers the case where the package metadata is unavailable.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from edgecv.backends.base import InferenceBackend

_BUILTIN: dict[str, str] = {
    "mock": "edgecv.backends.mock:MockBackend",
    "onnx": "edgecv.backends.onnx:OnnxBackend",
    "rknn": "edgecv.backends.rknn:RknnBackend",
}

_instances: dict[str, InferenceBackend] = {}


class BackendNotFoundError(KeyError):
    pass


def _entry_point_targets() -> dict[str, str]:
    targets: dict[str, str] = dict(_BUILTIN)
    try:
        for ep in entry_points(group="edgecv.backends"):
            targets[ep.name] = ep.value
    except Exception:
        # Metadata unavailable (e.g. source tree without install) — builtins suffice.
        pass
    return targets


def list_backends() -> list[str]:
    return sorted(_entry_point_targets())


def _load_class(target: str) -> type[InferenceBackend]:
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, attr)
    if not issubclass(cls, InferenceBackend):
        raise TypeError(f"{target} is not an InferenceBackend")
    return cls


def get_backend(name: str) -> InferenceBackend:
    if name in _instances:
        return _instances[name]
    targets = _entry_point_targets()
    if name not in targets:
        raise BackendNotFoundError(name)
    instance = _load_class(targets[name])()
    _instances[name] = instance
    return instance


def available_backends() -> list[str]:
    """Names whose runtime is importable and usable on this machine."""
    out: list[str] = []
    for name in list_backends():
        try:
            if get_backend(name).is_available():
                out.append(name)
        except Exception:
            continue
    return out

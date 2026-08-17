"""Adapter registry (torch-free). Each model contributes one Adapter; adapters
self-register on import (see adapters/__init__.py)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Adapter:
    name: str                                       # manifest model name, e.g. "siamfc_generic"
    build: Callable[[str], Any] | None = None       # torch path: checkpoint -> .eval() nn.Module
    export: Callable[..., str] | None = None        # upstream-exporter path (e.g. ultralytics yolo)
    dynamic_axes: dict | None = None                # optional; variable dims (e.g. YOLO det count)


_REGISTRY: dict[str, Adapter] = {}


def register(adapter: Adapter) -> None:
    _REGISTRY[adapter.name] = adapter


def get(name: str) -> Adapter:
    return _REGISTRY[name]


def registered_names() -> list[str]:
    return sorted(_REGISTRY)

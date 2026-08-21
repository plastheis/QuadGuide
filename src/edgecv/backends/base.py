"""Hardware-abstraction interfaces (ARCHITECTURE.md §10). Trackers depend on
these, never on a vendor runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from edgecv.models.manifest import ModelManifest


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]          # -1 for dynamic dims
    dtype: str                      # numpy dtype name, e.g. "float32", "int8"
    layout: str = "NCHW"            # informational
    quant: dict | None = None       # {"scale": ..., "zero_point": ...} for INT8, else None


@dataclass(frozen=True)
class IOSpec:
    inputs: tuple[TensorSpec, ...] = field(default_factory=tuple)
    outputs: tuple[TensorSpec, ...] = field(default_factory=tuple)


class Handle(ABC):
    """Async inference handle (NPUs pipeline). `wait` blocks for the result."""

    @abstractmethod
    def wait(self) -> dict[str, np.ndarray]: ...


class Model(ABC):
    @property
    @abstractmethod
    def io_spec(self) -> IOSpec: ...

    @abstractmethod
    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

    def infer_async(self, inputs: dict[str, np.ndarray]) -> Handle:
        """Optional. Default raises; backends that pipeline override this."""
        raise NotImplementedError(f"{type(self).__name__} does not support infer_async")

    @abstractmethod
    def close(self) -> None: ...


class InferenceBackend(ABC):
    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """True if this backend's runtime can actually load and run a model here."""

    @abstractmethod
    def load(self, manifest: ModelManifest) -> Model: ...

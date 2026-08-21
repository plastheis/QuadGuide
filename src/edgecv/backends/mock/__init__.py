"""Mock backend: canned, manifest-shaped outputs. Lets the full runtime/IPC/
fusion stack run with no model and no accelerator (ARCHITECTURE.md §10)."""

from __future__ import annotations

import numpy as np

from edgecv.backends.base import Handle, InferenceBackend, IOSpec, Model, TensorSpec
from edgecv.models.manifest import ModelManifest


def _specs(entries: list[dict]) -> tuple[TensorSpec, ...]:
    return tuple(
        TensorSpec(
            name=e["name"],
            shape=tuple(e["shape"]),
            dtype=e.get("dtype", "float32"),
            layout=e.get("layout", "NCHW"),
            quant=e.get("quant"),
        )
        for e in entries
    )


def _concrete_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    # Replace dynamic (-1) dims with 1 so a concrete array can be produced.
    return tuple(d if d > 0 else 1 for d in shape)


class _ImmediateHandle(Handle):
    def __init__(self, result: dict[str, np.ndarray]):
        self._result = result

    def wait(self) -> dict[str, np.ndarray]:
        return self._result


class MockModel(Model):
    def __init__(self, io_spec: IOSpec):
        self._io_spec = io_spec

    @property
    def io_spec(self) -> IOSpec:
        return self._io_spec

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            spec.name: np.zeros(_concrete_shape(spec.shape), dtype=np.dtype(spec.dtype))
            for spec in self._io_spec.outputs
        }

    def infer_async(self, inputs: dict[str, np.ndarray]) -> Handle:
        return _ImmediateHandle(self.infer(inputs))

    def close(self) -> None:  # nothing to release
        pass


class MockBackend(InferenceBackend):
    name = "mock"

    def is_available(self) -> bool:
        return True

    def load(self, manifest: ModelManifest) -> Model:
        io_spec = IOSpec(inputs=_specs(manifest.inputs), outputs=_specs(manifest.outputs))
        return MockModel(io_spec)

"""ONNXRuntime CPU backend (ARCHITECTURE.md §10). Lazy import: onnxruntime is an
optional extra, imported only when this backend is actually loaded."""

from __future__ import annotations

import numpy as np

from edgecv.backends.base import InferenceBackend, IOSpec, Model, TensorSpec
from edgecv.models.manifest import ModelManifest
from edgecv.models.paths import resolve_artifact_path

# ONNX tensor element type -> numpy dtype name (the common subset).
_ORT_TO_NP = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(float16)": "float16",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(int8)": "int8",
    "tensor(uint8)": "uint8",
}


def _dims(shape: list) -> tuple[int, ...]:
    return tuple(d if isinstance(d, int) and d > 0 else -1 for d in shape)


class OnnxModel(Model):
    def __init__(self, session):
        self._session = session
        inputs = tuple(
            TensorSpec(name=i.name, shape=_dims(i.shape),
                       dtype=_ORT_TO_NP.get(i.type, "float32"))
            for i in session.get_inputs()
        )
        outputs = tuple(
            TensorSpec(name=o.name, shape=_dims(o.shape),
                       dtype=_ORT_TO_NP.get(o.type, "float32"))
            for o in session.get_outputs()
        )
        self._io_spec = IOSpec(inputs=inputs, outputs=outputs)

    @property
    def io_spec(self) -> IOSpec:
        return self._io_spec

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        names = [o.name for o in self._io_spec.outputs]
        results = self._session.run(names, inputs)
        return dict(zip(names, results, strict=False))

    def close(self) -> None:
        self._session = None


class OnnxBackend(InferenceBackend):
    name = "onnx"

    def is_available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, manifest: ModelManifest) -> Model:
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - covered by is_available path
            raise RuntimeError(
                "onnxruntime is not installed; install with `pip install edgecv[onnx]`"
            ) from e
        artifact = manifest.artifacts.get("onnx")
        if not artifact or "path" not in artifact:
            raise ValueError(f"manifest {manifest.name!r} has no onnx artifact path")
        session = ort.InferenceSession(
            resolve_artifact_path(artifact["path"]), providers=["CPUExecutionProvider"]
        )
        return OnnxModel(session)

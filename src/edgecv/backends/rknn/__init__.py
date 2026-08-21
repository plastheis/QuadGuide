"""RKNN backend adapter (ARCHITECTURE.md §10). Lazy: rknn-toolkit-lite2 is NOT on
PyPI and is installed manually on-device. This adapter reports unavailability
cleanly off-device and raises an actionable error if used without the runtime.
It must be initialised inside the worker process, never the parent."""

from __future__ import annotations

from typing import Any

from edgecv.backends.base import (
    InferenceBackend,
    IOSpec,
    Model,
    TensorSpec,
)
from edgecv.models.manifest import ModelManifest
from edgecv.models.paths import apply_rknn_target, resolve_artifact_path

_INSTALL_HINT = (
    "rknn-toolkit-lite2 is not available. It is not on PyPI; install it manually "
    "on the Rockchip device (see README RKNN note). The [rknn] extra only registers "
    "this adapter."
)


def _import_rknnlite():
    from rknnlite.api import RKNNLite  # type: ignore

    return RKNNLite


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


class RknnModel(Model):
    """Wraps RKNNLite. Built INSIDE the using process only (ARCHITECTURE §14.7)."""

    def __init__(self, rknn: Any, io_spec: IOSpec, output_order: list[str]) -> None:
        self._rknn: Any = rknn
        self._io_spec = io_spec
        self._output_order = output_order
        # RKNNLite.inference defaults a 4-D input's data_format to NHWC and will
        # silently transpose whatever buffer it is handed to match — so feeding an
        # NCHW tensor without declaring its layout corrupts the data (channels get
        # read as spatial). Declare the inputs' layout so the runtime keeps the
        # buffer intact. (Empirically: omitting this drops backbone-vs-ONNX
        # correlation to ~0.38 and breaks tracking; declaring 'nchw' restores 1.0.)
        # RKNNLite takes a single data_format string applied to every input (a
        # list raises "Unsupport data format"), so all inputs must share a layout.
        layouts = {s.layout.lower() for s in io_spec.inputs}
        if len(layouts) > 1:
            raise ValueError(
                f"rknn backend needs a single input layout, got {sorted(layouts)}"
            )
        self._data_format = layouts.pop() if layouts else "nchw"

    @property
    def io_spec(self) -> IOSpec:
        return self._io_spec

    def infer(self, inputs: dict) -> dict:
        ordered = [inputs[s.name] for s in self._io_spec.inputs]
        results = self._rknn.inference(inputs=ordered, data_format=self._data_format)
        # RKNNLite returns outputs positionally; manifest.outputs must match the
        # compiled model's output order exactly (RKNNLite has no output-name API).
        # strict=True surfaces a count mismatch loudly instead of truncating.
        return dict(zip(self._output_order, results, strict=True))

    def close(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None


class RknnBackend(InferenceBackend):
    name = "rknn"

    def is_available(self) -> bool:
        try:
            _import_rknnlite()
        except Exception:
            return False
        return True

    def load(self, manifest: ModelManifest) -> Model:
        try:
            rknn_lite = _import_rknnlite()
        except Exception as e:
            raise RuntimeError(_INSTALL_HINT) from e
        artifact = manifest.artifacts.get("rknn")
        if not artifact or "path" not in artifact:
            raise ValueError(f"manifest {manifest.name!r} has no rknn artifact path")
        rknn = rknn_lite()
        # rknn blobs are per-SoC: fill the path's {target} token (if any) with the
        # active compile target before resolving against the model dir.
        resolved = resolve_artifact_path(apply_rknn_target(artifact["path"]))
        if rknn.load_rknn(resolved) != 0:
            raise RuntimeError(f"failed to load rknn model {resolved!r}")
        core_mask = artifact.get("npu_core") or 0
        if rknn.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError(f"rknn init_runtime failed for {resolved!r}")
        io_spec = IOSpec(
            inputs=_specs(manifest.inputs),
            outputs=_specs(manifest.outputs),
        )
        return RknnModel(rknn, io_spec, [o["name"] for o in manifest.outputs])

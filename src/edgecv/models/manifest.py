"""Model manifest schema + loader (ARCHITECTURE.md §10.1).

A manifest maps one logical model to per-backend artifacts plus preprocessing and
an I/O spec. Trackers depend on the manifest, never on a vendor artifact file.
I/O entries are kept as plain dicts here; backends translate them into TensorSpec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelManifest:
    name: str
    task: str
    preprocessing: dict = field(default_factory=dict)
    inputs: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    artifacts: dict[str, dict] = field(default_factory=dict)

    def artifact_for(self, backend: str) -> dict | None:
        return self.artifacts.get(backend)


def load_manifest(path: str | Path) -> ModelManifest:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} is not a mapping")
    name = data.get("name")
    task = data.get("task")
    if not name or not task:
        raise ValueError(f"manifest {path} must define 'name' and 'task'")
    io = data.get("io") or {}
    return ModelManifest(
        name=name,
        task=task,
        preprocessing=data.get("preprocessing") or {},
        inputs=io.get("inputs") or [],
        outputs=io.get("outputs") or [],
        artifacts=data.get("artifacts") or {},
    )

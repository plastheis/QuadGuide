"""RKNN (and backend-agnostic) split-model loading for NanoTrack.

NanoTrack is a split backbone+head architecture; the backbone and head are two
separate model artifacts. These tests cover NanoTrack.from_manifest, the loader
that builds both sub-models for a chosen backend (rknn on-device, onnx on the
host) from one logical manifest — so the rknn path "functions just like the onnx
version" without re-importing rknn-toolkit-lite2 in CI (the backend is injected).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from edgecv.backends.base import InferenceBackend, IOSpec, TensorSpec
from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.models.manifest import ModelManifest, load_manifest
from edgecv.trackers.nn.nanotrack import NanoTrack
from tests.edgecv._nn_stubs import ScriptedModel, nano_head_out, nano_z_feat

MANIFEST = Path(__file__).resolve().parents[2] / "src/edgecv/models/manifests/nanotrack.yaml"
S = 15  # score size


def _io(entries: list[dict]) -> tuple[TensorSpec, ...]:
    return tuple(TensorSpec(e["name"], tuple(e["shape"]), e.get("dtype", "float32"))
                 for e in entries)


class RecordingBackend(InferenceBackend):
    """Stand-in for the rknn backend: records each sub-manifest it loads and
    returns a scripted model with non-degenerate, manifest-shaped outputs (a
    centred cls peak + constant loc distances), so a full track runs cleanly."""

    name = "rknn"

    def __init__(self) -> None:
        self.loaded: list[ModelManifest] = []

    def is_available(self) -> bool:
        return True

    def load(self, manifest: ModelManifest):
        self.loaded.append(manifest)
        io = IOSpec(inputs=_io(manifest.inputs), outputs=_io(manifest.outputs))
        if len(manifest.inputs) == 1:          # backbone: one image -> feature
            outputs = [{manifest.outputs[0]["name"]: nano_z_feat()}]
        else:                                  # head: z_f, x_f -> cls, loc
            out = nano_head_out(S, S // 2, S // 2)
            outputs = [{manifest.outputs[0]["name"]: out["output1"],
                        manifest.outputs[1]["name"]: out["output2"]}]
        return ScriptedModel(io, outputs)


def _frame(h=240, w=320):
    return np.zeros((h, w, 3), np.uint8)


def _box():
    return BoundingBox(x=(160 - 20) / 320, y=(120 - 20) / 240, w=40 / 320, h=40 / 240)


def test_manifest_carries_split_rknn_artifacts_and_io():
    mf = load_manifest(MANIFEST)
    for key, in_names, out_names in (
        ("backbone", ["input"], ["output"]),
        ("head", ["input1", "input2"], ["output1", "output2"]),
    ):
        art = mf.artifacts[key]
        assert art["rknn"]["path"].endswith(".rknn")
        io = art["io"]
        assert [i["name"] for i in io["inputs"]] == in_names
        assert [o["name"] for o in io["outputs"]] == out_names


def test_from_manifest_loads_rknn_artifact_paths():
    be = RecordingBackend()
    NanoTrack.from_manifest(MANIFEST, backend="rknn", backend_obj=be)
    paths = sorted(m.artifacts["rknn"]["path"] for m in be.loaded)
    assert [Path(p).name for p in paths] == [
        "nanotrack_backbone_yolocrop.rknn",
        "nanotrack_head.rknn",
    ]


def test_from_manifest_builds_submodels_with_correct_io():
    be = RecordingBackend()
    NanoTrack.from_manifest(MANIFEST, backend="rknn", backend_obj=be)
    by_name = {m.name: m for m in be.loaded}
    bb = next(m for m in be.loaded if [i["name"] for i in m.inputs] == ["input"])
    hd = next(m for m in be.loaded if len(m.inputs) == 2)
    assert tuple(bb.outputs[0]["shape"]) == (1, 96, 16, 16)
    assert [i["name"] for i in hd.inputs] == ["input1", "input2"]
    assert [o["name"] for o in hd.outputs] == ["output1", "output2"]
    assert len(by_name) == 2  # backbone + head are distinct sub-manifests


def test_from_manifest_rknn_tracker_runs_end_to_end():
    # Build via the rknn path (backend injected) and prove it tracks like onnx:
    # init builds an 8x8 exemplar feature, update returns a real TrackResult.
    t = NanoTrack.from_manifest(MANIFEST, backend="rknn", backend_obj=RecordingBackend())
    assert t.name() == "NanoTrack"
    t.init(_frame(), _box())
    assert t.get_template().arrays["exemplar"].shape == (1, 96, 8, 8)
    res = t.update(_frame())
    assert isinstance(res.bbox, BoundingBox)
    assert res.confidence is not None
    assert res.seq == 1
    # all-zero mock cls -> fg prob 0.5 everywhere -> COASTING band.
    assert res.status in (TrackStatus.LOCKED, TrackStatus.COASTING, TrackStatus.LOST)


def test_from_manifest_passes_through_tracker_kwargs():
    t = NanoTrack.from_manifest(
        MANIFEST, backend="rknn", backend_obj=RecordingBackend(),
        score_lock=0.9, window_influence=0.1,
    )
    assert t._score_lock == 0.9
    assert t._window_influence == 0.1

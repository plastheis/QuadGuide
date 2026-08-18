import textwrap

import pytest

from edgecv.models.manifest import ModelManifest, load_manifest


def test_load_manifest_parses_yaml(tmp_path):
    p = tmp_path / "siamfc.yaml"
    p.write_text(textwrap.dedent("""
        name: siamfc_generic
        task: sot_template_matching
        preprocessing: {color: gray, exemplar: 127, search: 255}
        io:
          inputs:
            - {name: exemplar, shape: [1, 1, 127, 127], dtype: float32}
            - {name: search, shape: [1, 1, 255, 255], dtype: float32}
          outputs:
            - {name: score_map, shape: [1, 1, 17, 17], dtype: float32}
        artifacts:
          onnx: {path: siamfc_generic.onnx}
          rknn: {path: siamfc_generic.rk3588.rknn, quant: int8}
    """))
    man = load_manifest(p)
    assert man.name == "siamfc_generic"
    assert man.task == "sot_template_matching"
    assert man.inputs[0]["name"] == "exemplar"
    assert man.outputs[0]["shape"] == [1, 1, 17, 17]
    assert man.artifacts["rknn"]["quant"] == "int8"
    assert man.preprocessing["color"] == "gray"


def test_manifest_requires_name_and_task(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("io: {inputs: [], outputs: []}\nartifacts: {}\n")
    with pytest.raises(ValueError):
        load_manifest(p)


def test_artifact_for_backend_helper():
    man = ModelManifest(name="m", task="t", preprocessing={},
                        inputs=[], outputs=[],
                        artifacts={"onnx": {"path": "m.onnx"}})
    assert man.artifact_for("onnx") == {"path": "m.onnx"}
    assert man.artifact_for("rknn") is None

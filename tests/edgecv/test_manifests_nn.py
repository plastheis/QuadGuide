from pathlib import Path

from edgecv.models.manifest import load_manifest

MANIFESTS = Path("src/edgecv/models/manifests")


def test_siamfc_manifest_loads():
    m = load_manifest(MANIFESTS / "siamfc_generic.yaml")
    assert m.name == "siamfc_generic"
    assert {i["name"] for i in m.inputs} == {"exemplar", "search"}
    assert m.outputs[0]["name"] == "score_map"
    assert m.preprocessing["color"] == "rgb"


def test_yolo26n_manifest_loads():
    m = load_manifest(MANIFESTS / "yolo26n.yaml")
    assert m.name == "yolo26n"
    assert m.task == "detection"
    assert m.preprocessing["class_agnostic"] is True
    assert m.preprocessing["output_format"] == "yolov8"
    assert m.outputs[0]["shape"] == [1, 84, -1]


def test_yolo26s_manifest_loads():
    m = load_manifest(MANIFESTS / "yolo26s.yaml")
    assert m.name == "yolo26s"
    assert m.preprocessing["output_format"] == "yolov8"
    assert m.artifacts["onnx"]["path"] == "yolo26s.onnx"


def test_yolo_generic_is_retired():
    assert not (MANIFESTS / "yolo_generic.yaml").exists()


def test_yolo11n_manifest_loads():
    m = load_manifest(MANIFESTS / "yolo11n.yaml")
    assert m.name == "yolo11n"
    assert m.task == "detection"
    assert m.preprocessing["class_agnostic"] is True
    # rknn_model_zoo separated P2/P3/P4 head, decoded by decode_yolo_dfl.
    assert m.preprocessing["output_format"] == "rknn_dfl"
    assert m.preprocessing["strides"] == [4, 8, 16]
    assert m.preprocessing["reg_max"] == 16
    assert m.preprocessing["scale"] == 1.0   # raw 0-255 (norm baked into the rknn)
    # 9 outputs: 3 scales × {box, cls, score_sum}; first is the P2 box DFL tensor.
    assert len(m.outputs) == 9
    assert m.outputs[0]["shape"] == [1, 64, 160, 160]
    # YOLO gets its own NPU core (AcquireTrack two-core placement).
    assert m.artifacts["rknn"]["npu_core"] == 1
    assert m.artifacts["rknn"]["path"] == "yolo11n_p2p3p4_{target}_i8.rknn"
    assert m.artifacts["rknn"]["quant"] == "int8"


def test_nanotrack_has_npu_core():
    m = load_manifest(MANIFESTS / "nanotrack.yaml")
    # Both split halves pinned to the NanoTrack NPU core (head shares the backbone's).
    assert m.artifacts["backbone"]["rknn"]["npu_core"] == 2
    assert m.artifacts["head"]["rknn"]["npu_core"] == 2


def test_nanotrack_manifest_loads():
    m = load_manifest(MANIFESTS / "nanotrack.yaml")
    assert m.name == "nanotrack"
    assert m.task == "sot_template_matching"
    assert {i["name"] for i in m.inputs} == {"exemplar", "search"}
    assert [o["name"] for o in m.outputs] == ["cls", "loc"]
    assert m.preprocessing["penalty_k"] == 0.138
    assert m.preprocessing["window_influence"] == 0.455
    # Split two-model artifacts: each half carries per-backend paths + its own io.
    bb, hd = m.artifacts["backbone"], m.artifacts["head"]
    assert bb["onnx"]["path"] == "nanotrackv3_backbone.onnx"
    assert bb["rknn"]["path"] == "nanotrack_quant_{target}/nanotrack_backbone_yolocrop.rknn"
    assert bb["rknn"]["quant"] == "int8"
    assert hd["onnx"]["path"] == "nanotrackv3_head.onnx"
    assert hd["rknn"]["path"] == "nanotrack_quant_{target}/nanotrack_head.rknn"
    assert hd["rknn"]["quant"] == "fp16"
    assert [o["name"] for o in hd["io"]["outputs"]] == ["output1", "output2"]

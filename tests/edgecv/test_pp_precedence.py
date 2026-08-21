from edgecv.trackers.nn.base import UNSET, manifest_preprocessing, resolve_pp


def test_explicit_kwarg_wins_over_manifest_and_default():
    assert resolve_pp("rgb", {"color": "gray"}, "color", "bgr") == "rgb"


def test_manifest_wins_over_default_when_unset():
    assert resolve_pp(UNSET, {"color": "gray"}, "color", "rgb") == "gray"


def test_default_when_unset_and_absent_from_manifest():
    assert resolve_pp(UNSET, {}, "color", "rgb") == "rgb"


def test_manifest_preprocessing_none_is_empty():
    assert manifest_preprocessing(None) == {}


def test_manifest_preprocessing_reads_yaml():
    pp = manifest_preprocessing("src/edgecv/models/manifests/yolo26n.yaml")
    assert pp["output_format"] == "yolov8"

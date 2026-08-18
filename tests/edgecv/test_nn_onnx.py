import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from edgecv.core.bbox import BoundingBox  # noqa: E402
from edgecv.core.result import TrackResult, TrackStatus  # noqa: E402
from edgecv.models.manifest import load_manifest  # noqa: E402
from edgecv.trackers.nn.siamfc import SiamFC  # noqa: E402
from edgecv.trackers.nn.yolo import YoloDetector, YoloTracker  # noqa: E402
from tests.edgecv._onnx_synth import build_siamfc_onnx, build_yolo_onnx, build_yolo_onnx_v8  # noqa: E402

FH, FW = 240, 320


def _manifest_with_artifact(yaml_path, onnx_path):
    m = load_manifest(yaml_path)
    m.artifacts["onnx"] = {"path": str(onnx_path)}
    return m


def test_siamfc_runs_through_onnx_backend(tmp_path):
    model_path = tmp_path / "siamfc.onnx"
    build_siamfc_onnx(str(model_path))
    mf = _manifest_with_artifact("src/edgecv/models/manifests/siamfc_generic.yaml", model_path)
    with SiamFC(mf, backend="onnx") as t:
        box = BoundingBox(x=(160 - 20) / FW, y=(120 - 20) / FH, w=40 / FW, h=40 / FH)
        t.init(np.zeros((FH, FW, 3), np.uint8), box)
        res = t.update(np.random.default_rng(0).integers(0, 256, (FH, FW, 3), np.uint8))
    assert isinstance(res, TrackResult)
    assert 0.0 <= res.bbox.w <= 1.0 and res.seq == 1
    assert isinstance(res.status, TrackStatus)


def test_yolo_detector_runs_through_onnx_backend(tmp_path):
    model_path = tmp_path / "yolo.onnx"
    build_yolo_onnx(str(model_path), n=64, num=3, nc=1)
    mf = _manifest_with_artifact("src/edgecv/models/manifests/yolo26n.yaml", model_path)
    det = YoloDetector(mf, backend="onnx", input_size=64, conf_thresh=0.25,
                       output_format="yolov5")
    out = det.detect(np.zeros((64, 64, 3), np.uint8))
    det.close()
    assert out.boxes.shape[1] == 4
    assert len(out.scores) == 2          # two synthetic dets, no overlap -> both survive NMS
    assert out.boxes.min() >= 0.0


def test_yolo_detector_v8_runs_through_onnx_backend(tmp_path):
    model_path = tmp_path / "yolo_v8.onnx"
    build_yolo_onnx_v8(str(model_path), n=64, num=3, nc=1)
    # yolo26n manifest declares output_format: yolov8 — no override, so the decode
    # path is driven by the manifest end-to-end through ORT.
    mf = _manifest_with_artifact("src/edgecv/models/manifests/yolo26n.yaml", model_path)
    det = YoloDetector(mf, backend="onnx", input_size=64)
    out = det.detect(np.zeros((64, 64, 3), np.uint8))
    det.close()
    assert out.boxes.shape[1] == 4
    assert len(out.scores) == 2          # two synthetic dets, no overlap -> both survive NMS
    assert out.boxes.min() >= 0.0


def test_yolo_tracker_runs_through_onnx_backend(tmp_path):
    model_path = tmp_path / "yolo.onnx"
    build_yolo_onnx(str(model_path), n=64, num=3, nc=1)
    mf = _manifest_with_artifact("src/edgecv/models/manifests/yolo26n.yaml", model_path)
    with YoloTracker(mf, backend="onnx", input_size=64, conf_thresh=0.25,
                     output_format="yolov5") as t:
        box = BoundingBox(x=(160 - 20) / FW, y=(120 - 20) / FH, w=40 / FW, h=40 / FH)
        t.init(np.zeros((FH, FW, 3), np.uint8), box)
        res = t.update(np.zeros((FH, FW, 3), np.uint8))
    assert isinstance(res, TrackResult)
    assert res.status in (TrackStatus.LOCKED, TrackStatus.COASTING)
    assert res.seq == 1

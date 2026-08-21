import numpy as np
import pytest

from edgecv.backends.base import IOSpec, TensorSpec
from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.models.manifest import ModelManifest
from edgecv.trackers.nn.yolo import YoloDetector, YoloTracker
from tests.edgecv._nn_stubs import ScriptedModel

IN = 64  # small model input for tests


def _yolo_io(nc=80):
    return IOSpec(inputs=(TensorSpec("images", (1, 3, IN, IN), "float32"),),
                  outputs=(TensorSpec("output0", (1, -1, 5 + nc), "float32"),))


def _raw(dets, nc=80):
    """dets: list of (cx, cy, w, h, obj, cls_idx) in INPUT (letterbox) px."""
    out = np.zeros((1, len(dets), 5 + nc), np.float32)
    for i, (cx, cy, w, h, obj, ci) in enumerate(dets):
        out[0, i, :4] = [cx, cy, w, h]
        out[0, i, 4] = obj
        out[0, i, 5 + ci] = 1.0
    return {"output0": out}


def _detector(raw, **kw):
    return YoloDetector(model=ScriptedModel(_yolo_io(), [raw]), input_size=IN,
                        output_format="yolov5", **kw)


def test_detect_returns_normalised_xywh_and_score():
    # one detection centred in a square input -> centred normalised box
    det = _detector(_raw([(32, 32, 16, 16, 0.9, 3)]))
    out = det.detect(np.zeros((IN, IN, 3), np.uint8))
    assert out.boxes.shape == (1, 4)
    assert out.scores[0] == pytest.approx(0.9, abs=1e-3)  # obj * max(cls)=0.9*1.0
    bx, by, bw, bh = out.boxes[0]
    assert (bx + bw / 2) == pytest.approx(0.5, abs=0.05)


def test_detect_is_class_agnostic_and_pure():
    det = _detector(_raw([(32, 32, 16, 16, 0.8, 17), (10, 10, 8, 8, 0.7, 2)]))
    img = np.zeros((IN, IN, 3), np.uint8)
    out1 = det.detect(img)
    out2 = det.detect(img)            # purity: same result, no internal mutation
    assert len(out1.scores) == 2
    np.testing.assert_array_equal(out1.boxes, out2.boxes)


def test_detect_thresholds_low_confidence():
    det = _detector(_raw([(32, 32, 16, 16, 0.1, 0)]), conf_thresh=0.25)
    out = det.detect(np.zeros((IN, IN, 3), np.uint8))
    assert len(out.scores) == 0


FH, FW = 240, 320


def _tracker(maps, **kw):
    m = ScriptedModel(_yolo_io(), maps)
    return YoloTracker(model=m, input_size=IN, output_format="yolov5", **kw)


def _box(cx, cy, w=40, h=40):
    return BoundingBox(x=(cx - w / 2) / FW, y=(cy - h / 2) / FH, w=w / FW, h=h / FH)


def test_yolo_tracker_name():
    assert _tracker([_raw([])]).name() == "YOLO"


def test_association_prefers_near_over_far_highscore():
    # crop is centred on prev box (160,120). Two detections inside the crop:
    # one near crop-centre with decent score, one far corner with higher score.
    near = (IN / 2, IN / 2, 16, 16, 0.6, 0)
    far = (4, 4, 8, 8, 0.95, 1)
    t = _tracker([_raw([near, far])], assoc_sigma=0.5)
    t.init(np.zeros((FH, FW, 3), np.uint8), _box(160, 120))
    res = t.update(np.zeros((FH, FW, 3), np.uint8))
    cx, cy = res.bbox.to_pixels(FW, FH).center
    assert abs(cx - 160) < 30 and abs(cy - 120) < 30   # picked the near one
    assert res.status == TrackStatus.LOCKED


def test_box_adapts_to_detection_size():
    t = _tracker([_raw([(IN / 2, IN / 2, 32, 16, 0.9, 0)])])
    t.init(np.zeros((FH, FW, 3), np.uint8), _box(160, 120, w=40, h=40))
    res = t.update(np.zeros((FH, FW, 3), np.uint8))
    # detection is 2:1 (w:h); output aspect should differ from the square init box
    assert res.bbox.w > res.bbox.h


def test_misses_coast_then_lost():
    t = _tracker([_raw([]), _raw([]), _raw([])], max_misses=2)
    t.init(np.zeros((FH, FW, 3), np.uint8), _box(160, 120))
    r1 = t.update(np.zeros((FH, FW, 3), np.uint8))
    assert r1.status == TrackStatus.COASTING
    t.update(np.zeros((FH, FW, 3), np.uint8))
    r3 = t.update(np.zeros((FH, FW, 3), np.uint8))
    assert r3.status == TrackStatus.LOST


def test_nn_package_exports():
    import edgecv.trackers.nn as nn
    assert hasattr(nn, "SiamFC")
    assert hasattr(nn, "YoloTracker")
    assert hasattr(nn, "YoloDetector")


def _yolo_io_v8(nc=80):
    # one-to-many head: (1, 4+nc, N) — transposed vs v5, no objectness column
    return IOSpec(inputs=(TensorSpec("images", (1, 3, IN, IN), "float32"),),
                  outputs=(TensorSpec("output0", (1, 4 + nc, -1), "float32"),))


def _raw_v8(dets, nc=80):
    """dets: list of (cx, cy, w, h, cls_score, cls_idx) in INPUT (letterbox) px.
    Builds a (1, 4+nc, N) tensor (channels-first, the YOLO26 one-to-many layout)."""
    out = np.zeros((1, 4 + nc, len(dets)), np.float32)
    for j, (cx, cy, w, h, sc, ci) in enumerate(dets):
        out[0, :4, j] = [cx, cy, w, h]
        out[0, 4 + ci, j] = sc
    return {"output0": out}


def _detector_v8(raw, **kw):
    return YoloDetector(model=ScriptedModel(_yolo_io_v8(), [raw]),
                        input_size=IN, output_format="yolov8", **kw)


def test_yolov8_decode_no_objectness_max_class():
    # class score 0.9 in column for class 5; score must be 0.9 (no obj multiply)
    det = _detector_v8(_raw_v8([(32, 32, 16, 16, 0.9, 5)]))
    out = det.detect(np.zeros((IN, IN, 3), np.uint8))
    assert out.boxes.shape == (1, 4)
    assert out.scores[0] == pytest.approx(0.9, abs=1e-3)
    bx, by, bw, bh = out.boxes[0]
    assert (bx + bw / 2) == pytest.approx(0.5, abs=0.05)


def test_yolov8_thresholds_and_is_pure():
    det = _detector_v8(_raw_v8([(32, 32, 16, 16, 0.1, 0)]), conf_thresh=0.25)
    img = np.zeros((IN, IN, 3), np.uint8)
    assert len(det.detect(img).scores) == 0          # thresholded out
    det2 = _detector_v8(_raw_v8([(32, 32, 16, 16, 0.8, 2), (10, 10, 8, 8, 0.7, 9)]))
    o1, o2 = det2.detect(img), det2.detect(img)
    np.testing.assert_array_equal(o1.boxes, o2.boxes)  # purity


def _pp_manifest(**pp):
    return ModelManifest(name="t", task="detection", preprocessing=pp)


def test_detector_reads_preprocessing_from_manifest():
    mf = _pp_manifest(conf_thresh=0.5, input=128, output_format="yolov8", color="gray")
    det = YoloDetector(manifest=mf, model=ScriptedModel(_yolo_io_v8(), [_raw_v8([])]))
    assert det._conf == 0.5
    assert det._input_size == 128
    assert det._output_format == "yolov8"
    assert det._color == "gray"


def test_detector_explicit_kwarg_overrides_manifest():
    mf = _pp_manifest(conf_thresh=0.5)
    det = YoloDetector(manifest=mf, model=ScriptedModel(_yolo_io(), [_raw([])]),
                       conf_thresh=0.9, output_format="yolov5")
    assert det._conf == 0.9


def test_detector_default_when_absent_from_manifest():
    det = YoloDetector(model=ScriptedModel(_yolo_io(), [_raw([])]), output_format="yolov5")
    assert det._conf == 0.25          # hardcoded default
    assert det._output_format == "yolov5"


def test_tracker_reads_preprocessing_from_manifest():
    mf = _pp_manifest(conf_thresh=0.5, output_format="yolov8")
    t = YoloTracker(manifest=mf, model=ScriptedModel(_yolo_io_v8(), [_raw_v8([])]),
                    input_size=IN)
    assert t._detector._conf == 0.5
    assert t._detector._output_format == "yolov8"


def test_tracker_explicit_kwarg_overrides_manifest():
    mf = _pp_manifest(conf_thresh=0.5)
    t = YoloTracker(manifest=mf, model=ScriptedModel(_yolo_io(), [_raw([])]),
                    input_size=IN, conf_thresh=0.9, output_format="yolov5")
    assert t._detector._conf == 0.9

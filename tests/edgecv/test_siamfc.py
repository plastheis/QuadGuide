import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackStatus
from edgecv.models.manifest import ModelManifest
from edgecv.trackers.nn.siamfc import SiamFC
from tests.edgecv._nn_stubs import ScriptedModel, score_map_peaked, siam_io

SS = 17  # score_size


def _frame(h=240, w=320):
    return np.zeros((h, w, 3), np.uint8)


def _box():
    return BoundingBox(x=(160 - 20) / 320, y=(120 - 20) / 240, w=40 / 320, h=40 / 240)


def _siam(maps, **kw):
    return SiamFC(model=ScriptedModel(siam_io(SS), maps), **kw)


def test_name_and_instantiation():
    t = _siam([{"score_map": score_map_peaked(SS, 8, 8)}])
    assert t.name() == "SiamFC"


def test_init_builds_127_exemplar_template():
    # 3 maps because update() runs scale_num=3 infers; init runs 0 infers.
    t = _siam([{"score_map": score_map_peaked(SS, 8, 8)}])
    t.init(_frame(), _box())
    z = t.get_template().arrays["exemplar"]
    assert z.shape == (1, 3, 127, 127)
    assert t.status == TrackStatus.LOCKED


def test_centred_peak_keeps_centre():
    # all 3 scales return a centred peak -> no displacement, box centre unchanged
    maps = [{"score_map": score_map_peaked(SS, 8, 8)} for _ in range(3)]
    t = _siam(maps, window_influence=0.0)
    t.init(_frame(), _box())
    res = t.update(_frame())
    cx, cy = res.bbox.to_pixels(320, 240).center
    assert cx == pytest.approx(160.0, abs=2.0)
    assert cy == pytest.approx(120.0, abs=2.0)
    assert res.seq == 1


def test_offcentre_peak_moves_box_in_that_direction():
    # peak one cell to the +x of centre on every scale -> centre moves +x
    maps = [{"score_map": score_map_peaked(SS, 8, 9)} for _ in range(3)]
    t = _siam(maps, window_influence=0.0)
    t.init(_frame(), _box())
    res = t.update(_frame())
    cx, _ = res.bbox.to_pixels(320, 240).center
    assert cx > 161.0   # moved right


def test_winning_larger_scale_grows_box():
    # scales are searched ascending: [<1, 1, >1]; make the >1 scale (3rd call) win
    maps = [
        {"score_map": score_map_peaked(SS, 8, 8, peak=0.2)},
        {"score_map": score_map_peaked(SS, 8, 8, peak=0.2)},
        {"score_map": score_map_peaked(SS, 8, 8, peak=1.0)},
    ]
    t = _siam(maps, window_influence=0.0)
    t.init(_frame(), _box())
    w0 = t.get_template().bbox.w
    res = t.update(_frame())
    assert res.bbox.w > w0   # box grew toward the winning larger scale


def test_low_response_reports_lost():
    maps = [{"score_map": np.zeros((1, 1, SS, SS), np.float32)} for _ in range(3)]
    t = _siam(maps)
    t.init(_frame(), _box())
    res = t.update(_frame())
    assert res.status == TrackStatus.LOST


def test_nn_package_exports():
    import edgecv.trackers.nn as nn
    assert hasattr(nn, "SiamFC")
    assert hasattr(nn, "YoloTracker")
    assert hasattr(nn, "YoloDetector")


def test_manifest_preprocessing_reaches_siamfc():
    mf = ModelManifest(name="t", task="sot_template_matching",
                       preprocessing={"window_influence": 0.99, "context": 0.7})
    t = SiamFC(mf, model=ScriptedModel(siam_io(SS), [{"score_map": score_map_peaked(SS, 8, 8)}]))
    assert t._window_influence == 0.99
    assert t._context == 0.7


def test_siamfc_explicit_kwarg_overrides_manifest():
    mf = ModelManifest(name="t", task="sot_template_matching",
                       preprocessing={"window_influence": 0.99})
    t = SiamFC(mf, model=ScriptedModel(siam_io(SS), [{"score_map": score_map_peaked(SS, 8, 8)}]),
               window_influence=0.1)
    assert t._window_influence == 0.1

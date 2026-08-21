"""Unit tests for NanoTrackDetectorAdapter — the NN detector adapter kept as
reusable scaffolding for a future hybrid (the concrete MAFiD hybrid trackers
that previously consumed it have been removed)."""
from __future__ import annotations

from edgecv.fusion.calibrator import SigmoidCalibrator
from edgecv.trackers.hybrid.detector_adapter import NanoTrackDetectorAdapter
from tests.edgecv._nn_stubs import (
    ScriptedModel,
    nano_backbone_io,
    nano_head_io,
    nano_head_out,
    nano_z_feat,
)

S = 15


def _mock_models():
    bb = ScriptedModel(nano_backbone_io(), [{"output": nano_z_feat()}])
    hd = ScriptedModel(nano_head_io(S), [nano_head_out(S, S // 2, S // 2)])
    return bb, hd


def test_default_calibrator():
    cal = NanoTrackDetectorAdapter.default_calibrator
    assert isinstance(cal, SigmoidCalibrator)
    assert cal.centre == 0.5
    assert cal.steepness == 10.0


def test_constructor_from_config_with_mock_model():
    bb, hd = _mock_models()
    adapter = NanoTrackDetectorAdapter(
        config={
            "manifest": None,
            "backend": "auto",
            "backbone": bb,
            "head": hd,
            "model": bb,
            "score_lock": 0.6,
            "score_lost": 0.35,
        }
    )
    assert adapter._initialized is False
    assert adapter._needs_refresh is False
    assert adapter._nanotrack._score_lock == 0.6
    assert adapter._nanotrack._score_lost == 0.35
    adapter.close()


def test_request_refresh_with_mock_model():
    bb, hd = _mock_models()
    adapter = NanoTrackDetectorAdapter(
        config={"manifest": None, "backend": "auto",
                "backbone": bb, "head": hd, "model": bb}
    )
    assert adapter._needs_refresh is False
    adapter.request_refresh()
    assert adapter._needs_refresh is True
    adapter.close()


def test_close_is_idempotent_with_mock_model():
    bb, hd = _mock_models()
    adapter = NanoTrackDetectorAdapter(
        config={"manifest": None, "backend": "auto",
                "backbone": bb, "head": hd, "model": bb}
    )
    adapter.close()
    adapter.close()  # should not raise

"""EdgeCVTracker adapter — bridges EdgeCV trackers to QuadGuide's protocol.

Uses MOSSE (pure-numpy, no model / no NPU) so it runs in CI without an
accelerator. Verifies the structural protocol the tracker_worker relies on.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("edgecv")

from quadguide.core.messages import BoundingBox, TrackerHealth  # noqa: E402
from quadguide.perception.edgecv_adapter import EdgeCVTracker  # noqa: E402


def _frame(h: int = 120, w: int = 160) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def test_update_before_init_reports_no_lock():
    trk = EdgeCVTracker(tracker="mosse")
    out = trk.update(_frame())
    assert out.health == "no_lock"
    assert (out.bbox.x, out.bbox.y, out.bbox.w, out.bbox.h) == (0.0, 0.0, 0.0, 0.0)


def test_init_then_update_emits_valid_protocol_output():
    trk = EdgeCVTracker(tracker="mosse")
    f = _frame()
    trk.init(f, BoundingBox(0.4, 0.4, 0.2, 0.2))
    out = trk.update(f)

    # bbox is normalised and has the four attributes the worker reads.
    for v in (out.bbox.x, out.bbox.y, out.bbox.w, out.bbox.h):
        assert isinstance(v, float)
    # confidence is normalised into [0, 1] (raw MOSSE PSR is unbounded).
    assert 0.0 <= out.confidence <= 1.0
    # health is one of the strings TrackerHealth accepts.
    assert TrackerHealth(out.health) in TrackerHealth


def test_name_comes_from_underlying_tracker():
    assert EdgeCVTracker(tracker="mosse").name() == "MOSSE"


def test_reset_returns_to_no_lock():
    trk = EdgeCVTracker(tracker="mosse")
    f = _frame()
    trk.init(f, BoundingBox(0.4, 0.4, 0.2, 0.2))
    assert trk.update(f).health != "no_lock"
    trk.reset()
    assert trk.update(f).health == "no_lock"


def test_unknown_tracker_raises():
    with pytest.raises(ValueError, match="unknown EdgeCV tracker"):
        EdgeCVTracker(tracker="does_not_exist")


def test_loadable_via_tracker_worker_load_tracker():
    """The adapter is reachable through the same loader run.py uses."""
    from quadguide.perception.tracker_worker import load_tracker

    cfg = {
        "tracker": {
            "import": "quadguide.perception.edgecv_adapter:EdgeCVTracker",
            "params": {"tracker": "mosse"},
        }
    }
    trk = load_tracker(cfg)
    assert trk.name() == "MOSSE"

"""EdgeCVTracker adapter: acquire_track mapping + source-frame origin lineage.

Avoids spawning real AcquireTrack workers (no NPU/models here): builds a
lightweight 'mosse' adapter, then injects a stub EdgeCV tracker and flips the
acquire_track flags to exercise the always-update + health + origin_ns mapping.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from quadguide.perception.edgecv_adapter import (
    _HEALTH_BY_STATUS,
    EdgeCVTracker,
)


def _status(name):
    return SimpleNamespace(name=name)


def _eb(x, y, w, h):
    return SimpleNamespace(x=x, y=y, w=w, h=h)


class _StubEdge:
    """Minimal EdgeCV-tracker stand-in returning a scripted TrackResult-like obj."""

    def __init__(self, results):
        self._results = list(results)
        self.i = 0
        self.init_calls = []
        self.closed = False

    def update(self, frame):
        r = self._results[min(self.i, len(self._results) - 1)]
        self.i += 1
        return r

    def init(self, frame, bbox):
        self.init_calls.append(bbox)

    def close(self):
        self.closed = True


def _frame():
    return np.zeros((48, 64, 3), np.uint8)


def _acquire_adapter(results):
    # Build a no-model adapter, then graft the stub + acquire_track flags.
    a = EdgeCVTracker(tracker="mosse", color="rgb")
    a._tracker = _StubEdge(results)
    a._calibrator = None
    a._always_update = True
    a._async = True
    a._initialized = False
    return a


def test_mapping_dict_initializing_is_acquiring():
    assert _HEALTH_BY_STATUS["INITIALIZING"] == "acquiring"
    assert _HEALTH_BY_STATUS["LOCKED"] == "nominal"
    assert _HEALTH_BY_STATUS["COASTING"] == "uncertain"


def test_update_runs_before_init_and_reports_acquiring():
    res = SimpleNamespace(bbox=_eb(0.4, 0.4, 0.1, 0.1), confidence=0.8,
                          status=_status("INITIALIZING"), timestamp=5.0)
    a = _acquire_adapter([res])
    out = a.update(_frame())                 # not initialized → still runs
    assert out.health == "acquiring"
    assert out.bbox.x == pytest.approx(0.4)
    assert out.origin_ns == int(5.0 * 1e9)   # source-frame lineage forwarded


def test_locked_maps_to_nominal_with_origin():
    res = SimpleNamespace(bbox=_eb(0.5, 0.5, 0.2, 0.2), confidence=0.9,
                          status=_status("LOCKED"), timestamp=7.25)
    a = _acquire_adapter([res])
    out = a.update(_frame())
    assert out.health == "nominal"
    assert out.origin_ns == int(7.25 * 1e9)


def test_none_bbox_maps_to_lost():
    res = SimpleNamespace(bbox=None, confidence=None,
                          status=_status("LOST"), timestamp=1.0)
    a = _acquire_adapter([res])
    out = a.update(_frame())
    assert out.health == "lost"


def test_none_bbox_while_acquiring_stays_acquiring():
    # YOLO has no candidate in the crop this frame: bbox=None but still scanning
    # (INITIALIZING). Must report "acquiring" (zero bbox), not "lost" — otherwise
    # the acquire box flickers off and the tracker drops to lost before any lock.
    res = SimpleNamespace(bbox=None, confidence=None,
                          status=_status("INITIALIZING"), timestamp=2.0)
    a = _acquire_adapter([res])
    out = a.update(_frame())
    assert out.health == "acquiring"
    assert (out.bbox.x, out.bbox.y, out.bbox.w, out.bbox.h) == (0.0, 0.0, 0.0, 0.0)
    assert out.origin_ns == int(2.0 * 1e9)


def test_init_forwards_to_edge_tracker_commit():
    a = _acquire_adapter([SimpleNamespace(
        bbox=_eb(0, 0, 0.1, 0.1), confidence=0.5,
        status=_status("LOCKED"), timestamp=1.0)])
    a.init(_frame(), _eb(0.3, 0.3, 0.2, 0.2))   # non-zero → commit
    assert a._initialized is True
    assert len(a._tracker.init_calls) == 1

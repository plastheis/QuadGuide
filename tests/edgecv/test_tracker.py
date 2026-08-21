import numpy as np
import pytest

from edgecv.core.bbox import BoundingBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.core.tracker import Tracker


class _Dummy(Tracker):
    def __init__(self):
        self._status = TrackStatus.INITIALIZING
        self.closed = False

    def init(self, frame, bbox):
        self._status = TrackStatus.LOCKED

    def update(self, frame):
        return TrackResult(bbox=BoundingBox(0, 0, 0.1, 0.1), confidence=1.0,
                           status=self._status, timestamp=0.0, seq=0)

    @property
    def status(self):
        return self._status

    def name(self):
        return "Dummy"

    def close(self):
        self.closed = True


def test_tracker_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Tracker()


def test_context_manager_calls_close():
    with _Dummy() as t:
        t.init(np.zeros((4, 4), np.uint8), BoundingBox(0, 0, 0.5, 0.5))
        assert t.status is TrackStatus.LOCKED
    assert t.closed is True


def test_default_close_is_noop_for_inline_tracker():
    class _NoClose(_Dummy):
        pass
    # Should not raise even though we did not override close beyond _Dummy.
    _NoClose().__exit__(None, None, None)

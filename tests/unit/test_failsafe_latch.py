from quadguide.control.failsafe import LostDisarmLatch
from quadguide.core.messages import TrackerHealth

MS = 1_000_000        # ns per ms
HOLD = 300 * MS


def _latch(enabled=True, hold_ns=HOLD):
    return LostDisarmLatch(enabled=enabled, hold_ns=hold_ns)


def test_no_trip_before_hold():
    latch = _latch()
    assert latch.update(0, armed=True, health=TrackerHealth.LOST) is False
    assert latch.update(299 * MS, armed=True, health=TrackerHealth.LOST) is False


def test_trips_at_hold():
    latch = _latch()
    latch.update(0, armed=True, health=TrackerHealth.LOST)              # start debounce
    assert latch.update(300 * MS, armed=True, health=TrackerHealth.LOST) is True


def test_debounce_resets_on_non_lost_frame():
    latch = _latch()
    latch.update(0, armed=True, health=TrackerHealth.LOST)              # debounce starts at 0
    latch.update(250 * MS, armed=True, health=TrackerHealth.NOMINAL)    # non-LOST resets it
    latch.update(300 * MS, armed=True, health=TrackerHealth.LOST)       # new run starts at 300ms
    assert latch.update(400 * MS, armed=True, health=TrackerHealth.LOST) is False  # only 100ms in
    assert latch.update(600 * MS, armed=True, health=TrackerHealth.LOST) is True   # 300ms in → trip


def test_latch_persists_through_recovery():
    latch = _latch()
    latch.update(0, armed=True, health=TrackerHealth.LOST)
    assert latch.update(300 * MS, armed=True, health=TrackerHealth.LOST) is True
    assert latch.update(400 * MS, armed=True, health=TrackerHealth.NOMINAL) is True  # sticky


def test_cleared_by_operator_disarm():
    latch = _latch()
    latch.update(0, armed=True, health=TrackerHealth.LOST)
    assert latch.update(300 * MS, armed=True, health=TrackerHealth.LOST) is True
    assert latch.update(400 * MS, armed=False, health=TrackerHealth.LOST) is False   # disarm clears
    assert latch.update(500 * MS, armed=True, health=TrackerHealth.NOMINAL) is False  # clean slate


def test_disabled_never_trips():
    latch = _latch(enabled=False)
    latch.update(0, armed=True, health=TrackerHealth.LOST)
    assert latch.update(10_000 * MS, armed=True, health=TrackerHealth.LOST) is False


def test_not_armed_never_trips():
    latch = _latch()
    latch.update(0, armed=False, health=TrackerHealth.LOST)
    assert latch.update(300 * MS, armed=False, health=TrackerHealth.LOST) is False


def test_health_none_treated_as_not_lost():
    latch = _latch()
    latch.update(0, armed=True, health=TrackerHealth.LOST)
    latch.update(100 * MS, armed=True, health=None)                     # no estimate → resets
    assert latch.update(350 * MS, armed=True, health=TrackerHealth.LOST) is False  # fresh run

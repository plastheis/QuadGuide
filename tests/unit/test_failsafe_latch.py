from quadguide.control.failsafe import FailsafeLatch, arbitrate_failsafe
from quadguide.core.config import ConditionFailsafe, FailsafeAction
from quadguide.core.messages import FailsafeActionWire

MS = 1_000_000        # ns per ms
HOLD = 300 * MS


def _latch(enabled=True, hold_ns=HOLD):
    return FailsafeLatch(enabled=enabled, hold_ns=hold_ns)


# ── FailsafeLatch (generic debounce/latch on a `tripped` predicate) ──────────

def test_no_trip_before_hold():
    latch = _latch()
    assert latch.update(0, armed=True, tripped=True) is False
    assert latch.update(299 * MS, armed=True, tripped=True) is False


def test_trips_at_hold():
    latch = _latch()
    latch.update(0, armed=True, tripped=True)
    assert latch.update(300 * MS, armed=True, tripped=True) is True


def test_debounce_resets_on_non_tripped_tick():
    latch = _latch()
    latch.update(0, armed=True, tripped=True)
    latch.update(250 * MS, armed=True, tripped=False)   # resets debounce
    latch.update(300 * MS, armed=True, tripped=True)    # fresh run at 300ms
    assert latch.update(400 * MS, armed=True, tripped=True) is False  # 100ms in
    assert latch.update(600 * MS, armed=True, tripped=True) is True   # 300ms in


def test_latch_persists_through_recovery():
    latch = _latch()
    latch.update(0, armed=True, tripped=True)
    assert latch.update(300 * MS, armed=True, tripped=True) is True
    assert latch.update(400 * MS, armed=True, tripped=False) is True  # sticky


def test_cleared_by_operator_disarm():
    latch = _latch()
    latch.update(0, armed=True, tripped=True)
    assert latch.update(300 * MS, armed=True, tripped=True) is True
    assert latch.update(400 * MS, armed=False, tripped=True) is False  # disarm clears
    assert latch.update(500 * MS, armed=True, tripped=True) is False   # debounce reset


def test_disabled_never_trips():
    latch = _latch(enabled=False)
    latch.update(0, armed=True, tripped=True)
    assert latch.update(10_000 * MS, armed=True, tripped=True) is False


def test_not_armed_never_trips():
    latch = _latch()
    latch.update(0, armed=False, tripped=True)
    assert latch.update(300 * MS, armed=False, tripped=True) is False


# ── arbitrate_failsafe (precedence: DISARM > MODE; target-loss > watchdog) ────

_DISARM = ConditionFailsafe(enabled=True, action=FailsafeAction.DISARM)
_MODE_LAND = ConditionFailsafe(
    enabled=True, action=FailsafeAction.MODE, mode="LAND", custom_mode=9)
_MODE_ALTHOLD = ConditionFailsafe(
    enabled=True, action=FailsafeAction.MODE, mode="ALTHOLD", custom_mode=2)


def test_arbitrate_none_when_no_latch():
    action, mode = arbitrate_failsafe(False, _MODE_LAND, False, _MODE_LAND)
    assert action is FailsafeActionWire.NONE
    assert mode == 0


def test_arbitrate_mode_from_target_loss():
    action, mode = arbitrate_failsafe(True, _MODE_LAND, False, _MODE_LAND)
    assert action is FailsafeActionWire.SET_MODE
    assert mode == 9


def test_arbitrate_mode_from_watchdog_only():
    action, mode = arbitrate_failsafe(False, _MODE_LAND, True, _MODE_ALTHOLD)
    assert action is FailsafeActionWire.SET_MODE
    assert mode == 2


def test_arbitrate_disarm_beats_mode():
    action, mode = arbitrate_failsafe(True, _MODE_LAND, True, _DISARM)
    assert action is FailsafeActionWire.DISARM
    assert mode == 0


def test_arbitrate_target_loss_mode_wins_over_watchdog_mode():
    action, mode = arbitrate_failsafe(True, _MODE_LAND, True, _MODE_ALTHOLD)
    assert action is FailsafeActionWire.SET_MODE
    assert mode == 9  # target-loss precedence

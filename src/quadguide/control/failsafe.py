from __future__ import annotations

from quadguide.core.config import ConditionFailsafe, FailsafeAction
from quadguide.core.messages import FailsafeActionWire

__all__ = ["FailsafeLatch", "arbitrate_failsafe"]


class FailsafeLatch:
    """Debounced trip -> latch machine (pure state machine, no bus/clock).

    The caller passes monotonic ``now_ns`` and a boolean ``tripped`` predicate
    each tick; ``update`` returns whether the latch is engaged. Semantics:

    * Only trips while ``armed`` (the operator's commanded arm intent, i.e.
      ``arm/cmd`` — never the FC's actual armed state).
    * Trips when ``tripped`` is True continuously for ``hold_ns``.
    * Any non-tripped tick resets the debounce.
    * Once latched, stays latched until ``armed`` goes False (operator disarm) —
      the manual re-engage gate.
    * Disabled -> always returns False.

    Action-agnostic: the trip predicate (``health == LOST`` for target-loss,
    ``any watched topic stale`` for the watchdog) and the resulting action are
    supplied by the control worker, not this class.
    """

    def __init__(self, enabled: bool, hold_ns: int) -> None:
        self._enabled = enabled
        self._hold_ns = hold_ns
        self._trip_since: int | None = None
        self._latched = False

    def update(self, now_ns: int, armed: bool, tripped: bool) -> bool:
        if not self._enabled:
            return False
        if not armed:                       # operator disarm clears latch + debounce
            self._latched = False
            self._trip_since = None
            return False
        if self._latched:                   # sticky until 'not armed' clears it above
            return True
        if tripped:
            if self._trip_since is None:
                self._trip_since = now_ns
            elif now_ns - self._trip_since >= self._hold_ns:
                self._latched = True
        else:
            self._trip_since = None          # any non-tripped tick resets the debounce
        return self._latched


def arbitrate_failsafe(
    tl_latched: bool, tl: ConditionFailsafe,
    wd_latched: bool, wd: ConditionFailsafe,
) -> tuple[FailsafeActionWire, int]:
    """Resolve one effective failsafe action from the two condition latches.

    Precedence: DISARM beats MODE (more conservative). Among latched MODE
    conditions, target-loss beats watchdog. Returns (action, custom_mode);
    custom_mode is 0 unless the action is SET_MODE.
    """
    active: list[ConditionFailsafe] = []
    if tl_latched:
        active.append(tl)          # target-loss first → precedence among modes
    if wd_latched:
        active.append(wd)
    if not active:
        return FailsafeActionWire.NONE, 0
    if any(c.action is FailsafeAction.DISARM for c in active):
        return FailsafeActionWire.DISARM, 0
    chosen = active[0]             # all MODE; first is target-loss if latched
    return FailsafeActionWire.SET_MODE, int(chosen.custom_mode or 0)

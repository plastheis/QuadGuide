from __future__ import annotations

from quadguide.core.messages import TrackerHealth

__all__ = ["LostDisarmLatch"]


class LostDisarmLatch:
    """Debounced target-loss -> disarm latch (pure state machine, no bus/clock).

    The caller passes monotonic ``now_ns`` each tick; ``update`` returns whether
    the disarm latch is engaged. Semantics (design spec 2026-07-30):

    * Only trips while ``armed`` (the operator's commanded arm intent, i.e.
      ``arm/cmd`` — never the FC's actual armed state).
    * Trips when ``health == LOST`` continuously for ``hold_ns``.
    * Any non-LOST tick (including ``health is None``) resets the debounce.
    * Once latched, stays latched through health recovery; clears only when
      ``armed`` goes False (operator disarm) — the manual re-arm gate.
    * Disabled -> always returns False.
    """

    def __init__(self, enabled: bool, hold_ns: int) -> None:
        self._enabled = enabled
        self._hold_ns = hold_ns
        self._lost_since: int | None = None
        self._latched = False

    def update(self, now_ns: int, armed: bool, health) -> bool:
        if not self._enabled:
            return False
        if not armed:                       # operator disarm clears latch + debounce
            self._latched = False
            self._lost_since = None
            return False
        if self._latched:                   # sticky until 'not armed' clears it above
            return True
        if health == TrackerHealth.LOST:
            if self._lost_since is None:
                self._lost_since = now_ns
            elif now_ns - self._lost_since >= self._hold_ns:
                self._latched = True
        else:
            self._lost_since = None          # any non-LOST frame resets the debounce
        return self._latched

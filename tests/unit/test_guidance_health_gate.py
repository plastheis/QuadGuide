"""Guidance must ignore non-driving tracker health states (incl. ACQUIRING).

ACQUIRING carries pre-lock detection candidates for the HUD; it must never drive
flight. LOST/NO_LOCK mean no target. NOMINAL/UNCERTAIN do drive guidance.
"""

from quadguide.core.messages import TrackerHealth
from quadguide.guidance.worker import _NON_DRIVING_HEALTH


def test_non_driving_set_contents():
    assert set(_NON_DRIVING_HEALTH) == {
        TrackerHealth.LOST, TrackerHealth.NO_LOCK, TrackerHealth.ACQUIRING,
    }


def test_driving_states_not_ignored():
    assert TrackerHealth.NOMINAL not in _NON_DRIVING_HEALTH
    assert TrackerHealth.UNCERTAIN not in _NON_DRIVING_HEALTH


def test_acquiring_round_trips_on_wire():
    # Appended last → existing ordinals unchanged; new one round-trips.
    assert TrackerHealth._from_ord[TrackerHealth._ord[TrackerHealth.ACQUIRING]] \
        == TrackerHealth.ACQUIRING
    assert TrackerHealth._ord[TrackerHealth.NO_LOCK] == 3  # unchanged

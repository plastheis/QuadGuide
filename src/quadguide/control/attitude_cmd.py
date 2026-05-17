from __future__ import annotations
import math

from quadguide.core.messages import AccelCmd

_G = 9.81


def compute(accel: AccelCmd) -> tuple[float, float]:
    """Map body-frame acceleration command to roll/pitch setpoints (degrees).

    Small-angle mapping: roll = atan(ay/g), pitch = -atan(ax/g).
    Positive ay (rightward) → positive roll (right bank).
    Positive ax (forward)   → negative pitch (nose up).
    """
    roll_deg  =  math.degrees(math.atan2(accel.ay, _G))
    pitch_deg = -math.degrees(math.atan2(accel.ax, _G))
    return roll_deg, pitch_deg

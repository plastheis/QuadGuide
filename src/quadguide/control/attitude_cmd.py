from __future__ import annotations
import math

from quadguide.core.messages import AccelCmd

_G = 9.81


def compute(accel: AccelCmd) -> tuple[float, float]:
    """Map body-frame acceleration command to roll/pitch setpoints (degrees).

    Bore-sight is fixed at body +Z (up when level). A conventional nadir camera
    would use roll = +ay/g, pitch = -ax/g; the up-facing mount inverts both axes.
    These signs are a hard-coded property of the build and must be verified on
    the bench HIL rig (commanded centroid offset → negative feedback).
    """
    roll_deg  = -math.degrees(accel.ay / _G)
    pitch_deg =  math.degrees(accel.ax / _G)
    return roll_deg, pitch_deg

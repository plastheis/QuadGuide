from __future__ import annotations
import math

from quadguide.core.messages import AccelCmd

_G = 9.81


def compute(accel: AccelCmd) -> tuple[float, float]:
    """Map body-frame acceleration command to roll/pitch setpoints (degrees).

    Bore-sight is fixed at body +Z (up when level). Signs (roll = +ay/g,
    pitch = -ax/g) are a hard-coded property of the build, verified on the bench
    HIL rig for negative feedback (commanded centroid offset → tilt that drives
    the target back toward image centre).
    """
    roll_deg  =  math.degrees(accel.ay / _G)
    pitch_deg = -math.degrees(accel.ax / _G)
    return roll_deg, pitch_deg

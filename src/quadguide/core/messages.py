from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import struct

__all__ = [
    "TrackerHealth", "ActiveTracker", "ProcessState",
    "BoundingBox",
    "TrackerEstimate", "TargetEstimate", "AttitudeState", "IMUFrame",
    "AccelCmd", "ControlCmd", "LockOnCmd", "HealthReport",
    "FMT_TRACKER_ESTIMATE", "FMT_TARGET_ESTIMATE", "FMT_ATTITUDE_STATE",
    "FMT_IMU_FRAME", "FMT_ACCEL_CMD", "FMT_CONTROL_CMD",
    "FMT_LOCKON_CMD", "FMT_HEALTH_REPORT",
]


def _byte_enum(cls):
    """Decorator: adds _ord and _from_ord dicts to an Enum for O(1) wire encoding."""
    members = list(cls)
    cls._ord = {e: i for i, e in enumerate(members)}
    cls._from_ord = {i: e for i, e in enumerate(members)}
    return cls


@_byte_enum
class TrackerHealth(str, Enum):
    NOMINAL   = "nominal"
    UNCERTAIN = "uncertain"
    LOST      = "lost"
    NO_LOCK   = "no_lock"


@_byte_enum
class ActiveTracker(str, Enum):
    KCF   = "kcf"
    NANO  = "nano"
    FUSED = "fused"


@_byte_enum
class ProcessState(str, Enum):
    OK       = "ok"
    DEGRADED = "degraded"
    FAILSAFE = "failsafe"
    DEAD     = "dead"


# Format strings are the source of truth for wire layout.
# Byte-count comments show the arithmetic; if a comment disagrees with
# struct.calcsize(fmt), the format string wins.

FMT_TRACKER_ESTIMATE = "!QfffffB"
# Q(8) + bbox.x,y,w,h(4×f=16) + confidence(f=4) + health(B=1) = 29 bytes

FMT_TARGET_ESTIMATE = "!QfffffffBB"
# Q(8) + bbox.x,y,w,h(4×f=16) + centroid_x,y(2×f=8) + confidence(f=4)
#   + tracker_health(B=1) + active_tracker(B=1) = 38 bytes
# NOTE: architecture.md had "!QffffffBB" (6 f's = 34 bytes) — corrected here.

FMT_ATTITUDE_STATE = "!Qffffff"
# Q(8) + roll,pitch,yaw,roll_rate,pitch_rate,yaw_rate(6×f=24) = 32 bytes

FMT_IMU_FRAME = "!Qffffff"
# Q(8) + ax,ay,az,gx,gy,gz(6×f=24) = 32 bytes

FMT_ACCEL_CMD = "!Qff"
# Q(8) + ax,ay(2×f=8) = 16 bytes

FMT_CONTROL_CMD = "!Qffff"
# Q(8) + roll_deg,pitch_deg,yaw_rate_dps,throttle_norm(4×f=16) = 24 bytes

FMT_LOCKON_CMD = "!QHffff"
# Q(8) + seq(H=2) + bbox.x,y,w,h(4×f=16) = 26 bytes
# seq is uint16 (wraps at 65535). Comparison is always !=, never >; wraparound is safe.

FMT_HEALTH_REPORT = "!Q16sB"
# Q(8) + process(16s=16) + state(B=1) = 25 bytes
# detail is NOT on the wire — logged only.
# pack() truncates process name to 16 UTF-8 bytes before packing.


@dataclass(frozen=True)
class BoundingBox:
    x: float  # top-left x, normalised 0–1
    y: float  # top-left y, normalised 0–1
    w: float  # width, normalised 0–1
    h: float  # height, normalised 0–1

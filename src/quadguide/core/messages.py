from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import struct

__all__ = [
    "TrackerHealth", "ProcessState",
    "BoundingBox",
    "TrackerEstimate", "AttitudeState", "IMUFrame",
    "AccelCmd", "ControlCmd", "LockOnCmd", "HealthReport", "ArmCmd",
    "FMT_TRACKER_ESTIMATE", "FMT_ATTITUDE_STATE",
    "FMT_IMU_FRAME", "FMT_ACCEL_CMD", "FMT_CONTROL_CMD",
    "FMT_LOCKON_CMD", "FMT_HEALTH_REPORT", "FMT_ARM_CMD",
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
class ProcessState(str, Enum):
    OK       = "ok"
    DEGRADED = "degraded"
    FAILSAFE = "failsafe"
    DEAD     = "dead"


# Format strings are the source of truth for wire layout.
# Byte-count comments show the arithmetic; if a comment disagrees with
# struct.calcsize(fmt), the format string wins.

FMT_TRACKER_ESTIMATE = "!QfffffBI"
# Q(8) + bbox.x,y,w,h(4×f=16) + confidence(f=4) + health(B=1) + latency_ns(I=4) = 33 bytes

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

FMT_ARM_CMD = "!QB"
# Q(8) + armed(B=1) = 9 bytes


@dataclass(frozen=True)
class BoundingBox:
    x: float  # top-left x, normalised 0–1
    y: float  # top-left y, normalised 0–1
    w: float  # width, normalised 0–1
    h: float  # height, normalised 0–1


_ST_TRACKER_ESTIMATE = struct.Struct(FMT_TRACKER_ESTIMATE)
_ST_ATTITUDE_STATE   = struct.Struct(FMT_ATTITUDE_STATE)
_ST_IMU_FRAME        = struct.Struct(FMT_IMU_FRAME)
_ST_ACCEL_CMD        = struct.Struct(FMT_ACCEL_CMD)
_ST_CONTROL_CMD      = struct.Struct(FMT_CONTROL_CMD)
_ST_LOCKON_CMD       = struct.Struct(FMT_LOCKON_CMD)
_ST_HEALTH_REPORT    = struct.Struct(FMT_HEALTH_REPORT)
_ST_ARM_CMD          = struct.Struct(FMT_ARM_CMD)


@dataclass(frozen=True)
class TrackerEstimate:
    timestamp_ns: int
    bbox: BoundingBox
    confidence: float
    tracker_health: TrackerHealth
    latency_ns: int = 0  # set by tracker worker via dataclasses.replace; 0 = no frame yet

    def pack(self) -> bytes:
        return _ST_TRACKER_ESTIMATE.pack(
            self.timestamp_ns,
            self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h,
            self.confidence,
            TrackerHealth._ord[self.tracker_health],
            self.latency_ns,
        )

    @classmethod
    def unpack(cls, data: bytes) -> TrackerEstimate:
        ts, x, y, w, h, conf, health_b, latency = _ST_TRACKER_ESTIMATE.unpack(data)
        return cls(
            timestamp_ns=ts,
            bbox=BoundingBox(x, y, w, h),
            confidence=conf,
            tracker_health=TrackerHealth._from_ord[health_b],
            latency_ns=latency,
        )


@dataclass(frozen=True)
class AttitudeState:
    timestamp_ns: int
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    roll_rate_rps: float
    pitch_rate_rps: float
    yaw_rate_rps: float

    def pack(self) -> bytes:
        return _ST_ATTITUDE_STATE.pack(
            self.timestamp_ns,
            self.roll_rad, self.pitch_rad, self.yaw_rad,
            self.roll_rate_rps, self.pitch_rate_rps, self.yaw_rate_rps,
        )

    @classmethod
    def unpack(cls, data: bytes) -> AttitudeState:
        ts, roll, pitch, yaw, rr, pr, yr = _ST_ATTITUDE_STATE.unpack(data)
        return cls(ts, roll, pitch, yaw, rr, pr, yr)


@dataclass(frozen=True)
class IMUFrame:
    timestamp_ns: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

    def pack(self) -> bytes:
        return _ST_IMU_FRAME.pack(
            self.timestamp_ns,
            self.ax, self.ay, self.az,
            self.gx, self.gy, self.gz,
        )

    @classmethod
    def unpack(cls, data: bytes) -> IMUFrame:
        ts, ax, ay, az, gx, gy, gz = _ST_IMU_FRAME.unpack(data)
        return cls(ts, ax, ay, az, gx, gy, gz)


@dataclass(frozen=True)
class AccelCmd:
    timestamp_ns: int
    ax: float
    ay: float

    def pack(self) -> bytes:
        return _ST_ACCEL_CMD.pack(self.timestamp_ns, self.ax, self.ay)

    @classmethod
    def unpack(cls, data: bytes) -> AccelCmd:
        ts, ax, ay = _ST_ACCEL_CMD.unpack(data)
        return cls(ts, ax, ay)


@dataclass(frozen=True)
class ControlCmd:
    timestamp_ns: int
    roll_deg: float
    pitch_deg: float
    yaw_rate_dps: float
    throttle_norm: float

    def pack(self) -> bytes:
        return _ST_CONTROL_CMD.pack(
            self.timestamp_ns,
            self.roll_deg, self.pitch_deg,
            self.yaw_rate_dps, self.throttle_norm,
        )

    @classmethod
    def unpack(cls, data: bytes) -> ControlCmd:
        ts, roll, pitch, yaw_rate, throttle = _ST_CONTROL_CMD.unpack(data)
        return cls(ts, roll, pitch, yaw_rate, throttle)


@dataclass(frozen=True)
class LockOnCmd:
    timestamp_ns: int
    seq: int  # uint16; wraps at 65535. Always compare with !=, never >
    bbox: BoundingBox

    def pack(self) -> bytes:
        return _ST_LOCKON_CMD.pack(
            self.timestamp_ns, self.seq,
            self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h,
        )

    @classmethod
    def unpack(cls, data: bytes) -> LockOnCmd:
        ts, seq, x, y, w, h = _ST_LOCKON_CMD.unpack(data)
        return cls(ts, seq, BoundingBox(x, y, w, h))


@dataclass(frozen=True)
class HealthReport:
    timestamp_ns: int
    process: str    # max 16 UTF-8 bytes on the wire; longer names are truncated
    state: ProcessState
    detail: str     # NOT on the wire — logged only; always "" after unpack

    def pack(self) -> bytes:
        raw = self.process.encode("utf-8")
        if len(raw) > 16:
            # Slicing bytes at position 16 can split a multibyte codepoint, leaving
            # an invalid sequence that raises UnicodeDecodeError at the receiver.
            # Decode with errors="ignore" drops any incomplete trailing sequence.
            raw = raw[:16].decode("utf-8", errors="ignore").encode("utf-8")
        name_bytes = raw.ljust(16, b"\x00")
        return _ST_HEALTH_REPORT.pack(
            self.timestamp_ns,
            name_bytes,
            ProcessState._ord[self.state],
        )

    @classmethod
    def unpack(cls, data: bytes) -> HealthReport:
        ts, name_bytes, state_b = _ST_HEALTH_REPORT.unpack(data)
        return cls(
            timestamp_ns=ts,
            process=name_bytes.rstrip(b"\x00").decode("utf-8"),
            state=ProcessState._from_ord[state_b],
            detail="",
        )


@dataclass(frozen=True)
class ArmCmd:
    timestamp_ns: int
    armed: bool

    def pack(self) -> bytes:
        return _ST_ARM_CMD.pack(self.timestamp_ns, int(self.armed))

    @classmethod
    def unpack(cls, data: bytes) -> ArmCmd:
        ts, armed_b = _ST_ARM_CMD.unpack(data)
        return cls(timestamp_ns=ts, armed=bool(armed_b))



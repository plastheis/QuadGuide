import struct
import threading
import time
import pytest
from quadguide.core.messages import (
    TrackerHealth, ActiveTracker, ProcessState,
    BoundingBox, TrackerEstimate, TargetEstimate,
    AttitudeState, IMUFrame, AccelCmd, ControlCmd,
    LockOnCmd, HealthReport,
    FMT_TRACKER_ESTIMATE, FMT_TARGET_ESTIMATE,
    FMT_ATTITUDE_STATE, FMT_IMU_FRAME, FMT_ACCEL_CMD,
    FMT_CONTROL_CMD, FMT_LOCKON_CMD, FMT_HEALTH_REPORT,
)


class TestEnumOrdinals:
    def test_tracker_health_round_trip(self):
        for health in TrackerHealth:
            assert TrackerHealth._from_ord[TrackerHealth._ord[health]] == health

    def test_active_tracker_round_trip(self):
        for tracker in ActiveTracker:
            assert ActiveTracker._from_ord[ActiveTracker._ord[tracker]] == tracker

    def test_process_state_round_trip(self):
        for state in ProcessState:
            assert ProcessState._from_ord[ProcessState._ord[state]] == state

    def test_tracker_health_is_str(self):
        assert TrackerHealth.NOMINAL == "nominal"

    def test_active_tracker_is_str(self):
        assert ActiveTracker.KCF == "kcf"

    def test_process_state_is_str(self):
        assert ProcessState.OK == "ok"


class TestFormatSizes:
    def test_tracker_estimate(self):
        assert struct.calcsize(FMT_TRACKER_ESTIMATE) == 29

    def test_target_estimate(self):
        assert struct.calcsize(FMT_TARGET_ESTIMATE) == 38

    def test_attitude_state(self):
        assert struct.calcsize(FMT_ATTITUDE_STATE) == 32

    def test_imu_frame(self):
        assert struct.calcsize(FMT_IMU_FRAME) == 32

    def test_accel_cmd(self):
        assert struct.calcsize(FMT_ACCEL_CMD) == 16

    def test_control_cmd(self):
        assert struct.calcsize(FMT_CONTROL_CMD) == 24

    def test_lockon_cmd(self):
        assert struct.calcsize(FMT_LOCKON_CMD) == 26

    def test_health_report(self):
        assert struct.calcsize(FMT_HEALTH_REPORT) == 25


class TestRoundTrips:
    def test_tracker_estimate(self):
        msg = TrackerEstimate(
            timestamp_ns=1_000_000,
            bbox=BoundingBox(0.1, 0.2, 0.3, 0.4),
            confidence=0.9,
            tracker_health=TrackerHealth.NOMINAL,
        )
        assert TrackerEstimate.unpack(msg.pack()) == msg

    def test_target_estimate(self):
        msg = TargetEstimate(
            timestamp_ns=2_000_000,
            bbox=BoundingBox(0.1, 0.2, 0.3, 0.4),
            centroid_norm=(-0.1, 0.2),
            confidence=0.85,
            tracker_health=TrackerHealth.UNCERTAIN,
            active_tracker=ActiveTracker.FUSED,
        )
        assert TargetEstimate.unpack(msg.pack()) == msg

    def test_attitude_state(self):
        msg = AttitudeState(
            timestamp_ns=3_000_000,
            roll_rad=0.1, pitch_rad=0.2, yaw_rad=0.3,
            roll_rate_rps=0.01, pitch_rate_rps=0.02, yaw_rate_rps=0.03,
        )
        assert AttitudeState.unpack(msg.pack()) == msg

    def test_imu_frame(self):
        msg = IMUFrame(
            timestamp_ns=4_000_000,
            ax=1.0, ay=2.0, az=9.8,
            gx=0.1, gy=0.2, gz=0.3,
        )
        assert IMUFrame.unpack(msg.pack()) == msg

    def test_accel_cmd(self):
        msg = AccelCmd(timestamp_ns=5_000_000, ax=1.5, ay=-0.5)
        assert AccelCmd.unpack(msg.pack()) == msg

    def test_control_cmd(self):
        msg = ControlCmd(
            timestamp_ns=6_000_000,
            roll_deg=10.0, pitch_deg=-5.0,
            yaw_rate_dps=0.0, throttle_norm=0.5,
        )
        assert ControlCmd.unpack(msg.pack()) == msg

    def test_lockon_cmd(self):
        msg = LockOnCmd(
            timestamp_ns=7_000_000, seq=42,
            bbox=BoundingBox(0.3, 0.3, 0.2, 0.2),
        )
        assert LockOnCmd.unpack(msg.pack()) == msg

    def test_lockon_cmd_seq_max(self):
        # seq is uint16; 65535 must survive round-trip without overflow
        msg = LockOnCmd(timestamp_ns=0, seq=65535, bbox=BoundingBox(0, 0, 1, 1))
        assert LockOnCmd.unpack(msg.pack()).seq == 65535

    def test_health_report_round_trip(self):
        msg = HealthReport(
            timestamp_ns=8_000_000,
            process="camera",
            state=ProcessState.OK,
            detail="all good",
        )
        recovered = HealthReport.unpack(msg.pack())
        assert recovered.timestamp_ns == msg.timestamp_ns
        assert recovered.process == "camera"
        assert recovered.state == ProcessState.OK
        assert recovered.detail == ""  # detail is not on the wire

    def test_health_report_truncates_long_name(self):
        # process names > 16 UTF-8 bytes must be silently truncated, not raise
        msg = HealthReport(
            timestamp_ns=0, process="a" * 20,
            state=ProcessState.DEAD, detail="",
        )
        recovered = HealthReport.unpack(msg.pack())
        assert recovered.process == "a" * 16

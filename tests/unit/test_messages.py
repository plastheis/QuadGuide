import struct
import pytest
from quadguide.core.messages import (
    TrackerHealth, ActiveTracker, ProcessState,
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

import struct
import pytest
from quadguide.core.messages import (
    TrackerHealth, ProcessState,
    BoundingBox, TrackerEstimate,
    AttitudeState, IMUFrame, AccelCmd, ControlCmd,
    LockOnCmd, HealthReport, ArmCmd, FCStatus, FailsafeCmd,
    FMT_TRACKER_ESTIMATE,
    FMT_ATTITUDE_STATE, FMT_IMU_FRAME, FMT_ACCEL_CMD,
    FMT_CONTROL_CMD, FMT_LOCKON_CMD, FMT_HEALTH_REPORT, FMT_ARM_CMD,
    FMT_FC_STATUS, FMT_FAILSAFE_CMD,
)


class TestEnumOrdinals:
    def test_tracker_health_round_trip(self):
        for health in TrackerHealth:
            assert TrackerHealth._from_ord[TrackerHealth._ord[health]] == health

    def test_process_state_round_trip(self):
        for state in ProcessState:
            assert ProcessState._from_ord[ProcessState._ord[state]] == state

    def test_tracker_health_is_str(self):
        assert TrackerHealth.NOMINAL == "nominal"

    def test_process_state_is_str(self):
        assert ProcessState.OK == "ok"


class TestFormatSizes:
    def test_tracker_estimate(self):
        assert struct.calcsize(FMT_TRACKER_ESTIMATE) == 37

    def test_attitude_state(self):
        assert struct.calcsize(FMT_ATTITUDE_STATE) == 32

    def test_imu_frame(self):
        assert struct.calcsize(FMT_IMU_FRAME) == 32

    def test_accel_cmd(self):
        assert struct.calcsize(FMT_ACCEL_CMD) == 24

    def test_control_cmd(self):
        assert struct.calcsize(FMT_CONTROL_CMD) == 32

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
            origin_ns=999_000_000_000,  # absolute monotonic ts — exceeds uint32
        )
        r = TrackerEstimate.unpack(msg.pack())
        assert r.timestamp_ns == msg.timestamp_ns
        assert r.bbox.x == pytest.approx(msg.bbox.x, rel=1e-6)
        assert r.bbox.y == pytest.approx(msg.bbox.y, rel=1e-6)
        assert r.bbox.w == pytest.approx(msg.bbox.w, rel=1e-6)
        assert r.bbox.h == pytest.approx(msg.bbox.h, rel=1e-6)
        assert r.confidence == pytest.approx(msg.confidence, rel=1e-6)
        assert r.tracker_health == msg.tracker_health
        assert r.origin_ns == msg.origin_ns

    def test_attitude_state(self):
        msg = AttitudeState(
            timestamp_ns=3_000_000,
            roll_rad=0.1, pitch_rad=0.2, yaw_rad=0.3,
            roll_rate_rps=0.01, pitch_rate_rps=0.02, yaw_rate_rps=0.03,
        )
        r = AttitudeState.unpack(msg.pack())
        assert r.timestamp_ns == msg.timestamp_ns
        assert r.roll_rad == pytest.approx(msg.roll_rad, rel=1e-6)
        assert r.pitch_rad == pytest.approx(msg.pitch_rad, rel=1e-6)
        assert r.yaw_rad == pytest.approx(msg.yaw_rad, rel=1e-6)
        assert r.roll_rate_rps == pytest.approx(msg.roll_rate_rps, rel=1e-6)
        assert r.pitch_rate_rps == pytest.approx(msg.pitch_rate_rps, rel=1e-6)
        assert r.yaw_rate_rps == pytest.approx(msg.yaw_rate_rps, rel=1e-6)

    def test_imu_frame(self):
        msg = IMUFrame(
            timestamp_ns=4_000_000,
            ax=1.0, ay=2.0, az=9.8,
            gx=0.1, gy=0.2, gz=0.3,
        )
        r = IMUFrame.unpack(msg.pack())
        assert r.timestamp_ns == msg.timestamp_ns
        assert r.ax == pytest.approx(msg.ax, rel=1e-6)
        assert r.ay == pytest.approx(msg.ay, rel=1e-6)
        assert r.az == pytest.approx(msg.az, rel=1e-6)
        assert r.gx == pytest.approx(msg.gx, rel=1e-6)
        assert r.gy == pytest.approx(msg.gy, rel=1e-6)
        assert r.gz == pytest.approx(msg.gz, rel=1e-6)

    def test_accel_cmd(self):
        msg = AccelCmd(timestamp_ns=5_000_000, ax=1.5, ay=-0.5, origin_ns=4_900_000)
        r = AccelCmd.unpack(msg.pack())
        assert r.timestamp_ns == msg.timestamp_ns
        assert r.ax == pytest.approx(msg.ax, rel=1e-6)
        assert r.ay == pytest.approx(msg.ay, rel=1e-6)
        assert r.origin_ns == msg.origin_ns

    def test_control_cmd(self):
        msg = ControlCmd(
            timestamp_ns=6_000_000,
            roll_deg=10.0, pitch_deg=-5.0,
            yaw_rate_dps=0.0, throttle_norm=0.5,
            origin_ns=5_900_000,
        )
        r = ControlCmd.unpack(msg.pack())
        assert r.timestamp_ns == msg.timestamp_ns
        assert r.roll_deg == pytest.approx(msg.roll_deg, rel=1e-6)
        assert r.pitch_deg == pytest.approx(msg.pitch_deg, rel=1e-6)
        assert r.yaw_rate_dps == pytest.approx(msg.yaw_rate_dps, abs=1e-9)  # abs= required: rel= against 0.0 resolves to 0 tolerance
        assert r.throttle_norm == pytest.approx(msg.throttle_norm, rel=1e-6)
        assert r.origin_ns == msg.origin_ns

    def test_lockon_cmd(self):
        msg = LockOnCmd(
            timestamp_ns=7_000_000, seq=42,
            bbox=BoundingBox(0.3, 0.3, 0.2, 0.2),
        )
        r = LockOnCmd.unpack(msg.pack())
        assert r.timestamp_ns == msg.timestamp_ns
        assert r.seq == msg.seq
        assert r.bbox.x == pytest.approx(msg.bbox.x, rel=1e-6)
        assert r.bbox.y == pytest.approx(msg.bbox.y, rel=1e-6)
        assert r.bbox.w == pytest.approx(msg.bbox.w, rel=1e-6)
        assert r.bbox.h == pytest.approx(msg.bbox.h, rel=1e-6)

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

    def test_health_report_truncates_multibyte_at_boundary(self):
        # ☀ is 3 bytes (U+2600); 5 copies = 15 bytes, then "x" = 16 bytes total — fits
        # 6 copies = 18 bytes — must truncate to 5 copies (15 bytes) not slice mid-codepoint
        process = "☀" * 6  # 18 bytes; slicing at byte 16 cuts ☀ #6 mid-sequence
        msg = HealthReport(timestamp_ns=0, process=process, state=ProcessState.OK, detail="")
        recovered = HealthReport.unpack(msg.pack())
        # Must not raise UnicodeDecodeError; must produce valid UTF-8 string
        assert all(ord(c) == 0x2600 for c in recovered.process)
        assert len(recovered.process) == 5  # 5 × ☀ = 15 bytes fits in 16


class TestArmCmd:
    def test_format_size(self):
        assert struct.calcsize(FMT_ARM_CMD) == 9  # Q(8) + B(1)

    def test_round_trip_armed_true(self):
        msg = ArmCmd(timestamp_ns=1_000_000, armed=True)
        r = ArmCmd.unpack(msg.pack())
        assert r.timestamp_ns == 1_000_000
        assert r.armed is True

    def test_round_trip_armed_false(self):
        msg = ArmCmd(timestamp_ns=2_000_000, armed=False)
        r = ArmCmd.unpack(msg.pack())
        assert r.armed is False


class TestFCStatus:
    def test_format_size(self):
        assert struct.calcsize(FMT_FC_STATUS) == 13  # Q(8) + B(1) + I(4)

    def test_round_trip_armed(self):
        msg = FCStatus(timestamp_ns=7_000_000, armed=True, custom_mode=18)
        r = FCStatus.unpack(msg.pack())
        assert r.timestamp_ns == 7_000_000
        assert r.armed is True
        assert r.custom_mode == 18

    def test_round_trip_disarmed(self):
        msg = FCStatus(timestamp_ns=8_000_000, armed=False, custom_mode=0)
        r = FCStatus.unpack(msg.pack())
        assert r.armed is False
        assert r.custom_mode == 0


class TestFailsafeCmd:
    def test_format_size(self):
        assert struct.calcsize(FMT_FAILSAFE_CMD) == 9  # Q(8) + B(1)

    def test_round_trip_disarm_true(self):
        msg = FailsafeCmd(timestamp_ns=1_000_000, disarm=True)
        r = FailsafeCmd.unpack(msg.pack())
        assert r.timestamp_ns == 1_000_000
        assert r.disarm is True

    def test_round_trip_disarm_false(self):
        msg = FailsafeCmd(timestamp_ns=2_000_000, disarm=False)
        r = FailsafeCmd.unpack(msg.pack())
        assert r.disarm is False

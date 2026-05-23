import asyncio
import logging
import math
import struct

import pytest

from quadguide.link.crsf import (
    build_frame, CRSFParser,
    CRSF_ATTITUDE, CRSF_FLIGHT_MODE, CRSF_IMU_RAW, CRSF_RC_CHANNELS,
)
from quadguide.link.worker import _LinkState, _rx_loop
from quadguide.core.messages import AttitudeState, IMUFrame


class _FakeSerial:
    """Async-generator serial stub that yields a fixed byte sequence once."""
    def __init__(self, data: bytes):
        self._data = data

    async def read_stream(self):
        for b in self._data:
            yield b


class _FakeBus:
    def __init__(self):
        self.published: list[tuple[str, object]] = []

    def publish(self, topic: str, msg) -> None:
        self.published.append((topic, msg))

    def latest(self, topic: str):
        return None


def _run_rx(data: bytes) -> _FakeBus:
    serial = _FakeSerial(data)
    bus    = _FakeBus()
    state  = _LinkState()
    log    = logging.getLogger("test")
    asyncio.run(_rx_loop(serial, CRSFParser(), state, bus, log))
    return bus


def test_rx_loop_publishes_attitude():
    payload = struct.pack(">hhh", 1000, 500, -200)  # pitch, roll, yaw
    bus = _run_rx(build_frame(CRSF_ATTITUDE, payload))

    att_msgs = [m for t, m in bus.published if t == "fc/attitude"]
    assert len(att_msgs) == 1
    assert isinstance(att_msgs[0], AttitudeState)
    assert att_msgs[0].pitch_rad == pytest.approx(0.1,   rel=1e-4)
    assert att_msgs[0].roll_rad  == pytest.approx(0.05,  rel=1e-4)
    assert att_msgs[0].yaw_rad   == pytest.approx(-0.02, rel=1e-4)


def test_rx_loop_publishes_imu():
    payload = struct.pack(">hhhhhh", 0, 0, 1000, 1800, 0, 0)
    bus = _run_rx(build_frame(CRSF_IMU_RAW, payload))

    imu_msgs = [m for t, m in bus.published if t == "fc/imu"]
    assert len(imu_msgs) == 1
    assert isinstance(imu_msgs[0], IMUFrame)
    assert imu_msgs[0].az == pytest.approx(9.80665, rel=1e-5)
    assert imu_msgs[0].gx == pytest.approx(math.pi, rel=1e-5)


def test_rx_loop_imu_then_attitude_uses_gyro_for_rates():
    imu_payload = struct.pack(">hhhhhh", 0, 0, 0, 1800, 0, 0)  # gx = π rad/s
    att_payload = struct.pack(">hhh", 0, 0, 0)
    data = build_frame(CRSF_IMU_RAW, imu_payload) + build_frame(CRSF_ATTITUDE, att_payload)

    bus = _run_rx(data)

    att_msgs = [m for t, m in bus.published if t == "fc/attitude"]
    assert len(att_msgs) == 1
    assert att_msgs[0].roll_rate_rps == pytest.approx(math.pi, rel=1e-5)


def test_rx_loop_decodes_flight_mode_into_state():
    serial = _FakeSerial(build_frame(CRSF_FLIGHT_MODE, b"ANGLE\x00"))
    bus    = _FakeBus()
    state  = _LinkState()
    log    = logging.getLogger("test")
    asyncio.run(_rx_loop(serial, CRSFParser(), state, bus, log))
    assert state.flight_mode == "ANGLE"


def test_rx_loop_ignores_unknown_frames():
    bus = _run_rx(build_frame(CRSF_RC_CHANNELS, bytes(22)))
    assert not any(t == "fc/attitude" for t, _ in bus.published)
    assert not any(t == "fc/imu" for t, _ in bus.published)

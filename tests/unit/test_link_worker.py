import asyncio
import logging
import struct

from quadguide.link.crsf import build_frame, CRSFParser, CRSF_ATTITUDE
from quadguide.link.differentiator import AttitudeDifferentiator
from quadguide.link.worker import _rx_loop
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


def test_rx_loop_publishes_attitude_and_imu():
    payload = struct.pack(">hhh", 1000, 500, -200)  # pitch, roll, yaw (raw int16, decoder order)
    frame_bytes = build_frame(CRSF_ATTITUDE, payload)

    serial = _FakeSerial(frame_bytes)
    bus    = _FakeBus()
    diff   = AttitudeDifferentiator(alpha=1.0)
    parser = CRSFParser()
    log    = logging.getLogger("test")

    asyncio.run(_rx_loop(serial, parser, diff, bus, log))

    att_msgs = [m for t, m in bus.published if t == "fc/attitude"]
    imu_msgs = [m for t, m in bus.published if t == "fc/imu"]

    assert len(att_msgs) == 1
    assert len(imu_msgs) == 1
    assert isinstance(att_msgs[0], AttitudeState)
    assert isinstance(imu_msgs[0], IMUFrame)

    import pytest
    att = att_msgs[0]
    assert att.pitch_rad == pytest.approx(0.1,   rel=1e-4)
    assert att.roll_rad  == pytest.approx(0.05,  rel=1e-4)
    assert att.yaw_rad   == pytest.approx(-0.02, rel=1e-4)


def test_rx_loop_ignores_non_attitude_frames():
    from quadguide.link.crsf import CRSF_RC_CHANNELS
    frame_bytes = build_frame(CRSF_RC_CHANNELS, bytes(22))

    serial = _FakeSerial(frame_bytes)
    bus    = _FakeBus()
    diff   = AttitudeDifferentiator(alpha=1.0)
    parser = CRSFParser()
    log    = logging.getLogger("test")

    asyncio.run(_rx_loop(serial, parser, diff, bus, log))

    assert not any(t == "fc/attitude" for t, _ in bus.published)
    assert not any(t == "fc/imu" for t, _ in bus.published)

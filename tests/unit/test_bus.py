import pytest
from quadguide.core.bus import Bus, TOPICS
from quadguide.core.messages import AccelCmd, ControlCmd


@pytest.fixture
def bus():
    b = Bus(ring_depth=4)
    yield b
    b.close()


class TestTopicRegistry:
    def test_all_nine_topics_registered(self):
        b = Bus(ring_depth=2)
        try:
            expected = {
                "kcf/estimate", "nano/estimate", "target/estimate",
                "fc/attitude", "fc/imu", "guidance/accel",
                "control/cmd", "lockon/cmd", "system/health",
            }
            assert set(b._topics.keys()) == expected
        finally:
            b.close()

    def test_topics_constant_has_nine_entries(self):
        assert len(TOPICS) == 9


class TestLatest:
    def test_returns_none_when_empty(self, bus):
        assert bus.latest("guidance/accel") is None

    def test_unknown_topic_raises(self, bus):
        with pytest.raises(KeyError):
            bus.latest("nonexistent/topic")


class TestPublish:
    def test_publish_then_latest_returns_message(self, bus):
        msg = AccelCmd(timestamp_ns=1000, ax=1.0, ay=-0.5)
        bus.publish("guidance/accel", msg)
        assert bus.latest("guidance/accel") == msg

    def test_latest_returns_most_recent(self, bus):
        msg1 = AccelCmd(timestamp_ns=1000, ax=1.0, ay=0.0)
        msg2 = AccelCmd(timestamp_ns=2000, ax=2.0, ay=0.0)
        bus.publish("guidance/accel", msg1)
        bus.publish("guidance/accel", msg2)
        assert bus.latest("guidance/accel") == msg2

    def test_ring_wrap_returns_latest(self, bus):
        # ring_depth=4; publishing 5 messages wraps the ring
        msgs = [AccelCmd(timestamp_ns=i * 1000, ax=float(i), ay=0.0) for i in range(5)]
        for msg in msgs:
            bus.publish("guidance/accel", msg)
        assert bus.latest("guidance/accel") == msgs[-1]

    def test_publish_unknown_topic_raises(self, bus):
        msg = AccelCmd(timestamp_ns=0, ax=0.0, ay=0.0)
        with pytest.raises(KeyError):
            bus.publish("nonexistent/topic", msg)

    def test_different_topics_independent(self, bus):
        accel = AccelCmd(timestamp_ns=1, ax=1.0, ay=2.0)
        ctrl = ControlCmd(timestamp_ns=2, roll_deg=5.0, pitch_deg=-2.0,
                          yaw_rate_dps=0.0, throttle_norm=0.5)
        bus.publish("guidance/accel", accel)
        bus.publish("control/cmd", ctrl)
        assert bus.latest("guidance/accel") == accel
        assert bus.latest("control/cmd") == ctrl

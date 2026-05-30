import pytest
import threading
import time
from quadguide.core.bus import Bus, TOPICS
from quadguide.core.messages import AccelCmd, ControlCmd


@pytest.fixture
def bus():
    b = Bus(ring_depth=4)
    yield b
    b.close()


class TestTopicRegistry:
    def test_all_topics_registered(self):
        b = Bus(ring_depth=2)
        try:
            expected = {
                "target/estimate",
                "fc/attitude", "fc/imu", "guidance/accel",
                "control/cmd", "lockon/cmd", "system/health", "arm/cmd",
            }
            assert set(b._topics.keys()) == expected
        finally:
            b.close()

    def test_topics_constant_has_eight_entries(self):
        assert len(TOPICS) == 8


class TestLatest:
    def test_returns_none_when_empty(self, bus):
        assert bus.latest("guidance/accel") is None

    def test_unknown_topic_raises(self, bus):
        with pytest.raises(KeyError):
            bus.latest("nonexistent/topic")


class TestPublish:
    # Float values in these tests are chosen to be exactly representable as
    # IEEE 754 float32 so that == comparison survives the pack/unpack round-trip.
    # Use pytest.approx for any future tests with non-exact values (e.g. 0.1, 0.3).
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


class TestSubscribeOne:
    def test_blocks_until_publish(self, bus):
        msg = AccelCmd(timestamp_ns=999, ax=3.0, ay=1.5)

        def publisher():
            time.sleep(0.05)
            bus.publish("guidance/accel", msg)

        t = threading.Thread(target=publisher)
        t.start()
        start = time.monotonic()
        received = bus.subscribe_one("guidance/accel")
        elapsed = time.monotonic() - start
        t.join()

        assert received == msg
        assert elapsed >= 0.04, (
            f"subscribe_one returned too quickly ({elapsed:.3f}s) — "
            "pipe blocking is broken"
        )

    def test_unknown_topic_raises(self, bus):
        with pytest.raises(KeyError):
            bus.subscribe_one("nonexistent/topic")


class TestSubscribeAny:
    def test_wakes_on_first_publish(self, bus):
        msg = ControlCmd(
            timestamp_ns=5000, roll_deg=5.0, pitch_deg=-2.0,
            yaw_rate_dps=0.0, throttle_norm=0.5,
        )

        def publisher():
            time.sleep(0.02)
            bus.publish("control/cmd", msg)

        t = threading.Thread(target=publisher)
        t.start()
        topic, received = bus.subscribe_any(["guidance/accel", "control/cmd"])
        t.join()

        assert topic == "control/cmd"
        assert received == msg

    def test_unknown_topic_in_list_raises(self, bus):
        with pytest.raises(KeyError):
            bus.subscribe_any(["guidance/accel", "nonexistent/topic"])


class TestDetach:
    def test_detach_does_not_raise(self):
        b = Bus(ring_depth=2)
        b.detach()  # must not raise; no unlink, just close local references

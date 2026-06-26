import json
import pytest
import asyncio
from starlette.testclient import TestClient

from quadguide.core.messages import (
    BoundingBox, ControlCmd, LockOnCmd, TrackerEstimate, TrackerHealth,
)
from quadguide.ground.server import create_app


class _MockBus:
    def __init__(self):
        self.published: list[tuple[str, object]] = []

    def latest(self, topic: str):
        return None

    def publish(self, topic: str, msg) -> None:
        self.published.append((topic, msg))

    def detach(self) -> None:
        pass


class _MockFrameBuffer:
    def read_latest(self):
        return None, 0


@pytest.fixture
def client():
    app = create_app(_MockBus(), _MockFrameBuffer())
    with TestClient(app) as c:
        yield c


@pytest.fixture
def bus_client():
    bus = _MockBus()
    app = create_app(bus, _MockFrameBuffer())
    with TestClient(app) as c:
        yield bus, c


def test_lockon_returns_ok(client):
    resp = client.post("/lockon", json={"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_lockon_publishes_to_lockon_cmd(bus_client):
    bus, client = bus_client
    client.post("/lockon", json={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4})
    lockon = [(t, m) for t, m in bus.published if t == "lockon/cmd"]
    assert len(lockon) == 1
    cmd = lockon[0][1]
    assert isinstance(cmd, LockOnCmd)
    assert cmd.bbox.x == pytest.approx(0.1)
    assert cmd.bbox.y == pytest.approx(0.2)
    assert cmd.bbox.w == pytest.approx(0.3)
    assert cmd.bbox.h == pytest.approx(0.4)


def test_lockon_seq_starts_at_1(bus_client):
    bus, client = bus_client
    client.post("/lockon", json={"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2})
    cmd = [m for t, m in bus.published if t == "lockon/cmd"][0]
    assert cmd.seq == 1


def test_lockon_seq_increments(bus_client):
    bus, client = bus_client
    client.post("/lockon", json={"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2})
    client.post("/lockon", json={"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2})
    cmds = [m for t, m in bus.published if t == "lockon/cmd"]
    assert cmds[0].seq == 1
    assert cmds[1].seq == 2


def _read_one_sse_event(client) -> dict:
    # TestClient.stream() hangs on infinite SSE generators; drive _sse directly.
    from quadguide.ground.server import _sse

    async def _one():
        async for line in _sse(client.app):
            if line.startswith("data:"):
                return json.loads(line[len("data:"):].strip())
        raise AssertionError("no SSE event received")

    try:
        return asyncio.run(asyncio.wait_for(_one(), timeout=1.0))
    except asyncio.TimeoutError:
        raise AssertionError("SSE stream timed out")
    except Exception as e:
        raise AssertionError(f"Failed to read SSE event: {e}")


@pytest.fixture
def client_with_estimate():
    # End-to-end latency now comes from control/cmd: timestamp_ns - origin_ns.
    class _EstimateBus(_MockBus):
        def latest(self, topic: str):
            if topic == "target/estimate":
                return TrackerEstimate(
                    timestamp_ns=1_000_000_000,
                    bbox=BoundingBox(0.2, 0.2, 0.3, 0.3),
                    confidence=0.9,
                    tracker_health=TrackerHealth.NOMINAL,
                    origin_ns=995_000_000,
                )
            if topic == "control/cmd":
                # cum = timestamp_ns - origin_ns = 5 ms (glass→control-publish)
                return ControlCmd(
                    timestamp_ns=1_000_000_000,
                    roll_deg=0.0, pitch_deg=0.0,
                    yaw_rate_dps=0.0, throttle_norm=0.0,
                    origin_ns=995_000_000,
                )
            return None
    app = create_app(_EstimateBus(), _MockFrameBuffer())
    with TestClient(app) as c:
        yield c


def test_telemetry_includes_latency_keys(client):
    data = _read_one_sse_event(client)
    assert "latency_ms" in data
    assert "latency_avg_ms" in data


def test_telemetry_fc_armed_null_without_heartbeat(client):
    data = _read_one_sse_event(client)
    assert data["fc_armed"] is None
    assert data["fc_mode"] is None


def test_telemetry_fc_armed_reflects_fc_status():
    from quadguide.core.messages import FCStatus

    class _FCBus(_MockBus):
        def latest(self, topic: str):
            if topic == "fc/status":
                return FCStatus(timestamp_ns=1_000_000, armed=True, custom_mode=4)
            return None
    app = create_app(_FCBus(), _MockFrameBuffer())
    with TestClient(app) as c:
        data = _read_one_sse_event(c)
    assert data["fc_armed"] is True
    assert data["fc_mode"] == 4


def test_telemetry_latency_null_when_no_lineage(client):
    data = _read_one_sse_event(client)
    assert data["latency_ms"] is None
    assert data["latency_avg_ms"] is None


def test_telemetry_latency_ms_is_glass_to_control(client_with_estimate):
    data = _read_one_sse_event(client_with_estimate)
    assert data["latency_ms"] == pytest.approx(5.0, rel=0.01)


def test_telemetry_latency_avg_ms_after_one_sample(client_with_estimate):
    data = _read_one_sse_event(client_with_estimate)
    assert data["latency_avg_ms"] == pytest.approx(5.0, rel=0.01)


from quadguide.core.messages import ArmCmd


def test_arm_returns_ok(client):
    resp = client.post("/arm", json={"armed": True})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_arm_publishes_arm_cmd_true(bus_client):
    bus, client = bus_client
    client.post("/arm", json={"armed": True})
    arm_msgs = [(t, m) for t, m in bus.published if t == "arm/cmd"]
    assert len(arm_msgs) == 1
    assert isinstance(arm_msgs[0][1], ArmCmd)
    assert arm_msgs[0][1].armed is True


def test_arm_publishes_arm_cmd_false(bus_client):
    bus, client = bus_client
    client.post("/arm", json={"armed": False})
    arm_msgs = [(t, m) for t, m in bus.published if t == "arm/cmd"]
    assert len(arm_msgs) == 1
    assert arm_msgs[0][1].armed is False


def test_arm_missing_field_returns_422(client):
    resp = client.post("/arm", json={})
    assert resp.status_code == 422


def test_index_serves_verbose_by_default():
    app = create_app(_MockBus(), _MockFrameBuffer())
    with TestClient(app) as c:
        body = c.get("/").text
    assert 'id="right-col"' in body          # verbose-only marker


def test_index_serves_minimal_when_configured():
    app = create_app(_MockBus(), _MockFrameBuffer(), {"ground": {"ui_mode": "minimal"}})
    with TestClient(app) as c:
        body = c.get("/").text
    assert 'id="pip"' in body                # minimal-only marker

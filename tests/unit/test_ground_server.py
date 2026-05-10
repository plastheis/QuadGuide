import json
import pytest
from starlette.testclient import TestClient

from quadguide.core.messages import (
    ActiveTracker, BoundingBox, LockOnCmd, TargetEstimate, TrackerHealth,
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


@pytest.fixture
def client_with_estimate():
    class _EstimateBus(_MockBus):
        def latest(self, topic: str):
            if topic == "target/estimate":
                return TargetEstimate(
                    timestamp_ns=1_000_000_000,
                    bbox=BoundingBox(0.2, 0.2, 0.3, 0.3),
                    centroid_norm=(0.0, 0.0),
                    confidence=0.9,
                    tracker_health=TrackerHealth.NOMINAL,
                    active_tracker=ActiveTracker.CCV,
                    latency_ns=5_000_000,  # 5 ms
                )
            return None
    app = create_app(_EstimateBus(), _MockFrameBuffer())
    with TestClient(app) as c:
        yield c


def _read_one_sse_event(client, path: str) -> dict:
    # For SSE testing, we use a workaround to break out of infinite streaming.
    # We set a flag on the app to break after one event, run the request in a thread
    # with a timeout, then restore the flag.
    import threading
    import time

    # Set a test mode flag that will make _sse break after yielding one event
    client.app.state.test_mode_break_after_one = True

    result = []
    error = []

    def fetch():
        try:
            # Use stream() which should now break after getting one event
            with client.stream("GET", path) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        result.append(json.loads(line[len("data:"):].strip()))
                        return
                if not result:
                    error.append("No SSE data line received")
        except Exception as e:
            error.append(f"{type(e).__name__}: {e}")
        finally:
            client.app.state.test_mode_break_after_one = False

    thread = threading.Thread(target=fetch)
    thread.daemon = True
    thread.start()
    thread.join(timeout=2.0)

    if error:
        raise AssertionError(error[0])
    if not result:
        raise AssertionError("SSE stream timed out or no data received")
    return result[0]


def test_telemetry_includes_latency_keys(client):
    data = _read_one_sse_event(client, "/telemetry")
    assert "latency_ms" in data
    assert "latency_avg_ms" in data


def test_telemetry_latency_null_when_no_estimate(client):
    data = _read_one_sse_event(client, "/telemetry")
    assert data["latency_ms"] is None
    assert data["latency_avg_ms"] is None


def test_telemetry_latency_ms_matches_estimate(client_with_estimate):
    data = _read_one_sse_event(client_with_estimate, "/telemetry")
    assert data["latency_ms"] == pytest.approx(5.0, rel=0.01)


def test_telemetry_latency_avg_ms_after_one_sample(client_with_estimate):
    data = _read_one_sse_event(client_with_estimate, "/telemetry")
    assert data["latency_avg_ms"] == pytest.approx(5.0, rel=0.01)

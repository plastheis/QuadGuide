import time

from edgecv.runtime.orchestrator import Orchestrator, WorkerSpec


def _echo_worker(ctx, started, stop):
    """Module-level so it is picklable under spawn. Sets `started`, waits for stop."""
    started.set()
    while not stop.is_set():
        time.sleep(0.01)


def test_orchestrator_spawns_and_stops_worker():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    started = ctx.Event()
    stop = ctx.Event()
    orch = Orchestrator(mp_context="spawn")
    orch.add_worker(WorkerSpec(name="echo", target=_echo_worker,
                               args=(None, started, stop)))
    with orch:
        orch.start()
        assert started.wait(timeout=10.0), "worker never signalled start"
        assert orch.is_alive("echo")
        stop.set()
    # After context exit, the worker is reaped.
    assert not orch.is_alive("echo")


def test_close_is_idempotent():
    orch = Orchestrator(mp_context="spawn")
    orch.close()
    orch.close()  # must not raise

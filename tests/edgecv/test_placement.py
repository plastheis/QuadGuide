import textwrap

from edgecv.runtime.placement import (
    BoardProfile,
    ProcessPlacement,
    default_profile,
    load_profile,
)


def test_load_profile_parses_processes(tmp_path):
    p = tmp_path / "board.yaml"
    p.write_text(textwrap.dedent("""
        board: rk3588
        processes:
          caller:
            cpu_affinity: [4, 5, 6, 7]
            sched: {policy: FIFO, priority: 80}
          detector:
            cpu_affinity: [0, 1]
            npu_core: 0
            backend: rknn
    """))
    prof = load_profile(p)
    assert prof.board == "rk3588"
    assert prof.processes["caller"].cpu_affinity == [4, 5, 6, 7]
    assert prof.processes["caller"].sched == {"policy": "FIFO", "priority": 80}
    assert prof.processes["detector"].npu_core == 0
    assert prof.processes["detector"].backend == "rknn"


def test_default_profile_is_rk3588_and_has_caller_and_detector():
    prof = default_profile()
    assert prof.board == "rk3588"
    assert "caller" in prof.processes
    assert "detector" in prof.processes


def test_apply_affinity_is_safe_noop_when_unsupported(monkeypatch):
    # If os.sched_setaffinity is missing/raises, apply() must not crash the process.
    placement = ProcessPlacement(cpu_affinity=[0], sched=None, npu_core=None, backend=None)
    import edgecv.runtime.placement as mod
    monkeypatch.setattr(mod.os, "sched_setaffinity",
                        lambda pid, mask: (_ for _ in ()).throw(OSError("nope")),
                        raising=False)
    placement.apply()  # should swallow the error and return


def test_board_profile_constructs():
    bp = BoardProfile(board="x")
    assert bp.processes == {}

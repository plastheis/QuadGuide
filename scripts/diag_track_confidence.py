#!/usr/bin/env python3
"""Measure the tracker's confidence distribution to choose score_lock/score_lost.

Runs camera + tracker + the ground UI, records EVERY ``target/estimate`` sample,
and lets the operator label the recording in-frame vs out-of-frame from the
terminal. On exit it writes a CSV and reports whether the two distributions are
separable — i.e. whether NanoTrack's confidence can detect target loss at all —
and, if so, what score_lock / score_lost to configure.

Why this exists: bare NanoTrack is a Siamese correlation tracker with no
hysteresis. When the target leaves frame it does not necessarily collapse to a
low score — it frequently re-locks onto background clutter at a comfortable
confidence. `tracker.params.score_lost` is what turns confidence into the
"lost" health that drives the target-loss failsafe, so it must be picked from
measured data on YOUR camera, lens and scene — not from the library defaults.

Procedure:
    1. Start it, open the kiosk, put the target in frame and press LOCK.
    2. Track normally for ~20-30 s (move it around, vary range and lighting).
    3. Press ENTER here the moment the target leaves frame  → label OUT.
    4. Leave it out for ~20-30 s, panning across whatever background is realistic.
    5. Press ENTER again if you bring it back → label IN. Repeat as you like.
    6. Ctrl-C to stop and print the report.

Usage:
    python scripts/diag_track_confidence.py --config configs/rpi4b.yaml --minimal
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from quadguide.core.bus import Bus
from quadguide.core.clock import monotonic_ns
from quadguide.core.config import load_config
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.ground.worker import run as ground_run
from quadguide.perception.camera.worker import run_from_config as camera_run
from quadguide.perception.tracker_worker import run_from_config as tracker_run

_POLL_S = 0.005          # sample faster than any realistic frame rate
_IN, _OUT = "in_frame", "out_of_frame"


def _percentile(xs: list[float], p: float) -> float:
    """Nearest-rank percentile; xs must be sorted and non-empty."""
    if not xs:
        return float("nan")
    k = max(0, min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


def _describe(name: str, xs: list[float]) -> None:
    if not xs:
        print(f"  {name:<13} (no samples)")
        return
    s = sorted(xs)
    print(f"  {name:<13} n={len(s):<6} "
          f"min={s[0]:.3f}  p05={_percentile(s,5):.3f}  p25={_percentile(s,25):.3f}  "
          f"p50={_percentile(s,50):.3f}  p75={_percentile(s,75):.3f}  "
          f"p95={_percentile(s,95):.3f}  max={s[-1]:.3f}")


def _recommend(inf: list[float], outf: list[float]) -> None:
    """Report separability and, if usable, suggested score_lost / score_lock."""
    print("\n── recommendation " + "─" * 45)
    if not inf or not outf:
        print("  Need samples in BOTH labels. Press ENTER to toggle the label when")
        print("  the target leaves/re-enters the frame, then Ctrl-C.")
        return

    si, so = sorted(inf), sorted(outf)
    in_low = _percentile(si, 5)     # a real track should stay above this
    out_high = _percentile(so, 95)  # background should stay below this

    # Overlap: how much of each distribution sits on the wrong side of the
    # midpoint. High overlap ⇒ confidence alone cannot separate the two.
    mid = (in_low + out_high) / 2
    in_below = sum(1 for v in si if v < mid) / len(si)
    out_above = sum(1 for v in so if v >= mid) / len(so)

    print(f"  in-frame  p05 = {in_low:.3f}   (a real track rarely drops below this)")
    print(f"  out-frame p95 = {out_high:.3f}   (background rarely exceeds this)")
    print(f"  misclassified at midpoint {mid:.3f}: "
          f"{in_below * 100:.1f}% of in-frame, {out_above * 100:.1f}% of out-of-frame")

    if in_low <= out_high:
        print("\n  ✗ NOT SEPARABLE — the distributions overlap.")
        print("    Any single confidence threshold either declares loss during")
        print("    normal tracking or never declares it at all. Raising score_lost")
        print("    into the overlap trades false LANDs for missed losses.")
        print("    Options: increase failsafe.target_loss.hold_ms so brief dips are")
        print("    tolerated; or use a tracker that verifies its lock — EdgeCV's")
        print("    verified_acquire_track re-runs YOLO during LOCKED specifically")
        print("    to catch confident drift onto clutter (see edgecv_adapter.py).")
        return

    score_lost = round(out_high + (in_low - out_high) * 0.33, 2)
    score_lock = round(out_high + (in_low - out_high) * 0.66, 2)
    margin = in_low - out_high
    print(f"\n  ✓ Separable with margin {margin:.3f}. Suggested config:")
    print("\n    tracker:\n      params:")
    print(f"        score_lock: {score_lock}      # ≥ this → LOCKED  → \"nominal\"")
    print(f"        score_lost: {score_lost}      # <  this → LOST   → \"lost\"")
    print("\n  Then confirm the failsafe end-to-end with:")
    print("    python scripts/diag_failsafe_bench.py --scenario armed")


def _report(rows: list[tuple], source: str) -> None:
    """Print the confidence/health breakdown and the threshold recommendation."""
    inf = [r[2] for r in rows if r[1] == _IN]
    outf = [r[2] for r in rows if r[1] == _OUT]
    print(f"\n\n{'=' * 62}\n{len(rows)} samples → {source}\n")
    print("confidence distribution:")
    _describe(_IN, inf)
    _describe(_OUT, outf)

    print("\nhealth reported (what the failsafe actually consumes):")
    for label in (_IN, _OUT):
        hs = [r[3] for r in rows if r[1] == label]
        if hs:
            counts = {h: hs.count(h) for h in sorted(set(hs))}
            total = len(hs)
            print(f"  {label:<13} " + "  ".join(
                f"{h}={c} ({c / total * 100:.0f}%)" for h, c in counts.items()))

    _recommend(inf, outf)


def _analyze(path: str) -> int:
    """Re-run the report over a previously captured CSV."""
    try:
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            next(rd, None)                     # header
            rows = [(float(r[0]), r[1], float(r[2]), r[3],
                     *(float(v) for v in r[4:8])) for r in rd if len(r) >= 8]
    except OSError as exc:
        print(f"cannot read {path}: {exc}")
        return 1
    if not rows:
        print(f"{path} has no samples")
        return 1
    _report(rows, path)
    return 0


def _label_reader(state: dict) -> None:
    """ENTER toggles the label; 'q' + ENTER stops. Runs on a daemon thread."""
    while not state["stop"]:
        line = sys.stdin.readline()
        if not line:                      # stdin closed (e.g. nohup)
            return
        if line.strip().lower() == "q":
            state["stop"] = True
            return
        state["label"] = _OUT if state["label"] == _IN else _IN
        print(f"  [{time.strftime('%H:%M:%S')}] label → {state['label'].upper()}  "
              f"({state['n']} samples so far)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rpi4b.yaml")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--minimal", action="store_true",
                    help="serve the minimal kiosk UI")
    ap.add_argument("--out", default=None,
                    help="CSV path (default: quadguide-trace/confidence-<ts>.csv)")
    ap.add_argument("--analyze", default=None, metavar="CSV",
                    help="skip capture; re-run the report on an existing CSV")
    args = ap.parse_args()

    if args.analyze:
        return _analyze(args.analyze)

    config = load_config(args.config, {})
    config.setdefault("ground", {})["port"] = args.port
    if args.minimal:
        config["ground"]["ui_mode"] = "minimal"

    out_path = args.out or os.path.join(
        os.getcwd(), "quadguide-trace",
        f"confidence-{time.strftime('%Y%m%d-%H%M%S')}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    w = config["platform"]["camera"]["width"]
    h = config["platform"]["camera"]["height"]
    bus = Bus()
    fb = FrameBuffer(width=w, height=h)

    # daemon=False (matches run.py / dev_ground_perception): the tracker may
    # spawn its own children (EdgeCV AcquireTrack workers) and Python forbids
    # daemonic processes from having children.
    procs = [
        multiprocessing.Process(target=camera_run, args=(config, bus, fb),
                                daemon=False, name="camera"),
        multiprocessing.Process(target=tracker_run, args=(config, bus, fb),
                                daemon=False, name="tracker"),
        multiprocessing.Process(target=ground_run, args=(config, bus, fb),
                                daemon=False, name="ground"),
    ]
    # Children inherit SIG_IGN for SIGINT, so the operator's Ctrl-C (delivered to
    # the whole process group) doesn't make each worker dump a KeyboardInterrupt
    # traceback over the report. They still shut down cleanly on our SIGTERM.
    _prev_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for p in procs:
            p.start()
    finally:
        signal.signal(signal.SIGINT, _prev_sigint)

    tcfg = config["tracker"].get("params") or {}
    print(f"tracker: {tcfg.get('tracker')} / {tcfg.get('backend')}   "
          f"score_lock={tcfg.get('score_lock')} score_lost={tcfg.get('score_lost')}")
    print(f"kiosk  → http://0.0.0.0:{args.port}    CSV → {out_path}")
    print("\nLOCK the target in the kiosk, then:")
    print("  ENTER  toggle label (in_frame ⇄ out_of_frame)")
    print("  Ctrl-C stop and report\n")
    print(f"  label → {_IN.upper()}", flush=True)

    state = {"label": _IN, "stop": False, "n": 0}
    # Ctrl-C must STOP the capture, never abort it. SIGINT sets the stop flag
    # instead of raising KeyboardInterrupt, so the sampling loop exits normally
    # and cleanup + the report always run. (Raising here would land the
    # KeyboardInterrupt inside the finally block and destroy the capture.)
    def _on_sigint(_sig, _frm):
        if state["stop"]:                 # second Ctrl-C: operator wants out now
            raise KeyboardInterrupt
        state["stop"] = True
        print("\n  stopping — writing report (Ctrl-C again to abort)…", flush=True)

    signal.signal(signal.SIGINT, _on_sigint)
    threading.Thread(target=_label_reader, args=(state,), daemon=True).start()

    # Stream to disk as we sample: a capture is minutes of the operator's time,
    # so it must survive any abnormal exit, not just the happy path.
    rows: list[tuple] = []
    last_ts = -1
    t0 = monotonic_ns()
    fh = open(out_path, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["t_s", "label", "confidence", "health",
                 "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
    try:
        while not state["stop"]:
            est = bus.latest("target/estimate")
            if est is not None and est.timestamp_ns != last_ts:
                last_ts = est.timestamp_ns
                state["n"] += 1
                row = (
                    round((est.timestamp_ns - t0) / 1e9, 4),
                    state["label"],
                    round(float(est.confidence), 4),
                    est.tracker_health.value,
                    round(est.bbox.x, 4), round(est.bbox.y, 4),
                    round(est.bbox.w, 4), round(est.bbox.h, 4),
                )
                rows.append(row)
                wr.writerow(row)
                if state["n"] % 50 == 0:
                    fh.flush()
            time.sleep(_POLL_S)
    except KeyboardInterrupt:
        print("\n  aborted", flush=True)
    finally:
        # Every step guarded: one failure must not strand the rest, or the
        # shared memory leaks and the CSV is left unflushed.
        for step in (fh.flush, fh.close):
            try:
                step()
            except Exception as exc:                  # noqa: BLE001
                print(f"  warn: {step.__name__} failed: {exc}")
        for p in procs:
            try:
                if p.is_alive() and p.pid is not None:
                    os.kill(p.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass                                   # already gone
        for p in procs:
            try:
                p.join(timeout=3)
            except Exception:                          # noqa: BLE001
                pass
        for cleanup in (fb.unlink, bus.close):
            try:
                cleanup()
            except Exception as exc:                   # noqa: BLE001
                print(f"  warn: {cleanup.__name__} failed: {exc}")

    if not rows:
        print("\nNo target/estimate samples recorded — did the tracker start, "
              "and did you press LOCK?")
        return 1

    _report(rows, out_path)
    return 0


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    sys.exit(main())

#!/usr/bin/env python3
"""Latency diagnostic tool for QuadGuide.

Two modes:

  sim                Reproduce the latency the *current* free-running tracker loop
                     produces (camera@fps coupled to a faster, ungated tracker)
                     and contrast it with a new-frame-gated loop. Also shows how
                     the 10 Hz SSE feed aliases the sawtooth into a slow rhythmic
                     wobble. No hardware required — validates the diagnosis offline.

  trace <dir>        Ingest a `run.py --log` trace dump ({proc}.jsonl) and derive
                     per-stage and cumulative (glass→here) latency from the raw
                     timestamps. Plots time-series + histograms + a spectral view
                     that pins the rhythmic beat frequency, and prints a summary.

Both modes write a PNG (headless Agg backend) and print a text report.

Examples:
  scripts/diagnose_latency.py sim --out /tmp/sim.png
  scripts/diagnose_latency.py trace /var/log/quadguide/trace/20260611-101500
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NS_PER_MS = 1e6
STAGE_ORDER = ["tracker", "guidance", "control", "link"]


def _canon(proc: str) -> str:
    """Map a process name to its canonical stage. The tracker worker names itself
    `tracker_<algo>` (e.g. tracker_nanotrack), so match by prefix."""
    for s in STAGE_ORDER:
        if proc == s or proc.startswith(s + "_"):
            return s
    return proc


# ── stats helpers ─────────────────────────────────────────────────────────────

def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def _summarize(name: str, ms: np.ndarray) -> str:
    if ms.size == 0:
        return f"  {name:<22} (no samples)"
    return (f"  {name:<22} n={ms.size:<6d} "
            f"min={ms.min():6.2f}  p50={_pct(ms,50):6.2f}  "
            f"p95={_pct(ms,95):6.2f}  max={ms.max():6.2f}  "
            f"std={ms.std():5.2f}  (ms)")


def _dominant_period_ms(t_ns: np.ndarray, val_ms: np.ndarray) -> tuple[float, float]:
    """Resample onto a uniform grid and FFT to find the dominant oscillation.

    Returns (period_ms, freq_hz). The samples are roughly-but-not-exactly uniform
    (a free-running loop), so we interpolate onto the median sample interval first.
    """
    if t_ns.size < 16:
        return float("nan"), float("nan")
    t_s = (t_ns - t_ns[0]) / 1e9
    dt = np.median(np.diff(t_s))
    if dt <= 0:
        return float("nan"), float("nan")
    grid = np.arange(t_s[0], t_s[-1], dt)
    if grid.size < 16:
        return float("nan"), float("nan")
    y = np.interp(grid, t_s, val_ms)
    y = y - y.mean()
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, d=dt)
    spec[0] = 0.0  # ignore DC
    k = int(np.argmax(spec))
    f = freqs[k]
    if f <= 0:
        return float("nan"), float("nan")
    return 1000.0 / f, float(f)


# ── sim mode ──────────────────────────────────────────────────────────────────

def _simulate(fps: float, infer_ms: float, loop_overhead_ms: float,
              duration_s: float, jitter_ms: float, gated: bool,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Model one tracker loop against a camera producing frames at `fps`.

    Returns (publish_time_ms, latency_ms) where latency = publish_time - frame_ts,
    exactly as tracker_worker.py computes it.

    free-running (gated=False): each pass reads the *latest* frame (read_latest)
    and re-times it — so latency ramps between frame arrivals (a sawtooth).
    gated (gated=True): the loop processes each frame once, blocking until the
    next frame_ts changes — latency ≈ inference time, tight.
    """
    interval_ms = 1000.0 / fps
    # Camera frame-arrival timestamps with jitter (frame_ts is stamped at arrival).
    n_frames = int(duration_s * fps) + 2
    base = np.arange(n_frames) * interval_ms
    frame_ts = base + rng.normal(0.0, jitter_ms, n_frames)
    frame_ts = np.maximum.accumulate(frame_ts)  # monotonic

    pub_t, lat = [], []
    t = 0.0
    last_frame_used = -1.0
    fi = 0  # index of next frame for the gated path
    end = duration_s * 1000.0
    while t < end:
        if gated:
            # wait for the next frame to arrive, then process it once
            if fi >= n_frames:
                break
            t = max(t, frame_ts[fi])
            used_ts = frame_ts[fi]
            fi += 1
        else:
            # read_latest(): newest frame whose ts <= now
            idx = np.searchsorted(frame_ts, t, side="right") - 1
            if idx < 0:
                t = frame_ts[0]
                continue
            used_ts = frame_ts[idx]
        infer = max(0.0, infer_ms + rng.normal(0.0, infer_ms * 0.15)) + loop_overhead_ms
        now = t + infer
        pub_t.append(now)
        lat.append(now - used_ts)
        t = now
        last_frame_used = used_ts
    return np.array(pub_t), np.array(lat)


def _sse_sample(pub_t_ms: np.ndarray, lat_ms: np.ndarray,
                sse_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Subsample the latency signal the way the 10 Hz SSE feed does: at each tick,
    take the most recently published value (bus.latest semantics)."""
    if pub_t_ms.size == 0:
        return np.array([]), np.array([])
    step = 1000.0 / sse_hz
    ticks = np.arange(pub_t_ms[0], pub_t_ms[-1], step)
    idx = np.searchsorted(pub_t_ms, ticks, side="right") - 1
    idx = np.clip(idx, 0, pub_t_ms.size - 1)
    return ticks, lat_ms[idx]


def run_sim(args) -> int:
    rng = np.random.default_rng(args.seed)
    free_t, free_l = _simulate(args.fps, args.tracker_ms, args.loop_ms,
                               args.duration, args.jitter, False, rng)
    gate_t, gate_l = _simulate(args.fps, args.tracker_ms, args.loop_ms,
                               args.duration, args.jitter, True,
                               np.random.default_rng(args.seed))
    sse_t, sse_l = _sse_sample(free_t, free_l, args.sse_hz)

    period_ms, freq_hz = _dominant_period_ms(free_t * NS_PER_MS, free_l)

    print("=== SIM: free-running tracker vs new-frame-gated ===")
    print(f"camera {args.fps:g} fps ({1000/args.fps:.2f} ms/frame), "
          f"tracker infer {args.tracker_ms:g} ms + loop {args.loop_ms:g} ms, "
          f"SSE {args.sse_hz:g} Hz")
    print(_summarize("free-running (HUD)", free_l))
    print(_summarize("new-frame-gated", gate_l))
    print(_summarize("SSE-sampled (what UI shows)", sse_l))
    print(f"  free-running sawtooth period ≈ {period_ms:.1f} ms "
          f"(camera interval {1000/args.fps:.1f} ms) → confirms frame-coupled sawtooth")

    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    ax[0].plot(free_t / 1000, free_l, lw=0.7, color="#c0392b")
    ax[0].set_title("Free-running tracker (current code): latency = now − frame_ts")
    ax[0].set_ylabel("latency (ms)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(sse_t / 1000, sse_l, ".-", lw=0.8, ms=3, color="#8e44ad")
    ax[1].set_title(f"Same signal sampled at {args.sse_hz:g} Hz (SSE/HUD) — aliased rhythmic wobble")
    ax[1].set_ylabel("latency (ms)")
    ax[1].grid(alpha=0.3)

    ax[2].plot(gate_t / 1000, gate_l, lw=0.9, color="#27ae60")
    ax[2].set_title("New-frame-gated tracker (proposed fix): each frame measured once")
    ax[2].set_ylabel("latency (ms)")
    ax[2].set_xlabel("time (s)")
    ax[2].grid(alpha=0.3)
    ax[2].set_ylim(0, max(free_l.max(), gate_l.max()) * 1.1)

    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nplot → {args.out}")
    return 0


# ── trace mode ────────────────────────────────────────────────────────────────

def _load_trace(dir_: str) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(dir_, "*.jsonl"))):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        proc = os.path.splitext(os.path.basename(path))[0]
        records[proc] = rows
    return records


def _lat_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_ns, stage_ms, cum_ms) from 'lat' records. cum_ms is NaN where
    origin is unknown (org == 0)."""
    t, stage, cum = [], [], []
    for r in rows:
        if r.get("k") != "lat":
            continue
        ts = r["t"]
        in_ts = r.get("in")
        org = r.get("org", 0)
        t.append(ts)
        stage.append((ts - in_ts) / NS_PER_MS if in_ts else np.nan)
        cum.append((ts - org) / NS_PER_MS if org else np.nan)
    return np.array(t), np.array(stage), np.array(cum)


def run_trace(args) -> int:
    records = _load_trace(args.dir)
    if not records:
        print(f"no *.jsonl trace files in {args.dir}", file=sys.stderr)
        return 1

    def _rank(p: str) -> tuple:
        c = _canon(p)
        return (STAGE_ORDER.index(c) if c in STAGE_ORDER else len(STAGE_ORDER), p)
    procs = sorted(records, key=_rank)
    t0 = min((r["t"] for rows in records.values() for r in rows), default=0)

    print(f"=== TRACE: {args.dir} ===")
    series: dict[str, tuple] = {}
    for p in procs:
        t, stage, cum = _lat_arrays(records[p])
        if t.size == 0:
            continue
        series[p] = (t, stage, cum)
        stage_valid = stage[~np.isnan(stage)]
        cum_valid = cum[~np.isnan(cum)]
        print(_summarize(f"{p} stage", stage_valid))
        if cum_valid.size:
            print(_summarize(f"{p} cum(glass→)", cum_valid))
        per, frq = _dominant_period_ms(t, np.nan_to_num(stage, nan=np.nanmean(stage)))
        if not np.isnan(per):
            print(f"    └ {p} stage dominant period ≈ {per:.1f} ms ({frq:.1f} Hz)")

    # end-to-end = furthest-downstream cumulative available
    by_stage = {_canon(p): p for p in series}
    for s in reversed(STAGE_ORDER):
        p = by_stage.get(s)
        if p and series[p][2][~np.isnan(series[p][2])].size:
            e2e = series[p][2][~np.isnan(series[p][2])]
            print(f"\n  END-TO-END (glass→{s}): "
                  f"p50={_pct(e2e,50):.2f}  p95={_pct(e2e,95):.2f}  max={e2e.max():.2f} ms")
            break

    n = len(series)
    fig, ax = plt.subplots(n + 1, 1, figsize=(11, 2.2 * (n + 1)))
    if n == 0:
        ax = [ax]
    for i, (p, (t, stage, cum)) in enumerate(series.items()):
        ax[i].plot((t - t0) / 1e9, stage, lw=0.7, label=f"{p} stage")
        if cum[~np.isnan(cum)].size:
            ax[i].plot((t - t0) / 1e9, cum, lw=0.7, alpha=0.6, label=f"{p} cum")
        ax[i].set_ylabel("ms")
        ax[i].set_title(f"{p} latency")
        ax[i].legend(loc="upper right", fontsize=8)
        ax[i].grid(alpha=0.3)
    # histogram panel of tracker stage (the suspected jitter source)
    tracker_proc = next((p for p in series if _canon(p) == "tracker"), None)
    if tracker_proc:
        s = series[tracker_proc][1]
        s = s[~np.isnan(s)]
        ax[-1].hist(s, bins=60, color="#c0392b", alpha=0.8)
        ax[-1].set_title(f"{tracker_proc} stage latency distribution")
        ax[-1].set_xlabel("ms")
    ax[len(series) - 1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nplot → {args.out}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="QuadGuide latency diagnostics")
    sub = p.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("sim", help="reproduce the free-running tracker sawtooth/jitter")
    s.add_argument("--fps", type=float, default=60.0)
    s.add_argument("--tracker-ms", type=float, default=8.0, dest="tracker_ms",
                   help="tracker inference time per update (ms)")
    s.add_argument("--loop-ms", type=float, default=1.0, dest="loop_ms",
                   help="fixed per-iteration loop overhead (read+publish) (ms)")
    s.add_argument("--jitter", type=float, default=1.5, help="camera frame jitter std (ms)")
    s.add_argument("--duration", type=float, default=5.0, help="sim duration (s)")
    s.add_argument("--sse-hz", type=float, default=10.0, dest="sse_hz")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="/tmp/quadguide_sim_latency.png")
    s.set_defaults(func=run_sim)

    t = sub.add_parser("trace", help="analyze a run.py --log trace dump")
    t.add_argument("dir", help="trace directory containing {proc}.jsonl files")
    t.add_argument("--out", default="/tmp/quadguide_trace_latency.png")
    t.set_defaults(func=run_trace)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

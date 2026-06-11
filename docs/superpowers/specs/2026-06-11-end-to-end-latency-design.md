# End-to-End Latency Telemetry — Design Spec

**Date:** 2026-06-11
**Scope:** (1) Propagate a capture timestamp (`origin_ns`) down the message chain
so end-to-end "glass→actuation" latency is measurable. (2) Show that cumulative
latency on the HUD in place of the current capture→tracker number. (3) Add a
single-flag (`--log`) post-run diagnostic **trace dump** — raw per-iteration
latency timestamps + worker health/state — written to disk for **offline**
analysis by the Step-3 tool.

Supersedes the display half of `2026-05-10-latency-telemetry-design.md`.
Companion analysis: `2026-06-11-latency-jitter-diagnosis.md`.

---

## 0. Decisions (locked)

1. **Capture raw on-board, process offline.** Workers log raw timestamps; all
   statistics/plots happen in the Step-3 tool. No percentiles, no windows, no
   analysis baked into the flight code.
2. **Trace files, not a bus topic.** The bus is a *latest-value* transport
   (`bus.latest()` returns only the newest slot; `subscribe_one()` also returns
   `latest()`; ring depth 8; the only reader is the 10 Hz ground server; no
   external attach — diagnosis §5). Publishing raw high-rate samples through it
   would lose most of them and re-alias the rest. Each worker instead writes its
   **own append-only trace file**, RAM-buffered, flushed at shutdown.
3. **Propagate `origin_ns`** (the capture timestamp) verbatim down the chain so
   `cum = now − origin_ns` is exact along the real consumed lineage (diagnosis
   §"What origin_ns buys").
4. **HUD shows cumulative latency** derived from `origin_ns`, replacing the
   capture→tracker `latency_ns` it shows today.
5. **One flag: `--log`.** Toggles the whole trace dump across all workers.

### Assumption

All workers are `fork()`ed children sharing one `CLOCK_MONOTONIC`
(`core/clock.py:30-31`), so cross-process timestamp subtraction is valid with no
clock-sync. Holds because every stage runs on the SBC.

---

## 1. Message / wire changes (`core/messages.py`)

`origin_ns` is an **absolute** monotonic timestamp → 64 bits (`Q`).

| Type | Current fmt (B) | New fmt (B) | Change |
| --- | --- | --- | --- |
| `TrackerEstimate` | `!QfffffBI` (33) | `!QfffffBQ` (37) | **replace** `latency_ns:uint32` → `origin_ns:uint64` |
| `AccelCmd` | `!Qff` (16) | `!QffQ` (24) | **add** `origin_ns:uint64` |
| `ControlCmd` | `!Qffff` (24) | `!QffffQ` (32) | **add** `origin_ns:uint64` |

`origin_ns == 0` is the sentinel "no lineage" (pre-lock-on; control with no
upstream accel). Cumulative latency is defined only when `origin_ns > 0`.

Removing `latency_ns` from `TrackerEstimate` is safe — the only reader is
`ground/server.py` (grep-confirmed), reworked in §5. No new message types are
added (the trace is files, not bus messages).

---

## 2. `origin_ns` propagation (the chain)

| Stage | Sets `origin_ns` to | When |
| --- | --- | --- |
| tracker (`perception/tracker_worker.py`) | `frame_ts` (the SHM frame's capture stamp), else `0` | every estimate |
| guidance (`guidance/worker.py`) | `est.origin_ns` of the estimate it consumed | every published `AccelCmd` |
| control (`control/worker.py`) | `accel.origin_ns` if `accel is not None` else `0` | every `ControlCmd` |

**Lineage vs application:** control stamps `origin_ns` from the accel whenever an
accel exists, **independent** of arm/dwell/failsafe gating of roll/pitch. The
field tracks data lineage; whether the command is *applied* is separate. This
means end-to-end latency is observable whenever a lock exists, even disarmed
(useful for bench diagnosis).

---

## 3. Trace helper (`core/diagtrace.py`, new)

```python
class DiagTrace:
    """RAM-buffered, append-only diagnostic trace for one process.

    Disabled (enabled=False) → every method is a no-op (zero hot-path cost
    beyond a bool check). Enabled → records buffer in memory; flush() serialises
    once to {dir}/{process}.jsonl. Never writes to disk inside a worker loop —
    flush() is called from the post-loop / SIGTERM path only (safe for the
    SCHED_FIFO control loop)."""

    def __init__(self, process: str, *, enabled: bool, dir: str | None,
                 max_rows: int = 0): ...

    def latency(self, now_ns: int, input_ts_ns: int | None,
                origin_ns: int) -> None:
        # append (now, input_ts, origin); O(1), no serialisation
        ...

    def state(self, now_ns: int, **fields) -> None:    # periodic state snapshot
        ...

    def health(self, now_ns: int, state: str, detail: str = "") -> None:
        ...

    def flush(self) -> None:                            # serialise → JSONL, once
        ...
```

**File format — JSONL**, one record per line (heterogeneous records, trivial to
parse offline, cheap to append as tuples and serialise once at flush):

```json
{"t": 12345, "p": "tracker",  "k": "lat",    "in": 12300, "org": 12300}
{"t": 12346, "p": "guidance", "k": "lat",    "in": 12200, "org": 11980}
{"t": 12400, "p": "control",  "k": "state",  "armed": false, "failsafe": false, "throttle": 0.0}
{"t": 12500, "p": "link",     "k": "health", "state": "ok", "detail": "ANGLE"}
```

Offline derivation: `stage = t − in`, `cum = t − org` (when `org > 0`).
`max_rows = 0` → unbounded (bench runs are short); a positive value caps with a
ring to bound RAM on long runs.

---

## 4. Per-worker changes

Each worker: build a `DiagTrace(process, enabled=cfg.diag.trace, dir=…)` at
startup; record every iteration; record state at its **existing health
cadence**; `flush()` after the run loop (before `bus.detach()`), and in the
camera/link `finally` paths.

| Worker | `trace.latency(now, in, org)` each iter | `trace.state(...)` fields |
| --- | --- | --- |
| camera | — (no upstream; it *is* the origin) | `fps`, `frame_count` |
| tracker | `in=frame_ts, org=frame_ts` (when `frame_ts>0`) | `algo`, `health`, `confidence`, `locked` |
| guidance | `in=est.timestamp_ns, org=est.origin_ns` | `method`, `publishing`, `est_health` |
| control | `in=accel.timestamp_ns, org=accel.origin_ns` (when accel & `org>0`) | `armed`, `dwell_done`, `in_failsafe`, `fault`, `throttle` |
| link (TX) | `in=cmd.timestamp_ns, org=cmd.origin_ns` (when `org>0`) | `connected`, `flight_mode`, `armed` |

`health` records are written on `system/health` state transitions each worker
already tracks. The tracker logs latency **every iteration including pre-lock**
(reprocessed stale frames included) — that is the raw signal the Step-3 tool
uses to characterise the sawtooth/jitter.

---

## 5. Ground HUD — cumulative latency display

### `ground/server.py`
Replace the `estimate.latency_ns` block in `_sse()` with a read of
`control/cmd` (which now carries `origin_ns`):

```python
cmd = app.state.bus.latest("control/cmd")
cum_ms = ((cmd.timestamp_ns - cmd.origin_ns) / 1e6
          if cmd and cmd.origin_ns > 0 else None)
```

Using `cmd.timestamp_ns − cmd.origin_ns` (not `now − origin_ns`) measures
**glass→control-publish** and **excludes SSE polling delay** by construction.
Keep the existing 20-sample averaging deque for `latency_avg_ms`. SSE payload
keeps the same keys (`latency_ms`, `latency_avg_ms`) — now meaning cumulative,
not capture→track. (The control→link/TX hop is not visible live — it appears in
the trace file's `link` rows.)

### `ground/static/index.html`
LATENCY section label changes to reflect end-to-end (e.g. `latest (glass→ctrl)`);
the colour thresholds (green ≤ 50, yellow ≤ 100, red > 100 ms) now apply to a
meaningful total. Shows `—` before first lock-on.

---

## 6. Config & flag

### `core/config.py` — new `DiagConfig`
```python
@dataclass
class DiagConfig:
    trace: bool = False
    trace_dir: str | None = None     # default: {logging.dir}/trace/{run-ts}
    trace_max_rows: int = 0          # 0 = unbounded
```
`cfg_diag(config)` narrows the dict; absent `diag:` block → defaults.

### `scripts/run.py` — single flag
```python
parser.add_argument("--log", action="store_true",
                    help="Write a post-run diagnostic trace (latency + state) "
                         "to {logging.dir}/trace/<timestamp>/ for offline analysis")
```
`--log` injects `overrides["diag.trace"]="true"`. The trace dir is resolved once
in `run.py` (timestamped) and injected as `diag.trace_dir` so all workers share
one directory. Config flows to every worker already, so no other plumbing.

### `configs/rk3588.yaml`
```yaml
diag:
  trace: false        # enabled per-run via --log
```

---

## 7. Tests

- `test_messages.py` — round-trip + `calcsize` for new `TrackerEstimate`,
  `AccelCmd`, `ControlCmd` formats; `origin_ns` sentinel.
- `test_diagtrace.py` *(new)* — disabled = no-op/no file; enabled buffers and
  flushes valid JSONL; `lat`/`state`/`health` records; `max_rows` ring cap.
- `test_config.py` — `DiagConfig` defaults + override parsing.
- `test_ground_server.py` — `latency_ms` now = `cmd.timestamp_ns − origin_ns`;
  `null` when `origin_ns == 0`.
- Grep confirms no remaining `latency_ns` readers before deleting the field.

---

## 8. What is NOT changed

- Tracker loop structure — the new-frame gate (root-cause jitter fix) stays
  **deferred** per the agreed sequencing (measure first). The trace will *show*
  the sawtooth; fixing it is a later change.
- Bus mechanics — no new topics, no new bus API.
- Guidance/control algorithms — only `origin_ns` plumbing + trace calls.

---

## 9. Step-3 tool (preview)

`scripts/diagnose_latency.py`:
- `trace <dir>` — ingest the per-process JSONL files, align on the shared
  monotonic clock, derive per-stage and cumulative latency, plot time-series +
  histogram + autocorrelation/FFT (to pin the rhythmic beat frequency), and
  overlay worker state/health transitions. **Primary mode** — full-rate truth.
- `sim` — model camera@60 fps + free-running tracker to reproduce the sawtooth
  and demonstrate the beat with no rig. Validates the diagnosis offline.

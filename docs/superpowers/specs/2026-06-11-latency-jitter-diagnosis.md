# Latency Jitter — Diagnosis

**Date:** 2026-06-11
**Scope:** Root-cause the rhythmic 8–30 ms latency oscillation shown in the ground
station HUD, and assess whether the current latency metric is the right one for a
closed homing-guidance loop.

---

## 1. Symptom

- The HUD `LATENCY / latest` value bounces between **~8 ms and ~30 ms** in a
  **rhythmic** (periodic, not random) pattern.
- The bounce is present **even when the tracker is not initialized** (no lock-on).
- The operator's intuition — "rhythmic ⇒ bug, not load" — is correct.

---

## 2. What is measured today

Per the 2026-05-10 latency-telemetry spec, the displayed number is computed once,
in the tracker worker:

```
latency_ns = monotonic_ns()_after_update − frame_ts
```

- `frame_ts` — `monotonic_ns()` stamped by the camera worker **after**
  `cap.read()` returns (`perception/camera/sources.py:50`), written into the SHM
  ring with the frame (`core/frame_buffer.py:47-65`).
- `now_ns` — `monotonic_ns()` taken right after `tracker.update(frame)` returns
  (`perception/tracker_worker.py:160-161`).
- Carried on the wire as `TrackerEstimate.latency_ns`, a **uint32**
  (`FMT_TRACKER_ESTIMATE = "!QfffffBI"`, `core/messages.py:45`), clamped to
  `0xFFFF_FFFF` ns ≈ **4.29 s**. It stores a *delta*, not an absolute time —
  that is why 32 bits suffice.
- The ground server reads it directly via `bus.latest("target/estimate")` in the
  SSE loop (`ground/server.py:133-140`). Note: the 2026-05-10 spec routed this
  through fusion/`TargetEstimate`; that hop was removed in commit `7f57bfb`
  ("no internal tracker fusion"), so the UI now reads `TrackerEstimate` directly.

So the displayed latency is exactly **frame-capture → tracker-update-finished**:
ring-buffer dwell time + tracker inference. Everything downstream is excluded by
design.

---

## 3. Root cause of the jitter

### 3.1 The tracker loop is free-running with no new-frame gate

`TrackerWorker.run()` (`perception/tracker_worker.py:155-169`):

```python
while not self._stop:
    self._check_lockon()
    frame, frame_ts = self._fb.read_latest()      # always the NEWEST slot
    if frame is not None:
        out     = self._tracker.update(frame)
        now_ns  = monotonic_ns()
        latency = min(now_ns - frame_ts, 0xFFFF_FFFF) if frame_ts > 0 else 0
        ... publish ...
```

There is **no `RateLimiter` and no check that `frame_ts` changed** since the last
iteration. The loop spins as fast as the CPU allows and re-reads the *latest*
frame every pass.

`FrameBuffer.read_latest()` (`core/frame_buffer.py:67-86`) returns whatever the
camera last wrote — it does **not** block for a new frame and does not report
whether the frame is new.

### 3.2 This produces a sawtooth

The camera publishes frames at **60 fps → one new `frame_ts` every ~16.67 ms**
(`configs/rk3588.yaml:8`). The tracker loop is much faster than that, so between
two consecutive camera frames it reads the **same `frame_ts` many times**. On
each pass `frame_ts` is fixed but `now_ns` keeps advancing, so:

```
latency = now_ns − frame_ts   →  ramps up by one loop-period each pass,
                                  then snaps down when a new frame lands.
```

That is a **sawtooth** whose amplitude ≈ the camera frame interval (~16.67 ms)
and whose period = the camera frame interval.

### 3.3 Why it persists with the tracker uninitialized — the key confirmation

When there is no lock-on, `update()` returns almost instantly (the adapter's
no-lock path). The loop then spins *even faster*, so it samples the same
`frame_ts` even more times before the next frame — the sawtooth is still there,
just ramping from ~0 up to ~16.67 ms. **The tracking algorithm is never
involved.** This rules out the model/NPU as the source and proves the cause is
the loop structure + the measurement definition.

### 3.4 Why it looks *rhythmic*: two beats

1. **Loop beat.** The camera loop (~60 Hz) and the tracker loop (~1/loop-period)
   are independent free-running periodics coupled through "read latest." Their
   relative phase drifts slowly, so the sawtooth's sampled height drifts with a
   **beat envelope** at `|f_camera − f_tracker|`.
2. **SSE aliasing.** The HUD samples `bus.latest("target/estimate")` at a fixed
   **10 Hz** (`_SSE_RATE = 0.1`, `ground/server.py:130-134`). A ~60 Hz sawtooth
   sampled at 10 Hz is far above Nyquist (5 Hz) → **aliased**. The slow,
   regular wobble the operator sees on screen is largely the alias of the
   sawtooth, not the true signal. `bus.latest()` returns only the newest slot
   (`core/bus.py:103-115`), so the SSE grabs one arbitrary phase of the sawtooth
   per 100 ms and the intermediate publishes are dropped.

### 3.5 Why 8–30 ms rather than 0–16.7 ms

- **Floor (~8 ms, not 0):** each loop pass still costs the SHM read + (when
  locked) inference + publish, and `frame_ts` is stamped *after* `cap.read()`
  returns — so even a "fresh" frame already carries USB/V4L2 transfer age that
  is not in the number but the read/publish cost is.
- **Ceiling (~30 ms, not 16.7):** the camera worker is **not** CPU-pinned
  (`configs/rk3588.yaml:17-21` pins only tracker→core 4 and control→core 6), so
  frame delivery jitters; an occasional late/dropped frame stretches the
  interval to ~2× (~33 ms), pushing the sawtooth peak toward 30 ms.

These are estimates to be confirmed by measurement (Step 3); the *mechanism*
above does not depend on the exact numbers.

---

## 4. Why this metric is the wrong one for a guidance loop

ARCHITECTURE.md §4.1 makes the purpose explicit: this is a closed homing loop,
`camera → tracker → guidance → control → CRSF → FC`. What governs loop stability
is **the age of the information at the moment its control action reaches the FC**
(glass-to-actuation latency). The current metric measures only the *first and
smallest* segment and ignores the stages that dominate:

| Stage | Mechanism | Counted today? | Worst-case age added |
| --- | --- | --- | --- |
| sensor exposure → `cap.read()` returns | V4L2/USB transfer | ❌ (`frame_ts` is stamped *after* read) | not measured |
| capture → tracker estimate | free-running tracker | ✅ (this is the HUD number) | ~inference + ring dwell |
| estimate → `guidance/accel` | **50 Hz** `RateLimiter` (`guidance/worker.py:28`) | ❌ | **up to ~20 ms** |
| accel → `control/cmd` | **100 Hz** loop (`control/worker.py:34`) | ❌ | up to ~10 ms |
| `control/cmd` → CRSF on wire | **50 Hz fixed** TX (`link/worker.py:54-66`) | ❌ | **up to ~20 ms** |

Each consumer uses `bus.latest()` and therefore reads the freshest message while
silently skipping older ones — so each rate-limited stage adds 0…(its period) of
staleness. The two 50 Hz stages dominate the real loop latency, and the HUD
shows none of it. (Note: ARCHITECTURE.md §4.1 lists control at 250 Hz; the code
runs it at 100 Hz — `control/worker.py:34` — a separate doc/code drift worth
flagging.)

Net: the operator is watching the least important segment, rendered as an
aliased artifact.

---

## 5. Conclusions

1. **The jitter is a bug, not load.** It is the free-running tracker loop
   re-timestamping a stale frame, producing a sawtooth, doubly aliased (loop
   beat + 10 Hz SSE). Confirmed by its presence with the tracker uninitialized.
2. **Two independent fixes** (detailed in the Step-2 design spec):
   - *Presentation/root-cause:* gate the tracker loop on `frame_ts` changing so
     each frame is measured once; report rolling p50/p95/max instead of one
     aliased instantaneous sample.
   - *Scope:* propagate the capture timestamp end-to-end and measure
     glass-to-actuation age with a per-process breakdown.
3. **Measurement constraint for tooling (Step 3):** the `Bus` and `FrameBuffer`
   are created in the orchestrator parent and inherited across `fork`; their
   SHM segments are not named for external attach and their `multiprocessing`
   locks/Values are fork-inherited, not re-openable. An *external* process can
   therefore only tap the **10 Hz SSE** (which is the aliased view). Capturing
   the true full-rate signal requires **in-process instrumentation** (each
   worker logging its own stage latency). The diagnostic tool must account for
   this.

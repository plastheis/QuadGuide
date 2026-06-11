# QuadGuide

SBC-resident flight guidance stack for a manual lock-on target-tracking
quadcopter. Runs on a companion computer (RK3588 target) on the airframe:
camera → object tracker → proportional-navigation / pure-pursuit guidance →
roll/pitch setpoints to a madflight FC over UART (CRSF). See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design.

## Running

```bash
# full flight stack (all workers)
python scripts/run.py --config configs/rk3588.yaml

# camera + tracker + ground HUD only (no FC needed) — tracker tuning / bench
python scripts/dev_ground_perception.py --config configs/rk3588.yaml
```

Ground HUD: `http://<sbc-ip>:8080`. Override any config value with
`--set key.path=value` (e.g. `--set guidance.pure_pursuit.K=15`).

## Latency diagnostics

End-to-end latency telemetry and a post-run analysis tool. Full detail in
[`ARCHITECTURE.md` §13](ARCHITECTURE.md#13-latency-model--diagnostics).

**Capture a trace.** Add `--log` to either launcher to record a per-process
diagnostic trace (raw latency timestamps + worker state) for offline analysis:

```bash
python scripts/dev_ground_perception.py --config configs/rk3588.yaml --log
# ...lock on a target in the HUD, let it run, then Ctrl-C (graceful flush)
```

The trace dir is printed at startup (`{logging.dir}/trace/<ts>`, or
`./quadguide-trace/<ts>` if that isn't writable). It is off by default and meant
for bench/diagnosis runs, not production flight.

**Analyze it.**

```bash
# ingest a --log dump: per-stage + cumulative (glass→here) latency,
# distribution, and a spectral view that pins any rhythmic beat
python scripts/diagnose_latency.py trace ./quadguide-trace/<ts>

# no hardware: reproduce the free-running-loop sawtooth and the gated fix
python scripts/diagnose_latency.py sim
```

Latency lineage rides the message chain via `origin_ns` (the frame capture
timestamp), so any stage — and the trace — can report the true glass→actuation
age. The HUD's latency field shows the cumulative glass→control age.

## Tests

```bash
python -m pytest
```

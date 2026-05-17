# Guidance and Control Modules — Design Spec

Date: 2026-05-17

---

## Overview

Implements all stub files in `src/quadguide/guidance/` and `src/quadguide/control/`.
Both modules are pure-computation workers: no hardware ownership, no shared memory
frames. All inputs and outputs flow through the IPC bus.

---

## Design Decisions

### State vs pure functions

Stateful classes only where inter-frame state is genuinely required:
- `LOSRateEstimator` — holds previous centroid, previous timestamp, last lockon seq
- `ClosingVelEstimator` — holds previous bbox area, previous timestamp, EMA accumulator

Everything else is a pure function: `pronav`, `attitude_cmd.compute`, all limiter operations.
This matches the existing codebase pattern (fusion algorithms are classes; processing
helpers in nanotrack and link are pure functions).

### Lock-on reset in LOS
The `LOSRateEstimator` tracks the last `lockon/cmd` seq it processed — same pattern as
tracker workers. When `seq != _last_lockon_seq`, the differencer resets and returns
`(0.0, 0.0)` for that one sample. The guidance worker reads `bus.latest("lockon/cmd")`
each iteration and passes it into `los.update()`.

### Body-rate correction in LOS
Full rotation matrix using current `roll_rad`, `pitch_rad`, `yaw_rad` from `fc/attitude`.
Projects `[roll_rate_rps, pitch_rate_rps, yaw_rate_rps]` onto image-plane axes.
Valid at all attitudes, not just small-angle.

### Throttle hold
Throttle is a control-layer constant, not a guidance output. Control worker reads
`cfg_guidance(config).throttle_hold` at startup. `AccelCmd` wire format is unchanged.

### Slew rate dt
Fixed nominal `dt = 1/100` (control loop rate). Not measured per-loop — avoids
plumbing complexity for minor accuracy gain under normal jitter.

---

## guidance/

### `guidance/los.py`

```
class LOSRateEstimator:
    _prev_centroid: tuple[float, float] | None
    _prev_ts_ns: int
    _last_lockon_seq: int | None

    def update(
        self,
        centroid_norm: tuple[float, float],
        att: AttitudeState,
        lockon_cmd: LockOnCmd | None,
        now_ns: int,
    ) -> tuple[float, float]:
        1. If lockon_cmd is not None and lockon_cmd.seq != _last_lockon_seq:
               store seq, clear _prev_centroid, return (0.0, 0.0)
        2. If _prev_centroid is None or dt == 0:
               store centroid/ts, return (0.0, 0.0)
        3. raw_rate = (centroid_now - centroid_prev) / dt
        4. Build rotation matrix R from (roll_rad, pitch_rad, yaw_rad)
           Project [roll_rate_rps, pitch_rate_rps, yaw_rate_rps] through R
           onto image-plane axes → body_rate_correction: tuple[float, float]
        5. los_rate = raw_rate - body_rate_correction
        6. Store centroid/ts, return los_rate
```

No config tunables for LOS — rotation matrix is exact math.

### `guidance/closing_vel.py`

```
class ClosingVelEstimator:
    _prev_area: float | None
    _prev_ts_ns: int
    _ema_area_rate: float

    def update(
        self,
        bbox: BoundingBox,
        now_ns: int,
        cfg: GuidanceConfig,
    ) -> float:
        1. area = bbox.w * bbox.h
        2. If _prev_area is None or dt == 0: store area/ts, return cfg.closing_vel_fallback
        3. raw_rate = (area - _prev_area) / dt
        4. _ema_area_rate = alpha * raw_rate + (1 - alpha) * _ema_area_rate
        5. If abs(_ema_area_rate) < cfg.closing_vel_min_area_rate:
               log.debug("closing_vel: using fallback")
               return cfg.closing_vel_fallback
        6. return _ema_area_rate * cfg.closing_vel_area_scale
```

Config tunables:
- `guidance.closing_vel_ema_alpha` — EMA smoothing (0=heavy, 1=raw). Default: 0.3
- `guidance.closing_vel_min_area_rate` — fallback threshold. Default: 0.001
- `guidance.closing_vel_area_scale` — area-rate to m/s conversion. Default: 5.0

### `guidance/pronav.py`

Pure function, no state:

```python
def pronav(
    los_rate: tuple[float, float],
    closing_vel: float,
    N: float,
) -> tuple[float, float]:
    return N * closing_vel * los_rate[0], N * closing_vel * los_rate[1]
```

`N` comes from `cfg.guidance.N` (already in config).

### `guidance/worker.py`

Process entry point. 50 Hz `RateLimiter`.

```
log = setup_logging("guidance", config)
cfg = cfg_guidance(config)
los = LOSRateEstimator()
cv  = ClosingVelEstimator()
rate = RateLimiter(hz=50)
i = 0

loop (until SIGTERM):
    rate.sleep()
    est        = bus.latest("target/estimate")
    att        = bus.latest("fc/attitude")
    lockon_cmd = bus.latest("lockon/cmd")

    if est is None or att is None:
        continue
    if est.tracker_health in (TrackerHealth.LOST, TrackerHealth.NO_LOCK):
        continue

    now_ns  = monotonic_ns()
    los_r   = los.update(est.centroid_norm, att, lockon_cmd, now_ns)
    v_c     = cv.update(est.bbox, now_ns, cfg)
    ax, ay  = pronav(los_r, v_c, cfg.N)

    bus.publish("guidance/accel", AccelCmd(now_ns, ax, ay))

    i += 1
    if i % 10 == 0:
        bus.publish("system/health", HealthReport(now_ns, "guidance", ProcessState.OK, ""))

bus.detach()
```

---

## control/

### `control/attitude_cmd.py`

Pure function, no state:

```python
import math
_G = 9.81

def compute(accel: AccelCmd) -> tuple[float, float]:
    roll_deg  =  math.degrees(accel.ay / _G)
    pitch_deg = -math.degrees(accel.ax / _G)
    return roll_deg, pitch_deg
```

`AttitudeState` is not needed for small-angle mapping. Sign convention: positive
`accel.ay` (rightward) → positive roll (right bank); positive `accel.ax` (forward)
→ negative pitch (nose up).

### `control/limiter.py`

Three pure functions:

```python
def saturate(
    roll: float,
    pitch: float,
    limits: ControlLimitsConfig,
) -> tuple[float, float]:
    # clamp roll to ±limits.max_roll_deg
    # clamp pitch to ±limits.max_pitch_deg

def slew_rate(
    roll: float,
    pitch: float,
    prev_cmd: ControlCmd | None,
    limits: ControlLimitsConfig,
    dt: float,
) -> tuple[float, float]:
    # if prev_cmd is None: return (roll, pitch) unchanged
    # max_delta_roll  = limits.max_roll_rate_dps  * dt
    # max_delta_pitch = limits.max_pitch_rate_dps * dt
    # clamp (roll - prev_cmd.roll_deg)   to ±max_delta_roll
    # clamp (pitch - prev_cmd.pitch_deg) to ±max_delta_pitch

def failsafe_cmd(throttle_hold: float) -> ControlCmd:
    return ControlCmd(monotonic_ns(), 0.0, 0.0, 0.0, throttle_hold)
```

### `control/watchdog.py`

Thin constructor wrapper over `core/health.Watchdog`:

```python
def build_watchdog(cfg: WatchdogConfig, bus: Bus) -> Watchdog:
    return Watchdog([
        ("target/estimate", cfg.target_estimate_ms),
        ("fc/attitude",     cfg.fc_attitude_ms),
        ("guidance/accel",  cfg.guidance_accel_ms),
    ], bus)
```

No logic beyond wiring config to the existing `Watchdog` class in `core/health.py`.

### `control/worker.py`

Process entry point. 100 Hz `RateLimiter`, `SCHED_FIFO` on CPU core 3.

```
log      = setup_logging("control", config)
gcfg     = cfg_guidance(config)
acfg     = cfg_airframe(config)
wcfg     = cfg_watchdog(config)
pcfg     = cfg_platform(config)

platform.set_realtime(pcfg.realtime.control_cpu_core, pcfg.realtime.control_fifo_prio)

watchdog = build_watchdog(wcfg, bus)
rate     = RateLimiter(hz=100)
dt       = 1.0 / 100
prev_cmd: ControlCmd | None = None
i = 0

loop (until SIGTERM):
    rate.sleep()

    try:
        watchdog.check_all()
    except HealthFault as e:
        cmd = failsafe_cmd(gcfg.throttle_hold)
        bus.publish("control/cmd", cmd)
        log.warning("failsafe: %s", e)
        prev_cmd = cmd
        continue

    accel = bus.latest("guidance/accel")
    att   = bus.latest("fc/attitude")

    roll, pitch = attitude_cmd.compute(accel)
    roll, pitch = limiter.saturate(roll, pitch, acfg.control_limits)
    roll, pitch = limiter.slew_rate(roll, pitch, prev_cmd, acfg.control_limits, dt)

    cmd = ControlCmd(monotonic_ns(), roll, pitch, 0.0, gcfg.throttle_hold)
    bus.publish("control/cmd", cmd)
    prev_cmd = cmd

    i += 1
    if i % 20 == 0:
        bus.publish("system/health",
            HealthReport(monotonic_ns(), "control", ProcessState.OK, ""))

bus.detach()
```

---

## Config additions

Add to `configs/config.yaml` under the existing `guidance:` section:

```yaml
guidance:
  N: 4.0
  closing_vel_fallback: 2.0
  fov_horizontal_rad: 1.047
  throttle_hold: 0.55
  closing_vel_ema_alpha: 0.3
  closing_vel_min_area_rate: 0.001
  closing_vel_area_scale: 5.0
```

Add to `core/config.py` `GuidanceConfig`:

```python
@dataclass(frozen=True)
class GuidanceConfig:
    N: float
    closing_vel_fallback: float
    fov_horizontal_rad: float
    throttle_hold: float = 0.55
    closing_vel_ema_alpha: float = 0.3
    closing_vel_min_area_rate: float = 0.001
    closing_vel_area_scale: float = 5.0
```

No changes to `WatchdogConfig`, `AirframeConfig`, or wire formats.

---

## IPC bus — no new topics

All topics already declared in the IPC table. This implementation only reads and
writes existing topics.

---

## Files changed / created

| File | Action |
|------|--------|
| `src/quadguide/guidance/los.py` | implement |
| `src/quadguide/guidance/closing_vel.py` | implement |
| `src/quadguide/guidance/pronav.py` | implement |
| `src/quadguide/guidance/worker.py` | implement |
| `src/quadguide/control/attitude_cmd.py` | implement |
| `src/quadguide/control/limiter.py` | implement |
| `src/quadguide/control/watchdog.py` | implement |
| `src/quadguide/control/worker.py` | implement |
| `configs/config.yaml` | add 4 guidance tunables |
| `src/quadguide/core/config.py` | extend `GuidanceConfig` with 4 new fields |

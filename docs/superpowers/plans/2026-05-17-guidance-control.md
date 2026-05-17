# Guidance and Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all stub files in `guidance/` and `control/` so the system can compute proportional-navigation acceleration commands and convert them to attitude setpoints for the flight controller.

**Architecture:** `LOSRateEstimator` and `ClosingVelEstimator` are stateful classes holding inter-frame state; everything else (`pronav`, `attitude_cmd`, `limiter`) is pure functions. The guidance worker (50 Hz) produces `AccelCmd`; the control worker (100 Hz, SCHED_FIFO) consumes it and publishes `ControlCmd`. Lock-on resets are seq-gated via `lockon/cmd` (same pattern as tracker workers). Body-rate correction uses the full ZYX rotation matrix.

**Tech Stack:** Python 3.11+, `math` (stdlib only for algorithm files), `quadguide.core.{bus, clock, config, health, logging, messages}`, pytest

---

## File Map

| File | Action |
|------|--------|
| `src/quadguide/guidance/__init__.py` | create (empty) |
| `src/quadguide/guidance/pronav.py` | implement |
| `src/quadguide/guidance/los.py` | implement |
| `src/quadguide/guidance/closing_vel.py` | implement |
| `src/quadguide/guidance/worker.py` | implement |
| `src/quadguide/control/__init__.py` | create (empty) |
| `src/quadguide/control/attitude_cmd.py` | implement |
| `src/quadguide/control/limiter.py` | implement |
| `src/quadguide/control/watchdog.py` | implement |
| `src/quadguide/control/worker.py` | implement |
| `src/quadguide/core/config.py` | extend `GuidanceConfig` + `cfg_guidance()` |
| `configs/config.yaml` | add 4 guidance tunables |
| `tests/unit/test_config.py` | extend guidance accessor test |
| `tests/unit/test_pronav.py` | create |
| `tests/unit/test_los.py` | create |
| `tests/unit/test_closing_vel.py` | create |
| `tests/unit/test_attitude_cmd.py` | create |
| `tests/unit/test_limiter.py` | implement (file exists, currently empty) |

---

## Task 1: Extend GuidanceConfig and config.yaml

**Files:**
- Modify: `src/quadguide/core/config.py`
- Modify: `configs/config.yaml`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing tests**

Add these to the existing `TestAccessors` class in `tests/unit/test_config.py`:

```python
def test_cfg_guidance_new_fields(self):
    g = cfg_guidance(self.config)
    assert g.throttle_hold == pytest.approx(0.55)
    assert g.closing_vel_ema_alpha == pytest.approx(0.3)
    assert g.closing_vel_min_area_rate == pytest.approx(0.001)
    assert g.closing_vel_area_scale == pytest.approx(5.0)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/unit/test_config.py::TestAccessors::test_cfg_guidance_new_fields -v
```

Expected: `AttributeError: 'GuidanceConfig' object has no attribute 'throttle_hold'`

- [ ] **Step 3: Extend GuidanceConfig in config.py**

Replace the existing `GuidanceConfig` dataclass:

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

Replace the existing `cfg_guidance()` function:

```python
def cfg_guidance(d: dict) -> GuidanceConfig:
    g = d["guidance"]
    return GuidanceConfig(
        N=g["N"],
        closing_vel_fallback=g["closing_vel_fallback"],
        fov_horizontal_rad=g["fov_horizontal_rad"],
        throttle_hold=g.get("throttle_hold", 0.55),
        closing_vel_ema_alpha=g.get("closing_vel_ema_alpha", 0.3),
        closing_vel_min_area_rate=g.get("closing_vel_min_area_rate", 0.001),
        closing_vel_area_scale=g.get("closing_vel_area_scale", 5.0),
    )
```

- [ ] **Step 4: Add keys to configs/config.yaml**

Replace the existing `guidance:` section:

```yaml
guidance:
  N: 4.0
  closing_vel_fallback: 2.0
  fov_horizontal_rad: 1.047   # camera horizontal FoV (~60°)
  throttle_hold: 0.55         # constant throttle during tracking (0–1)
  closing_vel_ema_alpha: 0.3  # EMA smoothing for bbox area rate (0=heavy, 1=raw)
  closing_vel_min_area_rate: 0.001  # fallback if smoothed area rate is below this
  closing_vel_area_scale: 5.0       # area-rate (normalised/s) → m/s conversion
```

- [ ] **Step 5: Run tests to verify pass**

```bash
python -m pytest tests/unit/test_config.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/quadguide/core/config.py configs/config.yaml tests/unit/test_config.py
git commit -m "feat(config): extend GuidanceConfig with throttle_hold and closing_vel tunables"
```

---

## Task 2: guidance/__init__.py and pronav.py

**Files:**
- Create: `src/quadguide/guidance/__init__.py`
- Create: `src/quadguide/guidance/pronav.py`
- Create: `tests/unit/test_pronav.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_pronav.py`:

```python
import math
import pytest
from quadguide.guidance.pronav import pronav


def test_zero_los_rate_gives_zero_accel():
    ax, ay = pronav((0.0, 0.0), 2.0, 4.0)
    assert ax == 0.0
    assert ay == 0.0


def test_positive_los_x_gives_positive_ax():
    ax, ay = pronav((1.0, 0.0), 2.0, 4.0)
    assert math.isclose(ax, 8.0)   # N * v_c * los_x = 4 * 2 * 1
    assert ay == 0.0


def test_positive_los_y_gives_positive_ay():
    ax, ay = pronav((0.0, 1.0), 2.0, 4.0)
    assert ax == 0.0
    assert math.isclose(ay, 8.0)


def test_negative_closing_vel_flips_sign():
    ax, _ = pronav((1.0, 0.0), -2.0, 4.0)
    assert math.isclose(ax, -8.0)


def test_scales_with_N():
    ax, ay = pronav((1.0, 1.0), 1.0, 3.0)
    assert math.isclose(ax, 3.0)
    assert math.isclose(ay, 3.0)


def test_zero_closing_vel_gives_zero_regardless_of_los():
    ax, ay = pronav((10.0, 10.0), 0.0, 4.0)
    assert ax == 0.0
    assert ay == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/unit/test_pronav.py -v
```

Expected: `ModuleNotFoundError: No module named 'quadguide.guidance'`

- [ ] **Step 3: Create __init__.py and pronav.py**

Create `src/quadguide/guidance/__init__.py` (empty file):

```python
```

Create `src/quadguide/guidance/pronav.py`:

```python
from __future__ import annotations


def pronav(
    los_rate: tuple[float, float],
    closing_vel: float,
    N: float,
) -> tuple[float, float]:
    """Proportional navigation: a_cmd = N * V_c * los_rate."""
    return N * closing_vel * los_rate[0], N * closing_vel * los_rate[1]
```

- [ ] **Step 4: Run to verify pass**

```bash
python -m pytest tests/unit/test_pronav.py -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/guidance/__init__.py src/quadguide/guidance/pronav.py tests/unit/test_pronav.py
git commit -m "feat(guidance): implement pronav pure function"
```

---

## Task 3: guidance/los.py

**Files:**
- Create: `src/quadguide/guidance/los.py`
- Create: `tests/unit/test_los.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_los.py`:

```python
from __future__ import annotations
import math
import time

import pytest

from quadguide.guidance.los import LOSRateEstimator
from quadguide.core.messages import AttitudeState, BoundingBox, LockOnCmd

FOV_H = 1.047   # ~60 degrees horizontal
ASPECT = 640 / 480   # image width / height


def _att(roll=0.0, pitch=0.0, yaw=0.0, rr=0.0, pr=0.0, yr=0.0) -> AttitudeState:
    return AttitudeState(time.monotonic_ns(), roll, pitch, yaw, rr, pr, yr)


def _lockon(seq: int) -> LockOnCmd:
    return LockOnCmd(time.monotonic_ns(), seq, BoundingBox(0.4, 0.4, 0.1, 0.1))


class TestLOSReset:
    def test_first_call_returns_zero(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        result = est.update((0.1, 0.1), _att(), None, time.monotonic_ns())
        assert result == (0.0, 0.0)

    def test_second_call_without_lock_returns_nonzero(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), None, t0)
        result = est.update((0.2, 0.0), _att(), None, t0 + 20_000_000)
        assert result != (0.0, 0.0)
        assert result[0] > 0.0

    def test_new_lockon_seq_resets_and_returns_zero(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), None, t0)
        est.update((0.2, 0.2), _att(), None, t0 + 20_000_000)
        # New lockon should reset
        result = est.update((0.5, 0.5), _att(), _lockon(seq=1), t0 + 40_000_000)
        assert result == (0.0, 0.0)

    def test_same_lockon_seq_does_not_reset(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        lockon = _lockon(seq=1)
        est.update((0.0, 0.0), _att(), lockon, t0)
        result = est.update((0.2, 0.0), _att(), lockon, t0 + 20_000_000)
        assert result != (0.0, 0.0)

    def test_subsequent_lockon_with_different_seq_resets(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), _lockon(seq=1), t0)
        est.update((0.1, 0.0), _att(), _lockon(seq=1), t0 + 20_000_000)
        result = est.update((0.5, 0.0), _att(), _lockon(seq=2), t0 + 40_000_000)
        assert result == (0.0, 0.0)


class TestLOSRateComputation:
    def test_centroid_moving_right_gives_positive_x(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), None, t0)
        # Move 0.1 in x over 100 ms → raw_rate_x = 1.0 centroid/s
        result = est.update((0.1, 0.0), _att(), None, t0 + 100_000_000)
        assert result[0] > 0.0

    def test_centroid_stationary_zero_body_rate_gives_zero(self):
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.1, 0.1), _att(), None, t0)
        result = est.update((0.1, 0.1), _att(), None, t0 + 20_000_000)
        assert result == pytest.approx((0.0, 0.0), abs=1e-9)


class TestLOSBodyRateCorrection:
    def test_level_pitch_rate_subtracts_from_x(self):
        """Drone pitching at 1 rad/s with stationary centroid → negative LOS x rate."""
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), None, t0)
        att = _att(pr=1.0)   # pure pitch rate, level drone
        result = est.update((0.0, 0.0), att, None, t0 + 20_000_000)
        # correction_x = pitch_rate * 2/fov_h ≈ 1.91; raw_rate_x = 0
        # los_x = 0 - 1.91 ≈ -1.91
        expected_x = -(2.0 / FOV_H)
        assert result[0] == pytest.approx(expected_x, abs=0.02)

    def test_level_roll_rate_subtracts_from_y(self):
        """Drone rolling at 1 rad/s with stationary centroid → positive LOS y rate."""
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), None, t0)
        att = _att(rr=1.0)   # pure roll rate, level drone
        result = est.update((0.0, 0.0), att, None, t0 + 20_000_000)
        fov_v = FOV_H / ASPECT
        expected_y = 2.0 / fov_v   # roll moves centroid in y; sign = positive
        assert abs(result[1]) == pytest.approx(abs(expected_y), abs=0.02)

    def test_yaw_rate_has_no_image_effect_at_level(self):
        """Yaw rotation around boresight does not move a centred target."""
        est = LOSRateEstimator(FOV_H, ASPECT)
        t0 = time.monotonic_ns()
        est.update((0.0, 0.0), _att(), None, t0)
        att = _att(yr=2.0)   # pure yaw rate
        result = est.update((0.0, 0.0), att, None, t0 + 20_000_000)
        assert result == pytest.approx((0.0, 0.0), abs=1e-6)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/unit/test_los.py -v
```

Expected: `ModuleNotFoundError: No module named 'quadguide.guidance.los'`

- [ ] **Step 3: Implement los.py**

Create `src/quadguide/guidance/los.py`:

```python
from __future__ import annotations
import math

from quadguide.core.messages import AttitudeState, LockOnCmd


def _rot_matrix(roll: float, pitch: float, yaw: float) -> tuple:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr),
        (sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr),
        (-sp,      cp * sr,                 cp * cr               ),
    )


def _body_rate_correction(
    att: AttitudeState,
    fov_h: float,
    aspect: float,
) -> tuple[float, float]:
    R = _rot_matrix(att.roll_rad, att.pitch_rad, att.yaw_rad)
    p, q, r = att.roll_rate_rps, att.pitch_rate_rps, att.yaw_rate_rps

    # Camera boresight in inertial frame: R @ [0, 0, 1] = R column 2
    bx = R[0][2]; by = R[1][2]; bz = R[2][2]

    # Angular velocity in inertial frame: R @ [p, q, r]
    wx = R[0][0] * p + R[0][1] * q + R[0][2] * r
    wy = R[1][0] * p + R[1][1] * q + R[1][2] * r
    wz = R[2][0] * p + R[2][1] * q + R[2][2] * r

    # LOS angular rate in inertial = omega_i x boresight_i
    lx_i = wy * bz - wz * by
    ly_i = wz * bx - wx * bz

    # Project back to body/image frame: R.T @ [lx_i, ly_i, 0]
    lx_b = R[0][0] * lx_i + R[1][0] * ly_i
    ly_b = R[0][1] * lx_i + R[1][1] * ly_i

    # Scale from rad/s to centroid_norm/s (centroid_norm spans 2 across fov_h)
    fov_v = fov_h / aspect
    return lx_b * (2.0 / fov_h), ly_b * (2.0 / fov_v)


class LOSRateEstimator:
    """Line-of-sight rate estimator with lock-on seq reset and body-rate correction."""

    def __init__(self, fov_horizontal_rad: float, aspect: float) -> None:
        self._fov_h = fov_horizontal_rad
        self._aspect = aspect
        self._prev_centroid: tuple[float, float] | None = None
        self._prev_ts_ns: int = 0
        self._last_lockon_seq: int | None = None

    def update(
        self,
        centroid_norm: tuple[float, float],
        att: AttitudeState,
        lockon_cmd: LockOnCmd | None,
        now_ns: int,
    ) -> tuple[float, float]:
        if lockon_cmd is not None and lockon_cmd.seq != self._last_lockon_seq:
            self._last_lockon_seq = lockon_cmd.seq
            self._prev_centroid = None
            return (0.0, 0.0)

        if self._prev_centroid is None:
            self._prev_centroid = centroid_norm
            self._prev_ts_ns = now_ns
            return (0.0, 0.0)

        dt = (now_ns - self._prev_ts_ns) * 1e-9
        if dt <= 0.0:
            return (0.0, 0.0)

        raw_x = (centroid_norm[0] - self._prev_centroid[0]) / dt
        raw_y = (centroid_norm[1] - self._prev_centroid[1]) / dt

        corr_x, corr_y = _body_rate_correction(att, self._fov_h, self._aspect)

        self._prev_centroid = centroid_norm
        self._prev_ts_ns = now_ns

        return raw_x - corr_x, raw_y - corr_y
```

- [ ] **Step 4: Run to verify pass**

```bash
python -m pytest tests/unit/test_los.py -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/guidance/los.py tests/unit/test_los.py
git commit -m "feat(guidance): implement LOSRateEstimator with rotation-matrix body-rate correction"
```

---

## Task 4: guidance/closing_vel.py

**Files:**
- Create: `src/quadguide/guidance/closing_vel.py`
- Create: `tests/unit/test_closing_vel.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_closing_vel.py`:

```python
from __future__ import annotations
import time
import types

import pytest

from quadguide.guidance.closing_vel import ClosingVelEstimator
from quadguide.core.messages import BoundingBox

_CFG = types.SimpleNamespace(
    closing_vel_fallback=2.0,
    closing_vel_ema_alpha=1.0,    # alpha=1 → no smoothing, predictable tests
    closing_vel_min_area_rate=0.001,
    closing_vel_area_scale=5.0,
)


def _bbox(w: float, h: float) -> BoundingBox:
    return BoundingBox(0.0, 0.0, w, h)


class TestClosingVelFallback:
    def test_first_call_returns_fallback(self):
        est = ClosingVelEstimator()
        result = est.update(_bbox(0.3, 0.3), time.monotonic_ns(), _CFG)
        assert result == pytest.approx(2.0)

    def test_stationary_bbox_returns_fallback(self):
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.3, 0.3), t0, _CFG)
        result = est.update(_bbox(0.3, 0.3), t0 + 20_000_000, _CFG)
        assert result == pytest.approx(2.0)

    def test_tiny_area_change_returns_fallback(self):
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.3, 0.3), t0, _CFG)
        # area change: 0.09 → 0.090001 in 1s → rate ≈ 0.000001 < 0.001 threshold
        result = est.update(_bbox(0.300003, 0.300003), t0 + 1_000_000_000, _CFG)
        assert result == pytest.approx(2.0)


class TestClosingVelNormal:
    def test_growing_bbox_returns_positive(self):
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.2, 0.2), t0, _CFG)
        # area: 0.04 → 0.09 in 0.1s → area_rate = 0.5 → v_c = 0.5 * 5 = 2.5
        result = est.update(_bbox(0.3, 0.3), t0 + 100_000_000, _CFG)
        assert result == pytest.approx(2.5, abs=0.01)

    def test_shrinking_bbox_returns_negative(self):
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.3, 0.3), t0, _CFG)
        # area shrinks → negative v_c
        result = est.update(_bbox(0.2, 0.2), t0 + 100_000_000, _CFG)
        assert result < 0.0

    def test_scales_with_area_scale(self):
        cfg = types.SimpleNamespace(
            closing_vel_fallback=2.0,
            closing_vel_ema_alpha=1.0,
            closing_vel_min_area_rate=0.001,
            closing_vel_area_scale=10.0,   # double the default
        )
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.2, 0.2), t0, cfg)
        result = est.update(_bbox(0.3, 0.3), t0 + 100_000_000, cfg)
        assert result == pytest.approx(5.0, abs=0.01)   # 0.5 area_rate * 10


class TestClosingVelEMA:
    def test_alpha_one_is_raw(self):
        cfg = types.SimpleNamespace(
            closing_vel_fallback=2.0,
            closing_vel_ema_alpha=1.0,
            closing_vel_min_area_rate=0.001,
            closing_vel_area_scale=1.0,
        )
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.2, 0.2), t0, cfg)
        # area 0.04 → 0.09 in 1s → rate = 0.05 (per second for normalised area)
        result = est.update(_bbox(0.3, 0.3), t0 + 1_000_000_000, cfg)
        assert result == pytest.approx(0.05, rel=0.02)

    def test_ema_smooths_toward_zero(self):
        cfg = types.SimpleNamespace(
            closing_vel_fallback=0.0,    # disable fallback interference
            closing_vel_ema_alpha=0.5,
            closing_vel_min_area_rate=0.0,   # never fall back
            closing_vel_area_scale=1.0,
        )
        est = ClosingVelEstimator()
        t0 = time.monotonic_ns()
        est.update(_bbox(0.2, 0.2), t0, cfg)
        # raw_rate ≈ 0.05; ema = 0.5*0.05 + 0.5*0.0 = 0.025
        result = est.update(_bbox(0.3, 0.3), t0 + 1_000_000_000, cfg)
        assert result == pytest.approx(0.025, rel=0.05)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/unit/test_closing_vel.py -v
```

Expected: `ModuleNotFoundError: No module named 'quadguide.guidance.closing_vel'`

- [ ] **Step 3: Implement closing_vel.py**

Create `src/quadguide/guidance/closing_vel.py`:

```python
from __future__ import annotations
import logging

from quadguide.core.messages import BoundingBox

log = logging.getLogger(__name__)


class ClosingVelEstimator:
    """Estimates closing velocity from rate of change of bounding box area.

    Positive = target getting closer. Falls back to cfg.closing_vel_fallback
    when the EMA-smoothed area rate is below cfg.closing_vel_min_area_rate.
    """

    def __init__(self) -> None:
        self._prev_area: float | None = None
        self._prev_ts_ns: int = 0
        self._ema_area_rate: float = 0.0

    def update(self, bbox: BoundingBox, now_ns: int, cfg) -> float:
        area = bbox.w * bbox.h

        if self._prev_area is None:
            self._prev_area = area
            self._prev_ts_ns = now_ns
            return cfg.closing_vel_fallback

        dt = (now_ns - self._prev_ts_ns) * 1e-9
        if dt <= 0.0:
            return cfg.closing_vel_fallback

        raw_rate = (area - self._prev_area) / dt
        self._ema_area_rate = (
            cfg.closing_vel_ema_alpha * raw_rate
            + (1.0 - cfg.closing_vel_ema_alpha) * self._ema_area_rate
        )

        self._prev_area = area
        self._prev_ts_ns = now_ns

        if abs(self._ema_area_rate) < cfg.closing_vel_min_area_rate:
            log.debug("closing_vel: using fallback")
            return cfg.closing_vel_fallback

        return self._ema_area_rate * cfg.closing_vel_area_scale
```

- [ ] **Step 4: Run to verify pass**

```bash
python -m pytest tests/unit/test_closing_vel.py -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/guidance/closing_vel.py tests/unit/test_closing_vel.py
git commit -m "feat(guidance): implement ClosingVelEstimator with EMA and fallback"
```

---

## Task 5: guidance/worker.py

**Files:**
- Create: `src/quadguide/guidance/worker.py`

No unit tests — the worker loop requires a live bus. Verified by import check only.

- [ ] **Step 1: Implement worker.py**

Create `src/quadguide/guidance/worker.py`:

```python
from __future__ import annotations
import signal

from quadguide.core.bus import Bus
from quadguide.core.clock import RateLimiter, monotonic_ns
from quadguide.core.config import cfg_guidance, cfg_platform
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.logging import setup_logging
from quadguide.core.messages import (
    AccelCmd, HealthReport, ProcessState, TrackerHealth,
)
from quadguide.guidance.closing_vel import ClosingVelEstimator
from quadguide.guidance.los import LOSRateEstimator
from quadguide.guidance.pronav import pronav

__all__ = ["run"]

_HEALTH_EVERY = 10   # iterations; 50 Hz / 10 = 5 Hz health rate


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    log = setup_logging("guidance", config)
    gcfg = cfg_guidance(config)
    pcfg = cfg_platform(config)

    aspect = pcfg.camera.width / pcfg.camera.height
    los = LOSRateEstimator(gcfg.fov_horizontal_rad, aspect)
    cv = ClosingVelEstimator()
    rate = RateLimiter(hz=50)

    stop = False

    def _on_sigterm(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    i = 0
    log.info("guidance: started (N=%.1f, throttle_hold=%.2f)", gcfg.N, gcfg.throttle_hold)

    while not stop:
        rate.sleep()

        est        = bus.latest("target/estimate")
        att        = bus.latest("fc/attitude")
        lockon_cmd = bus.latest("lockon/cmd")

        if est is None or att is None:
            continue
        if est.tracker_health in (TrackerHealth.LOST, TrackerHealth.NO_LOCK):
            continue

        now_ns = monotonic_ns()
        los_r  = los.update(est.centroid_norm, att, lockon_cmd, now_ns)
        v_c    = cv.update(est.bbox, now_ns, gcfg)
        ax, ay = pronav(los_r, v_c, gcfg.N)

        bus.publish("guidance/accel", AccelCmd(now_ns, ax, ay))

        i += 1
        if i % _HEALTH_EVERY == 0:
            bus.publish(
                "system/health",
                HealthReport(monotonic_ns(), "guidance", ProcessState.OK, ""),
            )

    bus.detach()
    log.info("guidance: stopped")
```

- [ ] **Step 2: Verify import**

```bash
python -c "from quadguide.guidance.worker import run; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/quadguide/guidance/worker.py
git commit -m "feat(guidance): implement guidance worker (50 Hz pronav loop)"
```

---

## Task 6: control/__init__.py and attitude_cmd.py

**Files:**
- Create: `src/quadguide/control/__init__.py`
- Create: `src/quadguide/control/attitude_cmd.py`
- Create: `tests/unit/test_attitude_cmd.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_attitude_cmd.py`:

```python
import math
import time

import pytest

from quadguide.control.attitude_cmd import compute
from quadguide.core.messages import AccelCmd

_G = 9.81


def _accel(ax: float, ay: float) -> AccelCmd:
    return AccelCmd(time.monotonic_ns(), ax, ay)


def test_zero_accel_gives_zero_angles():
    roll, pitch = compute(_accel(0.0, 0.0))
    assert roll == 0.0
    assert pitch == 0.0


def test_full_g_lateral_gives_45_deg_roll():
    roll, pitch = compute(_accel(0.0, _G))
    assert math.isclose(roll, 45.0, abs_tol=0.01)


def test_negative_lateral_gives_negative_roll():
    roll, pitch = compute(_accel(0.0, -_G))
    assert math.isclose(roll, -45.0, abs_tol=0.01)


def test_full_g_forward_gives_minus_45_deg_pitch():
    # Positive ax (forward accel) → nose up → negative pitch setpoint
    roll, pitch = compute(_accel(_G, 0.0))
    assert math.isclose(pitch, -45.0, abs_tol=0.01)


def test_negative_forward_gives_positive_pitch():
    roll, pitch = compute(_accel(-_G, 0.0))
    assert math.isclose(pitch, 45.0, abs_tol=0.01)


def test_roll_and_pitch_independent():
    roll, pitch = compute(_accel(_G, _G))
    assert math.isclose(roll, 45.0, abs_tol=0.01)
    assert math.isclose(pitch, -45.0, abs_tol=0.01)


def test_small_accel_proportional():
    roll1, _ = compute(_accel(0.0, 1.0))
    roll2, _ = compute(_accel(0.0, 2.0))
    assert roll2 == pytest.approx(roll1 * 2, rel=0.01)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/unit/test_attitude_cmd.py -v
```

Expected: `ModuleNotFoundError: No module named 'quadguide.control'`

- [ ] **Step 3: Create __init__.py and attitude_cmd.py**

Create `src/quadguide/control/__init__.py` (empty file):

```python
```

Create `src/quadguide/control/attitude_cmd.py`:

```python
from __future__ import annotations
import math

from quadguide.core.messages import AccelCmd

_G = 9.81


def compute(accel: AccelCmd) -> tuple[float, float]:
    """Map body-frame acceleration command to roll/pitch setpoints (degrees).

    Small-angle mapping: roll = atan(ay/g), pitch = -atan(ax/g).
    Positive ay (rightward) → positive roll (right bank).
    Positive ax (forward)   → negative pitch (nose up).
    """
    roll_deg  =  math.degrees(math.atan2(accel.ay, _G))
    pitch_deg = -math.degrees(math.atan2(accel.ax, _G))
    return roll_deg, pitch_deg
```

- [ ] **Step 4: Run to verify pass**

```bash
python -m pytest tests/unit/test_attitude_cmd.py -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/control/__init__.py src/quadguide/control/attitude_cmd.py tests/unit/test_attitude_cmd.py
git commit -m "feat(control): implement attitude_cmd.compute (accel → roll/pitch setpoints)"
```

---

## Task 7: control/limiter.py

**Files:**
- Create: `src/quadguide/control/limiter.py`
- Implement: `tests/unit/test_limiter.py` (file exists, currently empty)

- [ ] **Step 1: Write tests**

Replace the contents of `tests/unit/test_limiter.py`:

```python
from __future__ import annotations
import time

import pytest

from quadguide.control.limiter import failsafe_cmd, saturate, slew_rate
from quadguide.core.config import ControlLimitsConfig
from quadguide.core.messages import ControlCmd

_LIMITS = ControlLimitsConfig(
    max_roll_deg=35.0,
    max_pitch_deg=35.0,
    max_roll_rate_dps=200.0,
    max_pitch_rate_dps=200.0,
)


def _cmd(roll: float, pitch: float, throttle: float = 0.55) -> ControlCmd:
    return ControlCmd(time.monotonic_ns(), roll, pitch, 0.0, throttle)


class TestSaturate:
    def test_within_limits_unchanged(self):
        roll, pitch = saturate(10.0, -10.0, _LIMITS)
        assert roll == 10.0
        assert pitch == -10.0

    def test_roll_clamped_positive(self):
        roll, _ = saturate(50.0, 0.0, _LIMITS)
        assert roll == 35.0

    def test_roll_clamped_negative(self):
        roll, _ = saturate(-50.0, 0.0, _LIMITS)
        assert roll == -35.0

    def test_pitch_clamped_positive(self):
        _, pitch = saturate(0.0, 50.0, _LIMITS)
        assert pitch == 35.0

    def test_pitch_clamped_negative(self):
        _, pitch = saturate(0.0, -50.0, _LIMITS)
        assert pitch == -35.0

    def test_at_limit_boundary_unchanged(self):
        roll, pitch = saturate(35.0, -35.0, _LIMITS)
        assert roll == 35.0
        assert pitch == -35.0


class TestSlewRate:
    def test_no_prev_passes_through_unchanged(self):
        roll, pitch = slew_rate(20.0, -20.0, None, _LIMITS, 0.01)
        assert roll == 20.0
        assert pitch == -20.0

    def test_large_positive_step_clamped(self):
        prev = _cmd(0.0, 0.0)
        # max_delta = 200 dps * 0.01 s = 2 deg
        roll, _ = slew_rate(10.0, 0.0, prev, _LIMITS, 0.01)
        assert roll == pytest.approx(2.0)

    def test_large_negative_step_clamped(self):
        prev = _cmd(0.0, 0.0)
        roll, _ = slew_rate(-10.0, 0.0, prev, _LIMITS, 0.01)
        assert roll == pytest.approx(-2.0)

    def test_small_step_passes_through(self):
        prev = _cmd(0.0, 0.0)
        roll, _ = slew_rate(1.0, 0.0, prev, _LIMITS, 0.01)
        assert roll == pytest.approx(1.0)

    def test_pitch_clamped_independently(self):
        prev = _cmd(0.0, 0.0)
        _, pitch = slew_rate(0.0, 10.0, prev, _LIMITS, 0.01)
        assert pitch == pytest.approx(2.0)

    def test_step_from_nonzero_prev(self):
        prev = _cmd(30.0, 0.0)
        # from 30 deg, requesting 35 deg: delta = 5 deg > 2 deg limit → clamp to 32 deg
        roll, _ = slew_rate(35.0, 0.0, prev, _LIMITS, 0.01)
        assert roll == pytest.approx(32.0)

    def test_exact_max_delta_allowed(self):
        prev = _cmd(0.0, 0.0)
        roll, _ = slew_rate(2.0, 0.0, prev, _LIMITS, 0.01)
        assert roll == pytest.approx(2.0)


class TestFailsafeCmd:
    def test_roll_pitch_yaw_are_zero(self):
        cmd = failsafe_cmd(0.55)
        assert cmd.roll_deg == 0.0
        assert cmd.pitch_deg == 0.0
        assert cmd.yaw_rate_dps == 0.0

    def test_throttle_matches_argument(self):
        cmd = failsafe_cmd(0.55)
        assert cmd.throttle_norm == pytest.approx(0.55)

    def test_different_throttle_value(self):
        cmd = failsafe_cmd(0.4)
        assert cmd.throttle_norm == pytest.approx(0.4)

    def test_returns_control_cmd_type(self):
        cmd = failsafe_cmd(0.55)
        assert isinstance(cmd, ControlCmd)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/unit/test_limiter.py -v
```

Expected: `ModuleNotFoundError: No module named 'quadguide.control.limiter'`

- [ ] **Step 3: Implement limiter.py**

Create `src/quadguide/control/limiter.py`:

```python
from __future__ import annotations

from quadguide.core.clock import monotonic_ns
from quadguide.core.config import ControlLimitsConfig
from quadguide.core.messages import ControlCmd


def saturate(
    roll: float,
    pitch: float,
    limits: ControlLimitsConfig,
) -> tuple[float, float]:
    roll  = max(-limits.max_roll_deg,  min(limits.max_roll_deg,  roll))
    pitch = max(-limits.max_pitch_deg, min(limits.max_pitch_deg, pitch))
    return roll, pitch


def slew_rate(
    roll: float,
    pitch: float,
    prev_cmd: ControlCmd | None,
    limits: ControlLimitsConfig,
    dt: float,
) -> tuple[float, float]:
    if prev_cmd is None:
        return roll, pitch
    max_dr = limits.max_roll_rate_dps  * dt
    max_dp = limits.max_pitch_rate_dps * dt
    roll  = prev_cmd.roll_deg  + max(-max_dr, min(max_dr,  roll  - prev_cmd.roll_deg))
    pitch = prev_cmd.pitch_deg + max(-max_dp, min(max_dp,  pitch - prev_cmd.pitch_deg))
    return roll, pitch


def failsafe_cmd(throttle_hold: float) -> ControlCmd:
    return ControlCmd(monotonic_ns(), 0.0, 0.0, 0.0, throttle_hold)
```

- [ ] **Step 4: Run to verify pass**

```bash
python -m pytest tests/unit/test_limiter.py -v
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/quadguide/control/limiter.py tests/unit/test_limiter.py
git commit -m "feat(control): implement limiter (saturate, slew_rate, failsafe_cmd)"
```

---

## Task 8: control/watchdog.py

**Files:**
- Create: `src/quadguide/control/watchdog.py`

Thin constructor wrapper over `core/health.Watchdog`. Tested indirectly by the full test suite (no isolated unit test needed — `build_watchdog` has no logic beyond wiring).

- [ ] **Step 1: Implement watchdog.py**

Create `src/quadguide/control/watchdog.py`:

```python
from __future__ import annotations

from quadguide.core.bus import Bus
from quadguide.core.config import WatchdogConfig
from quadguide.core.health import Watchdog


def build_watchdog(cfg: WatchdogConfig, bus: Bus) -> Watchdog:
    return Watchdog(
        [
            ("target/estimate", cfg.target_estimate_ms),
            ("fc/attitude",     cfg.fc_attitude_ms),
            ("guidance/accel",  cfg.guidance_accel_ms),
        ],
        bus,
    )
```

- [ ] **Step 2: Verify import**

```bash
python -c "from quadguide.control.watchdog import build_watchdog; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/quadguide/control/watchdog.py
git commit -m "feat(control): implement build_watchdog wrapper"
```

---

## Task 9: control/worker.py

**Files:**
- Create: `src/quadguide/control/worker.py`

No unit tests — requires live bus and platform. Verified by import check only.

- [ ] **Step 1: Implement worker.py**

Create `src/quadguide/control/worker.py`:

```python
from __future__ import annotations
import signal

from quadguide.core.bus import Bus
from quadguide.core.clock import RateLimiter, monotonic_ns
from quadguide.core.config import cfg_airframe, cfg_guidance, cfg_platform, cfg_watchdog
from quadguide.core.frame_buffer import FrameBuffer
from quadguide.core.health import FailsafeState, HealthFault
from quadguide.core.logging import setup_logging
from quadguide.core.messages import ControlCmd, HealthReport, ProcessState
from quadguide.control.attitude_cmd import compute as attitude_cmd_compute
from quadguide.control.limiter import failsafe_cmd, saturate, slew_rate
from quadguide.control.watchdog import build_watchdog
from quadguide.platform.adapter import PlatformAdapter

__all__ = ["run"]

_HEALTH_EVERY = 20   # iterations; 100 Hz / 20 = 5 Hz health rate
_DT = 1.0 / 100      # nominal loop period (s); fixed, not measured per-loop


def run(config: dict, bus: Bus, frame_buffer: FrameBuffer) -> None:
    log = setup_logging("control", config)
    gcfg = cfg_guidance(config)
    acfg = cfg_airframe(config)
    pcfg = cfg_platform(config)
    wcfg = cfg_watchdog(config)

    platform = PlatformAdapter(config)
    platform.set_realtime(pcfg.realtime.control_cpu_core, pcfg.realtime.control_fifo_prio)

    watchdog = build_watchdog(wcfg, bus)
    rate = RateLimiter(hz=100)

    prev_cmd: ControlCmd | None = None
    state = FailsafeState.NOMINAL
    stop = False

    def _on_sigterm(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    i = 0
    log.info(
        "control: started (100 Hz, core=%d, sched_fifo=%s)",
        pcfg.realtime.control_cpu_core,
        pcfg.realtime.control_sched_fifo,
    )

    while not stop:
        rate.sleep()

        try:
            watchdog.check_all()
            state = FailsafeState.NOMINAL
        except HealthFault as e:
            state = FailsafeState.LEVEL
            cmd = failsafe_cmd(gcfg.throttle_hold)
            bus.publish("control/cmd", cmd)
            log.warning("control: failsafe — %s", e)
            prev_cmd = cmd
            continue

        accel = bus.latest("guidance/accel")
        att   = bus.latest("fc/attitude")

        roll, pitch = attitude_cmd_compute(accel)
        roll, pitch = saturate(roll, pitch, acfg.control_limits)
        roll, pitch = slew_rate(roll, pitch, prev_cmd, acfg.control_limits, _DT)

        cmd = ControlCmd(monotonic_ns(), roll, pitch, 0.0, gcfg.throttle_hold)
        bus.publish("control/cmd", cmd)
        prev_cmd = cmd

        i += 1
        if i % _HEALTH_EVERY == 0:
            bus.publish(
                "system/health",
                HealthReport(monotonic_ns(), "control", ProcessState.OK, ""),
            )

    bus.detach()
    log.info("control: stopped (last state=%s)", state.value)
```

- [ ] **Step 2: Verify import**

```bash
python -c "from quadguide.control.worker import run; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Run full unit test suite**

```bash
python -m pytest tests/unit/ -v
```

Expected: all pass, no regressions

- [ ] **Step 4: Commit**

```bash
git add src/quadguide/control/worker.py
git commit -m "feat(control): implement control worker (100 Hz SCHED_FIFO loop with watchdog)"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All 8 stub files implemented; config additions done; `__init__.py` files created for guidance/ and control/; lock-on seq reset in LOS; body-rate correction via rotation matrix; throttle hold in guidance config; EMA closing-vel with fallback; slew rate using fixed nominal dt; failsafe publishes level cmd and continues loop.
- [x] **No placeholders:** All steps have complete code.
- [x] **Type consistency:** `LOSRateEstimator.update()` returns `tuple[float, float]` — used directly as `los_r` in worker and passed to `pronav()` which accepts `tuple[float, float]`. `ClosingVelEstimator.update()` returns `float` — used as `v_c`. `attitude_cmd_compute()` returns `tuple[float, float]` consumed by `saturate()` which also returns `tuple[float, float]`. `failsafe_cmd()` returns `ControlCmd` — published directly. All consistent.

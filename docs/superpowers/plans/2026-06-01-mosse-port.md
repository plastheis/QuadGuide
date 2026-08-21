# MOSSE Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement MOSSE as the first concrete CF tracker, satisfying the full transferable-filter contract (`build_filter`/`evaluate`/`get_filter`/`set_filter`/`response_map`/`psr`) and built entirely on `edgecv/trackers/cf/ops/`.

**Architecture:** Grayscale MOSSE (Bolme 2010), no scale adaptation. Filter is numerator/denominator (`A`,`B`) in the frequency domain, `H* = A/(B+λ)`. Pure module-level helpers (`_crop_patch`, `_preprocess`, `_subpixel_peak`, `_rand_warp`, `_bilinear_sample`) do the image math; the `Mosse` class wires them into the contract. **Convention (supersedes spec §4):** the desired-output Gaussian (`gaussian2d_labels`) peaks at the window **center** `(h//2, w//2)`, so a matched response peaks at center and target displacement is `peak − center` — there is NO fftshift wrap-around.

**Tech Stack:** Python 3.10+, numpy only (base wheel). Reuses ops `extract_raw`, `cos_window`, `fft2`, `ifft2`, `psr`; adds ops `gaussian2d_labels`, `fft_size`. Tests with pytest via `.venv/bin/python -m pytest`.

---

## File Structure

```
edgecv/trackers/cf/
├── ops/
│   ├── fft.py          # MODIFY — add fft_size()
│   ├── labels.py       # CREATE — gaussian2d_labels()
│   └── __init__.py     # MODIFY — export gaussian2d_labels, fft_size
├── mosse.py            # CREATE — module helpers + Mosse class
└── __init__.py         # MODIFY — export Mosse
tests/
├── test_cf_ops.py      # MODIFY — add fft_size + gaussian2d_labels tests
└── test_mosse.py       # CREATE — helper + tracker tests
```

Run all tests with: `.venv/bin/python -m pytest -q`

---

### Task 1: `fft_size` op

**Files:**
- Modify: `edgecv/trackers/cf/ops/fft.py`
- Modify: `edgecv/trackers/cf/ops/__init__.py`
- Test: `tests/test_cf_ops.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cf_ops.py` (extend the `from edgecv.trackers.cf.ops import (...)` block to include `fft_size`, then add):

```python
def test_fft_size_is_next_power_of_two():
    assert fft_size(1) == 1
    assert fft_size(2) == 2
    assert fft_size(3) == 4
    assert fft_size(64) == 64
    assert fft_size(65) == 128


def test_fft_size_is_monotonic_and_at_least_input():
    prev = 0
    for n in range(1, 130):
        s = fft_size(n)
        assert s >= n
        assert s >= prev
        prev = s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cf_ops.py::test_fft_size_is_next_power_of_two -v`
Expected: FAIL — `ImportError: cannot import name 'fft_size'`

- [ ] **Step 3: Write minimal implementation**

Append to `edgecv/trackers/cf/ops/fft.py`:

```python
def fft_size(n: int) -> int:
    """Smallest efficient transform length >= n. Numpy reference: next power of two.

    CF templates are a fixed size after init and transform every frame, so rounding
    the crop up to a power of two keeps the FFT fast without changing the algorithm.
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()
```

Add `fft_size` to the imports and `__all__` in `edgecv/trackers/cf/ops/__init__.py`:

```python
from edgecv.trackers.cf.ops.fft import fft2, fft_backends, fft_size, ifft2, set_fft_backend
```

and add `"fft_size",` to the `__all__` list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cf_ops.py -q`
Expected: PASS (all ops tests, including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/cf/ops/fft.py edgecv/trackers/cf/ops/__init__.py tests/test_cf_ops.py
git commit -m "feat(cf/ops): add fft_size power-of-two helper"
```

---

### Task 2: `gaussian2d_labels` op

**Files:**
- Create: `edgecv/trackers/cf/ops/labels.py`
- Modify: `edgecv/trackers/cf/ops/__init__.py`
- Test: `tests/test_cf_ops.py`

- [ ] **Step 1: Write the failing test**

Add `gaussian2d_labels` to the ops import block in `tests/test_cf_ops.py`, then add:

```python
def test_gaussian2d_labels_peaks_at_center():
    g = gaussian2d_labels((16, 24), sigma=2.0)
    assert g.shape == (16, 24)
    assert g.dtype == np.float32
    assert np.unravel_index(int(np.argmax(g)), g.shape) == (8, 12)
    assert g[8, 12] == pytest.approx(1.0)


def test_gaussian2d_labels_values_in_unit_interval():
    g = gaussian2d_labels((16, 16), sigma=2.0)
    assert g.min() > 0.0
    assert g.max() <= 1.0


def test_gaussian2d_labels_wider_sigma_has_more_support():
    narrow = gaussian2d_labels((32, 32), sigma=1.0)
    wide = gaussian2d_labels((32, 32), sigma=4.0)
    assert (wide > 0.5).sum() > (narrow > 0.5).sum()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cf_ops.py::test_gaussian2d_labels_peaks_at_center -v`
Expected: FAIL — `ImportError: cannot import name 'gaussian2d_labels'`

- [ ] **Step 3: Write minimal implementation**

Create `edgecv/trackers/cf/ops/labels.py`:

```python
"""Regression-target labels for CF training (ARCHITECTURE.md §6.1).

The desired correlation output: a 2D Gaussian peaked at the window centre. CF
trackers form ``G = fft2(gaussian2d_labels(...))`` and train the filter so a
matched patch produces this response. Peaked at centre means target displacement
reads directly as ``peak - centre`` (no fftshift wrap).
"""

from __future__ import annotations

import numpy as np


def gaussian2d_labels(size: tuple[int, int], sigma: float) -> np.ndarray:
    """Peak-normalised 2D Gaussian of shape ``size`` = (h, w), centred at (h//2, w//2)."""
    h, w = size
    ys = np.arange(h) - h // 2
    xs = np.arange(w) - w // 2
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.exp(-(xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
    return g.astype(np.float32)
```

Add to `edgecv/trackers/cf/ops/__init__.py` imports and `__all__`:

```python
from edgecv.trackers.cf.ops.labels import gaussian2d_labels
```

and add `"gaussian2d_labels",` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cf_ops.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/cf/ops/labels.py edgecv/trackers/cf/ops/__init__.py tests/test_cf_ops.py
git commit -m "feat(cf/ops): add gaussian2d_labels regression target"
```

---

### Task 3: `mosse.py` module + `_crop_patch` helper

**Files:**
- Create: `edgecv/trackers/cf/mosse.py`
- Test: `tests/test_mosse.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mosse.py`:

```python
import numpy as np
import pytest

from edgecv.trackers.cf.mosse import _crop_patch


def test_crop_patch_fully_inside_keeps_shape():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch = _crop_patch(frame, center=(5.0, 5.0), size=(6, 6))
    assert patch.shape == (6, 6)


def test_crop_patch_edge_pads_when_window_crosses_border():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch = _crop_patch(frame, center=(1.0, 1.0), size=(6, 6))
    assert patch.shape == (6, 6)
    # top-left is outside the frame; edge mode replicates frame[0, 0] == 0
    assert patch[0, 0] == frame[0, 0]


def test_crop_patch_preserves_channels():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    patch = _crop_patch(frame, center=(5.0, 5.0), size=(6, 6))
    assert patch.shape == (6, 6, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mosse.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'edgecv.trackers.cf.mosse'`

- [ ] **Step 3: Write minimal implementation**

Create `edgecv/trackers/cf/mosse.py` with the module header and the helper:

```python
"""MOSSE correlation-filter tracker (Bolme et al. 2010).

Grayscale, no scale adaptation. Implements the full transferable-filter contract
(ARCHITECTURE.md §6.1) on top of edgecv.trackers.cf.ops. Desired-output Gaussian
peaks at the window centre, so target displacement is peak - centre (no fftshift
wrap)."""

from __future__ import annotations

import time

import numpy as np

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.trackers.cf import ops
from edgecv.trackers.cf.base import CorrelationFilterTracker, EvalResult, FilterState


def _crop_patch(frame: np.ndarray, center: tuple[float, float],
                size: tuple[int, int]) -> np.ndarray:
    """Fixed-size patch centred at ``center`` (cx, cy) pixels, edge-padded at borders."""
    cx, cy = center
    th, tw = size
    h, w = frame.shape[0], frame.shape[1]
    x0 = int(round(cx - tw / 2.0))
    y0 = int(round(cy - th / 2.0))
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x0 + tw - w), max(0, y0 + th - h)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + tw), min(h, y0 + th)
    patch = frame[sy0:sy1, sx0:sx1]
    if px0 or px1 or py0 or py1:
        pad = [(py0, py1), (px0, px1)] + [(0, 0)] * (frame.ndim - 2)
        patch = np.pad(patch, pad, mode="edge")
    return patch
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mosse.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/cf/mosse.py tests/test_mosse.py
git commit -m "feat(cf/mosse): add border-safe _crop_patch helper"
```

---

### Task 4: `_bilinear_sample` + `_rand_warp` helpers

**Files:**
- Modify: `edgecv/trackers/cf/mosse.py`
- Test: `tests/test_mosse.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mosse.py` (extend the import to `from edgecv.trackers.cf.mosse import _bilinear_sample, _crop_patch, _rand_warp`):

```python
def test_bilinear_sample_identity_grid_returns_same_image():
    img = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
    ys, xs = np.indices((8, 8)).astype(np.float32)
    out = _bilinear_sample(img, xs, ys)
    np.testing.assert_allclose(out, img, atol=1e-5)


def test_rand_warp_preserves_shape_and_constant_image():
    rng = np.random.default_rng(1)
    const = np.full((16, 16), 5.0, np.float32)
    out = _rand_warp(const, rng)
    assert out.shape == (16, 16)
    np.testing.assert_allclose(out, 5.0, atol=1e-4)


def test_rand_warp_is_seed_deterministic():
    patch = np.random.default_rng(2).standard_normal((16, 16)).astype(np.float32)
    a = _rand_warp(patch, np.random.default_rng(7))
    b = _rand_warp(patch, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mosse.py -q`
Expected: FAIL — `ImportError: cannot import name '_bilinear_sample'`

- [ ] **Step 3: Write minimal implementation**

Add to `edgecv/trackers/cf/mosse.py` (after `_crop_patch`):

```python
def _bilinear_sample(img: np.ndarray, src_x: np.ndarray, src_y: np.ndarray) -> np.ndarray:
    """Sample ``img`` at floating (src_x, src_y) coords with clamped bilinear interpolation."""
    h, w = img.shape[0], img.shape[1]
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    wx = (src_x - x0).astype(np.float32)
    wy = (src_y - y0).astype(np.float32)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    if img.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    ia, ib = img[y0c, x0c], img[y0c, x1c]
    ic, idd = img[y1c, x0c], img[y1c, x1c]
    top = ia * (1.0 - wx) + ib * wx
    bot = ic * (1.0 - wx) + idd * wx
    return (top * (1.0 - wy) + bot * wy).astype(img.dtype)


def _rand_warp(patch: np.ndarray, rng: np.random.Generator,
               max_rot_deg: float = 2.0, max_scale: float = 0.02) -> np.ndarray:
    """Small random rotation+scale about the patch centre (Bolme init augmentation).

    Rotation/scale keep the target centred, so the centred desired-output Gaussian
    stays valid across augmented samples.
    """
    h, w = patch.shape[0], patch.shape[1]
    ang = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
    scale = 1.0 + rng.uniform(-max_scale, max_scale)
    cos_a = np.cos(ang) / scale
    sin_a = np.sin(ang) / scale
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.indices((h, w)).astype(np.float32)
    xr, yr = xs - cx, ys - cy
    src_x = cos_a * xr + sin_a * yr + cx
    src_y = -sin_a * xr + cos_a * yr + cy
    return _bilinear_sample(patch, src_x, src_y)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mosse.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/cf/mosse.py tests/test_mosse.py
git commit -m "feat(cf/mosse): add bilinear sampler and random-warp augmentation"
```

---

### Task 5: `_preprocess` + `_subpixel_peak` helpers

**Files:**
- Modify: `edgecv/trackers/cf/mosse.py`
- Test: `tests/test_mosse.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mosse.py` (extend import to include `_preprocess, _subpixel_peak`):

```python
def test_preprocess_constant_patch_is_all_zero():
    # z-score of a constant has zero std -> zero; windowing keeps it zero.
    patch = np.full((16, 16, 3), 100, np.uint8)
    window = np.ones((16, 16), np.float32)
    out = _preprocess(patch, window)
    assert out.shape == (16, 16)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 0.0, atol=1e-5)


def test_subpixel_peak_interpolates_fractional_offset():
    r = np.zeros((5, 5), np.float32)
    r[2, 1], r[2, 2], r[2, 3] = 2.0, 4.0, 3.0   # peak at (2,2), skewed toward +x
    py, px = _subpixel_peak(r)
    assert py == pytest.approx(2.0, abs=1e-6)
    assert px == pytest.approx(2.0 + 0.5 * (2.0 - 3.0) / (2.0 - 8.0 + 3.0), abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mosse.py -q`
Expected: FAIL — `ImportError: cannot import name '_preprocess'`

- [ ] **Step 3: Write minimal implementation**

Add to `edgecv/trackers/cf/mosse.py`:

```python
def _preprocess(patch: np.ndarray, window: np.ndarray) -> np.ndarray:
    """MOSSE preprocessing: grayscale -> log -> z-score -> cosine window."""
    gray = ops.extract_raw(patch)[..., 0]
    x = np.log(gray + 1.0)
    x = (x - x.mean()) / (x.std() + 1e-5)
    return (x * window).astype(np.float32)


def _subpixel_peak(response: np.ndarray) -> tuple[float, float]:
    """Refined (py, px) peak location via per-axis parabolic interpolation."""
    h, w = response.shape
    iy, ix = np.unravel_index(int(np.argmax(response)), response.shape)
    py, px = float(iy), float(ix)
    if 0 < ix < w - 1:
        left, ctr, right = response[iy, ix - 1], response[iy, ix], response[iy, ix + 1]
        denom = left - 2.0 * ctr + right
        if denom != 0:
            px += 0.5 * (left - right) / denom
    if 0 < iy < h - 1:
        up, ctr, down = response[iy - 1, ix], response[iy, ix], response[iy + 1, ix]
        denom = up - 2.0 * ctr + down
        if denom != 0:
            py += 0.5 * (up - down) / denom
    return py, px
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mosse.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/cf/mosse.py tests/test_mosse.py
git commit -m "feat(cf/mosse): add preprocess and subpixel-peak helpers"
```

---

### Task 6: `Mosse.__init__`, `build_filter`, `get_filter`, `name`

**Files:**
- Modify: `edgecv/trackers/cf/mosse.py`
- Modify: `edgecv/trackers/cf/__init__.py`
- Test: `tests/test_mosse.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mosse.py` (add `from edgecv.trackers.cf.mosse import Mosse` and `from edgecv.core.bbox import BoundingBox`):

```python
def _blob_frame(h=120, w=160, cx=80.0, cy=60.0, blob_sigma=6.0):
    ys, xs = np.indices((h, w)).astype(np.float32)
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * blob_sigma ** 2))
    img = (g * 255.0).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def _box_at(cx, cy, w_img, h_img, bw=40, bh=40):
    return BoundingBox(
        x=(cx - bw / 2) / w_img, y=(cy - bh / 2) / h_img,
        w=bw / w_img, h=bh / h_img)


def test_build_filter_produces_complex64_AB_of_template_size():
    frame = _blob_frame()
    t = Mosse(n_warps=2)
    state = t.build_filter(frame, _box_at(80, 60, 160, 120))
    th, tw = state.meta["template_size"]
    assert (th, tw) == (64, 64)
    assert state.arrays["A"].dtype == np.complex64
    assert state.arrays["B"].dtype == np.complex64
    assert state.arrays["A"].shape == (64, 64)
    assert state.meta["abi"] == "mosse-1"


def test_build_filter_is_pure_and_seed_deterministic():
    frame = _blob_frame()
    t = Mosse(n_warps=4, rng_seed=3)
    box = _box_at(80, 60, 160, 120)
    s1 = t.build_filter(frame, box)
    s2 = t.build_filter(frame, box)
    np.testing.assert_array_equal(s1.arrays["A"], s2.arrays["A"])
    np.testing.assert_array_equal(s1.arrays["B"], s2.arrays["B"])


def test_name_is_mosse():
    assert Mosse().name() == "MOSSE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mosse.py::test_name_is_mosse -v`
Expected: FAIL — `ImportError: cannot import name 'Mosse'`

- [ ] **Step 3: Write minimal implementation**

Add the class to `edgecv/trackers/cf/mosse.py` (after the helpers):

```python
class Mosse(CorrelationFilterTracker):
    def __init__(self, *, padding: float = 1.0, sigma: float = 2.0, eta: float = 0.125,
                 lmbda: float = 1e-3, n_warps: int = 8, psr_lock: float = 7.0,
                 psr_lost: float = 5.0, rng_seed: int = 0) -> None:
        self._padding = padding
        self._sigma = sigma
        self._eta = eta
        self._lmbda = lmbda
        self._n_warps = n_warps
        self._psr_lock = psr_lock
        self._psr_lost = psr_lost
        self._rng_seed = rng_seed
        self._state: FilterState | None = None
        self._G: np.ndarray | None = None
        self._response: np.ndarray | None = None
        self._psr: float = 0.0
        self._status: TrackStatus = TrackStatus.INITIALIZING
        self._seq: int = 0

    def name(self) -> str:
        return "MOSSE"

    def build_filter(self, frame: np.ndarray, bbox: BoundingBox) -> FilterState:
        h_img, w_img = frame.shape[0], frame.shape[1]
        pix = bbox.to_pixels(w_img, h_img)
        cx, cy = pix.center
        th = ops.fft_size(int(round(pix.h * self._padding)))
        tw = ops.fft_size(int(round(pix.w * self._padding)))
        window = ops.cos_window((th, tw))
        big_g = ops.fft2(ops.gaussian2d_labels((th, tw), self._sigma))
        rng = np.random.default_rng(self._rng_seed)
        a = np.zeros((th, tw), np.complex128)
        b = np.zeros((th, tw), np.complex128)
        for i in range(self._n_warps + 1):
            patch = _crop_patch(frame, (cx, cy), (th, tw))
            if i > 0:
                patch = _rand_warp(patch.astype(np.float32), rng)
            f = ops.fft2(_preprocess(patch, window))
            a += big_g * np.conj(f)
            b += f * np.conj(f)
        meta = {
            "template_size": (th, tw), "padding": self._padding, "sigma": self._sigma,
            "eta": self._eta, "lambda": self._lmbda, "feature": "raw",
            "preproc": "log_zscore", "abi": "mosse-1",
        }
        return FilterState(
            arrays={"A": a.astype(np.complex64), "B": b.astype(np.complex64)},
            bbox=bbox, meta=meta)

    def get_filter(self) -> FilterState:
        assert self._state is not None, "init() or set_filter() must run before get_filter()"
        return self._state
```

> Note: `_rand_warp` receives `patch.astype(np.float32)` so warping a `uint8` crop interpolates in float; `_preprocess` re-grayscales regardless, so passing a 3-channel float patch is fine (`extract_raw` handles both).

Add `Mosse` to `edgecv/trackers/cf/__init__.py`:

```python
from edgecv.trackers.cf.mosse import Mosse

__all__ = ["Mosse"]
```

(If `__init__.py` already has content, merge rather than overwrite: add the import and append `"Mosse"` to the existing `__all__`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mosse.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/cf/mosse.py edgecv/trackers/cf/__init__.py tests/test_mosse.py
git commit -m "feat(cf/mosse): add Mosse class with pure build_filter"
```

---

### Task 7: `evaluate`

**Files:**
- Modify: `edgecv/trackers/cf/mosse.py`
- Test: `tests/test_mosse.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mosse.py`:

```python
def test_evaluate_is_pure_and_returns_centered_peak_on_build_frame():
    frame = _blob_frame(cx=80.0, cy=60.0)
    t = Mosse(n_warps=2)
    box = _box_at(80, 60, 160, 120)
    state = t.build_filter(frame, box)
    a_before = state.arrays["A"].copy()

    er = t.evaluate(frame, state)

    # purity: evaluate must not mutate the state it was given
    np.testing.assert_array_equal(state.arrays["A"], a_before)
    th, tw = state.meta["template_size"]
    assert er.response_map.shape == (th, tw)
    assert np.isfinite(er.psr)
    # matched filter on its own build frame -> peak ~ centre -> box centre ~ unchanged
    cx, cy = er.bbox.to_pixels(160, 120).center
    assert cx == pytest.approx(80.0, abs=1.5)
    assert cy == pytest.approx(60.0, abs=1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mosse.py::test_evaluate_is_pure_and_returns_centered_peak_on_build_frame -v`
Expected: FAIL — `TypeError: Can't instantiate abstract class Mosse ... evaluate` (or AttributeError if partially defined)

- [ ] **Step 3: Write minimal implementation**

Add the `evaluate` method to the `Mosse` class:

```python
    def evaluate(self, frame: np.ndarray, state: FilterState) -> EvalResult:
        th, tw = state.meta["template_size"]
        lam = state.meta["lambda"]
        h_img, w_img = frame.shape[0], frame.shape[1]
        pix = state.bbox.to_pixels(w_img, h_img)
        cx, cy = pix.center
        window = ops.cos_window((th, tw))
        f = ops.fft2(_preprocess(_crop_patch(frame, (cx, cy), (th, tw)), window))
        h_conj = state.arrays["A"] / (state.arrays["B"] + lam)
        response = np.real(ops.ifft2(f * h_conj))
        py, px = _subpixel_peak(response)
        new_cx = cx + (px - tw // 2)
        new_cy = cy + (py - th // 2)
        new_pix = PixelBox(x=new_cx - pix.w / 2.0, y=new_cy - pix.h / 2.0, w=pix.w, h=pix.h)
        new_bbox = BoundingBox.from_pixels(new_pix, w_img, h_img)
        return EvalResult(bbox=new_bbox, response_map=response, psr=ops.psr(response))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mosse.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/cf/mosse.py tests/test_mosse.py
git commit -m "feat(cf/mosse): add pure evaluate (centered-peak displacement)"
```

---

### Task 8: `init`, `status`, `set_filter`, properties

**Files:**
- Modify: `edgecv/trackers/cf/mosse.py`
- Test: `tests/test_mosse.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mosse.py` (extend import to include `from edgecv.core.result import TrackStatus`):

```python
def test_init_builds_filter_and_locks():
    frame = _blob_frame()
    t = Mosse(n_warps=2)
    t.init(frame, _box_at(80, 60, 160, 120))
    assert t.status == TrackStatus.LOCKED
    assert t.get_filter().arrays["A"].shape == (64, 64)


def test_set_filter_with_search_box_moves_crop_center():
    frame = _blob_frame()
    t = Mosse(n_warps=2)
    t.init(frame, _box_at(80, 60, 160, 120))
    state = t.get_filter()
    search = _box_at(100, 70, 160, 120)
    t.set_filter(state, search_box=search)
    cx, cy = t.get_filter().bbox.to_pixels(160, 120).center
    assert cx == pytest.approx(100.0, abs=0.5)
    assert cy == pytest.approx(70.0, abs=0.5)


def test_get_set_round_trip_preserves_evaluation():
    frame = _blob_frame()
    t = Mosse(n_warps=2)
    t.init(frame, _box_at(80, 60, 160, 120))
    er1 = t.evaluate(frame, t.get_filter())
    t.set_filter(t.get_filter())
    er2 = t.evaluate(frame, t.get_filter())
    np.testing.assert_allclose(
        er1.bbox.to_pixels(160, 120).center, er2.bbox.to_pixels(160, 120).center, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mosse.py::test_init_builds_filter_and_locks -v`
Expected: FAIL — abstract method `status`/`set_filter` not implemented (`TypeError: Can't instantiate abstract class`)

- [ ] **Step 3: Write minimal implementation**

Add to the `Mosse` class:

```python
    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        self._state = self.build_filter(frame, bbox)
        th, tw = self._state.meta["template_size"]
        self._G = ops.fft2(ops.gaussian2d_labels((th, tw), self._state.meta["sigma"]))
        self._status = TrackStatus.LOCKED
        self._response = None
        self._psr = 0.0
        self._seq = 0

    def set_filter(self, state: FilterState, search_box: BoundingBox | None = None) -> None:
        self._state = state
        th, tw = state.meta["template_size"]
        self._G = ops.fft2(ops.gaussian2d_labels((th, tw), state.meta["sigma"]))
        if search_box is not None:
            scx, scy = search_box.center
            bw, bh = state.bbox.w, state.bbox.h
            self._state.bbox = BoundingBox(x=scx - bw / 2.0, y=scy - bh / 2.0, w=bw, h=bh)

    def _status_from(self, psr: float) -> TrackStatus:
        if psr >= self._psr_lock:
            return TrackStatus.LOCKED
        if psr >= self._psr_lost:
            return TrackStatus.COASTING
        return TrackStatus.LOST

    @property
    def status(self) -> TrackStatus:
        return self._status

    @property
    def response_map(self) -> np.ndarray:
        assert self._response is not None, "response_map is available only after update()"
        return self._response

    @property
    def psr(self) -> float:
        return self._psr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mosse.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/cf/mosse.py tests/test_mosse.py
git commit -m "feat(cf/mosse): add init, set_filter, status, and properties"
```

---

### Task 9: `update` — tracking, subpixel, failure freeze

**Files:**
- Modify: `edgecv/trackers/cf/mosse.py`
- Test: `tests/test_mosse.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mosse.py`:

```python
def test_update_tracks_translating_blob():
    t = Mosse(n_warps=4)
    t.init(_blob_frame(cx=80.0, cy=60.0), _box_at(80, 60, 160, 120))
    true_cx, true_cy = 80.0, 60.0
    for _ in range(5):
        true_cx += 3.0
        true_cy += 2.0
        res = t.update(_blob_frame(cx=true_cx, cy=true_cy))
    cx, cy = res.bbox.to_pixels(160, 120).center
    assert cx == pytest.approx(true_cx, abs=3.0)
    assert cy == pytest.approx(true_cy, abs=3.0)
    assert res.status == TrackStatus.LOCKED
    assert res.seq == 5


def test_update_resolves_subpixel_shift():
    t = Mosse(n_warps=4)
    t.init(_blob_frame(cx=80.0, cy=60.0), _box_at(80, 60, 160, 120))
    res = t.update(_blob_frame(cx=80.5, cy=60.0))
    cx, _ = res.bbox.to_pixels(160, 120).center
    assert 80.2 < cx < 80.8   # fractional, not snapped to an integer pixel


def test_update_on_noise_reports_lost_and_freezes_filter():
    t = Mosse(n_warps=4)
    t.init(_blob_frame(cx=80.0, cy=60.0), _box_at(80, 60, 160, 120))
    a_before = t.get_filter().arrays["A"].copy()
    noise = (np.random.default_rng(0).integers(0, 256, (120, 160, 3))).astype(np.uint8)
    res = t.update(noise)
    assert res.status == TrackStatus.LOST
    np.testing.assert_array_equal(t.get_filter().arrays["A"], a_before)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mosse.py::test_update_tracks_translating_blob -v`
Expected: FAIL — `TypeError: Can't instantiate abstract class Mosse` (abstract `update` from `Tracker` still unimplemented)

- [ ] **Step 3: Write minimal implementation**

Add the `update` method to the `Mosse` class:

```python
    def update(self, frame: np.ndarray) -> TrackResult:
        assert self._state is not None and self._G is not None, "init() must run before update()"
        er = self.evaluate(frame, self._state)
        self._response = er.response_map
        self._psr = er.psr
        self._status = self._status_from(er.psr)
        self._state.bbox = er.bbox
        if er.psr >= self._psr_lost:
            th, tw = self._state.meta["template_size"]
            h_img, w_img = frame.shape[0], frame.shape[1]
            cx, cy = er.bbox.to_pixels(w_img, h_img).center
            window = ops.cos_window((th, tw))
            f = ops.fft2(_preprocess(_crop_patch(frame, (cx, cy), (th, tw)), window))
            a_new = self._G * np.conj(f)
            b_new = f * np.conj(f)
            eta = self._eta
            self._state.arrays["A"] = (
                eta * a_new + (1.0 - eta) * self._state.arrays["A"]).astype(np.complex64)
            self._state.arrays["B"] = (
                eta * b_new + (1.0 - eta) * self._state.arrays["B"]).astype(np.complex64)
        self._seq += 1
        return TrackResult(bbox=er.bbox, confidence=er.psr, status=self._status,
                           timestamp=time.monotonic(), seq=self._seq)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mosse.py -q`
Expected: PASS (all MOSSE tests)

> If `test_update_tracks_translating_blob` is borderline, the likely cause is the
> per-frame motion being large relative to the template; the assertion tolerance
> (`abs=3.0`) and 5-frame run are chosen to stay well inside half the 64px template.
> Do not loosen tolerances to mask a real failure — debug with
> `superpowers:systematic-debugging` first.

- [ ] **Step 5: Commit**

```bash
git add edgecv/trackers/cf/mosse.py tests/test_mosse.py
git commit -m "feat(cf/mosse): add online update with PSR-gated learning and freeze"
```

---

### Task 10: Full-suite + lint/type gate

**Files:** none (verification only)

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (prior suite + new ops + MOSSE tests), no new skips.

- [ ] **Step 2: Run ruff**

Run: `.venv/bin/ruff check edgecv tests`
Expected: `All checks passed!`
If failures: fix them (common: import ordering — ruff's `I` rule; line length 100).

- [ ] **Step 3: Run mypy**

Run: `.venv/bin/mypy edgecv`
Expected: `Success: no issues found`
If failures: add precise annotations; do not add blanket `# type: ignore`.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore(cf/mosse): satisfy ruff and mypy"
```

(Skip this commit if Steps 1–3 were already clean.)

---

## Self-Review

**Spec coverage** (spec → task):
- §2 ops `gaussian2d_labels`, `fft_size` → Tasks 2, 1. ✓
- §3 math (G, preprocessing, A/B, H*=A/(B+λ), online update, seeded warps) → Tasks 5 (`_preprocess`), 6 (`build_filter`, warps, A/B), 7 (`evaluate`, H*), 9 (online update). ✓
- §4 peak localization (parabolic subpixel) → Task 5 (`_subpixel_peak`); **wrap handling intentionally dropped** — superseded by the centered-Gaussian convention (documented in the header and Task 7). ✓
- §5 `FilterState` arrays/meta (complex64, abi) → Task 6. ✓
- §6 contract methods (`init`/`update`/`build_filter`/`evaluate`/`get_filter`/`set_filter`/`response_map`/`psr`/`status`/`name`) → Tasks 6,7,8,9. ✓
- §6 single-source-of-truth `self._state.bbox` → Tasks 7,8,9 (no `self._center`). ✓
- §6.1 border-safe crop → Task 3. ✓
- §7 coordinate discipline (normalised boxes, PixelBox at boundary) → Tasks 6,7. ✓
- §8 test plan items 1–12 → Tasks 6–9 cover instantiation, init, build/evaluate purity, determinism, translation tracking, subpixel, failure freeze, set_filter search_box, get/set round-trip, coordinate invariants, border safety. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every test shows assertions. ✓

**Type/name consistency:** Helper names (`_crop_patch`, `_bilinear_sample`, `_rand_warp`, `_preprocess`, `_subpixel_peak`) and method names match across tasks; `meta` keys (`template_size`, `lambda`, `sigma`, `eta`, `abi`) consistent between `build_filter` (Task 6) and `evaluate`/`update`/`set_filter` (Tasks 7–9); `Mosse.__init__` param `lmbda` vs meta key `"lambda"` is deliberate (Python keyword). ✓

**Note for executor:** the `_blob_frame` and `_box_at` test helpers are defined once in Task 6 and reused by Tasks 7–9. If running tasks out of order, ensure both exist in `tests/test_mosse.py`.

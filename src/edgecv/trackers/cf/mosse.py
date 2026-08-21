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
from edgecv.fusion.calibrator import LinearCalibrator
from edgecv.trackers.cf import ops
from edgecv.trackers.cf.base import CorrelationFilterTracker, EvalResult, FilterState


def _crop_patch(
    frame: np.ndarray, center: tuple[float, float], size: tuple[int, int]
) -> np.ndarray:
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
    if sx0 >= sx1 or sy0 >= sy1:
        edge_y = int(np.clip(round(cy), 0, h - 1))
        edge_x = int(np.clip(round(cx), 0, w - 1))
        fill = frame[edge_y, edge_x]
        return np.broadcast_to(fill, (th, tw) + frame.shape[2:]).copy()
    patch = frame[sy0:sy1, sx0:sx1]
    if px0 or px1 or py0 or py1:
        pad = [(py0, py1), (px0, px1)] + [(0, 0)] * (frame.ndim - 2)
        patch = np.pad(patch, pad, mode="edge")
    return patch


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


def _rand_warp(
    patch: np.ndarray,
    rng: np.random.Generator,
    max_rot_deg: float = 2.0,
    max_scale: float = 0.02,
) -> np.ndarray:
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


class Mosse(CorrelationFilterTracker):
    default_calibrator = LinearCalibrator(low=3.0, high=15.0)


    def __init__(
        self,
        *,
        padding: float = 1.0,
        sigma: float = 2.0,
        eta: float = 0.125,
        lmbda: float = 1e-3,
        n_warps: int = 8,
        psr_lock: float = 7.0,
        psr_lost: float = 5.0,
        rng_seed: int = 0,
    ) -> None:
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
            "template_size": (th, tw),
            "padding": self._padding,
            "sigma": self._sigma,
            "eta": self._eta,
            "lambda": self._lmbda,
            "feature": "raw",
            "preproc": "log_zscore",
            "abi": "mosse-1",
        }
        return FilterState(
            arrays={"A": a.astype(np.complex64), "B": b.astype(np.complex64)},
            bbox=bbox,
            meta=meta,
        )

    def get_filter(self) -> FilterState:
        assert self._state is not None, "init() or set_filter() must run before get_filter()"
        return self._state

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

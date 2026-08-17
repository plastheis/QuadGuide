"""SiamFC tracker (ARCHITECTURE.md §6.2). Single two-input graph
(exemplar, search) -> score_map; multi-scale search adapts position and size.
Reference defaults: HonglinChu/SiamTrackers."""

from __future__ import annotations

import math
import time

import numpy as np

from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.trackers.cf.ops import psr
from edgecv.trackers.nn.base import UNSET, NNTracker, Template, resolve_pp
from edgecv.trackers.nn.preprocess import crop_with_context, resize_bilinear, to_input


def _hann2d(n: int) -> np.ndarray:
    h = np.hanning(n).astype(np.float32)
    win = np.outer(h, h)
    s = win.sum()
    return win / s if s > 0 else win


class SiamFC(NNTracker):
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 exemplar_size=UNSET, search_size=UNSET, context=UNSET,
                 total_stride=UNSET, response_up=UNSET, scale_num=UNSET,
                 scale_step=UNSET, scale_penalty=UNSET, scale_lr=UNSET,
                 window_influence=UNSET, color=UNSET, scale=UNSET,
                 score_lock=8.0, score_lost=4.0) -> None:
        super().__init__(manifest, backend=backend, model=model)
        pp = self._preprocessing
        self._exemplar_size = resolve_pp(exemplar_size, pp, "exemplar", 127)
        self._search_size = resolve_pp(search_size, pp, "search", 255)
        self._context = resolve_pp(context, pp, "context", 0.5)
        self._total_stride = resolve_pp(total_stride, pp, "total_stride", 8)
        self._response_up = resolve_pp(response_up, pp, "response_up", 16)
        self._scale_num = resolve_pp(scale_num, pp, "scale_num", 3)
        self._scale_step = resolve_pp(scale_step, pp, "scale_step", 1.0375)
        self._scale_penalty = resolve_pp(scale_penalty, pp, "scale_penalty", 0.9745)
        self._scale_lr = resolve_pp(scale_lr, pp, "scale_lr", 0.59)
        self._window_influence = resolve_pp(window_influence, pp, "window_influence", 0.176)
        self._color = resolve_pp(color, pp, "color", "rgb")
        self._scale = resolve_pp(scale, pp, "scale", 1.0)
        self._score_lock = score_lock
        self._score_lost = score_lost
        out = self._model.io_spec.outputs[0]
        self._out_name = out.name
        self._score_size = out.shape[-1]
        self._up_size = self._score_size * self._response_up
        self._hann = _hann2d(self._up_size)
        self._template: Template | None = None
        self._box: BoundingBox | None = None

    def name(self) -> str:
        return "SiamFC"

    def get_template(self) -> Template:
        assert self._template is not None, "init() must run first"
        return self._template

    def set_template(self, template: Template,
                     search_box: BoundingBox | None = None) -> None:
        self._template = template
        self._box = search_box if search_box is not None else template.bbox

    def _exemplar_side(self, pix: PixelBox) -> float:
        p = self._context * (pix.w + pix.h)
        return math.sqrt((pix.w + p) * (pix.h + p))

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        h_img, w_img = frame.shape[0], frame.shape[1]
        pix = bbox.to_pixels(w_img, h_img)
        s_z = self._exemplar_side(pix)
        patch, _ = crop_with_context(frame, pix.center, (s_z, s_z),
                                     (self._exemplar_size, self._exemplar_size))
        spec_z = self._model.io_spec.inputs[0]
        z = to_input(patch, spec_z, color=self._color, scale=self._scale)
        self._template = Template(arrays={"exemplar": z}, bbox=bbox, meta={"s_z": s_z})
        self._box = bbox
        self._status = TrackStatus.LOCKED
        self._seq = 0

    def _status_from(self, value: float) -> TrackStatus:
        if value >= self._score_lock:
            return TrackStatus.LOCKED
        if value >= self._score_lost:
            return TrackStatus.COASTING
        return TrackStatus.LOST

    def update(self, frame: np.ndarray) -> TrackResult:
        assert self._template is not None and self._box is not None, "init() first"
        h_img, w_img = frame.shape[0], frame.shape[1]
        pix = self._box.to_pixels(w_img, h_img)
        cx, cy = pix.center
        s_z = self._exemplar_side(pix)
        s_x = s_z * self._search_size / self._exemplar_size
        spec_x = self._model.io_spec.inputs[1]
        z = self._template.arrays["exemplar"]

        centre = self._scale_num // 2
        scales = self._scale_step ** (np.arange(self._scale_num) - centre)
        best = None  # (idx, factor, up, penalised_peak, raw_smap)
        for i, f in enumerate(scales):
            side = s_x * f
            patch, _ = crop_with_context(frame, (cx, cy), (side, side),
                                         (self._search_size, self._search_size))
            x = to_input(patch, spec_x, color=self._color, scale=self._scale)
            raw = self._model.infer({"exemplar": z, "search": x})[self._out_name]
            smap = np.asarray(raw, np.float32).reshape(self._score_size, self._score_size)
            up = resize_bilinear(smap[..., None], (self._up_size, self._up_size))[..., 0]
            penalty = 1.0 if i == centre else self._scale_penalty
            peak = float(up.max()) * penalty
            if best is None or peak > best[3]:
                best = (i, float(f), up, peak, smap)

        assert best is not None, "scale_num must be >= 1"
        _idx, factor, up, _peak, smap = best
        total = up.sum()
        resp = up / total if total > 0 else up
        resp = (1.0 - self._window_influence) * resp + self._window_influence * self._hann
        py, px = np.unravel_index(int(resp.argmax()), resp.shape)
        disp_x = (px - (self._up_size - 1) / 2.0) * self._total_stride / self._response_up
        disp_y = (py - (self._up_size - 1) / 2.0) * self._total_stride / self._response_up
        scale_x = (s_x * factor) / self._search_size
        new_cx = cx + disp_x * scale_x
        new_cy = cy + disp_y * scale_x

        scale_factor = (1.0 - self._scale_lr) + self._scale_lr * factor
        new_w = pix.w * scale_factor
        new_h = pix.h * scale_factor
        new_pix = PixelBox(x=new_cx - new_w / 2.0, y=new_cy - new_h / 2.0, w=new_w, h=new_h)
        self._box = BoundingBox.from_pixels(new_pix, w_img, h_img)

        conf = psr(smap)
        self._status = self._status_from(conf)
        self._seq += 1
        return TrackResult(bbox=self._box, confidence=float(conf), status=self._status,
                           timestamp=time.monotonic(), seq=self._seq)

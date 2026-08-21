"""NanoTrack V3 tracker (ARCHITECTURE.md §6.2). Split backbone+head architecture:
backbone model (255x255 -> 96ch 16x16 features) called twice (exemplar + search),
exemplar feature centre-cropped to 8x8, then head model (z_f, x_f) -> (cls, loc).
MobileNetV3-small-v3 + AdjustLayer + DepthwiseBAN anchor-free head.
Reference defaults: HonglinChu/SiamTrackers NanoTrack configv3."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from edgecv.backends.base import InferenceBackend
from edgecv.backends.registry import get_backend
from edgecv.core.bbox import BoundingBox, PixelBox
from edgecv.core.result import TrackResult, TrackStatus
from edgecv.models.manifest import ModelManifest, load_manifest
from edgecv.trackers.nn.base import UNSET, NNTracker, Template, resolve_pp, select_backend
from edgecv.trackers.nn.preprocess import crop_with_context, points_grid, to_input


def _split_submanifest(mf: ModelManifest, key: str, backend: str) -> ModelManifest:
    """A single-model manifest for one half (backbone/head) of a split NanoTrack.

    Carries that half's own io (so the RKNN backend, which has no name API, builds
    the right IOSpec and positional output order) and just that backend's artifact.
    """
    art = mf.artifacts.get(key)
    if not art:
        raise ValueError(
            f"manifest {mf.name!r} has no {key!r} artifact for split NanoTrack"
        )
    backend_art = art.get(backend)
    if not backend_art or "path" not in backend_art:
        raise ValueError(
            f"{key!r} artifact in manifest {mf.name!r} has no {backend!r} path"
        )
    io = art.get("io") or {}
    return ModelManifest(
        name=f"{mf.name}_{key}",
        task=mf.task,
        preprocessing=dict(mf.preprocessing),
        inputs=list(io.get("inputs") or []),
        outputs=list(io.get("outputs") or []),
        artifacts={backend: dict(backend_art)},
    )


def _hann2d(n: int) -> np.ndarray:
    h = np.hanning(n).astype(np.float32)
    return np.outer(h, h).reshape(-1)


def _softmax_fg(cls: np.ndarray) -> np.ndarray:
    """cls (1,2,S,S) logits -> foreground prob per location, flattened (S*S,)."""
    c = np.asarray(cls, np.float32).reshape(2, -1)
    c = c - c.max(axis=0, keepdims=True)
    e = np.exp(c)
    return (e / e.sum(axis=0, keepdims=True))[1]


class NanoTrack(NNTracker):
    def __init__(self, manifest=None, *, backend="auto", model=None,
                 backbone=None, head=None,
                 exemplar_size=UNSET, search_size=UNSET, context=UNSET,
                 stride=UNSET, base_size=UNSET, penalty_k=UNSET,
                 window_influence=UNSET, size_lr=UNSET, color=UNSET, scale=UNSET,
                 model_input=UNSET,
                 score_lock=0.6, score_lost=0.35) -> None:
        super().__init__(manifest, backend=backend, model=model or backbone)
        pp = self._preprocessing
        # exemplar_size / search_size are CONCEPTUAL for crop ratio; model_input
        # is the actual backbone input size (both crops resized to this).
        self._exemplar_size = resolve_pp(exemplar_size, pp, "exemplar", 127)
        self._search_size = resolve_pp(search_size, pp, "search", 255)
        self._model_input = resolve_pp(model_input, pp, "model_input", 255)
        self._context = resolve_pp(context, pp, "context", 0.5)
        self._stride = resolve_pp(stride, pp, "stride", 16)
        self._base_size = resolve_pp(base_size, pp, "base_size", 7)
        self._penalty_k = resolve_pp(penalty_k, pp, "penalty_k", 0.138)
        self._window_influence = resolve_pp(window_influence, pp, "window_influence", 0.455)
        self._size_lr = resolve_pp(size_lr, pp, "size_lr", 0.348)
        self._color = resolve_pp(color, pp, "color", "rgb")
        self._scale = resolve_pp(scale, pp, "scale", 1.0)
        self._score_lock = score_lock
        self._score_lost = score_lost

        # Two-model injection seam: explicit arg wins, else use the resolved
        # single model (backward compat), else load from manifest.
        self._backbone = backbone
        self._head = head
        if self._backbone is None and self._head is None:
            # Single-model (legacy) or manifest-loaded path
            self._head = self._model
            self._backbone = self._model  # same model for both

        # Read head output spec for cls/loc names and score size.
        head_out = self._head.io_spec.outputs
        names = [o.name for o in head_out]
        self._cls_name = "cls" if "cls" in names else names[0]
        self._loc_name = "loc" if "loc" in names else names[1]
        self._score_size = head_out[0].shape[-1]
        self._points = points_grid(self._stride, self._score_size)   # (2, S*S)
        self._hann = _hann2d(self._score_size)
        self._template: Template | None = None
        self._box: BoundingBox | None = None

    @classmethod
    def from_manifest(cls, manifest: ModelManifest | str | Path,
                      *, backend: str = "auto",
                      backend_obj: InferenceBackend | None = None,
                      **kwargs) -> NanoTrack:
        """Build a NanoTrack by loading its split backbone + head sub-models.

        Backend-agnostic: ``backend="rknn"`` runs on the Rockchip NPU, ``"onnx"``
        on the host/CI — both load the two artifacts named in the manifest's
        ``backbone``/``head`` entries. ``backend_obj`` injects a backend instance
        (tests; on-device the worker passes the resolved backend). Loading happens
        here, so for the rknn backend this MUST run inside the worker process that
        will use it (ARCHITECTURE.md §7.4, §14.7) — never in the parent.
        """
        mf = manifest if isinstance(manifest, ModelManifest) else load_manifest(manifest)
        name = select_backend(backend)
        be = backend_obj if backend_obj is not None else get_backend(name)
        backbone = be.load(_split_submanifest(mf, "backbone", name))
        head = be.load(_split_submanifest(mf, "head", name))
        # model=backbone satisfies the base single-model close() path without a
        # third load; NanoTrack.close() tears down backbone and head explicitly.
        return cls(mf, backend=name, model=backbone,
                   backbone=backbone, head=head, **kwargs)

    def name(self) -> str:
        return "NanoTrack"

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
        # Crop the SEARCH-sized window (s_x), not the exemplar window. The template
        # is the centre 8x8 of the 16x16 feature (z_feat[..., 4:12, 4:12] below);
        # that centre-crop only equals a true backbone(127 exemplar) when the 255
        # input spans s_x = s_z * search/exemplar (the central 8/16 of an s_x window
        # is exactly the s_z region at the right scale). Cropping s_z instead makes
        # the template ~2x too zoomed-in, so the box collapses onto a central feature
        # over successive frames. Must match the s_x used in update().
        s_x = s_z * self._search_size / self._exemplar_size
        patch, _ = crop_with_context(frame, pix.center, (s_x, s_x),
                                     (self._model_input, self._model_input))
        xf = to_input(patch, self._backbone.io_spec.inputs[0],
                      color=self._color, scale=self._scale)
        z_feat = np.asarray(
            self._backbone.infer({self._backbone.io_spec.inputs[0].name: xf})[
                self._backbone.io_spec.outputs[0].name
            ], np.float32,
        )  # (1, 96, 16, 16)
        z_f = z_feat[:, :, 4:12, 4:12]  # centre-crop 16->8

        self._template = Template(arrays={"exemplar": z_f}, bbox=bbox, meta={"s_z": s_z})
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
        # scale_z converts search-crop px to frame px.
        # With both crops at model_input: scale_z = model_input/s_x = exemplar/s_z
        # because s_x = s_z * search/exemplar.
        scale_z = self._exemplar_size / s_z
        z_f = self._template.arrays["exemplar"]  # (1, 96, 8, 8)

        # Crop search region, resize to model_input, feed through backbone.
        patch, _ = crop_with_context(frame, (cx, cy), (s_x, s_x),
                                     (self._model_input, self._model_input))
        xf = to_input(patch, self._backbone.io_spec.inputs[0],
                      color=self._color, scale=self._scale)
        x_feat = np.asarray(
            self._backbone.infer({self._backbone.io_spec.inputs[0].name: xf})[
                self._backbone.io_spec.outputs[0].name
            ], np.float32,
        )  # (1, 96, 16, 16)

        # Head: z_f (8x8), x_feat (16x16) -> cls, loc.
        head_in = {self._head.io_spec.inputs[0].name: z_f,
                   self._head.io_spec.inputs[1].name: x_feat}
        out = self._head.infer(head_in)

        score = _softmax_fg(out[self._cls_name])            # (S*S,)
        loc = np.asarray(out[self._loc_name], np.float32).reshape(4, -1)  # l,t,r,b
        px, py = self._points[0], self._points[1]           # search-crop px, centred at 0
        x1, y1 = px - loc[0], py - loc[1]
        x2, y2 = px + loc[2], py + loc[3]
        pred_cx, pred_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pred_w, pred_h = (x2 - x1), (y2 - y1)

        def _change(r):
            return np.maximum(r, 1.0 / r)

        def _sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        tw, th = pix.w * scale_z, pix.h * scale_z           # target size in search-crop px
        s_c = _change(_sz(pred_w, pred_h) / _sz(tw, th))
        r_c = _change((tw / th) / (pred_w / pred_h))
        penalty = np.exp(-(r_c * s_c - 1.0) * self._penalty_k)
        pscore = penalty * score
        pscore = (pscore * (1.0 - self._window_influence)
                  + self._hann * self._window_influence)
        best = int(pscore.argmax())

        lr = float(penalty[best] * score[best] * self._size_lr)
        new_cx = cx + float(pred_cx[best]) / scale_z
        new_cy = cy + float(pred_cy[best]) / scale_z
        new_w = pix.w * (1.0 - lr) + (float(pred_w[best]) / scale_z) * lr
        new_h = pix.h * (1.0 - lr) + (float(pred_h[best]) / scale_z) * lr

        new_pix = PixelBox(x=new_cx - new_w / 2.0, y=new_cy - new_h / 2.0,
                           w=new_w, h=new_h)
        self._box = BoundingBox.from_pixels(new_pix, w_img, h_img)

        conf = float(score[best])
        self._status = self._status_from(conf)
        self._seq += 1
        return TrackResult(bbox=self._box, confidence=conf, status=self._status,
                           timestamp=time.monotonic(), seq=self._seq)

    def close(self) -> None:
        if self._closed:
            return
        if self._backbone is not self._head and self._backbone is not None:
            self._backbone.close()
        if self._head is not None and self._head is not self._model:
            self._head.close()
        super().close()

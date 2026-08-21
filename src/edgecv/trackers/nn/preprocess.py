"""Pure numpy preprocessing for NN trackers (ARCHITECTURE.md §6.2, §7.4).

Module-level functions only, so a future spawned worker can import them. Numpy
reference today; RK RGA / DMA crop-resize can swap in behind this boundary later
with no tracker change (ARCHITECTURE §16). All crop<->frame and letterbox<->image
coordinate inversion lives here, in one place (ARCHITECTURE §5.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edgecv.backends.base import TensorSpec

try:  # optional accelerator: cv2 SIMD crop+resize (~50-100x the numpy gather).
    import cv2 as _cv2  # QuadGuide hard dep / EdgeCV `fast` extra; absent in CI.
except Exception:  # pragma: no cover - exercised by the numpy-fallback path
    _cv2 = None  # type: ignore[assignment]


def _sample_clamped(img: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Bilinear sample at float (gx, gy) frame coords; clamp = edge-replicate."""
    h, w = img.shape[0], img.shape[1]
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    wx = (gx - x0).astype(np.float32)
    wy = (gy - y0).astype(np.float32)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    if img.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    ia, ib = img[y0c, x0c], img[y0c, x1c]
    ic, idd = img[y1c, x0c], img[y1c, x1c]
    top = ia * (1.0 - wx) + ib * wx
    bot = ic * (1.0 - wx) + idd * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def _resize_bilinear_numpy(img: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Reference centre-aligned bilinear resize via a numpy gather."""
    oh, ow = out_hw
    h, w = img.shape[0], img.shape[1]
    xs = (np.arange(ow) + 0.5) * (w / ow) - 0.5
    ys = (np.arange(oh) + 0.5) * (h / oh) - 0.5
    gx, gy = np.meshgrid(xs, ys)
    return _sample_clamped(np.asarray(img, dtype=np.float32), gx, gy)


def resize_bilinear(img: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize (H,W[,C]) to out_hw with centre-aligned bilinear sampling.

    cv2 SIMD fast path (dominates the YOLO letterbox cost — ~80 ms → a few ms on
    a 640² resize), numpy reference fallback. cv2's INTER_LINEAR uses the SAME
    half-pixel grid (``src = (dst+0.5)*scale - 0.5``) and edge-replicate borders as
    the reference, so the LetterboxXform inversion is identical either way."""
    oh, ow = out_hw
    if _cv2 is None:
        return _resize_bilinear_numpy(img, out_hw)
    src = np.asarray(img, dtype=np.float32)
    out = _cv2.resize(src, (ow, oh), interpolation=_cv2.INTER_LINEAR)
    if src.ndim == 3 and out.ndim == 2:   # cv2 drops a trailing length-1 channel
        out = out[..., None]
    return np.ascontiguousarray(out, dtype=np.float32)


@dataclass(frozen=True)
class CropXform:
    center: tuple[float, float]    # crop centre in frame px (cx, cy)
    size_px: tuple[float, float]   # crop side in frame px (sh, sw)
    out_size: tuple[int, int]      # resized output (oh, ow)

    def to_frame(self, out_xy: tuple[float, float]) -> tuple[float, float]:
        """Map an output-patch pixel index ``(ox, oy)`` back to frame pixel
        coordinates; the ``+0.5`` term is the half-pixel sampling offset."""
        ox, oy = out_xy
        oh, ow = self.out_size
        sh, sw = self.size_px
        cx, cy = self.center
        fx = (cx - sw / 2.0) + (ox + 0.5) / ow * sw
        fy = (cy - sh / 2.0) + (oy + 0.5) / oh * sh
        return fx, fy


@dataclass(frozen=True)
class LetterboxXform:
    scale: float                   # uniform resize factor applied to the original
    pad: tuple[float, float]       # (pad_x, pad_y) added in output px
    out_size: tuple[int, int]      # (oh, ow)
    orig_size: tuple[int, int]     # (h, w)

    def to_orig_xyxy(self, xyxy: tuple[float, float, float, float]):
        x1, y1, x2, y2 = xyxy
        px, py = self.pad
        s = self.scale
        return ((x1 - px) / s, (y1 - py) / s, (x2 - px) / s, (y2 - py) / s)


def letterbox(
    image: np.ndarray, out_size: tuple[int, int], *, pad_value: float = 114.0
) -> tuple[np.ndarray, LetterboxXform]:
    """Aspect-preserving resize + symmetric pad into out_size (YOLO convention)."""
    oh, ow = out_size
    h, w = image.shape[0], image.shape[1]
    s = min(oh / h, ow / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = resize_bilinear(image, (nh, nw))
    ch = image.shape[2] if image.ndim == 3 else 1
    canvas = np.full((oh, ow, ch), pad_value, np.float32)
    pad_y = (oh - nh) / 2.0
    pad_x = (ow - nw) / 2.0
    y0, x0 = int(round(pad_y)), int(round(pad_x))
    block = resized if resized.ndim == 3 else resized[..., None]
    canvas[y0:y0 + nh, x0:x0 + nw] = block
    out = canvas if image.ndim == 3 else canvas[..., 0]
    return out, LetterboxXform(s, (float(x0), float(y0)), (oh, ow), (h, w))


def _crop_resize_numpy(
    frame: np.ndarray,
    center: tuple[float, float],
    size_px: tuple[float, float],
    out_size: tuple[int, int],
) -> np.ndarray:
    """Reference edge-replicate bilinear crop+resize via a numpy gather."""
    cx, cy = center
    sh, sw = size_px
    oh, ow = out_size
    fx = (cx - sw / 2.0) + (np.arange(ow) + 0.5) / ow * sw
    fy = (cy - sh / 2.0) + (np.arange(oh) + 0.5) / oh * sh
    gx, gy = np.meshgrid(fx, fy)
    patch = _sample_clamped(frame, gx, gy)
    if frame.ndim == 2:
        patch = patch.reshape(oh, ow)
    return patch


def _crop_resize(
    frame: np.ndarray,
    center: tuple[float, float],
    size_px: tuple[float, float],
    out_size: tuple[int, int],
) -> np.ndarray:
    """Edge-replicate bilinear crop+resize. cv2 fast path, numpy fallback.

    Both honour the SAME half-pixel sampling grid as `_crop_resize_numpy` — output
    pixel (ox, oy) samples frame coord ((cx-sw/2) + (ox+0.5)/ow*sw, …) — so the
    `CropXform` inversion (and every downstream box decode) is identical either way.
    """
    if _cv2 is None:
        return _crop_resize_numpy(frame, center, size_px, out_size)
    cx, cy = center
    sh, sw = size_px
    oh, ow = out_size
    # Separable affine mapping dst pixel -> src (frame) coord, matching the numpy
    # grid: src_x = a*ox + b with a = sw/ow, b = (cx-sw/2) + 0.5*a (the +0.5 is the
    # half-pixel centre offset — dropping it shifts the crop by half a sample).
    a = sw / ow
    c = sh / oh
    M = np.array(
        [[a, 0.0, (cx - sw / 2.0) + 0.5 * a],
         [0.0, c, (cy - sh / 2.0) + 0.5 * c]],
        np.float32,
    )
    patch = _cv2.warpAffine(
        frame, M, (ow, oh),
        flags=_cv2.INTER_LINEAR | _cv2.WARP_INVERSE_MAP,
        borderMode=_cv2.BORDER_REPLICATE,
    )
    return patch.astype(np.float32, copy=False)


def crop_with_context(
    frame: np.ndarray,
    center: tuple[float, float],
    size_px: tuple[float, float],
    out_size: tuple[int, int],
) -> tuple[np.ndarray, CropXform]:
    """Crop a (sh, sw)-px window centred at `center`, edge-replicate at borders,
    resize to out_size in one gather. Returns the patch and the inversion transform.

    Uses a cv2 SIMD crop+resize when available, falling back to a numpy gather;
    both share the half-pixel grid that `CropXform` inverts (ARCHITECTURE §6.2/§16)."""
    sh, sw = size_px
    oh, ow = out_size
    patch = _crop_resize(frame, center, size_px, out_size)
    return patch, CropXform(center, (sh, sw), (oh, ow))


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img[..., None]
    if img.shape[2] == 1:
        return img
    w = np.array([0.299, 0.587, 0.114], np.float32)
    return (img[..., :3] * w).sum(axis=2, keepdims=True)


def to_input(patch: np.ndarray, spec: TensorSpec, *, color: str = "rgb",
             scale: float = 1.0 / 255.0, mean=None, std=None) -> np.ndarray:
    """Colour-convert, normalise, pack to NCHW, cast to spec.dtype (quantise if INT8)."""
    img = patch.astype(np.float32)
    if color == "gray":
        img = _to_gray(img)
    elif img.ndim == 2:
        img = img[..., None]
    img = img * scale
    if mean is not None:
        img = ((img - np.asarray(mean, np.float32))
               / np.asarray(std, np.float32)).astype(np.float32)
    chw = np.transpose(img, (2, 0, 1))[None]          # 1,C,H,W
    if spec.quant:
        q = np.round(chw / spec.quant["scale"]) + spec.quant["zero_point"]
        info = np.iinfo(np.dtype(spec.dtype))
        return np.clip(q, info.min, info.max).astype(spec.dtype)
    return chw.astype(np.dtype(spec.dtype))


def points_grid(stride: int, size: int) -> np.ndarray:
    """Anchor-free point centres for a size×size head, in search-image pixels
    centred at 0. Returns (2, size*size): row 0 = x, row 1 = y, flattened
    row-major (index = row*size + col), matching a (C, S, S)->(C, S*S) reshape.
    Mirrors NanoTrack's generate_points: ori = -(size//2)*stride."""
    ori = -(size // 2) * stride
    coords = (ori + stride * np.arange(size)).astype(np.float32)
    gx, gy = np.meshgrid(coords, coords)          # gx[r,c]=coords[c], gy[r,c]=coords[r]
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=0)


def decode_yolo_dfl(outputs, strides, *, reg_max: int = 16,
                    conf_thresh: float = 0.25):
    """Decode the rknn_model_zoo-style separated YOLOv8/11 head.

    ``outputs`` is the list of per-scale head tensors in compiled order, grouped
    per stride: each group is ``[box_reg (1, 4*reg_max, H, W), cls (1, nc, H, W),
    ...]`` — a trailing score-sum tensor (if present) is ignored. ``cls`` is taken
    as already sigmoid-activated (probabilities). Returns ``(xyxy_px, scores)`` in
    input-letterbox pixels; the caller inverts the letterbox to original coords.

    Anchor-free DFL: the 64-channel box output is ``(4, reg_max)`` per cell;
    softmax over the bins, expectation → l/t/r/b distances (in cells), projected
    from the cell centre ``(col+0.5, row+0.5)`` and scaled by ``stride``.
    """
    per_scale = max(1, len(outputs) // len(strides))
    bins = np.arange(reg_max, dtype=np.float32)
    all_xyxy, all_score = [], []
    for s, stride in enumerate(strides):
        reg = np.asarray(outputs[s * per_scale + 0], np.float32)[0]   # (4*reg_max,H,W)
        cls = np.asarray(outputs[s * per_scale + 1], np.float32)[0]   # (nc,H,W)
        _, H, W = reg.shape
        # Threshold on score FIRST, then DFL-decode only the surviving cells —
        # the high-res P2 grid (160²) has thousands of cells but few detections,
        # so the per-cell softmax should run on the kept handful, not all of them.
        score = cls.max(axis=0).reshape(-1)                           # (H*W,) class-agnostic
        idx = np.nonzero(score >= conf_thresh)[0]
        if idx.size == 0:
            continue
        sc = score[idx]
        reg = reg.reshape(4, reg_max, H * W)[:, :, idx]               # (4,reg_max,K)
        reg = reg - reg.max(axis=1, keepdims=True)
        e = np.exp(reg)
        dist = ((e / e.sum(axis=1, keepdims=True))
                * bins[None, :, None]).sum(axis=1)                    # (4,K) l,t,r,b
        col = (idx % W).astype(np.float32) + 0.5                      # cell centre x
        row = (idx // W).astype(np.float32) + 0.5                     # cell centre y
        x1 = (col - dist[0]) * stride
        y1 = (row - dist[1]) * stride
        x2 = (col + dist[2]) * stride
        y2 = (row + dist[3]) * stride
        all_xyxy.append(np.stack([x1, y1, x2, y2], axis=1))
        all_score.append(sc)
    if all_xyxy:
        return (np.concatenate(all_xyxy, 0).astype(np.float32),
                np.concatenate(all_score, 0).astype(np.float32))
    return np.empty((0, 4), np.float32), np.empty((0,), np.float32)


def class_agnostic_nms(boxes_xyxy: np.ndarray, scores: np.ndarray,
                       iou_thresh: float) -> np.ndarray:
    """Greedy NMS over a single pool (class labels ignored). Returns kept indices."""
    if len(scores) == 0:
        return np.empty((0,), np.int64)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return np.array(keep, np.int64)

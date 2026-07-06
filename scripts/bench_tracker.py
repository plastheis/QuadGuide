#!/usr/bin/env python3
"""Compare NanoTrack ONNX precision variants for QuadGuide (CPU/onnxruntime).

The repo now vendors two backbone/head pairs in models/:

  fp32   nanotrackv3_backbone.onnx     + nanotrackv3_head.onnx     (original)
  quant  nanotrack_backbone_int8.onnx  + nanotrack_head_fp16.onnx  (new)

Neither the nanotrack.yaml manifest nor the onnx InferenceBackend has a slot
for the quant pair on CPU today (the manifest's "onnx" artifact is always the
fp32 path; int8/fp16 only exist there under "rknn" for the NPU). So this
script loads all four .onnx files directly and drives two independent
NanoTrack instances via dependency injection (backbone=, head=), bypassing
load_tracker()/the manifest entirely.

Two comparisons:

  parity   Same crop in, both variants out: backbone feature correlation,
           and head cls/loc correlation both isolated (fp32 features into
           each head) and end-to-end (each backbone feeding its own head).
           Validates the manifest's "~0.97 feature corr", "cls/loc corr 1.0"
           claims against these exact files.

  track    Two full trackers (init+update) run independently over a
           synthesized moving-target sequence, each maintaining its own box
           estimate. Reports per-model latency (backbone/head split) and how
           far each variant's track drifts from the known ground truth and
           from each other.

No camera/hardware required — the sequence is synthesized (deterministic,
seeded). Point --video at real footage to bench on it instead of synthetic
frames (parity mode still needs the synthetic ground-truth trajectory, so it
only applies to the synthetic path).

Examples:
  scripts/bench_tracker.py parity
  scripts/bench_tracker.py track --frames 300 --threads 1
  scripts/bench_tracker.py track --video clip.mp4 --frames 300
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edgecv.backends.base import Model  # noqa: E402
from edgecv.backends.onnx import OnnxModel  # noqa: E402
from edgecv.core.bbox import BoundingBox, PixelBox  # noqa: E402
from edgecv.trackers.nn.nanotrack import NanoTrack  # noqa: E402
from edgecv.trackers.nn.preprocess import crop_with_context, to_input  # noqa: E402

MODEL_FILES = {
    "backbone_fp32": "nanotrackv3_backbone.onnx",
    "head_fp32":     "nanotrackv3_head.onnx",
    "backbone_int8": "nanotrack_backbone_int8.onnx",
    "head_fp16":     "nanotrack_head_fp16.onnx",
}


# ── model loading ────────────────────────────────────────────────────────────

def _load_onnx(path: Path, threads: int) -> OnnxModel:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=so,
                                    providers=["CPUExecutionProvider"])
    return OnnxModel(session)


class TimedModel(Model):
    """Wraps a Model, recording each infer() call's wall time (ms)."""

    def __init__(self, inner: Model) -> None:
        self._inner = inner
        self.times_ms: list[float] = []

    @property
    def io_spec(self):
        return self._inner.io_spec

    def infer(self, inputs):
        t0 = time.monotonic_ns()
        out = self._inner.infer(inputs)
        self.times_ms.append((time.monotonic_ns() - t0) / 1e6)
        return out

    def close(self) -> None:
        self._inner.close()


def _resolve_paths(model_dir: Path, overrides: dict[str, str | None]) -> dict[str, Path]:
    paths = {}
    for key, default_name in MODEL_FILES.items():
        override = overrides.get(key)
        p = Path(override) if override else model_dir / default_name
        if not p.exists():
            raise SystemExit(f"model not found: {p}")
        paths[key] = p
    return paths


# ── synthetic sequence ───────────────────────────────────────────────────────

def _smooth_noise(rng: np.random.Generator, h: int, w: int, lo_res: int = 12) -> np.ndarray:
    """Low-res random field upsampled to (h, w), as a cheap Perlin-ish texture."""
    lo = rng.uniform(0.0, 1.0, (lo_res, lo_res)).astype(np.float32)
    ys = (np.arange(h) + 0.5) * (lo_res / h) - 0.5
    xs = (np.arange(w) + 0.5) * (lo_res / w) - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, lo_res - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, lo_res - 1)
    y1 = np.clip(y0 + 1, 0, lo_res - 1)
    x1 = np.clip(x0 + 1, 0, lo_res - 1)
    wy = np.clip(ys - y0, 0.0, 1.0)[:, None]
    wx = np.clip(xs - x0, 0.0, 1.0)[None, :]
    top = lo[y0][:, x0] * (1 - wx) + lo[y0][:, x1] * wx
    bot = lo[y1][:, x0] * (1 - wx) + lo[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def _background(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    bg = np.stack([_smooth_noise(rng, h, w) for _ in range(3)], axis=-1)
    bg = bg * 180.0 + 30.0  # keep off pure black/white so bilinear sampling is meaningful
    return bg.astype(np.float32)


def _trajectory(n: int, w: int, h: int, rng: np.random.Generator):
    """Lissajous path + slow radius breathing, kept well inside frame bounds."""
    margin = 0.28
    cx0, cy0 = w / 2.0, h / 2.0
    ax, ay = w * margin, h * margin
    fx, fy = rng.uniform(0.6, 1.1), rng.uniform(0.6, 1.1)
    phase = rng.uniform(0, 2 * np.pi)
    r0 = min(w, h) * 0.10
    t = np.arange(n) / max(n - 1, 1) * 2 * np.pi
    cx = cx0 + ax * np.sin(fx * t)
    cy = cy0 + ay * np.sin(fy * t + phase)
    r = r0 * (1.0 + 0.15 * np.sin(0.7 * t))
    return cx.astype(np.float32), cy.astype(np.float32), r.astype(np.float32)


def _draw_target(frame: np.ndarray, cx: float, cy: float, r: float,
                  rng: np.random.Generator) -> np.ndarray:
    h, w = frame.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    edge = r * 0.15
    alpha = np.clip(1.0 - (d - (r - edge)) / (2 * edge), 0.0, 1.0)  # soft disc mask
    speckle = rng.uniform(0.7, 1.3, (h, w)).astype(np.float32)
    target_color = np.array([210.0, 90.0, 60.0], np.float32)
    target = target_color[None, None, :] * speckle[..., None]
    out = frame * (1 - alpha[..., None]) + target * alpha[..., None]
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def _synthetic_sequence(n: int, w: int, h: int, seed: int):
    rng = np.random.default_rng(seed)
    bg = _background(rng, h, w)
    cx, cy, r = _trajectory(n, w, h, rng)
    frames = [_draw_target(bg, cx[i], cy[i], r[i], rng) for i in range(n)]
    return frames, cx, cy, r


def _video_frames(path: str, limit: int):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.astype(np.float32))
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {path}")
    return frames


def _bbox_px(cx: float, cy: float, r: float, w: int, h: int) -> BoundingBox:
    box = PixelBox(x=cx - r, y=cy - r, w=2 * r, h=2 * r)
    return BoundingBox.from_pixels(box, w, h)


# ── stats helpers ────────────────────────────────────────────────────────────

def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def _summarize(name: str, ms) -> str:
    a = np.asarray(ms, np.float64)
    if a.size == 0:
        return f"  {name:<28} (no samples)"
    return (f"  {name:<28} n={a.size:<5d} "
            f"p50={_pct(a,50):6.3f}  p95={_pct(a,95):6.3f}  "
            f"max={a.max():6.3f}  mean={a.mean():6.3f}  (ms)")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _softmax_fg(cls: np.ndarray) -> np.ndarray:
    c = np.asarray(cls, np.float32).reshape(2, -1)
    c = c - c.max(axis=0, keepdims=True)
    e = np.exp(c)
    return (e / e.sum(axis=0, keepdims=True))[1]


def _iou_px(a: BoundingBox, b: BoundingBox, w: int, h: int) -> float:
    pa, pb = a.to_pixels(w, h), b.to_pixels(w, h)
    ax1, ay1, ax2, ay2 = pa.x, pa.y, pa.x + pa.w, pa.y + pa.h
    bx1, by1, bx2, by2 = pb.x, pb.y, pb.x + pb.w, pb.y + pb.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pa.w * pa.h + pb.w * pb.h - inter
    return inter / union if union > 0 else float("nan")


# ── parity mode ──────────────────────────────────────────────────────────────

def run_parity(args) -> int:
    model_dir = Path(args.model_dir)
    overrides = {"backbone_fp32": args.backbone_fp32, "head_fp32": args.head_fp32,
                 "backbone_int8": args.backbone_int8, "head_fp16": args.head_fp16}
    paths = _resolve_paths(model_dir, overrides)
    for k, p in paths.items():
        print(f"  {k:<14} {p}  ({p.stat().st_size / 1024:.0f} KiB)")

    bb32 = _load_onnx(paths["backbone_fp32"], args.threads)
    hd32 = _load_onnx(paths["head_fp32"], args.threads)
    bb8 = _load_onnx(paths["backbone_int8"], args.threads)
    hd16 = _load_onnx(paths["head_fp16"], args.threads)

    frames, cx, cy, r = _synthetic_sequence(args.frames, args.width, args.height, args.seed)
    exemplar_size, search_size, model_input, context = 127, 255, 255, 0.5

    def _search_crop(frame, c, radius):
        # square target (w=h=2*radius); mirrors NanoTrack._exemplar_side + the
        # s_x = s_z * search/exemplar relation used in init()/update().
        side = 2 * radius
        pad = context * 2 * side
        s_z = np.sqrt((side + pad) * (side + pad))
        s_x = s_z * search_size / exemplar_size
        patch, _ = crop_with_context(frame, c, (s_x, s_x), (model_input, model_input))
        return patch

    spec_in = bb32.io_spec.inputs[0]
    # exemplar feature, computed once from frame 0 at the initial box (both variants).
    z_patch = _search_crop(frames[0], (cx[0], cy[0]), r[0])
    z_in = to_input(z_patch, spec_in, color="rgb", scale=1.0)
    z32_full = np.asarray(bb32.infer({spec_in.name: z_in})["output"], np.float32)
    z8_full = np.asarray(bb8.infer({spec_in.name: z_in})["output"], np.float32)
    z32, z8 = z32_full[:, :, 4:12, 4:12], z8_full[:, :, 4:12, 4:12]

    sample_idx = np.linspace(0, args.frames - 1, num=min(args.parity_samples, args.frames),
                             dtype=int)
    feat_corr, feat_maxdiff = [], []
    cls_iso_corr, loc_iso_corr = [], []
    cls_e2e_corr, loc_e2e_corr = [], []
    hd32_in = hd32.io_spec.inputs
    hd16_in = hd16.io_spec.inputs

    for i in sample_idx:
        patch = _search_crop(frames[i], (cx[i], cy[i]), r[i])
        xf = to_input(patch, spec_in, color="rgb", scale=1.0)
        x32 = np.asarray(bb32.infer({spec_in.name: xf})["output"], np.float32)
        x8 = np.asarray(bb8.infer({spec_in.name: xf})["output"], np.float32)
        feat_corr.append(_corr(x32, x8))
        feat_maxdiff.append(float(np.abs(x32 - x8).max()))

        out32 = hd32.infer({hd32_in[0].name: z32, hd32_in[1].name: x32})
        out_iso = hd16.infer({hd16_in[0].name: z32, hd16_in[1].name: x32})  # fp32 feats -> fp16 head
        out_e2e = hd16.infer({hd16_in[0].name: z8, hd16_in[1].name: x8})    # int8 feats -> fp16 head

        cls_iso_corr.append(_corr(_softmax_fg(out32["output1"]), _softmax_fg(out_iso["output1"])))
        loc_iso_corr.append(_corr(out32["output2"], out_iso["output2"]))
        cls_e2e_corr.append(_corr(_softmax_fg(out32["output1"]), _softmax_fg(out_e2e["output1"])))
        loc_e2e_corr.append(_corr(out32["output2"], out_e2e["output2"]))

    print(f"\n=== PARITY: {len(sample_idx)} sampled frames, same crop into both variants ===")
    print(f"  backbone feature corr (int8 vs fp32):     "
          f"mean={np.nanmean(feat_corr):.4f}  min={np.nanmin(feat_corr):.4f}  "
          f"max|diff|={max(feat_maxdiff):.3f}")
    print(f"  head cls corr, isolated (fp32 feats):     mean={np.nanmean(cls_iso_corr):.4f}  "
          f"min={np.nanmin(cls_iso_corr):.4f}")
    print(f"  head loc corr, isolated (fp32 feats):     mean={np.nanmean(loc_iso_corr):.4f}  "
          f"min={np.nanmin(loc_iso_corr):.4f}")
    print(f"  cls corr, end-to-end (int8 feats->fp16):  mean={np.nanmean(cls_e2e_corr):.4f}  "
          f"min={np.nanmin(cls_e2e_corr):.4f}")
    print(f"  loc corr, end-to-end (int8 feats->fp16):  mean={np.nanmean(loc_e2e_corr):.4f}  "
          f"min={np.nanmin(loc_e2e_corr):.4f}")

    for m in (bb32, hd32, bb8, hd16):
        m.close()
    return 0


# ── track mode ────────────────────────────────────────────────────────────────

def _build_tracker(backbone: Model, head: Model) -> NanoTrack:
    return NanoTrack(manifest=None, backend="onnx", backbone=backbone, head=head)


def run_track(args) -> int:
    model_dir = Path(args.model_dir)
    overrides = {"backbone_fp32": args.backbone_fp32, "head_fp32": args.head_fp32,
                 "backbone_int8": args.backbone_int8, "head_fp16": args.head_fp16}
    paths = _resolve_paths(model_dir, overrides)

    tbb32 = TimedModel(_load_onnx(paths["backbone_fp32"], args.threads))
    thd32 = TimedModel(_load_onnx(paths["head_fp32"], args.threads))
    tbb8 = TimedModel(_load_onnx(paths["backbone_int8"], args.threads))
    thd16 = TimedModel(_load_onnx(paths["head_fp16"], args.threads))

    if args.video:
        frames = _video_frames(args.video, args.frames)
        h, w = frames[0].shape[:2]
        cx = cy = r = None
    else:
        frames, cx, cy, r = _synthetic_sequence(args.frames, args.width, args.height, args.seed)
        h, w = args.height, args.width
    n = len(frames)

    init_box = (_bbox_px(cx[0], cy[0], r[0], w, h) if cx is not None
                else BoundingBox(0.375, 0.375, 0.25, 0.25))

    fp32 = _build_tracker(tbb32, thd32)
    quant = _build_tracker(tbb8, thd16)
    fp32.init(frames[0], init_box)
    quant.init(frames[0], init_box)

    rows = []
    for i in range(1, n):
        t0 = time.monotonic_ns()
        out32 = fp32.update(frames[i])
        t1 = time.monotonic_ns()
        outq = quant.update(frames[i])
        t2 = time.monotonic_ns()

        gt = _bbox_px(cx[i], cy[i], r[i], w, h) if cx is not None else None
        row = {
            "frame": i,
            "fp32_x": out32.bbox.x, "fp32_y": out32.bbox.y,
            "fp32_w": out32.bbox.w, "fp32_h": out32.bbox.h,
            "fp32_conf": out32.confidence, "fp32_status": out32.status.name,
            "fp32_total_ms": (t1 - t0) / 1e6,
            "quant_x": outq.bbox.x, "quant_y": outq.bbox.y,
            "quant_w": outq.bbox.w, "quant_h": outq.bbox.h,
            "quant_conf": outq.confidence, "quant_status": outq.status.name,
            "quant_total_ms": (t2 - t1) / 1e6,
            "iou_variants": _iou_px(out32.bbox, outq.bbox, w, h),
        }
        if gt is not None:
            row["iou_fp32_gt"] = _iou_px(out32.bbox, gt, w, h)
            row["iou_quant_gt"] = _iou_px(outq.bbox, gt, w, h)
        rows.append(row)

    fp32.close()
    quant.close()

    print(f"\n=== TRACK: {n} frames, {'video ' + args.video if args.video else 'synthetic'} "
          f"({w}x{h}), intra_op_num_threads={args.threads} ===")
    print(" latency (excludes each session's first Run(), which pays one-time arena setup):")
    print(_summarize("fp32 backbone", tbb32.times_ms[1:]))
    print(_summarize("fp32 head", thd32.times_ms[1:]))
    print(_summarize("fp32 total/frame", [r_["fp32_total_ms"] for r_ in rows]))
    print(_summarize("int8 backbone", tbb8.times_ms[1:]))
    print(_summarize("fp16 head", thd16.times_ms[1:]))
    print(_summarize("quant total/frame", [r_["quant_total_ms"] for r_ in rows]))
    fp32_mean = np.mean([r_["fp32_total_ms"] for r_ in rows])
    quant_mean = np.mean([r_["quant_total_ms"] for r_ in rows])
    print(f"  speedup (fp32/quant): {fp32_mean / quant_mean:.2f}x")

    lock32 = np.mean([r_["fp32_status"] == "LOCKED" for r_ in rows])
    lockq = np.mean([r_["quant_status"] == "LOCKED" for r_ in rows])
    print(f"\n locked fraction: fp32={lock32:.1%}  quant={lockq:.1%}")
    print(f" bbox IoU between variants: mean={np.mean([r_['iou_variants'] for r_ in rows]):.3f}  "
          f"min={np.min([r_['iou_variants'] for r_ in rows]):.3f}")
    if cx is not None:
        print(f" bbox IoU vs ground truth: fp32 mean={np.mean([r_['iou_fp32_gt'] for r_ in rows]):.3f}  "
              f"quant mean={np.mean([r_['iou_quant_gt'] for r_ in rows]):.3f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w_.writeheader()
            w_.writerows(rows)
        print(f"\n csv -> {args.csv}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(11, 9))
        ax[0].hist(tbb32.times_ms[1:], bins=40, alpha=0.6, label="fp32 backbone")
        ax[0].hist(tbb8.times_ms[1:], bins=40, alpha=0.6, label="int8 backbone")
        ax[0].set_title("backbone latency")
        ax[0].set_xlabel("ms")
        ax[0].legend()

        ax[1].hist(thd32.times_ms[1:], bins=40, alpha=0.6, label="fp32 head")
        ax[1].hist(thd16.times_ms[1:], bins=40, alpha=0.6, label="fp16 head")
        ax[1].set_title("head latency")
        ax[1].set_xlabel("ms")
        ax[1].legend()

        frame_ids = [r_["frame"] for r_ in rows]
        ax[2].plot(frame_ids, [r_["fp32_conf"] for r_ in rows], label="fp32 confidence")
        ax[2].plot(frame_ids, [r_["quant_conf"] for r_ in rows], label="quant confidence")
        if cx is not None:
            ax[2].plot(frame_ids, [r_["iou_fp32_gt"] for r_ in rows], "--", alpha=0.6,
                      label="fp32 IoU vs gt")
            ax[2].plot(frame_ids, [r_["iou_quant_gt"] for r_ in rows], "--", alpha=0.6,
                      label="quant IoU vs gt")
        ax[2].set_title("confidence / accuracy over the sequence")
        ax[2].set_xlabel("frame")
        ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(args.out, dpi=110)
        print(f" plot -> {args.out}")

    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-dir", default="models",
                   help="dir holding the four vendored .onnx files (default: models)")
    p.add_argument("--backbone-fp32", default=None)
    p.add_argument("--head-fp32", default=None)
    p.add_argument("--backbone-int8", default=None)
    p.add_argument("--head-fp16", default=None)
    p.add_argument("--threads", type=int, default=1,
                   help="onnxruntime intra_op_num_threads (1 matches the rpi4b's "
                        "single-core-pinned tracker deployment)")


def _add_seq_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--seed", type=int, default=0)


def main() -> int:
    p = argparse.ArgumentParser(description="NanoTrack ONNX precision benchmark")
    sub = p.add_subparsers(dest="mode", required=True)

    par = sub.add_parser("parity", help="numerical parity: same crop, both variants")
    _add_model_args(par)
    _add_seq_args(par)
    par.add_argument("--parity-samples", type=int, default=8,
                     help="number of frames sampled across the sequence for parity checks")
    par.set_defaults(func=run_parity)

    trk = sub.add_parser("track", help="full tracker: latency + drift over a sequence")
    _add_model_args(trk)
    _add_seq_args(trk)
    trk.add_argument("--video", default=None, help="bench on real footage instead of synthetic")
    trk.add_argument("--csv", default=None, help="write per-frame comparison rows here")
    trk.add_argument("--out", default="/tmp/quadguide_nanotrack_precision.png")
    trk.add_argument("--no-plot", action="store_true")
    trk.set_defaults(func=run_track)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

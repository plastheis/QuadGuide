"""Adapter exposing EdgeCV trackers through QuadGuide's tracker protocol.

EdgeCV (``~/EdgeCV``) is a generic single-object-tracking library: the caller
owns frames and ``tracker.update(frame)`` returns an EdgeCV ``TrackResult``
(``bbox: BoundingBox | None``, ``confidence: float | None``, ``status:
TrackStatus``). QuadGuide's ``tracker_worker`` instead expects the structural
protocol in ARCHITECTURE.md §6.3 — ``name``/``init``/``update``/``reset``/
``close`` where ``update()`` returns an object with ``.bbox.{x,y,w,h}`` (0–1),
``.confidence`` ∈ [0,1] and ``.health`` ∈
{"nominal","uncertain","lost","no_lock"}.

This module is the impedance match between the two — exactly analogous to the
built-in :class:`OpenCVTrackerAdapter` for ``cv2``. It keeps EdgeCV free of any
QuadGuide concern (and vice-versa: QuadGuide never imports a vendor runtime):

* **All EdgeCV imports are lazy** (inside ``__init__``). ``run.py`` forks the
  tracker worker and ``load_tracker`` runs in the *child*, so the RKNN context
  is created in the process that uses it — never in the parent before fork
  (EdgeCV ARCHITECTURE §7.4 / §14.7).
* **Frames are converted BGR→RGB.** QuadGuide cameras emit BGR; EdgeCV trackers
  expect RGB (MOSSE uses luma weights, NanoTrack's manifest declares ``color:
  rgb``). Pass ``color: rgb`` in params to skip the swap.
* **Confidence is normalised to 0–1.** EdgeCV CF trackers report raw PSR
  (≈2–50); the tracker's own ``default_calibrator`` maps it to [0,1] for the
  HUD. NN scores are already 0–1 and just clamped.

Wire it via config (``module:Class`` resolved by ``load_tracker``)::

    tracker:
      import: quadguide.perception.edgecv_adapter:EdgeCVTracker
      params:
        tracker: nanotrack          # mosse | nanotrack | siamfc | yolo
        backend: rknn               # auto | onnx | rknn | mock
        model_dir: /home/radxa/EdgeCV/models   # sets EDGECV_MODEL_DIR
        # remaining keys forwarded to the EdgeCV tracker constructor, e.g.
        # score_lock: 0.6
"""
from __future__ import annotations

import os
from collections import namedtuple

import numpy as np

__all__ = ["EdgeCVTracker"]

# Structural output types the worker reads (same shape OpenCVTrackerAdapter uses).
_TrackerOutput = namedtuple("_TrackerOutput", "bbox confidence health")
_BBox = namedtuple("_BBox", "x y w h")

_NO_LOCK = _TrackerOutput(_BBox(0.0, 0.0, 0.0, 0.0), 0.0, "no_lock")
_LOST = _TrackerOutput(_BBox(0.0, 0.0, 0.0, 0.0), 0.0, "lost")

# EdgeCV TrackStatus.name → QuadGuide TrackerHealth string.
_HEALTH_BY_STATUS = {
    "LOCKED": "nominal",
    "COASTING": "uncertain",
    "INITIALIZING": "uncertain",
    "LOST": "lost",
}

# Short name → manifest file for single-model NN trackers (under EdgeCV's
# packaged manifests dir). MOSSE needs no model; NanoTrack loads two.
_NN_MANIFESTS = {
    "siamfc": "siamfc_generic.yaml",
    "yolo": "yolo26n.yaml",
}


class EdgeCVTracker:
    """Wrap an EdgeCV tracker in QuadGuide's structural tracker protocol."""

    def __init__(
        self,
        *,
        tracker: str = "mosse",
        backend: str = "auto",
        model_dir: str | None = None,
        color: str = "bgr",
        **tracker_params: object,
    ) -> None:
        if color not in ("bgr", "rgb"):
            raise ValueError(f"color must be 'bgr' or 'rgb', got {color!r}")
        self._color = color

        # Point EdgeCV's artifact resolver at the model blobs before any load.
        if model_dir is not None:
            os.environ["EDGECV_MODEL_DIR"] = str(model_dir)

        self._tracker = self._build(tracker.lower(), backend, tracker_params)
        # CF trackers carry a default calibrator (raw PSR → 0–1); NN trackers
        # generally do not (scores already 0–1).
        self._calibrator = getattr(self._tracker, "default_calibrator", None)
        self._initialized = False

    # ── construction ──────────────────────────────────────────────────────
    @staticmethod
    def _manifests_dir():
        import edgecv
        from pathlib import Path

        return Path(edgecv.__file__).resolve().parent / "models" / "manifests"

    def _build(self, name: str, backend: str, params: dict):
        """Construct the EdgeCV tracker selected by ``name``.

        Mirrors EdgeCV's ``tools/track_webcam.py:build_tracker`` but is
        device-oriented: ``backend='auto'`` prefers RKNN on the NPU.
        """
        if name == "mosse":
            from edgecv.trackers.cf import Mosse

            return Mosse(**params)

        from edgecv.models.manifest import load_manifest
        from edgecv.trackers.nn.base import select_backend

        resolved = select_backend(backend)

        if name == "nanotrack":
            from edgecv.trackers.nn import NanoTrack

            manifest = load_manifest(self._manifests_dir() / "nanotrack.yaml")
            return NanoTrack.from_manifest(manifest, backend=resolved, **params)

        if name in _NN_MANIFESTS:
            from edgecv.trackers.nn import SiamFC, YoloTracker

            cls = {"siamfc": SiamFC, "yolo": YoloTracker}[name]
            manifest = load_manifest(self._manifests_dir() / _NN_MANIFESTS[name])
            return cls(manifest, backend=resolved, **params)

        raise ValueError(
            f"unknown EdgeCV tracker {name!r}; expected one of "
            f"mosse, nanotrack, {', '.join(_NN_MANIFESTS)}"
        )

    # ── frame colour ──────────────────────────────────────────────────────
    def _rgb(self, frame: np.ndarray) -> np.ndarray:
        if self._color == "bgr" and frame.ndim == 3 and frame.shape[2] == 3:
            return np.ascontiguousarray(frame[..., ::-1])
        return frame

    # ── protocol ──────────────────────────────────────────────────────────
    def name(self) -> str:
        return self._tracker.name()

    def init(self, frame, bbox) -> None:
        from edgecv.core.bbox import BoundingBox

        eb = BoundingBox(
            x=float(bbox.x),
            y=float(bbox.y),
            w=max(0.0, float(bbox.w)),
            h=max(0.0, float(bbox.h)),
        )
        self._tracker.init(self._rgb(frame), eb)
        self._initialized = True

    def update(self, frame):
        if not self._initialized:
            return _NO_LOCK
        res = self._tracker.update(self._rgb(frame))
        if res.bbox is None:
            return _LOST
        b = res.bbox
        health = _HEALTH_BY_STATUS.get(res.status.name, "uncertain")
        return _TrackerOutput(
            _BBox(b.x, b.y, max(0.0, b.w), max(0.0, b.h)),
            self._normalize_confidence(res.confidence),
            health,
        )

    def reset(self) -> None:
        # Soft clear: next update() reports no_lock until re-init. No hardware
        # release — EdgeCV trackers re-init cleanly via init().
        self._initialized = False

    def close(self) -> None:
        self._tracker.close()

    # ── helpers ───────────────────────────────────────────────────────────
    def _normalize_confidence(self, raw) -> float:
        if raw is None:
            return 0.0
        if self._calibrator is not None:
            return float(self._calibrator.calibrate(float(raw)))
        return float(max(0.0, min(1.0, float(raw))))

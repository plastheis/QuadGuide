"""Deterministic stub Models for NN-tracker tests (no weights, no backend)."""
from __future__ import annotations

import numpy as np

from edgecv.backends.base import IOSpec, Model, TensorSpec


class ScriptedModel(Model):
    """Returns pre-set output arrays per infer() call, cycling through `outputs`.

    `outputs` is a list of dicts {output_name: ndarray}. io_spec is supplied so the
    tracker can read names/shapes. infer() ignores its inputs (geometry is driven
    entirely by the scripted outputs)."""

    def __init__(self, io_spec: IOSpec, outputs: list[dict[str, np.ndarray]]):
        self._io_spec = io_spec
        self._outputs = outputs
        self.calls = 0
        self.closed = False

    @property
    def io_spec(self) -> IOSpec:
        return self._io_spec

    def infer(self, inputs: dict[str, np.ndarray]):
        out = self._outputs[self.calls % len(self._outputs)]
        self.calls += 1
        return out

    def close(self) -> None:
        self.closed = True


def siam_io(score_size: int = 17) -> IOSpec:
    return IOSpec(
        inputs=(TensorSpec("exemplar", (1, 3, 127, 127), "float32"),
                TensorSpec("search", (1, 3, 255, 255), "float32")),
        outputs=(TensorSpec("score_map", (1, 1, score_size, score_size), "float32"),))


def score_map_peaked(score_size: int, cy: int, cx: int, peak: float = 1.0) -> np.ndarray:
    m = np.zeros((1, 1, score_size, score_size), np.float32)
    m[0, 0, cy, cx] = peak
    return m


def nano_io(score_size: int = 15) -> IOSpec:
    return IOSpec(
        inputs=(TensorSpec("exemplar", (1, 3, 127, 127), "float32"),
                TensorSpec("search", (1, 3, 255, 255), "float32")),
        outputs=(TensorSpec("cls", (1, 2, score_size, score_size), "float32"),
                 TensorSpec("loc", (1, 4, score_size, score_size), "float32")))


def nano_backbone_io() -> IOSpec:
    """Split backbone: 255x255 input -> 96ch 16x16 features."""
    return IOSpec(
        inputs=(TensorSpec("input", (1, 3, 255, 255), "float32"),),
        outputs=(TensorSpec("output", (1, 96, 16, 16), "float32"),))


def nano_head_io(score_size: int = 15) -> IOSpec:
    """Split head: z_f (1,96,8,8) + x_f (1,96,16,16) -> cls, loc."""
    return IOSpec(
        inputs=(TensorSpec("input1", (1, 96, 8, 8), "float32"),
                TensorSpec("input2", (1, 96, 16, 16), "float32")),
        outputs=(TensorSpec("output1", (1, 2, score_size, score_size), "float32"),
                 TensorSpec("output2", (1, 4, score_size, score_size), "float32")))


def nano_z_feat() -> np.ndarray:
    """Dummy backbone output: (1, 96, 16, 16) feature for z_f centre-crop."""
    return np.zeros((1, 96, 16, 16), np.float32)


def nano_head_out(score_size: int, cy: int, cx: int,
                  left: float = 8.0, top: float = 8.0,
                  right: float = 8.0, bottom: float = 8.0,
                  fg: float = 8.0) -> dict[str, np.ndarray]:
    """Single head output dict with cls/loc for NanoTrack tests."""
    return {"output1": cls_peaked(score_size, cy, cx, fg),
            "output2": loc_const(score_size, left, top, right, bottom)}


def cls_peaked(score_size: int, cy: int, cx: int, fg: float = 8.0) -> np.ndarray:
    """cls logits (1,2,S,S): bg channel 0, fg channel 1 with a high logit at (cy,cx)
    so softmax fg prob ≈ 1 there and 0.5 elsewhere."""
    m = np.zeros((1, 2, score_size, score_size), np.float32)
    m[0, 1, cy, cx] = fg
    return m


def loc_const(score_size: int, left: float, top: float,
              right: float, bottom: float) -> np.ndarray:
    """loc (1,4,S,S) with constant l,t,r,b distances at every location."""
    m = np.zeros((1, 4, score_size, score_size), np.float32)
    m[0, 0] = left
    m[0, 1] = top
    m[0, 2] = right
    m[0, 3] = bottom
    return m

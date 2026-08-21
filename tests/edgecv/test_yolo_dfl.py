"""Decode for the rknn_model_zoo-style separated YOLO head (DFL box + cls per scale).

The user's yolo11n_p2p3p4 rknn emits 9 outputs (3 scales × {box (1,64,H,W),
cls (1,1,H,W), score-sum (1,1,H,W)}). decode_yolo_dfl turns those into xyxy
pixel boxes + scores for NMS.
"""

from __future__ import annotations

import numpy as np

from edgecv.trackers.nn.preprocess import decode_yolo_dfl


def _scale_outputs(H, W, reg_max=16, nc=1):
    reg = np.zeros((1, 4 * reg_max, H, W), np.float32)
    cls = np.zeros((1, nc, H, W), np.float32)
    sumc = np.zeros((1, 1, H, W), np.float32)
    return reg, cls, sumc


def test_single_scale_one_box_decodes_to_known_xyxy():
    H = W = 2
    stride = 8
    reg_max = 16
    reg, cls, sumc = _scale_outputs(H, W, reg_max)
    # Target cell (row=0, col=1): high score, DFL one-hot at bin=2 for all 4 sides.
    cls[0, 0, 0, 1] = 0.9
    for side in range(4):
        reg[0, side * reg_max + 2, 0, 1] = 50.0   # softmax → ~1 at bin 2 → dist≈2
    xyxy, score = decode_yolo_dfl([reg, cls, sumc], strides=[stride],
                                  reg_max=reg_max, conf_thresh=0.25)
    assert xyxy.shape == (1, 4)
    assert score[0] == np.float32(0.9)
    # anchor centre (col+0.5, row+0.5)=(1.5,0.5); dist=2 each side.
    ax, ay, d = 1.5, 0.5, 2.0
    expected = np.array([(ax - d) * stride, (ay - d) * stride,
                         (ax + d) * stride, (ay + d) * stride], np.float32)
    assert np.allclose(xyxy[0], expected, atol=1e-3)


def test_threshold_filters_low_scores():
    H = W = 4
    reg, cls, sumc = _scale_outputs(H, W)
    cls[0, 0] = 0.1  # all below default conf
    xyxy, score = decode_yolo_dfl([reg, cls, sumc], strides=[8], conf_thresh=0.25)
    assert xyxy.shape == (0, 4)
    assert score.shape == (0,)


def test_three_scales_concatenate():
    outs = []
    strides = [4, 8, 16]
    for hw in (8, 4, 2):
        reg, cls, sumc = _scale_outputs(hw, hw)
        cls[0, 0, 0, 0] = 0.8   # one detection per scale
        outs += [reg, cls, sumc]
    xyxy, score = decode_yolo_dfl(outs, strides=strides, conf_thresh=0.25)
    assert xyxy.shape == (3, 4)
    assert np.allclose(score, 0.8)


def test_handles_two_outputs_per_scale():
    """Some exports omit the score-sum tensor (2 per scale)."""
    H = W = 2
    reg, cls, _ = _scale_outputs(H, W)
    cls[0, 0, 0, 0] = 0.7
    xyxy, score = decode_yolo_dfl([reg, cls], strides=[8], conf_thresh=0.25)
    assert xyxy.shape == (1, 4)
    assert score[0] == np.float32(0.7)

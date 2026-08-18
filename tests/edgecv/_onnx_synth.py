"""Synthetic ONNX models for the onnx-backend integration test. No trained weights:
the graphs are shape-correct and consume their inputs so ORT is happy, but their
outputs are (deliberately) not meaningful detections/score-maps."""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_siamfc_onnx(path: str, score_size: int = 17) -> None:
    ex = helper.make_tensor_value_info("exemplar", TensorProto.FLOAT, [1, 3, 127, 127])
    se = helper.make_tensor_value_info("search", TensorProto.FLOAT, [1, 3, 255, 255])
    sc = helper.make_tensor_value_info(
        "score_map", TensorProto.FLOAT, [1, 1, score_size, score_size]
    )
    # AveragePool(255, k=15, s=15) -> [1,3,17,17]; mean over channels -> [1,1,17,17];
    # add scalar mean(exemplar) so both inputs are consumed.
    pool = helper.make_node("AveragePool", ["search"], ["pooled"],
                            kernel_shape=[15, 15], strides=[15, 15])
    redc = helper.make_node("ReduceMean", ["pooled"], ["pooled_c"],
                            axes=[1], keepdims=1)            # [1,1,17,17]
    rm = helper.make_node("ReduceMean", ["exemplar"], ["ex_mean"], keepdims=0)  # scalar
    add = helper.make_node("Add", ["pooled_c", "ex_mean"], ["score_map"])
    graph = helper.make_graph([pool, redc, rm, add], "siamfc_stub", [ex, se], [sc])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_yolo_onnx_v8(path: str, n: int = 64, num: int = 3, nc: int = 1) -> None:
    """YOLO26 one-to-many layout: channels-first (1, 4+nc, num), NO objectness.
    Box in channels [:4], class scores in channels [4:]. Shape-correct, not trained."""
    img = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, n, n])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + nc, num])
    dets = np.zeros((1, 4 + nc, num), np.float32)
    dets[0, :4, 0] = [n / 2, n / 2, 16, 16]   # centred
    dets[0, 4, 0] = 0.9
    dets[0, :4, 1] = [4, 4, 8, 8]             # corner
    dets[0, 4, 1] = 0.95                       # third column stays zero -> thresholded out
    const = helper.make_node("Constant", [], ["dets_out"],
                             value=numpy_helper.from_array(dets, "dets"))
    zinit = numpy_helper.from_array(np.array(0.0, np.float32), "zero_scalar")
    rm = helper.make_node("ReduceMean", ["images"], ["img_mean"], keepdims=0)  # scalar
    mul = helper.make_node("Mul", ["img_mean", "zero_scalar"], ["zeroed"])
    # consumes images, value unchanged
    add = helper.make_node("Add", ["dets_out", "zeroed"], ["output0"])
    graph = helper.make_graph([const, rm, mul, add], "yolo_v8_stub", [img], [out],
                              initializer=[zinit])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_yolo_onnx(path: str, n: int = 64, num: int = 3, nc: int = 1) -> None:
    img = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, n, n])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, num, 5 + nc])
    dets = np.zeros((1, num, 5 + nc), np.float32)
    dets[0, 0] = [n / 2, n / 2, 16, 16, 0.9] + [1.0] * nc   # centred, high score
    dets[0, 1] = [4, 4, 8, 8, 0.95] + [1.0] * nc            # corner, higher score
    const = helper.make_node("Constant", [], ["dets_out"],
                             value=numpy_helper.from_array(dets, "dets"))
    zinit = numpy_helper.from_array(np.array(0.0, np.float32), "zero_scalar")
    rm = helper.make_node("ReduceMean", ["images"], ["img_mean"], keepdims=0)  # scalar
    mul = helper.make_node("Mul", ["img_mean", "zero_scalar"], ["zeroed"])
    # consumes images, value unchanged
    add = helper.make_node("Add", ["dets_out", "zeroed"], ["output0"])
    graph = helper.make_graph([const, rm, mul, add], "yolo_stub", [img], [out],
                              initializer=[zinit])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)

"""YOLO26 conversion adapter (ARCHITECTURE.md §11). Drives the Ultralytics exporter
to a ONE-TO-MANY (NMS-required) ONNX graph; the generic ONNX->RKNN step then handles
INT8. Unlike torch adapters (e.g. siamfc), Ultralytics owns the architecture and the
exporter, so this registers an `export` hook instead of `build` — convert_lib.run()
branches on it and skips the torch parity harness. See design spec §7.

The one-to-many head (`nms=False`) is mandatory: the NMS-free end-to-end head is
unavailable on RKNN, and we need onnx (x86) and rknn (device) to emit the same
(1, 4+nc, N) tensor for the `yolov8` decoder (design spec §3)."""

from __future__ import annotations

import shutil
from pathlib import Path

from convert_lib.registry import Adapter, register


def _export(checkpoint: str, onnx_out: str, manifest) -> str:
    from ultralytics import YOLO  # lazy: host-only [dev] dep, only needed at convert time

    imgsz = int(manifest.inputs[0]["shape"][-1])
    produced = YOLO(checkpoint).export(format="onnx", nms=False, imgsz=imgsz, opset=13)
    Path(onnx_out).parent.mkdir(parents=True, exist_ok=True)
    if str(produced) != str(onnx_out):
        shutil.move(str(produced), str(onnx_out))
    return onnx_out


register(Adapter(name="yolo26n", export=_export))
register(Adapter(name="yolo26s", export=_export))

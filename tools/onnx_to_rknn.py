"""ONNX -> RKNN CLI (host-only). Thin wrapper over convert_lib.rknn for converting an
ONNX produced elsewhere (e.g. an ultralytics export). See tools/CONVERSION.md.

Usage (on a host with rknn-toolkit2):
    python tools/onnx_to_rknn.py --onnx models/m.onnx --out models/m.rk3588.rknn \
        --target rk3588 --calibration-dir calib/ --inputs exemplar search
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_lib.rknn import rknn_convert  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX -> RKNN (host-only)")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--calibration-dir", default=None,
                    help="folder of representative images; enables INT8 quantisation")
    ap.add_argument("--inputs", nargs="+", default=["exemplar", "search"],
                    help="model input names (order must match the ONNX graph)")
    args = ap.parse_args()
    rknn_convert(args.onnx, args.out, args.target, args.calibration_dir, args.inputs)


if __name__ == "__main__":
    main()

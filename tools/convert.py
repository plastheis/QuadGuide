"""CLI dispatcher for model conversion (ARCHITECTURE.md §11). Host-only.

Adds tools/ to sys.path so `import convert_lib` resolves when run as a script:
    python tools/convert.py --model siamfc_generic --checkpoint models/siamfc.pth
    python tools/convert.py --model siamfc_generic --checkpoint models/siamfc.pth \\
        --rknn --calib calib/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_lib import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a tracker checkpoint to ONNX (+ optional RKNN)")
    ap.add_argument("--model", required=True, help="manifest model name, e.g. siamfc_generic")
    ap.add_argument("--checkpoint", required=True, help="path to the .pth state_dict")
    ap.add_argument("--out", default=None,
                    help="ONNX output path (default: the manifest's resolved artifact path)")
    ap.add_argument("--rknn", action="store_true", help="also convert the ONNX to RKNN")
    ap.add_argument("--target", default="rk3588", help="RKNN target platform (default: rk3588)")
    ap.add_argument("--calib", default=None, help="calibration image dir for INT8 RKNN")
    args = ap.parse_args()
    run(args.model, args.checkpoint, args.out,
        rknn=args.rknn, target=args.target, calib=args.calib)


if __name__ == "__main__":
    main()

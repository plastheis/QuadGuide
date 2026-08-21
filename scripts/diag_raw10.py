#!/usr/bin/env python3
"""Verify genuine 10-bit raw capture from the OV9281 via Picamera2RawCamera.

    sudo systemctl stop quadguide
    sudo .venv/bin/python scripts/diag_raw10.py --config configs/rpi4b_raw10.yaml
    sudo systemctl start quadguide

Prints per-frame stats and PASS/FAIL on the "genuinely >8-bit" check, and writes
a tonemapped PNG + a raw .npy to /tmp/raw10 for inspection.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from quadguide.core.config import cfg_platform, load_config
from quadguide.ground.overlay import tonemap
from quadguide.perception.camera.sources import Picamera2RawCamera

OUT = "/tmp/raw10"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rpi4b_raw10.yaml")
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pcfg = cfg_platform(load_config(args.config))
    cam = Picamera2RawCamera(pcfg.camera)
    cam.open()
    try:
        last = None
        levels = 0
        for i in range(args.frames):
            frame, _ts = cam.read()
            last = frame
            levels = int(np.unique(frame).size)
            low2 = int(np.count_nonzero(frame & 0x3))   # populated low 2 bits ⇒ true 10-bit
            print(f"frame {i:02d} {frame.shape} dtype={frame.dtype} "
                  f"min={frame.min():4d} mean={frame.mean():7.1f} "
                  f"p95={np.percentile(frame, 95):6.1f} max={frame.max():4d} "
                  f"levels={levels} low2bits={low2}")
    finally:
        cam.close()

    if last is None:
        print("FAIL: no frames captured")
        return 1
    np.save(f"{OUT}/frame.npy", last)
    try:
        import cv2
        cv2.imwrite(f"{OUT}/frame.png", tonemap(last, mode="percentile"))
    except Exception:
        pass

    p95 = float(np.percentile(last, 95))
    genuine_10bit = levels > 256 and last.dtype == np.uint16
    sky_headroom = p95 < 1000                     # below the 10-bit clip (1023)
    print(f"\nlevels={levels} (>256 ⇒ true >8-bit), dtype={last.dtype}, "
          f"sky p95={p95:.0f} (<1023 clip)")
    print("PASS" if (genuine_10bit and sky_headroom) else
          "FAIL: not genuine 10-bit or sky clipped")
    return 0 if (genuine_10bit and sky_headroom) else 1


if __name__ == "__main__":
    raise SystemExit(main())

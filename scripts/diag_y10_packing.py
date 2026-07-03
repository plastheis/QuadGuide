#!/usr/bin/env python3
"""Diagnose the real RAW10/Y10 byte packing the rkcif node delivers for the OV9281.

Run with the camera FREE (stop quadguide first):

    sudo systemctl stop quadguide
    sudo .venv/bin/python scripts/diag_y10_packing.py
    sudo systemctl start quadguide

It captures one raw frame, prints the per-position byte statistics of each
5-byte group (the LSB-pack byte has a distinctly low mean), and writes a PNG for
each candidate unpacking so we can see which one removes the vertical striping.
"""
import subprocess, sys, os
import numpy as np

DEV = "/dev/video0"
W, H = 1280, 800
OUT = "/tmp/y10diag"
os.makedirs(OUT, exist_ok=True)
RAW = os.path.join(OUT, "raw.bin")

# 1. capture one raw frame
r = subprocess.run(
    ["v4l2-ctl", "-d", DEV,
     "--set-fmt-video=width=%d,height=%d,pixelformat=Y10 " % (W, H),
     "--stream-mmap", "--stream-count=1", "--stream-to=" + RAW],
    capture_output=True, text=True,
)
sys.stderr.write(r.stderr)
size = os.path.getsize(RAW)
stride = size // H
print(f"captured {size} bytes  stride={stride}  ({stride/W:.3f} B/px)")

buf = np.fromfile(RAW, dtype=np.uint8)[:stride * H].reshape(H, stride)
packed = W * 5 // 4                      # 1600 real data bytes per row
g = buf[:, :packed].reshape(H, -1, 5).astype(np.uint16)   # (H, W/4, 5)

# 2. per-position stats: the LSB-pack byte stands out (low mean, full 0..255 only
#    on MSB bytes). Whichever position has the lowest mean is the LSB-pack byte.
print("\nper-group-position byte means (LSB-pack byte = the odd one out):")
means = [float(g[..., k].mean()) for k in range(5)]
for k, m in enumerate(means):
    print(f"  pos {k}: mean={m:7.2f}  std={float(g[...,k].std()):6.2f}")
lsb_pos = int(np.argmin(means))
print(f"  -> likely LSB-pack byte at position {lsb_pos}")

# 3. render every candidate: drop one position, keep the other 4 as MSB columns.
try:
    import cv2
    have_cv2 = True
except Exception:
    have_cv2 = False

for drop in range(5):
    keep = [k for k in range(5) if k != drop]
    cols = np.stack([g[..., k] for k in keep], axis=-1).reshape(H, W).astype(np.uint8)
    p = os.path.join(OUT, f"cand_drop{drop}.png")
    if have_cv2:
        cv2.imwrite(p, cols)
    else:
        cols.astype(np.uint8).tofile(p + ".gray")
    tag = "  <-- current code assumes drop=4" if drop == 4 else ""
    tag += "  <-- stats say try this" if drop == lsb_pos else ""
    print(f"  wrote {p}{tag}")

print("\nLook at the cand_drop*.png — the correct one has NO vertical striping.")

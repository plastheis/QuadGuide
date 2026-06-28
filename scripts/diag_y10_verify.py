#!/usr/bin/env python3
"""Verify 640x400 native capture (FoV + brightness) and pick gain/exposure.

    sudo systemctl stop quadguide
    sudo .venv/bin/python scripts/diag_y10_verify.py
    sudo systemctl start quadguide

Writes PNGs to /tmp/y10verify for inspection.
"""
import os, re, subprocess, sys
import numpy as np
sys.path.insert(0, "src")
from quadguide.perception.camera.sources import unpack_raw10_to_gray8

DEV, MEDIA = "/dev/video0", "/dev/media0"
OUT = "/tmp/y10verify"
os.makedirs(OUT, exist_ok=True)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


graph = run(["media-ctl", "-d", MEDIA, "-p"])
ent = re.search(r"entity \d+: (\S[^\n(]*ov9281[^\n(]*) \(", graph).group(1).strip()
subdev = re.search(r"ov9281[^\n(]*\(.*?device node name (/dev/v4l-subdev\d+)",
                   graph, re.S).group(1)


def capture(w, h, gain, exp, tag):
    run(["media-ctl", "-d", MEDIA, "--set-v4l2",
         '"%s":0[fmt:Y10_1X10/%dx%d]' % (ent, w, h)])
    run(["v4l2-ctl", "-d", DEV, "--set-fmt-video=width=%d,height=%d,pixelformat=Y10 " % (w, h)])
    run(["v4l2-ctl", "-d", subdev, "-c", "analogue_gain=%d,exposure=%d" % (gain, exp)])
    fmt = run(["v4l2-ctl", "-d", DEV, "--get-fmt-video"])
    stride = int(re.search(r"Bytes per Line\s*:\s*(\d+)", fmt).group(1))
    raw = OUT + f"/{tag}.bin"
    run(["v4l2-ctl", "-d", DEV, "--set-fmt-video=width=%d,height=%d,pixelformat=Y10 " % (w, h),
         "--stream-mmap", "--stream-count=1", "--stream-to=" + raw])
    a = np.fromfile(raw, np.uint8)
    if a.size < stride * h:
        print(f"  {tag}: SHORT READ ({a.size} < {stride*h})"); return
    img = unpack_raw10_to_gray8(a, w, h, stride)
    try:
        import cv2; cv2.imwrite(OUT + f"/{tag}.png", img)
    except Exception:
        pass
    print(f"  {tag:24s} {w}x{h} g={gain:>3} e={exp:>4} stride={stride} "
          f"mean={img.mean():6.1f} p5={np.percentile(img,5):3.0f} p95={np.percentile(img,95):3.0f}")


print("FoV reference (full sensor) vs native 640x400 at the SAME gain/exposure:")
capture(1280, 800, 16, 800, "ref_1280x800_g16_e800")
capture(640, 400, 16, 800, "n640_g16_e800")   # 4x brighter than ref => binned/full-FoV
print("\n640x400 gain/exposure sweep (pick the one with p95 ~210-245, not clipped):")
for g, e in [(32, 600), (48, 800), (64, 800), (48, 1200), (64, 1600)]:
    capture(640, 400, g, e, f"n640_g{g}_e{e}")

run(["v4l2-ctl", "-d", subdev, "-c", "analogue_gain=16,exposure=800"])
print(f"\nimages in {OUT}; sensor restored to g16/e800")

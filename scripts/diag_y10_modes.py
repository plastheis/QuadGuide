#!/usr/bin/env python3
"""Probe OV9281/rkcif capabilities so we can wire resolution, fps and exposure to config.

Run with the camera FREE:

    sudo systemctl stop quadguide
    sudo .venv/bin/python scripts/diag_y10_modes.py
    sudo systemctl start quadguide
"""
import re, subprocess, os
import numpy as np

DEV, MEDIA = "/dev/video0", "/dev/media0"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


# locate the ov9281 sensor subdev node from the media graph
graph = run(["media-ctl", "-d", MEDIA, "-p"])
m = re.search(r"entity \d+: (\S[^\n]*ov9281[^\n(]*)\(.*?device node name (/dev/v4l-subdev\d+)",
              graph, re.S)
ent = m.group(1).strip() if m else "?"
subdev = m.group(2) if m else "?"
print(f"sensor entity : {ent}")
print(f"sensor subdev : {subdev}\n")

# 1. native sensor modes (pad sizes the driver actually offers)
print("=== sensor pad frame sizes (mbus Y10_1X10) ===")
print(run(["v4l2-ctl", "-d", subdev, "--list-subdev-framesizes",
           "pad=0,code=0x200b"]).strip() or "(none reported)")
print("\n=== sensor frame intervals (fps) per size ===")
for w, h in [(1280, 800), (1280, 720), (640, 480), (640, 400), (320, 200)]:
    out = run(["v4l2-ctl", "-d", subdev, "--list-subdev-frameintervals",
               f"pad=0,code=0x200b,width={w},height={h}"]).strip()
    if out and "Invalid" not in out and "error" not in out.lower():
        print(f"  {w}x{h}: {out.splitlines()[-1] if out else ''}")

# 2. can the node capture a requested non-native size? try 640x400 and 640x480
print("\n=== try set-fmt on capture node (what the sensor really gives back) ===")
for w, h in [(640, 400), (640, 480), (1280, 800)]:
    run(["media-ctl", "-d", MEDIA, "--set-v4l2",
         '"%s":0[fmt:Y10_1X10/%dx%d]' % (ent, w, h)])
    run(["v4l2-ctl", "-d", DEV,
         "--set-fmt-video=width=%d,height=%d,pixelformat=Y10 " % (w, h)])
    got = run(["v4l2-ctl", "-d", DEV, "--get-fmt-video"])
    gw = re.search(r"Width/Height\s*:\s*(\d+)/(\d+)", got)
    gs = re.search(r"Bytes per Line\s*:\s*(\d+)", got)
    # does a 1-frame capture actually succeed?
    cap = run(["v4l2-ctl", "-d", DEV,
               "--set-fmt-video=width=%d,height=%d,pixelformat=Y10 " % (w, h),
               "--stream-mmap", "--stream-count=1", "--stream-to=/dev/null"])
    ok = "VIDIOC" not in cap and "error" not in cap.lower()
    print(f"  asked {w}x{h:>4} -> got {gw.group(1) if gw else '?'}x{gw.group(2) if gw else '?'}"
          f"  stride={gs.group(1) if gs else '?'}  stream_ok={ok}")

# 3. fps control via S_PARM on the node
print("\n=== fps via --set-parm on node ===")
for fps in [30, 60, 90]:
    out = run(["v4l2-ctl", "-d", DEV, "--set-parm=%d" % fps])
    got = run(["v4l2-ctl", "-d", DEV, "--get-parm"])
    fm = re.search(r"Frames per second:\s*([\d.]+)", got)
    print(f"  asked {fps:>3} -> {fm.group(1) if fm else out.strip()}")

# 4. exposure/gain effect on brightness (mean of MSB byte 0 of each group)
print("\n=== brightness vs exposure/gain (mean of unpacked MSB) ===")
# restore a known native mode first
run(["media-ctl", "-d", MEDIA, "--set-v4l2", '"%s":0[fmt:Y10_1X10/1280x800]' % ent])
run(["v4l2-ctl", "-d", DEV, "--set-fmt-video=width=1280,height=800,pixelformat=Y10 "])

def mean_brightness(tag):
    p = f"/tmp/y10diag/probe_{tag}.bin"
    os.makedirs("/tmp/y10diag", exist_ok=True)
    run(["v4l2-ctl", "-d", DEV, "--set-fmt-video=width=1280,height=800,pixelformat=Y10 ",
         "--stream-mmap", "--stream-count=1", "--stream-to=" + p])
    a = np.fromfile(p, np.uint8)
    if a.size < 1792 * 800:
        return None
    return float(a[:1792 * 800].reshape(800, 1792)[:, :1600:5].mean())  # byte0 of groups

for gain, exp in [(16, 800), (64, 800), (128, 1600), (248, 2500), (248, 3600)]:
    run(["v4l2-ctl", "-d", subdev, "-c", "analogue_gain=%d,exposure=%d" % (gain, exp)])
    b = mean_brightness(f"g{gain}_e{exp}")
    print(f"  gain={gain:>3} exposure={exp:>4} -> mean MSB byte ~ {b}")

# restore defaults
run(["v4l2-ctl", "-d", subdev, "-c", "analogue_gain=16,exposure=800"])
print("\ndone. (restored gain=16 exposure=800)")

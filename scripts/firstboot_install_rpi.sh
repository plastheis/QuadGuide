#!/usr/bin/env bash
# firstboot_install_rpi.sh — one-shot board-side bootstrap for QuadGuide on a fresh
# Raspberry Pi 4B running Raspberry Pi OS Lite (Bookworm, 64-bit) with a USB UVC
# camera. Run ONCE, at the board's local console, as root, with the Pi on WIRED
# ETHERNET (it needs internet for apt/pip/git).
#
#   sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/plastheis/QuadGuide/main/scripts/firstboot_install_rpi.sh)"
#
# or, if you already cloned the repo:   sudo bash scripts/firstboot_install_rpi.sh
#
# This is the RPi 4B sibling of firstboot_install.sh (which is ROCK 5C only). It is
# idempotent — safe to re-run. It does the parts that CAN be automated:
#   1. apt system deps (incl. GStreamer-enabled OpenCV — the pip wheel has none)
#   2. clone/update QuadGuide + EdgeCV into the login user's home
#   3. build the venv (--system-site-packages so it sees apt's cv2/numpy) + pip deps
#      (EdgeCV pulls onnxruntime — the CPU inference runtime; no RKNN/NPU on RPi)
#   4. enable the GPIO UART (/dev/serial0) for the MAVLink link to the FC
#   5. install the systemd service + WiFi AP (delegates to scripts/install_sbc.sh)
# The ONNX tracker weights are VENDORED in QuadGuide's models/ dir (committed to
# the repo, so the clone in step 2 brings them along) — it warns if they're absent.
set -euo pipefail

# ── tunables (override via env: GIT_OWNER=foo sudo -E bash ...) ────────────────
GIT_OWNER="${GIT_OWNER:-plastheis}"
QG_REPO="${QG_REPO:-https://github.com/${GIT_OWNER}/QuadGuide.git}"
EDGECV_REPO="${EDGECV_REPO:-https://github.com/${GIT_OWNER}/EdgeCV.git}"   # tracker library/runtime (models vendored in QuadGuide)
QG_BRANCH="${QG_BRANCH:-main}"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "run as root: sudo bash $0" >&2; exit 1; }

# Resolve the real login user (so clones/venv aren't owned by root). RPi OS's
# default user is 'pi', but Bookworm's imager lets you set a custom one — prefer
# whoever invoked sudo, fall back to 'pi'.
LOGIN_USER="${SUDO_USER:-pi}"
if ! id "$LOGIN_USER" >/dev/null 2>&1; then LOGIN_USER="pi"; fi
HOME_DIR="$(getent passwd "$LOGIN_USER" | cut -d: -f6)"
HOME_DIR="${HOME_DIR:-/home/$LOGIN_USER}"
QG_DIR="$HOME_DIR/quadguide"
EDGECV_DIR="$HOME_DIR/EdgeCV"
n_models=0   # set in step 2; referenced in the summary even if EdgeCV clone fails
run_user() { sudo -u "$LOGIN_USER" -H "$@"; }
say "user=$LOGIN_USER home=$HOME_DIR  quadguide=$QG_DIR  edgecv=$EDGECV_DIR"

# ── 1. system packages ────────────────────────────────────────────────────────
say "1/5 apt system dependencies"
export DEBIAN_FRONTEND=noninteractive
apt-get update
# Hard requirements. python3-opencv = OpenCV WITH V4L2 + GStreamer (the pip wheel
# lacks both → USBCamera's cv2.VideoCapture can't open /dev/video0). numpy/pyserial
# from apt so the venv (--system-site-packages) shares one ABI. v4l-utils for the
# USB camera (USBCamera uses v4l2-ctl to pin a constant frame rate) + bring-up.
# NOTE vs the ROCK script: no i2c-tools / device-tree-compiler (no CSI overlay),
# no GStreamer plugins (the v4l2 backend opens the device directly), no rknnlite2
# (no NPU — EdgeCV runs on the CPU via onnxruntime, installed in step 3).
apt-get install -y --no-install-recommends \
    git git-lfs python3 python3-venv python3-pip python3-dev \
    python3-opencv python3-numpy python3-serial \
    v4l-utils build-essential
# Enable Git LFS smudge so `git clone` auto-downloads EdgeCV's *.onnx model files
# (large binaries committed via LFS arrive as pointer stubs otherwise).
git lfs install --system 2>/dev/null || run_user git lfs install 2>/dev/null || true

# ── 2. clone/update repos ─────────────────────────────────────────────────────
say "2/5 clone/update QuadGuide + EdgeCV (as $LOGIN_USER)"
clone_or_pull() {  # $1=url $2=dir $3=branch
    if [ -d "$2/.git" ]; then
        run_user git -C "$2" pull --ff-only || warn "git pull failed in $2 (continuing)"
    else
        run_user git clone --branch "$3" --depth 1 "$1" "$2"
    fi
    # Fetch LFS objects (no-op if the repo doesn't use LFS).
    run_user git -C "$2" lfs pull 2>/dev/null || true
}
clone_or_pull "$QG_REPO" "$QG_DIR" "$QG_BRANCH"
# The ONNX tracker weights are VENDORED in QuadGuide at models/ (committed to the
# repo, so the clone above brought them along). configs/rpi4b.yaml's model_dir
# points here. This is independent of the EdgeCV *library* clone below.
n_models=$(find "$QG_DIR/models" -name '*.onnx' 2>/dev/null | wc -l)
if [ "$n_models" -gt 0 ]; then
    echo "  vendored ONNX models present: $n_models *.onnx under $QG_DIR/models"
else
    warn "no *.onnx found in $QG_DIR/models — the ONNX tracker can't load. Commit the"
    warn "nanotrack .onnx weights to QuadGuide's models/ dir (or, if they're Git LFS,"
    warn "run  (cd $QG_DIR && git lfs pull) ) then re-run."
fi
if [ "$QG_DIR" != "/home/pi/quadguide" ]; then
    warn "QuadGuide is at $QG_DIR but configs/rpi4b.yaml sets model_dir to"
    warn "/home/pi/quadguide/models — run as the 'pi' user, or edit model_dir."
fi
# EdgeCV LIBRARY (runtime + packaged tracker manifests) — still required so the
# adapter can `import edgecv`; its OWN model blobs are NOT used here (we vendor the
# .onnx in QuadGuide instead).
if ! clone_or_pull "$EDGECV_REPO" "$EDGECV_DIR" main; then
    warn "EdgeCV clone failed — fix EDGECV_REPO and re-run, or set tracker.import to a"
    warn "cv2 tracker (e.g. cv2:TrackerKCF) in configs/rpi4b.yaml to run without EdgeCV."
fi

# ── 3. venv + python deps ─────────────────────────────────────────────────────
say "3/5 python venv + dependencies"
if [ ! -x "$QG_DIR/.venv/bin/python" ]; then
    run_user python3 -m venv --system-site-packages "$QG_DIR/.venv"
fi
PIP="$QG_DIR/.venv/bin/pip"
run_user "$PIP" install --upgrade pip wheel
# QuadGuide runtime deps not satisfied by the apt packages above:
run_user "$PIP" install pyyaml pymavlink fastapi "uvicorn[standard]"
# Install the quadguide package ITSELF (editable). Without this, `import quadguide`
# fails — the code lives in src/ and is not on sys.path otherwise, so run.py dies
# with ModuleNotFoundError before any worker starts.
run_user "$PIP" install -e "$QG_DIR"
# EdgeCV (editable) — pulls its own deps incl. onnxruntime (the CPU inference
# runtime the ONNX tracker backend uses on the RPi 4B; there is no NPU here).
if [ -f "$EDGECV_DIR/pyproject.toml" ] || [ -f "$EDGECV_DIR/setup.py" ]; then
    run_user "$PIP" install -e "$EDGECV_DIR" || warn "EdgeCV pip install failed — see EdgeCV README"
fi

# ── 4. GPIO UART for the FC link (/dev/serial0) ───────────────────────────────
say "4/5 enable GPIO UART for /dev/serial0 (MAVLink to the FC)"
# RPi OS Bookworm keeps boot files under /boot/firmware; older releases use /boot.
BOOTDIR=/boot/firmware
[ -f "$BOOTDIR/config.txt" ] || BOOTDIR=/boot
CONFIG_TXT="$BOOTDIR/config.txt"
CMDLINE_TXT="$BOOTDIR/cmdline.txt"
if [ -f "$CONFIG_TXT" ]; then
    # enable_uart=1 turns the UART on; dtoverlay=disable-bt frees the PL011 from the
    # Bluetooth modem so /dev/serial0 → ttyAMA0 on GPIO14/15 (idempotent appends).
    grep -q '^enable_uart=1'        "$CONFIG_TXT" || echo 'enable_uart=1'        >> "$CONFIG_TXT"
    grep -q '^dtoverlay=disable-bt' "$CONFIG_TXT" || echo 'dtoverlay=disable-bt' >> "$CONFIG_TXT"
    echo "  enabled UART in $CONFIG_TXT (enable_uart=1, dtoverlay=disable-bt)"
else
    warn "no $CONFIG_TXT — enable the UART manually (raspi-config → Interface → Serial Port:"
    warn "login shell = No, serial hardware = Yes)."
fi
# Drop the serial login console so getty doesn't fight MAVLink for the port.
if [ -f "$CMDLINE_TXT" ]; then
    sed -i 's/console=serial0,[0-9]* //; s/console=ttyAMA0,[0-9]* //' "$CMDLINE_TXT"
    echo "  removed serial login console from $CMDLINE_TXT"
fi
systemctl disable --now hciuart 2>/dev/null || true
systemctl disable --now serial-getty@ttyS0.service serial-getty@ttyAMA0.service 2>/dev/null || true

# ── 5. systemd service + WiFi AP ──────────────────────────────────────────────
say "5/5 systemd autostart + WiFi AP"
chmod +x "$QG_DIR/scripts/install_sbc.sh" 2>/dev/null || true
bash "$QG_DIR/scripts/install_sbc.sh" "configs/rpi4b.yaml"

# ── summary ───────────────────────────────────────────────────────────────────
say "DONE — software installed, service enabled, WiFi AP 'drone' configured."
cat <<EOF

Reboot to apply the UART change:   sudo reboot

ONNX tracker weights are vendored in $QG_DIR/models ($n_models *.onnx found above).
If that count is 0 the tracker worker can't load and quadguide.service crash-loops
(expected) — commit the nanotrack .onnx weights to QuadGuide's models/ dir (or, if
they're Git LFS,  (cd $QG_DIR && git lfs pull) ) then re-run. To run WITHOUT EdgeCV
at all, set tracker.import to a cv2 tracker (e.g. cv2:TrackerKCF) in configs/rpi4b.yaml.

Wire the FC to GPIO14 (TXD, pin 8) / GPIO15 (RXD, pin 10) / GND — cross TX↔RX.
Confirm the link node after reboot:   ls -l /dev/serial0   (→ ttyAMA0)

Verify:   systemctl status quadguide ; journalctl -u quadguide -f
Web UI:   http://10.42.0.1:8080   (join WiFi 'drone' / pass 'drone123').
EOF

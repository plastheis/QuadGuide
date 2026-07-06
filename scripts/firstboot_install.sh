#!/usr/bin/env bash
# firstboot_install.sh — one-shot board-side bootstrap for QuadGuide on a fresh
# Radxa ROCK 5C (RK3588S2) image. Run ONCE, at the board's local console, as root,
# with the board on WIRED ETHERNET (it needs internet for apt/pip/git).
#
#   sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/plastheis/QuadGuide/main/scripts/firstboot_install.sh)"
#
# or, if you already cloned the repo:   sudo bash scripts/firstboot_install.sh
#
# It is idempotent — safe to re-run. It does the parts that CAN be automated:
#   1. apt system deps (incl. GStreamer-enabled OpenCV — the pip wheel has none)
#   2. clone/update QuadGuide + EdgeCV into the login user's home
#   3. build the venv (--system-site-packages so it sees apt's cv2/numpy) + pip deps
#   4. compile + enable the OV9281 device-tree overlay
#   5. install the systemd service + WiFi AP (delegates to scripts/install_sbc.sh)
# It clearly PRINTS the two steps that need your hardware and cannot be automated:
#   - the OV9281 kernel driver (not in the shipped kernel — see §A printout)
#   - EdgeCV's RKNN models (per-model artifacts EdgeCV ships separately)
set -euo pipefail

# ── tunables (override via env: GIT_OWNER=foo sudo -E bash ...) ────────────────
GIT_OWNER="${GIT_OWNER:-plastheis}"
QG_REPO="${QG_REPO:-https://github.com/${GIT_OWNER}/QuadGuide.git}"
EDGECV_REPO="${EDGECV_REPO:-https://github.com/${GIT_OWNER}/EdgeCV.git}"   # ships its own .rknn models
QG_BRANCH="${QG_BRANCH:-main}"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "run as root: sudo bash $0" >&2; exit 1; }

# Resolve the real login user (so clones/venv aren't owned by root).
LOGIN_USER="${SUDO_USER:-radxa}"
if ! id "$LOGIN_USER" >/dev/null 2>&1; then LOGIN_USER="radxa"; fi
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
# Hard requirements. python3-opencv = OpenCV WITH GStreamer (the pip wheel has
# none → CSICamera fails); Debian's build links the system GStreamer that the
# Radxa image already ships. numpy/pyserial from apt so the venv
# (--system-site-packages) shares one ABI. i2c-tools/v4l-utils for camera bring-up.
apt-get install -y --no-install-recommends \
    git git-lfs python3 python3-venv python3-pip python3-dev \
    python3-opencv python3-numpy python3-serial \
    i2c-tools v4l-utils \
    device-tree-compiler build-essential
# (network-manager is pre-installed on Radxa OS; install_sbc.sh validates it.
#  We don't re-request it — that risks an apt downgrade like the gstreamer one.)
# GStreamer plugins are best-effort and MUST NOT abort the install: the Radxa
# image already ships GStreamer (with Rockchip-accelerated plugins), and forcing
# these can demand a DOWNGRADE of the vendor gstreamer1.0-plugins-good, which apt
# refuses under -y. We deliberately do NOT pass --allow-downgrades (that would
# replace the hardware-accelerated plugins with vanilla Debian ones). If apt
# declines, the already-installed plugins are kept and we continue.
apt-get install -y --no-install-recommends \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    || warn "gstreamer plugins left as-is (already present, or apt declined a downgrade)"
# Enable Git LFS smudge so `git clone` auto-downloads EdgeCV's *.rknn model files
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
if clone_or_pull "$EDGECV_REPO" "$EDGECV_DIR" main; then
    # Models ship in the EdgeCV repo at EdgeCV/models; configs/rk3588.yaml expects
    # them at /home/radxa/EdgeCV/models (the canonical radxa-user path).
    n_models=$(find "$EDGECV_DIR/models" -name '*.rknn' 2>/dev/null | wc -l) || true
    if [ "$n_models" -gt 0 ]; then
        echo "  EdgeCV models present: $n_models *.rknn under $EDGECV_DIR/models"
    else
        warn "no *.rknn found in $EDGECV_DIR/models — if EdgeCV uses Git LFS, run"
        warn "  (cd $EDGECV_DIR && git lfs pull)   then check $EDGECV_DIR/models"
    fi
    if [ "$EDGECV_DIR" != "/home/radxa/EdgeCV" ]; then
        warn "EdgeCV is at $EDGECV_DIR but configs/rk3588.yaml points model_dir at"
        warn "/home/radxa/EdgeCV/models — run as the 'radxa' user, or edit model_dir."
    fi
else
    warn "EdgeCV clone failed — fix EDGECV_REPO and re-run, or set tracker.import to a"
    warn "cv2 tracker (e.g. cv2:TrackerKCF) in configs/rk3588.yaml to run without EdgeCV."
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
# EdgeCV (editable) — pulls its own deps incl. onnxruntime; RKNN runtime is below.
if [ -f "$EDGECV_DIR/pyproject.toml" ] || [ -f "$EDGECV_DIR/setup.py" ]; then
    run_user "$PIP" install -e "$EDGECV_DIR" || warn "EdgeCV pip install failed — see EdgeCV README"
fi
# RKNN runtime for NPU inference. Radxa ships it via apt; fall back to a no-op so a
# missing NPU runtime never aborts the install (MOSSE/cv2 trackers still work).
apt-get install -y python3-rknnlite2 2>/dev/null || \
    warn "python3-rknnlite2 not installed — needed for EdgeCV NPU trackers (nanotrack/yolo)."

# ── 4. OV9281 overlay (+ driver check) ────────────────────────────────────────
say "4/5 OV9281 camera overlay"
KREL="$(uname -r)"
if zcat /proc/config.gz 2>/dev/null | grep -q '^CONFIG_VIDEO_OV9281=' || \
   grep -q '^CONFIG_VIDEO_OV9281=' "/boot/config-$KREL" 2>/dev/null; then
    OV9281_DRIVER=yes
else
    OV9281_DRIVER=no
fi
OVL_SRC="$QG_DIR/overlays/rock-5c-ov9281.dts"
if [ -f "$OVL_SRC" ]; then
    dtc -@ -I dts -O dtb -o /boot/dtbo/rock-5c-ov9281.dtbo "$OVL_SRC" \
        && echo "  compiled /boot/dtbo/rock-5c-ov9281.dtbo"
    # enable it in the U-Boot extlinux overlays line (idempotent)
    UENV=/boot/uEnv.txt
    if [ -f "$UENV" ]; then
        if grep -q '^overlays=' "$UENV"; then
            grep -q 'rock-5c-ov9281' "$UENV" || \
                sed -i 's/^overlays=\(.*\)$/overlays=\1 rock-5c-ov9281/' "$UENV"
        else
            echo 'overlays=rock-5c-ov9281' >> "$UENV"
        fi
        echo "  enabled rock-5c-ov9281 in $UENV"
    else
        warn "no $UENV — enable the overlay via 'rsetup' → Overlays instead."
    fi
else
    warn "overlay source missing at $OVL_SRC (old checkout?) — skipping camera overlay."
fi

# ── 5. systemd service + WiFi AP ──────────────────────────────────────────────
say "5/5 systemd autostart + WiFi AP"
chmod +x "$QG_DIR/scripts/install_sbc.sh" 2>/dev/null || true
bash "$QG_DIR/scripts/install_sbc.sh" "configs/rk3588.yaml"

# ── summary ───────────────────────────────────────────────────────────────────
say "DONE — software installed, service enabled, WiFi AP 'drone' configured."
cat <<EOF

Reboot to load the camera overlay:   sudo reboot

After reboot, two things still need YOUR hardware (cannot be automated from afar):

  A) OV9281 KERNEL DRIVER — $( [ "$OV9281_DRIVER" = yes ] && echo "present ✓" || echo "MISSING.")
EOF
if [ "$OV9281_DRIVER" != yes ]; then cat <<'EOF'
     The shipped kernel has no ov9281 driver (CONFIG_VIDEO_OV9281 not set). Build it:
       - grab rockchip-linux/kernel rk-6.1 drivers/media/i2c/ov9281.c
       - in a dir with it + a Makefile "obj-m += ov9281.o":
           make -C /usr/src/linux-headers-$(uname -r) M=$PWD modules
           sudo cp ov9281.ko /usr/lib/modules/$(uname -r)/updates/ && sudo depmod -a
           echo ov9281 | sudo tee /etc/modules-load.d/ov9281.conf
     Until then the camera worker can't open and quadguide.service crash-loops (expected).
EOF
fi
cat <<EOF
  B) EdgeCV MODELS — cloned with the EdgeCV repo into $EDGECV_DIR/models
     ($n_models *.rknn found above). If that count is 0, the repo uses Git LFS and
     the objects didn't smudge: run  (cd $EDGECV_DIR && git lfs pull).

Verify:   systemctl status quadguide ; journalctl -u quadguide -f
Web UI:   http://10.42.0.1:8080   (join WiFi 'drone' / pass 'drone123'), or over the HaLow bridge.
EOF

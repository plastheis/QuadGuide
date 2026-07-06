#!/usr/bin/env bash
# install_sbc.sh — set up quadguide systemd autostart + WiFi AP on a Radxa ROCK 5C.
#
# Run once as root from the repo:
#     sudo ./scripts/install_sbc.sh [CONFIG_PATH]
#
# Preview without root / without changing anything (renders unit + nmcli plan):
#     ./scripts/install_sbc.sh --dry-run [CONFIG_PATH]
#
# Env overrides: AP_SSID (default drone), AP_PSK (default drone123),
#                WIFI_IFACE (default: first nmcli wifi device), LOG_DIR (/var/log/quadguide)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
UNIT_TEMPLATE="$REPO_DIR/systemd/quadguide.service"
UNIT_DEST="/etc/systemd/system/quadguide.service"

DRY_RUN=0
CONFIG="configs/rk3588.yaml"
for arg in "$@"; do
    case "$arg" in
        --dry-run|--render-only) DRY_RUN=1 ;;
        -h|--help) grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --*) echo "unknown flag: $arg" >&2; exit 2 ;;
        *) CONFIG="$arg" ;;
    esac
done

AP_SSID="${AP_SSID:-drone}"
AP_PSK="${AP_PSK:-drone123}"
LOG_DIR="${LOG_DIR:-/var/log/quadguide}"

render_unit() {
    [ -f "$UNIT_TEMPLATE" ] || { echo "missing template: $UNIT_TEMPLATE" >&2; exit 1; }
    sed \
        -e "s|@REPO_DIR@|$REPO_DIR|g" \
        -e "s|@PYTHON@|$PYTHON|g" \
        -e "s|@CONFIG@|$CONFIG|g" \
        -e "s|@LOG_DIR@|$LOG_DIR|g" \
        "$UNIT_TEMPLATE"
}

ap_iface() {
    if [ -n "${WIFI_IFACE:-}" ]; then echo "$WIFI_IFACE"; return 0; fi
    nmcli -t -f DEVICE,TYPE device status 2>/dev/null | awk -F: '$2=="wifi"{print $1; exit}'
}

if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== rendered $UNIT_DEST ==="
    render_unit
    echo
    echo "=== WiFi AP plan ==="
    echo "iface : $(ap_iface || true)"
    echo "ssid  : $AP_SSID"
    echo "psk   : $AP_PSK  (wpa-psk / 2.4GHz / ipv4 shared / autoconnect)"
    echo "url   : http://10.42.0.1:8080"
    exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "must run as root (use sudo). Preview with: $0 --dry-run" >&2
    exit 1
fi
[ -x "$PYTHON" ] || { echo "venv python missing/not executable: $PYTHON" >&2; exit 1; }

echo "[1/2] installing $UNIT_DEST"
mkdir -p "$LOG_DIR"
render_unit > "$UNIT_DEST"
systemctl daemon-reload
systemctl enable --now quadguide.service

echo "[2/2] configuring WiFi AP '$AP_SSID'"
command -v nmcli >/dev/null || { echo "nmcli not found — NetworkManager required" >&2; exit 1; }
[ "$(nmcli -t -f RUNNING general 2>/dev/null)" = "running" ] \
    || { echo "NetworkManager is not running" >&2; exit 1; }
IFACE="$(ap_iface)"
[ -n "$IFACE" ] || { echo "no wifi device found (see: nmcli device status)" >&2; exit 1; }
nmcli connection delete qg-ap >/dev/null 2>&1 || true
nmcli connection add type wifi ifname "$IFACE" con-name qg-ap autoconnect yes ssid "$AP_SSID"
nmcli connection modify qg-ap \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    ipv4.method shared \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.proto rsn \
    wifi-sec.pairwise ccmp \
    wifi-sec.group ccmp \
    wifi-sec.pmf disable \
    wifi-sec.psk "$AP_PSK"
nmcli connection up qg-ap

cat <<EOF

Done.
  Service  : quadguide.service (enabled; follow with: journalctl -u quadguide -f)
  WiFi AP  : SSID '$AP_SSID'  PSK '$AP_PSK'  (2.4GHz, iface $IFACE)
  Ground UI: http://10.42.0.1:8080  (also reachable over the HaLow bridge)
EOF

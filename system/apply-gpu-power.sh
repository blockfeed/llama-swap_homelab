#!/usr/bin/env bash
# =============================================================================
# apply-gpu-power.sh — Apply persistent GPU power management fix
# =============================================================================
# Installs all three layers of the GPU D3hot prevention fix:
#   1. Immediate: writes to sysfs now (takes effect instantly, lost on reboot)
#   2. Boot-persistent: udev rule (survives reboots, applies at device probe)
#   3. Service-lifecycle: systemd drop-in (ties power state to llama-swap)
#
# Must be run as root (sudo).
#
# Usage:
#   sudo ./system/apply-gpu-power.sh
#
# To undo:
#   sudo rm /etc/udev/rules.d/99-amdgpu-power.rules
#   sudo rm /etc/systemd/system/llama-swap.service.d/gpu-power.conf
#   sudo udevadm control --reload-rules
#   sudo systemctl daemon-reload
#   echo auto > /sys/class/drm/card0/device/power/control
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSFS_POWER="/sys/class/drm/card0/device/power/control"
UDEV_DEST="/etc/udev/rules.d/99-amdgpu-power.rules"
DROPIN_DEST="/etc/systemd/system/llama-swap.service.d/gpu-power.conf"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Error: must be run as root (sudo $0)" >&2
  exit 1
fi

echo "=== GPU power management fix ==="
echo ""

# 1. Immediate sysfs write — takes effect now, lost on reboot without udev rule
echo "[1/3] Immediate: disabling runtime PM via sysfs..."
if [[ -f "$SYSFS_POWER" ]]; then
  echo on > "$SYSFS_POWER"
  echo "      Wrote 'on' → $SYSFS_POWER"
  echo "      Current power_state: $(cat /sys/class/drm/card0/device/power_state 2>/dev/null || echo 'unknown')"
else
  echo "      WARNING: $SYSFS_POWER not found — is amdgpu loaded?" >&2
  echo "      Check: ls /sys/class/drm/ to find the correct card device." >&2
fi

echo ""

# 2. udev rule — applies at next device probe / reboot
echo "[2/3] Boot-persistent: installing udev rule..."
install -m 644 "$SCRIPT_DIR/99-amdgpu-power.rules" "$UDEV_DEST"
udevadm control --reload-rules
echo "      Installed: $UDEV_DEST"
echo "      Rules reloaded (existing device not re-triggered; reboot for full activation)"

echo ""

# 3. Systemd drop-in — ties GPU power state to llama-swap service lifecycle
echo "[3/3] Service-lifecycle: installing systemd drop-in..."
mkdir -p "$(dirname "$DROPIN_DEST")"
install -m 644 "$SCRIPT_DIR/gpu-power.conf" "$DROPIN_DEST"
systemctl daemon-reload
echo "      Installed: $DROPIN_DEST"
echo "      systemd configuration reloaded"

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  sudo systemctl restart llama-swap   # apply drop-in to running service"
echo "  systemctl cat llama-swap            # verify effective configuration"
echo ""
echo "Verification:"
echo "  cat $SYSFS_POWER                    # should read 'on'"
echo "  cat /sys/class/drm/card0/device/power_state  # should read 'D0'"

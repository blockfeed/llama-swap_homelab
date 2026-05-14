#!/bin/bash
# =============================================================================
# Deploy Script: Install Container Configs to System Locations
# =============================================================================
# Purpose: Copy the container configuration files from this repo to their
#          system locations and enable the container.
#
# This script is idempotent — safe to run multiple times.
#
# Usage:
#   sudo ./deploy.sh
#
# What it does:
#   1. Copies llama-container.nspawn to /etc/systemd/nspawn/
#   2. Creates drop-in directory and copies gpu.conf + llama-swap-dep.conf
#   3. Runs machinectl enable llama-container
#   4. Runs systemctl daemon-reload
#
# See also:
#   bootstrap.sh  — Create the container rootfs (run before deploy)
#   README.md     — Full documentation
# =============================================================================

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="llama-container"
NSPAWN_DIR="/etc/systemd/nspawn"
DROPIN_DIR="/etc/systemd/system/systemd-nspawn@${CONTAINER_NAME}.service.d"

# --- Color helpers ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Preflight ---
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (sudo)."
    exit 1
fi

# --- Step 1: Deploy .nspawn config ---
info "Deploying nspawn config..."
mkdir -p "${NSPAWN_DIR}"
cp "${SCRIPT_DIR}/llama-container.nspawn" "${NSPAWN_DIR}/llama-container.nspawn"
chmod 644 "${NSPAWN_DIR}/llama-container.nspawn"
info "  → ${NSPAWN_DIR}/llama-container.nspawn"

# --- Step 2: Deploy systemd drop-ins ---
info "Deploying systemd drop-ins..."
mkdir -p "${DROPIN_DIR}"

for file in gpu.conf llama-swap-dep.conf; do
    if [[ -f "${SCRIPT_DIR}/${file}" ]]; then
        cp "${SCRIPT_DIR}/${file}" "${DROPIN_DIR}/${file}"
        chmod 644 "${DROPIN_DIR}/${file}"
        info "  → ${DROPIN_DIR}/${file}"
    else
        warn "  ${file} not found in ${SCRIPT_DIR}, skipping."
    fi
done

# --- Step 3: Enable container ---
info "Enabling container with machinectl..."
machinectl enable "${CONTAINER_NAME}" 2>/dev/null || true
info "  Container '${CONTAINER_NAME}' enabled."

# --- Step 4: Reload systemd ---
info "Reloading systemd..."
systemctl daemon-reload
info "  systemd reloaded."

# --- Summary ---
echo ""
info "================================================================"
info "  Deployment complete!"
info "================================================================"
echo ""
info "Container configuration deployed to system locations."
info ""
info "To start the container (after bootstrapping rootfs):"
echo ""
echo "  sudo machinectl start ${CONTAINER_NAME}"
echo "  sudo machinectl list"
echo "  ssh -p 2223 <user>@<host-ip>"
echo ""
info "To check drop-in status:"
echo "  systemctl cat systemd-nspawn@${CONTAINER_NAME}"
echo ""
info "To stop the container:"
echo "  sudo machinectl stop ${CONTAINER_NAME}"
echo ""
info "GPU device GID on this host: $(stat -c '%g' /dev/kfd 2>/dev/null || echo 'N/A')"
info "The container user must be in group 'render' with the same GID as the host."

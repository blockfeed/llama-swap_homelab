#!/bin/bash
# =============================================================================
# Bootstrap Script: Create systemd-nspawn Container Rootfs
# =============================================================================
# Purpose: Create a minimal Arch Linux rootfs for the llama-container
#          systemd-nspawn instance using pacstrap.
#
# Prerequisites:
#   - Arch Linux host
#   - arch-install-scripts package (provides pacstrap)
#   - root/sudo access
#
# Usage:
#   sudo ./bootstrap.sh
#
# What it does:
#   1. Checks prerequisites (root, pacstrap, existing container)
#   2. Pacstraps base + openssh packages into /var/lib/machines/llama-container
#   3. Enables sshd inside the container
#   4. Configures SSH port 2223 in the container's sshd_config
#   5. Reports post-bootstrap steps (GPU group config)
#
# Post-bootstrap steps (manual, inside container):
#   systemd-nspawn -D /var/lib/machines/llama-container
#   groupadd --force --gid 993 render
#   usermod -aG render <your-ssh-user>
#   passwd <your-ssh-user>
#   exit
#
# See also:
#   deploy.sh  — Deploy configs to system locations (run after bootstrap)
# =============================================================================

set -euo pipefail

# --- Configuration ---
CONTAINER_NAME="llama-container"
ROOTFS="/var/lib/machines/${CONTAINER_NAME}"
GPU_GID="993"
GPU_GROUP="render"

# --- Color helpers ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Preflight checks ---
info "Checking prerequisites..."

if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (sudo)."
    exit 1
fi

if ! command -v pacstrap &>/dev/null; then
    error "pacstrap not found. Install arch-install-scripts:"
    echo "  sudo pacman -S arch-install-scripts"
    exit 1
fi

if [[ -d "${ROOTFS}" ]]; then
    warn "Container rootfs already exists at ${ROOTFS}"
    read -r -p "Overwrite? This will DELETE the existing rootfs. [y/N] " reply
    if [[ "${reply}" =~ ^[Yy]$ ]]; then
        info "Removing existing rootfs..."
        rm -rf "${ROOTFS}"
    else
        error "Aborted by user."
        exit 1
    fi
fi

# --- Bootstrap rootfs ---
info "Bootstrapping minimal Arch Linux rootfs at ${ROOTFS}..."
if ! pacstrap -c "${ROOTFS}" base openssh python-huggingface-hub; then
    error "pacstrap failed. Check network connectivity and disk space."
    exit 1
fi
info "Rootfs bootstrap complete."

# --- Enable sshd ---
info "Enabling sshd inside container..."
systemd-nspawn -D "${ROOTFS}" systemctl enable sshd
info "sshd enabled."

# --- Configure SSH port ---
info "Setting SSH port to 2223..."
SSHD_CONFIG="${ROOTFS}/etc/ssh/sshd_config"
if [[ -f "${SSHD_CONFIG}" ]]; then
    # Uncomment and set Port directive
    sed -i 's/^#Port 22/Port 2223/' "${SSHD_CONFIG}"
    # If no Port line exists at all, append one
    if ! grep -q '^Port ' "${SSHD_CONFIG}"; then
        echo "Port 2223" >> "${SSHD_CONFIG}"
    fi
    info "SSH port configured to 2223."
else
    warn "sshd_config not found at ${SSHD_CONFIG}. Check pacstrap output."
fi

# --- Done ---
echo ""
info "================================================================"
info "  Container rootfs bootstrap complete!"
info "================================================================"
echo ""
info "Container rootfs: ${ROOTFS}"
echo ""
info "Next steps (run AFTER deploying configs with deploy.sh):"
echo ""
echo "  1. Deploy configuration files:"
echo "     sudo ./deploy.sh"
echo ""
echo "  2. Enter the container to set up users and GPU access:"
echo "     sudo systemd-nspawn -D ${ROOTFS}"
echo ""
echo "  3. Inside the container:"
echo "     groupadd --force --gid ${GPU_GID} ${GPU_GROUP}"
echo "     useradd -m -G ${GPU_GROUP} <your-ssh-user>"
echo "     passwd <your-ssh-user>"
echo "     exit"
echo ""
echo "  4. Start the container:"
echo "     sudo machinectl start ${CONTAINER_NAME}"
echo ""
echo "  5. Verify:"
echo "     sudo machinectl list"
echo "     ssh -p 2223 <your-ssh-user>@<host-ip>"
echo ""
info "GPU device GID on this host: ${GPU_GID} (group '${GPU_GROUP}')"
info "The container user must be in the '${GPU_GROUP}' group with GID ${GPU_GID}."

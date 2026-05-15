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
#   2. Pacstraps base + openssh + python-huggingface-hub into /var/lib/machines/llama-container
#   3. Enables sshd inside the container
#   4. Configures SSH port 2223 in the container's sshd_config
#   5. Creates user and GPU render group inside the container
#   6. Injects host SSH public key for passwordless access
#   7. Reports post-bootstrap steps (deploy configs, start container)
#
# No manual post-bootstrap steps required — user and SSH key are set up
# automatically. After bootstrap, run deploy.sh then start the container.
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
CONTAINER_USER="<user>"
CONTAINER_UID="1000"
# SSH public key for the container user
SSH_AUTHORIZED_KEY="<ssh_public_key>"

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

# --- Create user and GPU group, inject SSH key ---
info "Creating render group (GID ${GPU_GID}) inside container..."
systemd-nspawn -D "${ROOTFS}" groupadd --force --gid ${GPU_GID} ${GPU_GROUP}

info "Creating user '${CONTAINER_USER}' (UID ${CONTAINER_UID}) inside container..."
systemd-nspawn -D "${ROOTFS}" useradd -m -u ${CONTAINER_UID} -G ${GPU_GROUP} ${CONTAINER_USER}
info "User '${CONTAINER_USER}' created and added to '${GPU_GROUP}' group."

info "Installing SSH authorized key for ${CONTAINER_USER}..."
USER_SSH_DIR="${ROOTFS}/home/${CONTAINER_USER}/.ssh"
mkdir -p "${USER_SSH_DIR}"
echo "${SSH_AUTHORIZED_KEY}" > "${USER_SSH_DIR}/authorized_keys"
chmod 700 "${USER_SSH_DIR}"
chmod 600 "${USER_SSH_DIR}/authorized_keys"
chown -R ${CONTAINER_UID}:${CONTAINER_UID} "${USER_SSH_DIR}"
info "SSH key installed."

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
echo "  2. Start the container:"
echo "     sudo machinectl start ${CONTAINER_NAME}"
echo ""
echo "  3. Verify:"
echo "     sudo machinectl list"
echo "     ssh -p 2223 ${CONTAINER_USER}@<host-ip>"
echo ""
info "User '${CONTAINER_USER}' is already configured with SSH key access."
info "GPU render group (GID ${GPU_GID}) is set up."

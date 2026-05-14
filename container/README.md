# systemd-nspawn Container: llama-container

Isolate llama-swap inference tooling (benchmarking, test harnesses, custom
development) in a lightweight systemd-nspawn container on Arch Linux.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Host (Arch Linux, RX 7900 XTX)                     │
│                                                      │
│  ┌───────────────────────┐                           │
│  │ llama-swap.service    │  Host inference server    │
│  │ (llama-swap + cpp)    │  BindsTo → container      │
│  └───────────┬───────────┘                           │
│              │ BindsTo / After                       │
│              ▼                                       │
│  ┌───────────────────────┐  ┌────────────────────┐   │
│  │ systemd-nspawn@       │  │ /etc/llama-swap/   │   │
│  │ llama-container       │◄─┤                    │   │
│  │                       │  │ Bind (rw)          │   │
│  │ ┌─────────────────┐   │  ├────────────────────┤   │
│  │ │ sshd (Port 2223) │   │  │ /opt/llama/        │   │
│  │ │ systemd inside   │   │◄─┤                    │   │
│  │ │ tools / scripts  │   │  │ Bind (rw)          │   │
│  │ └─────────────────┘   │  └────────────────────┘   │
│  │                       │  ┌────────────────────┐   │
│  │ GPU access via        │  │ /dev/kfd           │   │
│  │ DeviceAllow +         │◄─┤ /dev/dri/renderD128│   │
│  │ BindPaths             │  └────────────────────┘   │
│  └───────────────────────┘                           │
└─────────────────────────────────────────────────────┘
```

### Design Decisions

- **VirtualEthernet=no**: The container shares the host network namespace.
  SSH listens on port 2223 (host IP, non-standard SSH port). No separate
  IP or bridge setup required.

- **BindsTo=llama-swap.service**: The container lives and dies with the
  inference service. If llama-swap stops, the container stops. This is a
  hard dependency (not a soft Wants).

- **Bind (rw) for /etc/llama-swap**: The container can read and write the
  configuration. This allows containerized tools to modify config.yaml
  (e.g., for benchmarking different model sets) without host-side steps.

- **Bind (rw) for /opt/llama**: Model files are writable so the container
  can download and manage its own models using `huggingface-cli`. The
  `python-huggingface-hub` package is included in the bootstrap.

- **DeviceAllow + BindPaths**: Both are required for GPU access.
  `DeviceAllow` grants cgroup access; `BindPaths` mounts the device node
  into the container's `/dev/`. Without `DeviceAllow`, systemd inside the
  container blocks access even with the device node present.

## File Inventory

| File | Purpose |
|------|---------|
| `llama-container.nspawn` | systemd-nspawn config: boot, bind mounts, networking |
| `gpu.conf` | Systemd drop-in: GPU device access for the container |
| `llama-swap-dep.conf` | Systemd drop-in: `BindsTo=llama-swap.service` |
| `bootstrap.sh` | Create the container rootfs with pacstrap |
| `deploy.sh` | Copy configs to system locations and enable |
| `README.md` | This documentation |

### Deployed Locations

| Repo File | Target Path |
|-----------|-------------|
| `llama-container.nspawn` | `/etc/systemd/nspawn/llama-container.nspawn` |
| `gpu.conf` | `/etc/systemd/system/systemd-nspawn@llama-container.service.d/gpu.conf` |
| `llama-swap-dep.conf` | `/etc/systemd/system/systemd-nspawn@llama-container.service.d/llama-swap-dep.conf` |

## Prerequisites

- **Arch Linux host** (the node running llama-swap)
- **Packages**:
  - `arch-install-scripts` (provides `pacstrap`)
  - `systemd` (provides `systemd-nspawn`, `machinectl`)
  - `openssh` (on the host, for SSH access to the container)
- **Root/sudo access** for all bootstrap and deploy commands
- **llama-swap.service** must be installed and operational (the container binds to it)

## Setup Procedure

Two phases: **bootstrap** (create the root filesystem), then **deploy** (install
system configs). Run both as root on the llama-swap host (<host-ip>).

### Phase 1: Bootstrap the Container Rootfs

```bash
# Copy this directory to the host if not already present:
# scp -r container/ user@<host-ip>:/etc/llama-swap/

# Install prerequisite if missing:
sudo pacman -S arch-install-scripts

# Run the bootstrap script:
sudo ./bootstrap.sh
```

The bootstrap script will:

1. Check for root and `pacstrap`
2. Warn and prompt before overwriting an existing rootfs
3. Run `pacstrap -c /var/lib/machines/llama-container base openssh`
4. Enable `sshd` inside the container
5. Set SSH port to **2223** in the container's `/etc/ssh/sshd_config`

### Phase 2: Deploy Configuration Files

```bash
sudo ./deploy.sh
```

The deploy script will:

1. Copy `llama-container.nspawn` → `/etc/systemd/nspawn/llama-container.nspawn`
2. Create `/etc/systemd/system/systemd-nspawn@llama-container.service.d/`
3. Copy `gpu.conf` and `llama-swap-dep.conf` into the drop-in directory
4. Run `machinectl enable llama-container`
5. Run `systemctl daemon-reload`

### Phase 3: Configure Users and GPU Access Inside the Container

Enter the container to create a user and set up GPU group membership:

```bash
sudo systemd-nspawn -D /var/lib/machines/llama-container
```

Inside the container:

```bash
# Create the 'render' group with the host's GID (993)
groupadd --force --gid 993 render

# Create your user and add to render group
useradd -m -G render <your-ssh-user>

# Set a password for SSH access
passwd <your-ssh-user>

# Exit the container
exit
```

**Why GID 993?** On the host, `/dev/kfd` and `/dev/dri/renderD128` are owned by
group `render` with GID 993. ROCm checks group membership at the GID level, not
the group name. The container must have a group with the same GID for GPU access
to work.

### Phase 4: Start the Container

```bash
# Start the container
sudo machinectl start llama-container

# Verify it's running
sudo machinectl list

# Check status
sudo machinectl status llama-container

# Test SSH access
ssh -p 2223 <your-ssh-user>@<host-ip>
```

## Verification Checklist

After setup is complete, verify each of these:

- [ ] **Container is running**: `sudo machinectl list` shows `llama-container`
- [ ] **SSH access works**: `ssh -p 2223 <user>@<host-ip>`
- [ ] **GPU devices visible inside**:
  ```bash
  ssh -p 2223 <user>@<host-ip> ls -la /dev/kfd /dev/dri/renderD128
  ```
- [ ] **Config bind mount works**:
  ```bash
  ssh -p 2223 <user>@<host-ip> ls /etc/llama-swap/config.yaml
  ```
- [ ] **Model files accessible**:
  ```bash
  ssh -p 2223 <user>@<host-ip> ls /opt/llama/models/
  ```
- [ ] **ROCm detects GPU**:
  ```bash
  ssh -p 2223 <user>@<host-ip> rocm-smi --showhw
  ```
- [ ] **Drop-in configs loaded**:
  ```bash
  systemctl cat systemd-nspawn@llama-container
  # Should show DeviceAllow and BindPaths directives
  ```
- [ ] **Container auto-starts with llama-swap**:
  ```bash
  sudo systemctl stop llama-swap
  sudo machinectl list  # Should show container stopped
  sudo systemctl start llama-swap
  sleep 5
  sudo machinectl list  # Should show container running
  ```

## Usage

### Accessing the Container

```bash
# SSH (from any host on the local network)
ssh -p 2223 <user>@<host-ip>

# Direct shell (from the host)
sudo machinectl shell llama-container
```

### Running Inference Tools

Once inside the container, you have access to:

- Writable model files at `/opt/llama/models/` for downloads and management
- Read-write config at `/etc/llama-swap/config.yaml`
- `huggingface-cli` available for model downloads
- GPU via `/dev/kfd` and `/dev/dri/renderD128`
- Full network access (same namespace as host)

### Downloading Models with HuggingFace CLI

Because `/opt/llama` is writable, you can download models directly from
inside the container:

```bash
# Inside the container
huggingface-cli download lmstudio-community/Qwen3.6-35B-A3B-UD-IQ4_XS-GGUF \
  --local-dir /opt/llama/models/qwen36 \
  --local-dir-use-symlinks False
```

Downloaded models are immediately visible to the host's llama-swap (no
restart needed with `-watch-config`).

Example — run a benchmark from inside the container:

```bash
# Inside the container
llama-bench --model /opt/llama/models/qwen36/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --n-gpu-layers 999
```

### Container Lifecycle

```bash
# Start
sudo machinectl start llama-container

# Stop
sudo machinectl stop llama-container

# Restart
sudo machinectl restart llama-container

# Status
sudo machinectl status llama-container

# Shell
sudo machinectl shell llama-container

# Poweroff (from inside)
sudo poweroff
```

### Viewing Logs

```bash
# Container boot log (systemd journal from inside)
journalctl -u systemd-nspawn@llama-container

# Follow
journalctl -u systemd-nspawn@llama-container -f

# SSH access logs (from inside the container)
sudo machinectl shell llama-container /usr/bin/journalctl -u sshd
```

## Maintenance

### Updating Container Packages

```bash
# Enter the container
sudo machinectl shell llama-container

# Update packages (inside the container)
pacman -Syu
exit
```

Or run a single command:

```bash
sudo machinectl shell llama-container /usr/bin/pacman -Syu --noconfirm
```

### Backup and Restore

```bash
# Backup rootfs (while container is stopped)
sudo machinectl stop llama-container
sudo tar czf /tmp/llama-container-backup-$(date +%Y%m%d).tar.gz \
  /var/lib/machines/llama-container/

# Restore from backup
sudo tar xzf /tmp/llama-container-backup-YYYYMMDD.tar.gz -C /
```

### Rebuilding from Scratch

```bash
# Stop and remove container
sudo machinectl stop llama-container
sudo machinectl disable llama-container
sudo rm -rf /var/lib/machines/llama-container
# Re-run bootstrap.sh and deploy.sh
sudo ./bootstrap.sh && sudo ./deploy.sh
```

## Troubleshooting

### GPU Not Detected Inside Container

```bash
# Check devices are bound
ls -la /dev/kfd /dev/dri/renderD128

# Check group membership
id

# Verify GID matches host
stat -c '%g' /dev/kfd
getent group render
```

If the GID doesn't match (993), recreate the group:

```bash
groupdel render 2>/dev/null || true
groupadd --force --gid 993 render
usermod -aG render <your-ssh-user>
# Log out and back in for group change to take effect
```

### Container Fails to Start

```bash
# Check journal
journalctl -u systemd-nspawn@llama-container -n 50

# Verify .nspawn config syntax
systemd-analyze verify /etc/systemd/nspawn/llama-container.nspawn

# Verify drop-in syntax
systemd-analyze verify /etc/systemd/system/systemd-nspawn@llama-container.service.d/*.conf

# Check rootfs exists
ls /var/lib/machines/llama-container/

# Re-run deploy to ensure all configs are in place
sudo ./deploy.sh
```

### SSH Connection Refused

```bash
# Check if container is running
sudo machinectl status llama-container

# Check SSH is running inside
sudo machinectl shell llama-container /usr/bin/systemctl status sshd

# Verify port configuration
sudo machinectl shell llama-container /usr/bin/grep '^Port ' /etc/ssh/sshd_config

# Check host firewall (if applicable)
sudo iptables -L -n | grep 2223
```

### Container Shuts Down Immediately

This usually means PID 1 exited. Common causes:

- No `Boot=yes` in the .nspawn config (systemd never started as PID 1)
- Missing `/etc/systemd/nspawn/llama-container.nspawn` or syntax error
- Rootfs incomplete (re-run `bootstrap.sh`)

## Teardown

To completely remove the container and all its configuration:

```bash
# Stop and disable
sudo machinectl stop llama-container
sudo machinectl disable llama-container

# Remove rootfs
sudo rm -rf /var/lib/machines/llama-container/

# Remove configs
sudo rm /etc/systemd/nspawn/llama-container.nspawn
sudo rm -rf /etc/systemd/system/systemd-nspawn@llama-container.service.d/

# Reload systemd
sudo systemctl daemon-reload
```

## GPU Device Reference

| Device | Path | Major:Minor | Purpose |
|--------|------|-------------|---------|
| KFD | `/dev/kfd` | 235:0 | Kernel Fusion Driver (ROCm compute) |
| Render Node | `/dev/dri/renderD128` | 226:128 | AMDGPU DRM render node |

**Group**: `render` with GID **993** on the llama-swap host.

## Related

- [`../system/`](../system/README.md) — Host-level systemd configuration for llama-swap
- [`../config.yaml`](../config.yaml) — Model definitions and routing
- [`../README.md`](../README.md) — Full system documentation

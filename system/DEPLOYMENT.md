# Deployment Guide

## Quick Start

```bash
# deploy the network binding override
sudo mkdir -p /etc/systemd/system/llama-swap.service.d/
sudo cp system/listen.conf /etc/systemd/system/llama-swap.service.d/
sudo systemctl daemon-reload
sudo systemctl restart llama-swap

# install GPU power management fix
sudo ./system/apply-gpu-power.sh
sudo systemctl restart llama-swap

# verify
systemctl status llama-swap --no-pager
ss -tlnp | grep 12434
curl -s http://localhost:12434/v1/models | python3 -m json.tool
```

## Detailed Steps

### 1. Prepare Drop-in Directory

```bash
sudo mkdir -p /etc/systemd/system/llama-swap.service.d/
ls -la /etc/systemd/system/llama-swap.service.d/
```

### 2. Deploy Network Override

**Option A: Copy (stable, recommended for production)**
```bash
sudo cp system/listen.conf /etc/systemd/system/llama-swap.service.d/
sudo chmod 644 /etc/systemd/system/llama-swap.service.d/listen.conf
```

**Option B: Symlink (auto-updates with repo)**
```bash
sudo ln -sf "$(pwd)/system/listen.conf" /etc/systemd/system/llama-swap.service.d/listen.conf
```

| | Copy | Symlink |
|---|---|---|
| Survives repo moves | ✓ | ✗ |
| Auto-updates | ✗ | ✓ |
| Works with package upgrades | ✓ | ✓ |

### 3. Deploy GPU Power Fix

```bash
sudo ./system/apply-gpu-power.sh
```

This installs the udev rule and systemd drop-in in one shot. See [`README.md`](README.md) for what each layer does.

### 4. Reload and Restart

```bash
sudo systemctl daemon-reload
sudo systemctl restart llama-swap
```

### 5. Verify

```bash
# service is active
systemctl is-active llama-swap

# listening on LAN IP, not localhost
ss -tlnp | grep 12434

# API is reachable
curl -s http://localhost:12434/v1/models | head -5

# GPU is in D0
cat /sys/class/drm/card0/device/power_state   # expects: D0

# recent logs look clean
journalctl -u llama-swap -n 20 --no-pager
```

## Rollback

### Network Override

```bash
sudo rm /etc/systemd/system/llama-swap.service.d/listen.conf
sudo systemctl daemon-reload
sudo systemctl restart llama-swap
# service falls back to 127.0.0.1:12434
```

### GPU Power Fix

```bash
sudo rm /etc/udev/rules.d/99-amdgpu-power.rules
sudo rm /etc/systemd/system/llama-swap.service.d/gpu-power.conf
sudo udevadm control --reload-rules
sudo systemctl daemon-reload
sudo systemctl restart llama-swap
echo auto > /sys/class/drm/card0/device/power/control
```

## Configuration Validation

Before reloading after a config change:

```bash
# YAML syntax
python3 -c "import yaml; yaml.safe_load(open('/etc/llama-swap/config.yaml'))"

# systemd override syntax
sudo systemd-analyze verify /etc/systemd/system/llama-swap.service.d/listen.conf

# effective unit configuration
systemctl cat llama-swap
```

## Troubleshooting Checklist

- [ ] Drop-in file exists in `/etc/systemd/system/llama-swap.service.d/`
- [ ] File permissions are 644
- [ ] `systemctl daemon-reload` was run after any file change
- [ ] Service is active: `systemctl is-active llama-swap`
- [ ] Listening on expected IP: `ss -tlnp | grep 12434`
- [ ] No port conflicts: `sudo lsof -i :12434`
- [ ] Config YAML is valid
- [ ] Model GGUF files exist at paths specified in config.yaml
- [ ] Firewall allows port 12434 from expected client IPs

## Common Commands

```bash
# service
sudo systemctl start|stop|restart llama-swap
systemctl status llama-swap
systemctl cat llama-swap

# logs
journalctl -u llama-swap -n 100
journalctl -u llama-swap -f
journalctl -u llama-swap --since "1 hour ago"

# network
ss -tlnp | grep -E '12434|808[0-9]'
curl -s http://localhost:12434/v1/models

# GPU
cat /sys/class/drm/card0/device/power_state
cat /sys/class/drm/card0/device/power/control
rocm-smi --showmeminfo vram

# maintenance
sudo systemctl daemon-reload
sudo systemctl reset-failed llama-swap
sudo systemctl enable llama-swap
```

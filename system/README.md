# Systemd Configuration

This directory contains systemd drop-in overrides and supporting files for the llama-swap service.

## File Structure

```
system/
├── README.md                   # this file
├── listen.conf                 # override: bind to LAN IP instead of localhost
├── 99-amdgpu-power.rules       # udev rule: disable GPU D3hot at boot
└── apply-gpu-power.sh          # one-shot script: install all GPU power layers

Active locations (after deployment):
/etc/systemd/system/llama-swap.service.d/
├── listen.conf                 # network binding override
└── gpu-power.conf              # GPU power management (ExecStartPre/StopPost)

/etc/udev/rules.d/
└── 99-amdgpu-power.rules       # boot-time GPU power control

/usr/lib/systemd/system/
└── llama-swap.service          # base unit (package-managed, do not edit)
```

## Network Binding

The base unit file (package-managed) binds to `127.0.0.1:12434`. The `listen.conf` drop-in overrides this to a LAN IP so that remote clients — Open WebUI, mobile frontends, other VMs — can reach the inference API directly.

```ini
[Service]
ExecStart=
ExecStart=/usr/bin/llama-swap -config /etc/llama-swap/config.yaml -watch-config -listen <LAN_IP>:12434
```

The empty `ExecStart=` clears the base unit's value before setting the new one. This is required for ExecStart overrides; without it, systemd appends rather than replaces.

`-watch-config` enables hot-reload when `config.yaml` changes. You can add or remove model entries without restarting the service — only global setting or macro changes require a restart.

**Security:** bind to a specific LAN IP rather than `0.0.0.0`, and restrict port 12434 to known client addresses via firewall rules. llama-swap has no built-in authentication.

### Deployment

```bash
sudo mkdir -p /etc/systemd/system/llama-swap.service.d/

# copy (stable; manual update required when listen.conf changes)
sudo cp system/listen.conf /etc/systemd/system/llama-swap.service.d/

# or symlink (auto-updates with repo; breaks if path changes)
sudo ln -sf "$(pwd)/system/listen.conf" /etc/systemd/system/llama-swap.service.d/listen.conf

sudo systemctl daemon-reload
sudo systemctl restart llama-swap
```

Verify:
```bash
systemctl cat llama-swap          # should show both base unit and drop-in
ss -tlnp | grep 12434             # should show <LAN_IP>:12434, not 127.0.0.1
```

### Rollback

```bash
sudo rm /etc/systemd/system/llama-swap.service.d/listen.conf
sudo systemctl daemon-reload
sudo systemctl restart llama-swap
# service falls back to 127.0.0.1:12434
```

## GPU Power Management

### The Problem

Linux runtime power management suspends the AMD GPU to D3hot (a PCI sleep state) after extended idle periods. When the llama-swap TTL fires and a model unloads, the GPU goes idle and enters D3hot. On the next inference request:

1. GPU wakes from D3hot
2. Memory clock ramps from 96 MHz back to 1249 MHz (hardware maximum)
3. Ramp-up takes 2–5 seconds

During ramp-up, early-response tokens generate well below the hardware ceiling. The latency isn't just inconvenient — it distorts throughput measurements if you're benchmarking at model load time rather than measuring steady-state.

### Root Cause Data

Characterization benchmark (Qwen3.5-27B Q4_K_M, `-ngl 999 -fa 1 -ctk q8_0 -ctv q8_0`):

| Measurement | Value |
|---|---|
| tg128 throughput ceiling | 30.44 t/s |
| mclk at idle (D3hot) | 96 MHz |
| mclk during inference | 1249 MHz (hardware max, level 3 of 4) |
| Memory bandwidth efficiency | ~52% of 960 GB/s theoretical |
| PCIe link | 4.0 x16 — not a bottleneck |
| GTT usage during inference | ~21 MB — no VRAM spill |

The 30 t/s ceiling is the practical hardware limit at these settings. Improving beyond it requires upstream ROCm/llama.cpp kernel optimization. Setting `power_dpm_force_performance_level=high` was tested and performs slightly worse than `auto` — the driver manages clocks better within D0 when left to its own judgment.

### The Fix

Three layers, installed together by `apply-gpu-power.sh`:

**Layer 1 — Immediate (sysfs write):**
```bash
echo on > /sys/class/drm/card0/device/power/control
```
Takes effect immediately. Lost on reboot without the udev rule.

**Layer 2 — Boot-persistent (udev rule):**
`99-amdgpu-power.rules` matches any `amdgpu`-bound PCI device and writes `on` to its power control node at device probe time. The GPU starts in D0 before any service touches it.

**Layer 3 — Service-lifecycle (systemd drop-in):**
The `gpu-power.conf` drop-in uses `ExecStartPre`/`ExecStopPost` with the `+` prefix (grants root privileges, bypassing `DynamicUser=yes`) to force D0 when llama-swap starts and restore `auto` when it stops:

```ini
[Service]
ExecStartPre=+/bin/sh -c 'echo on > /sys/class/drm/card0/device/power/control'
ExecStopPost=+/bin/sh -c 'echo auto > /sys/class/drm/card0/device/power/control'
```

The `auto` restore on stop means the GPU can still suspend during periods when the inference service isn't running — this fix doesn't permanently disable power management system-wide.

**Effect on other GPU workloads:** none. During active gaming or rendering the GPU is always in D0; D3hot only occurs during extended idle. The `power_dpm_force_performance_level` stays at `auto`, so clock management under load is unchanged.

### Installing

```bash
sudo ./system/apply-gpu-power.sh
sudo systemctl restart llama-swap
```

### Verification

```bash
cat /sys/class/drm/card0/device/power/control   # should read: on
cat /sys/class/drm/card0/device/power_state      # should read: D0
journalctl -u llama-swap -n 5                   # check ExecStartPre ran
```

## Service Reference

### Base Unit Settings

| Directive | Value | Effect |
|---|---|---|
| `DynamicUser` | `yes` | Ephemeral unprivileged user per start |
| `StateDirectory` | `%p` | Auto-creates `/var/lib/llama-swap/` |
| `WorkingDirectory` | `%S/%p` | CWD for spawned model processes |
| `Restart` | `on-failure` | Auto-restart on non-zero exit |
| `RestartSec` | `3` | Delay before restart attempt |
| `StartLimitBurst` | `3` | Max 3 restarts within 30 s window |

### Port Allocation

| Port | Purpose |
|---|---|
| `12434` | llama-swap proxy (configurable via `-listen`) |
| `8081+` | Model instance backends (internal, not externally exposed) |

Model backend ports are bound to the proxy host and only accessible via llama-swap. Clients always talk to port 12434.

### Common Operations

```bash
# service management
sudo systemctl start llama-swap
sudo systemctl stop llama-swap
sudo systemctl restart llama-swap

# status
systemctl status llama-swap
systemctl cat llama-swap         # show effective configuration including overrides

# logs
journalctl -u llama-swap -f
journalctl -u llama-swap -n 50

# verify after config change
curl -s http://localhost:12434/v1/models | python3 -m json.tool
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Service binds to 127.0.0.1 | Drop-in not loaded | `systemctl daemon-reload`, check drop-in file exists |
| Health check timeout at startup | Large model taking >180 s to load | Increase `healthCheckTimeout` in config.yaml |
| Service enters failed state | Hit restart limit | `systemctl reset-failed llama-swap`, check logs |
| Models not appearing in /v1/models | Config syntax error | `python3 -c "import yaml; yaml.safe_load(open('config.yaml'))"` |
| GPU stays in D3hot | udev rule not loaded or wrong card path | Check `ls /sys/class/drm/` for actual card device name |

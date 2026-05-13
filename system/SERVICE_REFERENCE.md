# Service Reference

## Unit File Overview

| File | Location | Managed by |
|---|---|---|
| Base unit | `/usr/lib/systemd/system/llama-swap.service` | Package manager (do not edit) |
| Drop-in directory | `/etc/systemd/system/llama-swap.service.d/` | This repo |
| Network override | `listen.conf` | This repo |

## Base Unit (Key Directives)

```ini
[Service]
DynamicUser=yes
StateDirectory=%p
WorkingDirectory=%S/%p
Environment=HOME=%S/%p
Restart=on-failure
RestartSec=3
StartLimitBurst=3
StartLimitInterval=30
ExecStart=/usr/bin/llama-swap -config %E/llama-swap/config.yaml -watch-config -listen 127.0.0.1:12434

[Install]
WantedBy=multi-user.target
```

## Directive Reference

| Directive | Value | Notes |
|---|---|---|
| `DynamicUser` | `yes` | Ephemeral unprivileged user; UID/GID assigned at runtime |
| `StateDirectory` | `%p` → `llama-swap` | Auto-creates `/var/lib/llama-swap/`; owned by the dynamic user |
| `WorkingDirectory` | `%S/%p` → `/var/lib/llama-swap` | CWD for the process and any spawned model backends |
| `Environment=HOME` | `%S/%p` | Sets HOME for the dynamic user; model backends inherit it |
| `Restart` | `on-failure` | Auto-restart on non-zero exit or signal kill |
| `RestartSec` | `3` | 3-second delay before restart attempt |
| `StartLimitBurst` | `3` | Service enters `failed` state after 3 restarts in 30 s |
| `StartLimitInterval` | `30` | Window for the burst limit |

## Systemd Variable Expansion

| Variable | Expands to | Example |
|---|---|---|
| `%p` | Service name (without `.service`) | `llama-swap` |
| `%S` | State directory root | `/var/lib` |
| `%E` | Configuration root | `/etc` |

## ExecStart Flags

| Flag | Default | Description |
|---|---|---|
| `-config` | `/etc/llama-swap/config.yaml` | Model routing configuration |
| `-watch-config` | (flag) | Hot-reload on config.yaml changes; no restart needed for model add/remove |
| `-listen` | `127.0.0.1:12434` | Bind address; overridden by `listen.conf` drop-in |

## Port Allocation

| Port | Purpose | Exposed |
|---|---|---|
| `12434` | llama-swap control/proxy | Yes (LAN, via override) |
| `8081+` | Model instance backends | No (localhost only, proxied) |

Port range for backends is set by `startPort: 8081` in `config.yaml`. Each loaded model gets its own port, incrementing by 1. Clients never talk to these ports directly.

## Directory Structure

```
/var/lib/llama-swap/        # runtime state (auto-created by StateDirectory)
/etc/llama-swap/            # configuration root
├── config.yaml             # model routing
└── system/                 # systemd templates (this directory)
/opt/llama/models/          # GGUF model files
```

## Restart Behavior

The service auto-restarts on failure with a 3-second delay. After 3 restarts within 30 seconds, it enters `failed` state and stops retrying. To clear the failed state:

```bash
sudo systemctl reset-failed llama-swap
sudo systemctl start llama-swap
```

## Logging

All llama-swap and model backend output goes to the system journal (`logToStdout: "both"` in config.yaml):

```bash
journalctl -u llama-swap -f           # follow
journalctl -u llama-swap -n 50        # recent
journalctl -u llama-swap -b           # since last boot
```

## Security Notes

- **DynamicUser:** The process runs as an ephemeral unprivileged user. The GPU power management drop-in uses `ExecStartPre=+` (the `+` grants root) to write sysfs nodes that the dynamic user can't reach.
- **Model ports:** Backend ports (8081+) are bound to the proxy host and only accessible via the llama-swap proxy. They are not externally exposed.
- **Network:** Bind to a specific LAN IP in `listen.conf`, not `0.0.0.0`. Pair with firewall rules.
- **Config directory:** `/etc/llama-swap/` should be owned by the user managing the service, not the dynamic user.

## Performance Options

Add to the `[Service]` section via a drop-in if needed:

```ini
# Thread count (llama-server inherits)
Environment="LLAMA_CPP_THREADS=8"

# Memory cap (prevents GTT spill from consuming all system RAM)
MemoryMax=32G
```

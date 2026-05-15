# Systemd Deployment Guide for llama-swap

## Quick Start

```bash
# Deploy the override
sudo cp /etc/llama-swap/system/listen.conf /etc/systemd/system/llama-swap.service.d/

# Apply changes
sudo systemctl daemon-reload
sudo systemctl restart llama-swap

# Verify
systemctl status llama-swap --no-pager
ss -tlnp | grep 12434
```

## Detailed Deployment Steps

### Step 1: Prepare Systemd Directory

```bash
# Ensure drop-in directory exists
sudo mkdir -p /etc/systemd/system/llama-swap.service.d/

# Verify permissions
ls -la /etc/systemd/system/llama-swap.service.d/
```

### Step 2: Deploy Override File

**Option A: Copy (Production)**
```bash
sudo cp /etc/llama-swap/system/listen.conf /etc/systemd/system/llama-swap.service.d/
sudo chmod 644 /etc/systemd/system/llama-swap.service.d/listen.conf
```

**Option B: Symlink (Development)**
```bash
sudo ln -sf /etc/llama-swap/system/listen.conf /etc/systemd/system/llama-swap.service.d/listen.conf
```

### Step 3: Reload Systemd

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Verify override is loaded
systemctl cat llama-swap | grep -A5 "Listen Address"
```

### Step 4: Restart Service

```bash
# Stop current service
sudo systemctl stop llama-swap

# Start with new configuration
sudo systemctl start llama-swap

# Or restart in one command
sudo systemctl restart llama-swap
```

### Step 5: Verify Deployment

```bash
# Check service status
systemctl is-active llama-swap
# Expected: active

# Check listening address
ss -tlnp | grep 12434
# Expected: <host-ip>:12434

# Test endpoint
curl -s http://<host-ip>:12434/models | head -5

# Check logs
journalctl -u llama-swap -n 10 --no-pager
```

## Rollback Procedure

```bash
# Backup current override (if exists)
sudo cp /etc/systemd/system/llama-swap.service.d/listen.conf \
        /etc/systemd/system/llama-swap.service.d/listen.conf.backup

# Remove override
sudo rm /etc/systemd/system/llama-swap.service.d/listen.conf

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart llama-swap

# Verify reverted to defaults
ss -tlnp | grep 12434
# Expected: 127.0.0.1:12434 (localhost only)
```

## Monitoring and Maintenance

### Enable Auto-Start on Boot

```bash
sudo systemctl enable llama-swap
sudo systemctl is-enabled llama-swap
# Expected: enabled
```

### View Service Logs

```bash
# Recent logs
journalctl -u llama-swap -n 50

# Follow logs in real-time
journalctl -u llama-swap -f

# Logs since service start
journalctl -u llama-swap -b

# Logs from specific time
journalctl -u llama-swap --since "2026-04-13 08:00:00"
```

### Check Resource Usage

```bash
# Memory usage
systemctl show llama-swap | grep Memory

# Process list
ps aux | grep llama-swap

# Network connections
ss -tlnp | grep -E '12434|808[0-9]'
```

## Configuration Validation

### Validate Override Syntax

```bash
# Check for syntax errors
sudo systemd-analyze verify /etc/systemd/system/llama-swap.service.d/listen.conf

# Dry-run verification
sudo systemd-analyze verify llama-swap.service
```

### Validate Config File

```bash
# Check YAML syntax
python3 -c "import yaml; yaml.safe_load(open('/etc/llama-swap/config.yaml'))"

# Check model paths exist
grep "gguf" /etc/llama-swap/config.yaml | while read line; do
  model=$(echo $line | grep -oP '(?<=model: ).*')
  if [ -f "$model" ]; then
    echo "✓ $model"
  else
    echo "✗ $model (NOT FOUND)"
  fi
done
```

## Troubleshooting Checklist

- [ ] Override file exists in `/etc/systemd/system/llama-swap.service.d/`
- [ ] File permissions are 644
- [ ] `systemctl daemon-reload` was executed
- [ ] Service is active: `systemctl is-active llama-swap`
- [ ] Listening on correct IP: `ss -tlnp | grep 12434`
- [ ] No port conflicts: `sudo lsof -i :12434`
- [ ] Config file is valid YAML
- [ ] Model files exist at specified paths
- [ ] Firewall allows port 12434 (if applicable)

## Common Commands Reference

```bash
# Service management
sudo systemctl start llama-swap
sudo systemctl stop llama-swap
sudo systemctl restart llama-swap
sudo systemctl reload llama-swap

# Status and information
systemctl status llama-swap
systemctl is-active llama-swap
systemctl is-enabled llama-swap
systemctl show llama-swap

# Configuration
systemctl cat llama-swap
sudo systemctl edit llama-swap  # Temporary override

# Logs
journalctl -u llama-swap -n 100
journalctl -u llama-swap -f
journalctl -u llama-swap --since "1 hour ago"

# Network
ss -tlnp | grep 12434
curl http://127.0.0.1:12434/models
curl http://<host-ip>:12434/health

# Maintenance
sudo systemctl daemon-reload
sudo systemctl reset-failed llama-swap
sudo systemctl enable llama-swap
```

## Production Checklist

Before deploying to production:

- [ ] Override file tested in staging
- [ ] Backup of current configuration created
- [ ] Maintenance window scheduled (if needed)
- [ ] Monitoring and alerting configured
- [ ] Firewall rules updated
- [ ] Client applications configured for new endpoint
- [ ] Rollback procedure documented and tested
- [ ] Team notified of changes

## Support

For issues:
1. Check logs: `journalctl -u llama-swap -n 50`
2. Verify configuration: `systemctl cat llama-swap`
3. Review documentation: `/etc/llama-swap/system/README.md`
4. Check GitHub issues: https://github.com/<USER>/llama-swap

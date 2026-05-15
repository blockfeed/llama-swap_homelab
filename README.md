# llama-swap homelab

Configuration files for a multi-model LLM inference server on a single consumer GPU — twenty-plus model variants (instruct, thinking, vision, agent) over a local OpenAI-compatible API, built on [llama-swap](https://github.com/mostlygeek/llama-swap). The intent is to document and version-control one practical approach: an AMD Radeon RX 7900 XTX, 24 GB of VRAM, models tuned to fit within it one at a time without latency from reloads.

That meant working through KV cache quantization, per-model context sizing, GPU power management, and MTP speculative decoding. MTP models depend on a patched llama-server — Arch Linux PKGBUILD at [blockfeed/llama-cpp-mtp-hip](https://github.com/blockfeed/llama-cpp-mtp-hip).

---

## Hardware & Stack

| | |
|---|---|
| GPU | AMD Radeon RX 7900 XTX (RDNA3, gfx1100) |
| VRAM | 24 GB GDDR6 |
| Compute | ROCm 7.2.1, HIP backend |
| llama.cpp | MTP-patched build ([am17an/mtp-clean](https://github.com/am17an/llama.cpp/tree/mtp-clean), see companion package below) |
| Orchestrator | [llama-swap](https://github.com/mostlygeek/llama-swap) |
| OS | Arch Linux, Zen kernel |

The llama.cpp binary is a custom Arch Linux package built from the MTP speculative decoding branch. The PKGBUILD and prebuilt package live at [blockfeed/llama-cpp-mtp-hip](https://github.com/blockfeed/llama-cpp-mtp-hip). Everything else is stock AUR/pacman packages.

---

## Architecture

Each model entry in `config.yaml` defines a command to launch a llama-server backend. llama-swap starts it on first request, routes traffic, and tears it down after an idle TTL. Only one large model fits in VRAM at a time — demand-paged model scheduling with a shared GPU.

The service is managed by systemd with a drop-in override to bind to the LAN rather than localhost — details in [`system/`](system/README.md).

### Macro System

Configuration is built around a macro system that centralizes sampling parameters and server flags. Each macro expands to a complete llama-server command via `${macro_name}` substitution, keeping model entries short and consistent:

```yaml
macros:
  "base": >
    /usr/bin/llama-server
    --n-gpu-layers 999
    --cache-type-k q4_0
    --cache-type-v q4_0
    --flash-attn on
    ...

  "qwen36_agent": >
    ${base}
    --reasoning off
    --temp 0.7
    --top-p 0.8
    ...
```

One constraint worth knowing: llama-swap silently ignores `--ctx-size` when it appears in a macro that itself references another macro. To avoid this, macros only ever reference `${base}` — one level deep, no deeper. Context size always goes in the model entry itself.

### Multi-Variant Models

The Qwen3.6-35B-A3B model runs five behavioral profiles (agent, code, casual, heavy thinking, planning) from a **single llama-server process**. Separate model entries would force a full server restart on every profile switch (~5.5 s reload). The `setParamsByID` filter injects different sampling parameters and template arguments per request without restarting:

```yaml
"qwen36-35b-a3b":
  cmd: >
    ${qwen36_agent}
    --model /opt/llama/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
    --ctx-size 102400
  filters:
    setParamsByID:
      "qwen36-agent":
        temperature: 0.7
        chat_template_kwargs: {enable_thinking: false, preserve_thinking: true}
      "qwen36-chat":
        temperature: 1.0
        chat_template_kwargs: {enable_thinking: true, preserve_thinking: true}
```

The server starts with thinking disabled and agent-optimized parameters. A request tagged `qwen36-chat` gets thinking enabled and temperature raised to 1.0 without the model ever unloading.

The canonical entry is `qwen36-35b-a3b`; all the above are aliases. `enable_thinking` in `setParamsByID` overrides the server-level `--reasoning off` default per-request.

| Alias | Thinking | Temp | Use case |
|---|---|---|---|
| `qwen36-agent` / `hermes-agent` | OFF | 0.7 | Tool-calling, agentic workflows |
| `qwen36-code` / `agent-primary` | OFF | 0.6 | Precise coding |
| `qwen36-paseo` / `paseo` | OFF | 0.7 | Casual interactive (no preserve_thinking overhead) |
| `qwen36-chat` | ON | 1.0 | Research, ideas, heavy reasoning |
| `qwen36-plan` | ON | 1.0 | Planning, design discussion |
| `qwen36-35b-a3b-mtp-*` / `hermes-agent` / `agent-primary` / `qwen36-agent` / `qwen36-code` / `qwen36-chat` / `qwen36-plan` / `qwen36-paseo` / `paseo` | OFF→ON | 0.6–1.0 | Same aliases via `setParamsByID` (single MTP process) |


---

## VRAM Budgeting

The hard limit is **93% VRAM utilization**. Above that, ROCm begins spilling into GTT (system RAM accessed over PCIe), which tanks throughput and eventually causes segfaults in the HIP runtime. On a 24 GB card, the working ceiling is roughly 22.3 GB.

Two things keep the budget manageable across a wide range of models:

**KV cache quantization.** The base macro forces `--cache-type-k q4_0 --cache-type-v q4_0` on every model. This quantizes the attention key/value cache from fp16 to 4-bit, roughly halving its VRAM footprint. At 100k tokens of context, the KV cache in fp16 would consume 10–12 GB; q4_0 brings it under 4 GB for most model families with minimal quality impact at chat context lengths.

**Empirical context sizing.** Every model reports a native context ceiling in its metadata, but running at that ceiling isn't practical on 24 GB. The approach: load the model, run a few inference passes, check `rocm-smi --showmeminfo vram` under load, and drop context until there's comfortable headroom below the ceiling. The values in `config.yaml` are the results of that process.

| Model | Context | Approx. VRAM |
|---|---|---|
| Qwen3.6-35B-A3B (MoE, IQ4_XS) | 102,400 | ~20 GB |
| Qwen3.6-35B-A3B-MTP (MoE, IQ4_NL) | 65,536 | ~22 GB |
| Qwen3.6-27B (dense, Q4_K_XL) | 94,208 | ~22 GB |
| Qwen3.6-27B-MTP (dense, Q4_K_M) | 65,536 | ~22 GB |
| Mistral-Small-3.2-24B (Q4_K_XL) | 92,160 | ~20 GB |
| GLM-4.7-Flash-23B (MLA, Q4_K_XL) | 65,536 | ~15 GB |
| Qwen3.5-9B (Q5_K_XL) | 122,880 | ~14 GB |
| Rocinante-X-12B (Q8_0) | 16,384 | ~14 GB |

Only one model above ~15 GB fits in VRAM at a time. llama-swap evicts automatically on first request to a different model.

---

## Parameter Tuning

Sampling parameters are based on [Unsloth's published recommendations](https://unsloth.ai) for each model family, used as a starting point and adjusted for the specific use case.

### Qwen3.6 Modes

Qwen3.6 has a hybrid reasoning architecture with two operating modes, controlled by `--reasoning on/off`:

| Mode | Temp | top_p | top_k | presence_penalty | Use case |
|---|---|---|---|---|---|
| Non-thinking (agent) | 0.7 | 0.8 | 20 | 1.5 | tool-calling, structured output |
| Non-thinking (code) | 0.6 | 0.95 | 20 | 0.0 | precise coding, low-diversity output |
| Thinking (general) | 1.0 | 0.95 | 20 | 1.5 | research, planning, exploration |

The presence_penalty difference between code (0.0) and general (1.5) is intentional: coding tasks benefit from repetition (common patterns, boilerplate), while conversational and reasoning tasks benefit from topic diversity.

### Template Incompatibilities

Qwen3.5 and Qwen3.6 look similar but use incompatible Jinja2 chat templates for thinking mode:

- **Qwen3.5**: `--chat-template-kwargs '{"enable_thinking": true}'`
- **Qwen3.6**: `--reasoning on` with `--chat-template-kwargs '{"preserve_thinking": true}'`

Cross-applying either will appear to work but produce wrong behavior — usually thinking tokens leaking into the output or being silently dropped. Each model family has a dedicated macro to prevent this.

### GLM-4.7 (MLA Architecture)

GLM-4.7-Flash uses DeepSeek2 Multi-Linear Attention. Running with default `--parallel 4` collapses throughput from ~60 t/s to 2–8 t/s under concurrent load. `--parallel 1` is not a tuning choice here — it's a hard requirement for the attention implementation.

---

## MTP Speculative Decoding

[MTP (Multi-Token Prediction)](https://github.com/ggml-org/llama.cpp/pull/22673) is a speculative decoding mode where auxiliary "draft heads" — extra output layers trained alongside the main model — predict multiple future tokens per forward pass. The main model verifies each prediction and accepts or rejects it; accepted tokens cost no additional compute. Rejected tokens fall back to standard decoding for that step.

Unlike draft-model approaches (e.g. eagle3), MTP requires no second model in VRAM. The draft heads are baked into the GGUF. The tradeoff is needing a patched llama-server — the [MTP PR](https://github.com/ggml-org/llama.cpp/pull/22673) added this capability. An Arch Linux package built from that branch is available at [blockfeed/llama-cpp-mtp-hip](https://github.com/blockfeed/llama-cpp-mtp-hip). The standard llama-server binary does not support `--spec-type mtp`.

Flags:
```
--spec-type mtp        # enable MTP speculative decoding
--spec-draft-n-max 2   # draft up to 2 tokens per step (Unsloth recommendation)
--parallel 1           # hard requirement — MTP does not support concurrent slots
```

**Measured throughput (RX 7900 XTX, warm model, 200-token generation):**

| Model | Mode | Tokens/sec |
|---|---|---|
| Qwen3.6-35B-A3B-MTP (MoE) | agent (non-thinking) | 77.3 t/s |
| Qwen3.6-35B-A3B-MTP (MoE) | thinking | 81.4 t/s |
| Qwen3.6-27B-MTP (dense) | agent (non-thinking) | 31.8 t/s |
| Qwen3.6-27B-MTP (dense) | thinking | 37.2 t/s |

The MoE model is faster despite having more total parameters because it only activates ~3.5B parameters per token — the 35B figure is total expert weights, not active compute. The dense 27B activates all 27B on every step.

Thinking mode runs faster than agent mode on both models. Chain-of-thought tokens are more predictable than direct-answer output (common reasoning phrases, logical connectives, repeated structural patterns), so the draft heads achieve higher acceptance rates on reasoning traces. The effect is more pronounced on the 27B (+17%) than the 35B-A3B (+5%).

One note on the 27B MTP configuration: Unsloth's published parameters for this model recommend `presence_penalty=0.0` in thinking mode, which differs from the 35B-A3B recommendation of 1.5. Each model has a dedicated macro to keep this straight — see `MODEL-CATALOG.md` for the reasoning.

---

## GPU Power Management

Linux runtime power management suspends the AMD GPU to D3hot (a PCI sleep state) after an idle period. After a model unloads and the GPU sits idle, the memory clock ramps down from 1249 MHz to 96 MHz. The next inference request waits for clock ramp-up — 2–5 seconds during which early tokens generate at a fraction of the hardware ceiling.

The fix is three layers, all in the `system/` directory:

**Immediate (`apply-gpu-power.sh` layer 1):** writes `on` to the GPU's sysfs runtime PM control node, disabling D3hot right now.

**Boot-persistent (`99-amdgpu-power.rules`):** a udev rule applies the same write at device probe time, so the GPU starts in D0 after every reboot before llama-swap has a chance to start.

**Service-lifecycle (systemd drop-in `ExecStartPre`/`ExecStopPost`):** forces D0 when llama-swap starts, restores `auto` when it stops — meaning the GPU can still suspend during periods when the inference service isn't running.

```bash
sudo ./system/apply-gpu-power.sh   # installs all three layers
```

Some measurements that characterized the problem during investigation:

| | |
|---|---|
| Hardware throughput ceiling (Qwen3.5-27B Q4_K_M, tg128) | 30.44 t/s |
| mclk at idle (D3hot) | 96 MHz |
| mclk during inference | 1249 MHz (hardware max) |
| Memory bandwidth efficiency | ~52% of 960 GB/s theoretical |
| PCIe link | 4.0 x16 — not a bottleneck |

The 30 t/s ceiling is the practical hardware limit at those settings; the bandwidth efficiency figure is normal for ROCm on RDNA3. Setting `power_dpm_force_performance_level=high` was tested and slightly underperforms `auto` — the driver manages clocks better when left to its own judgment.

An additional optimization: capping the GPU's maximum shader clock (SCLK) to 2301 MHz instead of the ~2543 MHz boost clock. This trades roughly 3–4% peak throughput for a significant reduction in power draw and fan noise, with no meaningful impact on inference quality. This cap is applied via `ExecStartPre` in the systemd drop-in.

---

## Model Inventory

| Model ID | Architecture | Quant | Context | Role |
|---|---|---|---|---|
| `qwen36-agent` / `hermes-agent` | Qwen3.6-35B-A3B (MoE) | IQ4_XS | 102,400 | agent |
| `qwen36-code` / `agent-primary` | Qwen3.6-35B-A3B (MoE) | IQ4_XS | 102,400 | agent |
| `qwen36-paseo` | Qwen3.6-35B-A3B (MoE) | IQ4_XS | 102,400 | user |
| `qwen36-chat` | Qwen3.6-35B-A3B (MoE) | IQ4_XS | 102,400 | user |
| `qwen36-plan` | Qwen3.6-35B-A3B (MoE) | IQ4_XS | 102,400 | user |
| `qwen36-35b-a3b-mtp-*` | Qwen3.6-35B-A3B-MTP (MoE) | IQ4_NL | 65,536 | agent/user |
| `qwen36-27b-mtp-agent` | Qwen3.6-27B-MTP (dense) | Q4_K_M | 65,536 | agent |
| `qwen36-27b-mtp-reasoning` | Qwen3.6-27B-MTP (dense) | Q4_K_M | 65,536 | user |
| `qwen36-27b-reasoning` | Qwen3.6-27B (dense) | Q4_K_XL | 94,208 | user |
| `agent-mistral` | Mistral-Small-3.2-24B | Q4_K_XL | 92,160 | agent |
| `glm47-agent` | GLM-4.7-Flash-23B (MLA) | Q4_K_XL | 65,536 | agent |
| `glm47-thinking` | GLM-4.7-Flash-23B (MLA) | Q4_K_XL | 65,536 | user |
| `qwen35-9b` | Qwen3.5-9B | Q5_K_XL | 122,880 | agent |
| `qwen35-27b-abliterated` | Qwen3.5-27B (MoE) | Q4_K_M | 90,112 | user |
| `rocinante-x-12b` | Rocinante-X-12B (Nemo) | Q8_0 | 16,384 | user |
| `grape2-thinking` | GRaPE-2-Pro (Qwen3.5 base) | Q4_K_M | 65,536 | user |
| `qwen36-vision-agent` | Qwen3.6-35B-A3B + mmproj | IQ4_XS | 102,400 | agent |
| `thinking-primary-vision` | Qwen3.6-27B + mmproj | Q4_K_XL | 94,208 | user |
| `agent-primary-vision` | Qwen3.6-35B-A3B + mmproj | IQ4_XS | 102,400 | agent |
| `qwen3vl-30b` | Qwen3-VL-30B-A3B | Q4_K_M | 65,536 | user |
| `qwen2vl-7b` | Qwen2-VL-7B | Q6_K_L | 65,536 | user |

The first five model IDs (qwen36-agent through qwen36-plan) are aliases on a single llama-server process via `setParamsByID`. The `qwen36-35b-a3b-mtp-*` entries share the same mechanism — the MTP model runs on a single process with the same set of aliases, each getting different sampling parameters injected per-request. The rest are independent processes that evict each other from VRAM on demand.

---

## Container Setup

The `container/` directory contains a systemd-nspawn setup for isolating
llama-swap inference tooling (benchmarking, test harnesses, custom development)
from the host. Includes GPU passthrough (RX 7900 XTX), host filesystem bind
mounts, and bootstrap/deploy scripts. See
[container/README.md](container/README.md) for the architecture, setup guide,
and troubleshooting.

| File | Purpose |
|---|---|
| `bootstrap.sh` | First-time container provisioning (packages, users, sshd) |
| `deploy.sh` | Deploys llama-swap config into the container |
| `gpu.conf` | systemd tmpfiles.d entry for GPU device permissions |
| `llama-container.nspawn` | systemd-nspawn unit definition |
| `llama-swap-dep.conf` | Systemd service dependency wiring |

---

## Known Issues

### MTP crash under concurrent requests: `recurrent state read/write is not supported with partial rollback`

**Affects:** All MTP models (`Qwen3.6-35B-A3B-MTP`, `Qwen3.6-27B-MTP`)

**Symptom:** llama-server aborts mid-generation when a second request arrives while an MTP model is actively processing a prompt.

**Root cause:** The [am17an/mtp-clean](https://github.com/am17an/mtp-clean) branch implements MTP draft heads using `llama_memory_recurrent`. When a concurrent request triggers `server_slot::prompt_save` (for KV prefix reuse), it hits `llama_memory_recurrent::state_write`, which unconditionally aborts:

```
GGML_ABORT("recurrent state read/write is not supported with partial rollback");
```

The recurrent memory used for draft heads cannot be serialized. This is a missing feature in the upstream MTP branch, not a build or configuration problem.

**Fix:** Add `--no-cache-prompt` to both MTP model definitions in `config.yaml`. This prevents the server from ever calling `prompt_save`.

**Trade-off:** Disabling prompt caching means repeated requests with the same system prompt or prefix won't get KV reuse across turns. Given `--parallel 1` and MTP's throughput advantage (~1.5–2x from speculative decoding), this is the correct trade-off.

**Upstream tracking:** The bug lives in `llama_memory_recurrent::state_write` in the [am17an/mtp-clean](https://github.com/am17an/mtp-clean) branch. Watch that repo for a fix if you want to re-enable prompt caching.

---

## Operational Notes

Validate config syntax before reloading:
```bash
python3 -c "import yaml; yaml.safe_load(open('config.yaml'))"
sudo systemctl restart llama-swap
curl -s http://localhost:12434/v1/models | python3 -m json.tool
```

Check VRAM under load (idle readings are misleading — KV cache allocates on first request):
```bash
rocm-smi --showmeminfo vram
```

Diagnose GPU power state:
```bash
cat /sys/class/drm/card0/device/power_state    # D0 = active, D3hot = suspended
cat /sys/class/drm/card0/device/power/control  # "on" = runtime PM disabled
```

The `healthCheckTimeout` in `config.yaml` is set to 180 s. Large MoE models take 60–90 s to load all layers on first request; the default 30 s timeout causes false-positive health check failures before the model is ready.

# llama-swap homelab

Configuration for a multi-model LLM inference server on a single consumer GPU — five model families (instruct, thinking, vision, agent) over a local OpenAI-compatible API, built on [llama-swap](https://github.com/mostlygeek/llama-swap). The intent is to document one practical approach: an AMD Radeon RX 7900 XTX, 24 GB of VRAM, models tuned to fit within it one at a time without latency from reloads.

All primary inference models use MTP (Multi-Token Prediction) speculative decoding where supported. That meant working through KV cache quantization, context sizing methodology, and the constraints of two different MTP implementations — Qwen3.6 uses internal draft heads baked into the GGUF (MTP mainline since b9180); Gemma 4 uses an external draft model (PR #23398 merged into mainline). Both run on the standard system llama-server.

---

## Hardware & Stack

| | |
|---|---|
| GPU | AMD Radeon RX 7900 XTX (RDNA3, gfx1100) |
| VRAM | 24 GB GDDR6 |
| Compute | ROCm 7.2.1, HIP backend |
| llama.cpp | Standard build; all models use `/usr/bin/llama-server` |
| Orchestrator | [llama-swap](https://github.com/mostlygeek/llama-swap) |
| OS | Arch Linux, Zen kernel |

---

## Architecture

Each model entry in `config.yaml` defines a command to launch a llama-server backend. llama-swap starts it on first request, routes traffic, and tears it down after an idle TTL. Only one large model fits in VRAM at a time — demand-paged model scheduling with a shared GPU.

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

  "qwen36_mtp_agent": >
    ${base}
    --parallel 1
    --cache-ram 0
    --spec-type draft-mtp
    --spec-draft-n-max 2
    ...
```

One constraint worth knowing: llama-swap silently ignores `--ctx-size` when it appears in a macro that itself references another macro. To avoid this, macros only ever reference `${base}` — one level deep, no deeper. Context size always goes in the model entry itself.

The `gemma4_mtp` macro is an exception: it is self-contained and does not reference `${base}` at all. This is required because Gemma 4 MTP uses a different binary from the system llama-server. Referencing `${base}` would pull in the wrong binary path.

### Multi-Variant Models

The Qwen3.6-35B-A3B-MTP model runs two behavioral profiles (agent and planning/thinking) from a **single llama-server process**. Separate model entries would force a full server restart on every profile switch (~5.5 s reload). The `setParamsByID` filter injects different sampling parameters and template arguments per request without restarting:

```yaml
"qwen36-35b-a3b-mtp":
  cmd: >
    ${qwen36_mtp_agent}
    --model /opt/llama/models/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
    --ctx-size 262144
  filters:
    setParamsByID:
      "qwen36-code":
        temperature: 0.6
        chat_template_kwargs: {enable_thinking: false, preserve_thinking: true}
      "qwen36-plan":
        temperature: 1.0
        chat_template_kwargs: {enable_thinking: true, preserve_thinking: true}
```

The Gemma 4 model uses the same mechanism for its `gemma4-code` and `gemma4-plan` aliases.

| Model | Alias | Thinking | Temp | Use case |
|---|---|---|---|---|
| `qwen36-35b-a3b-mtp` | `qwen36-code` | OFF | 0.6 | Precise coding |
| `qwen36-35b-a3b-mtp` | `qwen36-plan` | ON | 1.0 | Planning, research |
| `gemma4-26b-a4b-mtp` | `gemma4-code` | OFF | 1.0 | Agent/tool-calling |
| `gemma4-26b-a4b-mtp` | `gemma4-plan` | ON | 1.0 | Reasoning, planning |

---

## VRAM Budgeting

The hard limit is **93% VRAM utilization**. Above that, ROCm begins spilling into GTT (system RAM accessed over PCIe), which tanks throughput and eventually causes segfaults in the HIP runtime. On a 24 GB card, the working ceiling is roughly 22.3 GB.

Two things keep the budget manageable:

**KV cache quantization.** The base macro forces `--cache-type-k q4_0 --cache-type-v q4_0` on every model. This quantizes the attention key/value cache from fp16 to 4-bit, roughly halving its VRAM footprint. At 100k tokens of context, the KV cache in fp16 would consume 10–12 GB; q4_0 brings it under 4 GB for most model families with minimal quality impact at chat context lengths.

**Empirical context sizing.** Every model reports a native context ceiling in its metadata, but running at that ceiling isn't practical on 24 GB. The approach: load the model, run inference, check `rocm-smi --showmeminfo vram` under load, and drop context until there's comfortable headroom below the ceiling. The values in `config.yaml` are the results of that process — see [Context Sizing](#context-sizing) below.

| Model | Context | Approx. VRAM |
|---|---|---|
| Qwen3.6-35B-A3B-MTP (MoE, IQ4_NL) | 262,144 | ~21.6 GB |
| Gemma 4 26B-A4B-MTP (MoE, Q4_K_XL + Q2_K draft) | 262,144 | ~22 GB est. |
| Qwen3.6-35B-A3B Vision (MoE, IQ4_XS + F16 mmproj) | 262,144 | ~22 GB est. |
| Huihui 35B Abliterated (MoE, Q4_K_M) | 245,760 | ~21.5 GB |
| Rocinante-X-12B (Nemo, Q8_0) | 188,416 | ~14 GB |

Only one model above ~15 GB fits in VRAM at a time. llama-swap evicts automatically on first request to a different model.

### Context Sizing

Some GGUF files have a hard context ceiling baked into the model metadata (`general.context_length`). Check with `llama-server --model /path/to/model.gguf --model-info` before setting `--ctx-size` above that value — exceeding the baked-in ceiling produces garbage output.

Within that ceiling, the practical limit is VRAM. A binary-search script (`maximize_ctx.py`, not included) automates the search: starting from the current value, it doubles the step size until OOM or the GGUF ceiling, then narrows to 2048-token precision. For MTP models it also runs a post-swap reload test to catch GPU deadlock before committing the value. The `--ctx-size` values in `config.yaml` are outputs of that process.

---

## Parameter Tuning

Sampling parameters are based on Unsloth's published recommendations for each model family, used as a starting point and adjusted for the specific use case.

### Qwen3.6 Modes

| Mode | Temp | top_p | top_k | presence_penalty | Use case |
|---|---|---|---|---|---|
| Non-thinking (agent) | 0.7 | 0.8 | 20 | 1.5 | tool-calling, structured output |
| Non-thinking (code) | 0.6 | 0.95 | 20 | 0.0 | precise coding, low-diversity output |
| Thinking (general) | 1.0 | 0.95 | 20 | 1.5 | research, planning, exploration |

The presence_penalty difference between code (0.0) and general (1.5) is intentional: coding tasks benefit from repetition (common patterns, boilerplate), while reasoning tasks benefit from topic diversity.

### Template Incompatibilities

Qwen3.6 and Gemma 4 both support a thinking mode but use incompatible mechanisms. Cross-applying either causes wrong behavior — thinking tokens leak into output or disappear silently.

| Family | Thinking activation | Notes |
|---|---|---|
| **Qwen3.6** | `--reasoning on` + `{"preserve_thinking": true}` | `preserve_thinking` retains prior-turn traces across non-thinking turns |
| **Gemma 4** | `{"enable_thinking": true}` only | No `--reasoning` flag; no `preserve_thinking` |

Each model family has a dedicated macro to prevent confusion.

---

## MTP Speculative Decoding

[MTP (Multi-Token Prediction)](https://github.com/ggml-org/llama.cpp/pull/22673) is a speculative decoding mode where auxiliary "draft heads" — extra output layers trained alongside the main model — predict multiple future tokens per forward pass. The main model verifies each prediction and accepts or rejects it; accepted tokens cost no additional compute. Rejected tokens fall back to standard decoding for that step.

All primary inference models use MTP where the architecture supports it:

| Model | MTP type | Notes |
|---|---|---|
| `qwen36-35b-a3b-mtp` | Internal heads | Draft heads embedded in GGUF; system binary |
| `gemma4-26b-a4b-mtp` | External drafter | Separate `--model-draft` GGUF; system binary (PR #23398 merged) |
| `qwen36-35b-a3b-vision` | None | `--mmproj` incompatible with `--spec-type draft-mtp` |
| `rocinante-x-12b` | None | Standard single-token inference |
| `huihui-35b-abliterated` | None | Standard single-token inference |

**Internal heads (Qwen3.6-35B-A3B-MTP):** Draft heads are embedded in the main GGUF. No second model file needed. Runs on the standard system llama-server — MTP is mainline since b9180.

**External drafter (Gemma 4 26B-A4B-MTP):** Draft heads are in a separate GGUF. Both `--model` and `--model-draft` are required. Runs on the standard system llama-server — Gemma 4 MTP support landed in PR #23398 and is now in mainline.

Required flags for both patterns:
```
--spec-type draft-mtp  # enable MTP speculative decoding
--spec-draft-n-max N   # 2 for Qwen3.6, 3 for Gemma 4
--parallel 1           # hard requirement — MTP does not support concurrent slots
--cache-ram 0          # required — prevents prompt_save abort; see Known Issues
-b 512 -ub 512         # compute scratch buffer at b=2048 exhausts VRAM headroom
```

**Measured throughput (RX 7900 XTX, warm model):**

| Model | Tokens/sec | Draft acceptance | Notes |
|---|---|---|---|
| Qwen3.6-35B-A3B-MTP | 75–103 t/s (typically 80–85) | ~70% avg (range 54–95%) | internal heads, n-max=2 |
| Gemma 4 26B-A4B-MTP | ~65–70 t/s | ~50–55% | external drafter, n-max=3 |

Qwen3.6 thinking mode runs slightly faster than agent mode because chain-of-thought tokens are more predictable — draft heads achieve higher acceptance rates on reasoning traces than on direct-answer output.

MTP is not applied to dense models on this hardware — see Known Issues.

---

## Model Inventory

| Model ID | Architecture | Quant | Context | MTP | Role |
|---|---|---|---|---|---|
| `qwen36-35b-a3b-mtp` | Qwen3.6-35B-A3B (MoE) | IQ4_NL | 262,144 | Yes (internal) | developer |
| `qwen36-code` | alias → `qwen36-35b-a3b-mtp` | — | — | — | developer |
| `qwen36-plan` | alias → `qwen36-35b-a3b-mtp` | — | — | — | developer |
| `gemma4-26b-a4b-mtp` | Gemma 4 26B-A4B (MoE) | Q4_K_XL + Q2_K | 262,144 | Yes (external) | developer |
| `gemma4-code` | alias → `gemma4-26b-a4b-mtp` | — | — | — | developer |
| `gemma4-plan` | alias → `gemma4-26b-a4b-mtp` | — | — | — | developer |
| `rocinante-x-12b` | Rocinante-X-12B (Nemo) | Q8_0 | 188,416 | No | user |
| `qwen36-35b-a3b-vision` | Qwen3.6-35B-A3B + mmproj | IQ4_XS + F16 | 262,144 | No | developer |
| `agent-primary-vision` | alias → `qwen36-35b-a3b-vision` | — | — | — | developer |
| `qwen36-chat-vision` | alias → `qwen36-35b-a3b-vision` | — | — | — | developer |
| `huihui-35b-abliterated` | Qwen3.6-35B-A3B abliterated (MoE) | Q4_K_M | 245,760 | No | user |
| `huihui-35b-abliterated-think` | alias → `huihui-35b-abliterated` | — | — | — | user |

Aliases on a single llama-server process via `setParamsByID` — no reload on variant switch.

---

## Known Issues

### MTP crash under concurrent requests: `recurrent state read/write is not supported with partial rollback`

**Status:** Resolved — fix applied in `config.yaml` (`qwen36_mtp_agent` macro).

**Affects:** All MTP models using internal draft heads (`Qwen3.6-35B-A3B-MTP`).

**Symptom:** llama-server aborts mid-generation when a second request arrives while an MTP model is actively processing a prompt.

**Root cause:** The MTP branch implements draft heads using `llama_memory_recurrent`. When a concurrent request triggers `server_slot::prompt_save` (for KV prefix reuse), it hits `llama_memory_recurrent::state_write`, which unconditionally aborts:

```
GGML_ABORT("recurrent state read/write is not supported with partial rollback");
```

The recurrent memory used for draft heads cannot be serialized. This is a missing feature in the upstream MTP branch, not a build or configuration problem.

**Fix:** `--cache-ram 0` in the `qwen36_mtp_agent` macro. This nulls the server-side prompt cache `unique_ptr`, preventing `prompt_save` from ever being called.

**Trade-off:** No KV prefix reuse across turns for repeated system prompt prefixes. Given `--parallel 1` and MTP's throughput advantage (~1.5–2x from speculative decoding), this is the correct trade-off.

---

### Dense 27B + MTP: GPU spin-hang on second load

**Status:** Resolved — MTP not used on any dense models in this config.

**Affects:** Dense 27B models using `--spec-type draft-mtp` on ROCm gfx1100.

**Symptom:** Model loads, passes health check, serves 1–N requests — then after a model swap and reload, hangs silently. ~99% CPU. No OOM error. No crash log. GPU kernel stuck in spin-wait indefinitely; requires process kill.

**Root cause:** ROCm flash attention and the MTP speculative decoding compute graph deadlock on the **second load** of a dense 27B model after a swap. The first load works because GPU compute queues are clean. After eviction and reload, cached queue state is incompatible with the MTP draft verification graph. Sparse MoE models (Qwen3.6-35B-A3B) are unaffected.

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

The `healthCheckTimeout` in `config.yaml` is set to 180 s. Large MoE models take 60–90 s to load all layers on first request; the default 30 s timeout causes false-positive health check failures before the model is ready.

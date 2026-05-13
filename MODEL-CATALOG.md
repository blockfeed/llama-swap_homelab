# Model Catalog

Per-model documentation: architecture, quantization, sampling parameters, VRAM estimates, and measured throughput. Includes comparison of local config against upstream recommendations where they diverge.

Hardware: AMD Radeon RX 7900 XTX (24 GB VRAM, gfx1100, ROCm 7.2.1)

---

## Critical Constraints

These apply globally and are worth understanding before reading individual model entries.

**VRAM ceiling = 93%.** Above ~22.3 GB on this card, ROCm spills into GTT (system RAM via PCIe). This tanks throughput and eventually causes segfaults in the HIP runtime. Always check `rocm-smi --showmeminfo vram` under load — not at idle, which is misleading.

**KV cache quantization.** All models use `--cache-type-k q4_0 --cache-type-v q4_0` via the `${base}` macro. This halves KV cache VRAM versus fp16. At 100k context the KV cache in fp16 would be 10–12 GB; q4_0 brings it under 4 GB. Quality impact is negligible for conversational contexts.

**Macro nesting.** llama-swap silently ignores `--ctx-size` when it appears in a macro that references another macro. All macros here expand `${base}` exactly one level deep. Context size always goes in the model entry.

**Flash Attention.** `--flash-attn on` requires the literal string `on`, not a boolean flag. Using `--flash-attn` alone without a value has no effect.

**Qwen3.5 vs Qwen3.6 templates.** These model families look similar but use incompatible Jinja2 chat templates for thinking mode. See [Template Incompatibilities](#template-incompatibilities) below.

---

## Macro Reference

### `base` — Global defaults

```yaml
"base": >
  /usr/bin/llama-server
  --port ${PORT}
  --n-gpu-layers 999
  --cache-type-k q4_0
  --cache-type-v q4_0
  --host 0.0.0.0
  --jinja
  --flash-attn on
  -ub 2048
  -b 2048
```

| Parameter | Why |
|---|---|
| `--n-gpu-layers 999` | Offload all layers to GPU; no CPU fallback. |
| `--cache-type-k/v q4_0` | 4-bit KV cache quantization. Halves VRAM at long context. |
| `--host 0.0.0.0` | Bind all interfaces; required for LAN clients reaching the backend via llama-swap. |
| `--jinja` | Use Jinja2 templates from GGUF metadata. Required for correct system prompt handling across Qwen, Mistral, GLM families. |
| `--flash-attn on` | Flash Attention 2. Faster inference and reduced memory bandwidth on RDNA3. |
| `-ub/-b 2048` | Batch sizes. Balanced for 24 GB — larger values increase throughput but risk OOM with large MoE models. |

---

### `glm47_agent` — GLM-4.7 agent mode

```yaml
"glm47_agent": >
  ${base}
  --parallel 1
  --temp 0.7
  --top-p 1.0
  --top-k 0
  --min-p 0.01
  --repeat-penalty 1.0
```

`--parallel 1` is mandatory. GLM-4.7 uses DeepSeek2 Multi-Linear Attention (MLA). With the default `n_parallel=4`, throughput collapses from ~60 t/s to 2–8 t/s under concurrent load. This is not a tuning choice — the attention implementation does not parallelize well in llama-server's concurrent request handler.

Parameters match upstream GLM-4.7 recommendations: `top-p 1.0`, no top-k, `min-p 0.01` to trim the very low probability tail.

---

### `glm47_thinking` — GLM-4.7 thinking mode

```yaml
"glm47_thinking": >
  ${base}
  --parallel 1
  --temp 1.0
  --top-p 0.95
  --top-k 20
  --min-p 0.01
  --repeat-penalty 1.0
```

Higher temperature for exploratory reasoning. `top-k 20` constrains vocabulary to keep thinking traces focused without over-constraining creative reasoning.

---

### `qwen35_thinking` — Qwen3.5 thinking mode

```yaml
"qwen35_thinking": >
  ${base}
  --reasoning on
  --temp 1.0
  --top-p 0.95
  --top-k 20
  --min-p 0.0
  --presence-penalty 1.5
  --repeat-penalty 1.0
```

`--reasoning on` is the Qwen3.5 thinking mechanism. **Do not use this on Qwen3.6 models** — they use `--reasoning on` with `preserve_thinking` instead of `enable_thinking`. Cross-applying causes thinking tokens to leak into output or disappear silently.

---

### `qwen36_agent` — Qwen3.6 agent / non-thinking

```yaml
"qwen36_agent": >
  ${base}
  --reasoning off
  --chat-template-kwargs '{"preserve_thinking":true}'
  --temp 0.7
  --top-p 0.8
  --top-k 20
  --min-p 0.0
  --presence-penalty 1.5
```

`preserve_thinking=true` retains prior-turn thinking context in multi-turn workflows even when the current response doesn't use thinking mode. Important for agent sessions where earlier reasoning steps should remain in context.

Parameters match Unsloth Qwen3.6 non-thinking general task recommendations.

---

### `qwen36_thinking` — Qwen3.6 thinking mode

```yaml
"qwen36_thinking": >
  ${base}
  --reasoning on
  --chat-template-kwargs '{"preserve_thinking":true}'
  --temp 1.0
  --top-p 0.95
  --top-k 20
  --min-p 0.0
  --presence-penalty 1.5
  --repeat-penalty 1.0
```

Parameters match Unsloth Qwen3.6 thinking general task recommendations.

---

### `qwen36_code` — Qwen3.6 precise coding

```yaml
"qwen36_code": >
  ${base}
  --reasoning off
  --chat-template-kwargs '{"preserve_thinking":true}'
  --temp 0.6
  --top-p 0.95
  --top-k 20
  --min-p 0.0
  --presence-penalty 0.0
  --repeat-penalty 1.0
```

`presence_penalty=0.0` for coding: code benefits from repetition (common patterns, boilerplate). The standard 1.5 presence penalty suppresses it unnecessarily.

---

### `qwen36_mtp_agent` — Qwen3.6 MTP agent (shared: 35B-A3B and 27B)

```yaml
"qwen36_mtp_agent": >
  ${base}
  --parallel 1
  --reasoning off
  --chat-template-kwargs '{"preserve_thinking":true}'
  --temp 0.7
  --top-p 0.8
  --top-k 20
  --min-p 0.0
  --presence-penalty 1.5
  --spec-type mtp
  --spec-draft-n-max 2
```

- `--spec-type mtp`: enables Multi-Token Prediction speculative decoding. Requires the MTP-patched llama-server from [am17an/mtp-clean](https://github.com/am17an/llama.cpp/tree/mtp-clean). See [blockfeed/llama-cpp-mtp-hip](https://github.com/blockfeed/llama-cpp-mtp-hip) for the Arch package.
- `--spec-draft-n-max 2`: draft up to 2 tokens per step. Unsloth recommendation for Qwen3.6 MTP — higher values have diminishing returns.
- `--parallel 1`: MTP does not support concurrent request slots. Hard requirement.

---

### `qwen36_mtp_thinking` — Qwen3.6 MTP thinking (35B-A3B)

```yaml
"qwen36_mtp_thinking": >
  ${base}
  --parallel 1
  --reasoning on
  --chat-template-kwargs '{"preserve_thinking":true}'
  --temp 1.0
  --top-p 0.95
  --top-k 20
  --min-p 0.0
  --presence-penalty 1.5
  --repeat-penalty 1.0
  --spec-type mtp
  --spec-draft-n-max 2
```

`presence_penalty=1.5` per Unsloth Qwen3.6-35B-A3B-MTP documentation. MTP thinking mode runs faster than agent mode because chain-of-thought tokens are more predictable — draft heads achieve higher acceptance rates on reasoning traces.

---

### `qwen36_27b_mtp_thinking` — Qwen3.6 MTP thinking (27B only)

```yaml
"qwen36_27b_mtp_thinking": >
  ${base}
  --parallel 1
  --reasoning on
  --chat-template-kwargs '{"preserve_thinking":true}'
  --temp 1.0
  --top-p 0.95
  --top-k 20
  --min-p 0.0
  --presence-penalty 0.0
  --repeat-penalty 1.0
  --spec-type mtp
  --spec-draft-n-max 2
```

`presence_penalty=0.0` per Unsloth Qwen3.6-27B-MTP documentation. This differs from the 35B-A3B recommendation (1.5). The 27B dense model's internal diversity mechanism handles topic spread differently. Each model has a dedicated macro to make this difference explicit.

| | 35B-A3B-MTP thinking | 27B-MTP thinking |
|---|---|---|
| `presence_penalty` | 1.5 | **0.0** |
| All other params | same | same |

---

### `rocinante_general` — Rocinante-X-12B

```yaml
"rocinante_general": >
  ${base}
  --temp 0.7
  --top-p 0.9
  --top-k 40
  --min-p 0.0
  --repeat-penalty 1.1
```

`repeat-penalty 1.1` counters the Nemo architecture's tendency toward repetitive output in long generations. Author (TheDrummer) recommended parameters.

---

### `grape2_thinking` — GRaPE-2-Pro

```yaml
"grape2_thinking": >
  ${base}
  --temp 0.6
  --top-p 0.95
  --top-k 20
  --min-p 0
  --repeat-penalty 1.0
```

GRaPE-2-Pro uses `<thinking_mode>` prompt tags for reasoning depth control — not `enable_thinking` or `--reasoning`. The tag goes at the **end of the user prompt** (not the system prompt):

```
<thinking_mode=auto>    # adaptive
<thinking_mode=high>    # 8k-16k thinking tokens
<thinking_mode=minimal> # near-zero thinking
```

---

## Template Incompatibilities

The Qwen3.5 and Qwen3.6 model families use different Jinja2 chat templates for thinking mode. Cross-applying either causes silent misbehavior.

| Family | Thinking flag | Template keyword |
|---|---|---|
| Qwen3.5 | `--reasoning on` | `enable_thinking` |
| Qwen3.6 | `--reasoning on` | `preserve_thinking` |
| Qwen3-VL | `--reasoning on` | Qwen3.5 family (not Qwen3.6) |
| Qwen2-VL | (none needed) | own ChatML template |

Vision models (`--mmproj`) also require `--n-gpu-layers 100,100` (format: `cpu_layers,gpu_layers`) and cannot be used with MTP (`--spec-type mtp`).

---

## Model Catalog

### qwen36-35b-a3b (multi-variant)

| | |
|---|---|
| **Architecture** | Qwen3.6-35B-A3B, MoE (128 experts, 3 activated ≈ 3.5B active params) |
| **GGUF** | UD-IQ4_XS (~17.5 GB) |
| **Context** | 102,400 |
| **Macro** | `qwen36_agent` (base); per-variant via `setParamsByID` |
| **Role** | developer (agent variants) / user (thinking variants) |

Single process serving five profiles without reload. The server launches once; `setParamsByID` injects sampling params and `chat_template_kwargs` per request.

| Alias | Thinking | Temp | Presence penalty | Use case |
|---|---|---|---|---|
| `qwen36-agent` | off | 0.7 | 1.5 | tool-calling, multi-turn |
| `qwen36-code` | off | 0.6 | 0.0 | precise coding |
| `qwen36-paseo` / `paseo` | off | 0.7 | 1.5 | casual chat (no prior-turn thinking) |
| `qwen36-chat` | on | 1.0 | 1.5 | heavy reasoning, research |
| `qwen36-plan` | on | 1.0 | 1.5 | planning, design discussion |

**VRAM estimate:** ~17.5 GB weights + ~2.5 GB KV cache at 102k context (q4_0) ≈ 20 GB.

---

### agent-mistral

| | |
|---|---|
| **Architecture** | Mistral-Small-3.2-24B-Instruct-2506, dense |
| **GGUF** | UD-Q4_K_XL (~16 GB) |
| **Context** | 92,160 |
| **Macro** | `qwen36_agent` |
| **Role** | developer |

Alternative agent backend. Good fallback for long-context tasks or when the Qwen3.6 process is occupied.

The `qwen36_agent` macro includes `--chat-template-kwargs` arguments that Mistral ignores (different template family). These are harmless but worth noting if sampling behavior seems off — a dedicated `mistral_agent` macro would be more correct.

**VRAM estimate:** ~15 GB weights + ~5.5 GB KV cache at 92k context ≈ 20.5 GB.

---

### qwen35-9b

| | |
|---|---|
| **Architecture** | Qwen3.5-9B, dense |
| **GGUF** | UD-Q5_K_XL (~6.3 GB) |
| **Context** | 122,880 |
| **Macro** | `qwen35_agent` |
| **Role** | developer |

Fastest agent backend at this VRAM footprint. Fits comfortably in 24 GB alongside almost any other model, though running two large models concurrently on this hardware isn't useful anyway.

**Estimated throughput:** ~120–150 t/s (dense 9B, very light VRAM footprint).

---

### glm47-agent / glm47-thinking

| | |
|---|---|
| **Architecture** | GLM-4.7-Flash-REAP-23B-A3B, DeepSeek2/MLA |
| **GGUF** | UD-Q4_K_XL (~16.5 GB) |
| **Context** | 65,536 |
| **Macros** | `glm47_agent` / `glm47_thinking` |
| **Role** | developer / user |

MLA (Multi-Linear Attention) architecture with compressed KV storage. The `--parallel 1` constraint is not optional — see macro reference above.

Context is halved from the native 131k because MLA's KV structure is larger per token than standard GQA. 65k stays under the VRAM ceiling; 131k does not.

**Measured throughput:** ~60 t/s (agent), ~50 t/s (thinking).

---

### qwen36-35b-a3b-mtp-agent / qwen36-35b-a3b-mtp-chat

| | |
|---|---|
| **Architecture** | Qwen3.6-35B-A3B, MoE + MTP draft heads |
| **GGUF** | UD-IQ4_NL (~18.9 GB) |
| **Context** | 102,400 |
| **Macros** | `qwen36_mtp_agent` / `qwen36_mtp_thinking` |
| **Role** | developer / user |

MTP variant of the 35B-A3B. Same architecture and weights as the non-MTP model, with additional draft head layers baked into the GGUF enabling speculative decoding without a separate draft model.

MTP requires the [am17an/mtp-clean](https://github.com/am17an/llama.cpp/tree/mtp-clean) llama-server build. See [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) for the implementation. The standard llama.cpp binary does not support `--spec-type mtp`.

**Measured throughput (warm model, 200-token generation, RX 7900 XTX):**

| Profile | Tokens/sec |
|---|---|
| Agent (non-thinking) | 77.3 t/s |
| Thinking | 81.4 t/s |

Thinking mode is faster (+5%) because reasoning tokens are more predictable — draft heads achieve higher acceptance rates on chain-of-thought output than on direct answers.

**Parameter comparison:**

| | Upstream (Unsloth) | config.yaml |
|---|---|---|
| temperature (thinking) | 1.0 | 1.0 |
| top_p (thinking) | 0.95 | 0.95 |
| presence_penalty (thinking) | 1.5 | 1.5 |
| temperature (agent) | 0.7 | 0.7 |
| presence_penalty (agent) | 1.5 | 1.5 |
| spec-draft-n-max | 2 | 2 |

All parameters match upstream recommendations exactly.

**VRAM estimate:** ~18.9 GB weights (IQ4_NL, slightly heavier than IQ4_XS) + KV cache ≈ ~22 GB at 102k context. Monitor with `rocm-smi --showmeminfo vram` after first inference.

---

### qwen36-27b-mtp-agent / qwen36-27b-mtp-reasoning

| | |
|---|---|
| **Architecture** | Qwen3.6-27B, dense + MTP draft heads |
| **GGUF** | Q4_K_M (~15.7 GB) |
| **Context** | 94,208 |
| **Macros** | `qwen36_mtp_agent` / `qwen36_27b_mtp_thinking` |
| **Role** | developer / user |

Dense (non-MoE) MTP model. All 27B parameters activate per token — more compute per step than the sparse MoE variants. MTP partially offsets this with speculative throughput gains.

**Measured throughput (warm model, 200-token generation, RX 7900 XTX):**

| Profile | Tokens/sec |
|---|---|
| Agent (non-thinking) | 31.8 t/s |
| Thinking | 37.2 t/s |

Roughly 2.4× slower than the 35B-A3B-MTP agent (77.3 t/s) despite fewer total parameters. The MoE model only activates ~3.5B params per token; the dense 27B activates all 27B. Dense models pay full compute per step regardless of MTP acceptance rate.

Thinking mode is faster than agent (+17%) — larger MTP speedup than the 35B-A3B because the proportional cost of thinking token prediction is lower relative to the base forward pass cost.

**Critical parameter difference:** the reasoning (thinking) profile uses `presence_penalty=0.0`, not 1.5. This is a model-specific Unsloth recommendation that differs from the 35B-A3B. The dedicated `qwen36_27b_mtp_thinking` macro prevents confusion with the shared thinking macro.

**Parameter comparison:**

| | Upstream (Unsloth 27B-MTP) | config.yaml |
|---|---|---|
| temperature (thinking) | 1.0 | 1.0 |
| top_p (thinking) | 0.95 | 0.95 |
| presence_penalty (thinking) | **0.0** | **0.0** |
| temperature (agent) | 0.7 | 0.7 |
| presence_penalty (agent) | 1.5 | 1.5 |

**VRAM estimate:** ~15.7 GB weights + KV cache ≈ ~22 GB at 94k context. Dense 27B is near the VRAM ceiling — verify with `rocm-smi --showmeminfo vram` after the first request.

---

### qwen36-27b-reasoning (non-MTP)

| | |
|---|---|
| **Architecture** | Qwen3.6-27B, dense |
| **GGUF** | UD-Q4_K_XL (~17 GB) |
| **Context** | 94,208 |
| **Macro** | `qwen36_thinking` |
| **Role** | user |

Non-MTP dense 27B. Higher-quality quantization (Q4_K_XL vs Q4_K_M in the MTP variant) at slightly more VRAM. Use when output quality matters more than throughput, or when the MTP binary isn't available.

**VRAM estimate:** ~17 GB weights + KV cache ≈ ~22–23 GB. At the VRAM ceiling.

---

### qwen35-27b-abliterated

| | |
|---|---|
| **Architecture** | Qwen3.5-27B-A3B, MoE (Huihui abliterated merge) |
| **GGUF** | Q4_K_M (~15 GB) |
| **Context** | 90,112 |
| **Macro** | `qwen35_thinking` |
| **Role** | user |

Abliteration removes refusal vectors from the model's activation space. **Reduced epistemic honesty** — the model may produce confident output where the base model would hedge or decline. Not for agent/infrastructure tasks; interactive sessions only.

Uses Qwen3.5 Jinja template (`enable_thinking`), not Qwen3.6 (`preserve_thinking`).

---

### rocinante-x-12b

| | |
|---|---|
| **Architecture** | Rocinante-X-12B-v1b (Mistral-Nemo derivative) |
| **GGUF** | Q8_0 (~13 GB) |
| **Context** | 16,384 |
| **Macro** | `rocinante_general` |
| **Role** | user |

Q8_0 quantization for maximum quality at this model size. Short context (16k) saves VRAM; voice and casual sessions rarely need more. Uses Mistral v3 Tekken tokenizer — system prompts should be injected as user/assistant pairs, not as Nemo-style `[SYSTEM_PROMPT]` blocks.

**Estimated throughput:** ~90–100 t/s. Fastest model in the catalog by token rate.

---

### grape2-thinking

| | |
|---|---|
| **Architecture** | GRaPE-2-Pro (Qwen3.5-27B base, fine-tuned) |
| **GGUF** | Q4_K_M (~18 GB) |
| **Context** | 65,536 |
| **Macro** | `grape2_thinking` |
| **Role** | user |

Uses custom `<thinking_mode>` prompt tags instead of the standard `enable_thinking` flag. Reasoning depth is controlled at prompt time:

| Tag | Thinking budget |
|---|---|
| `<thinking_mode=minimal>` | ~0 tokens |
| `<thinking_mode=low>` | <1k tokens |
| `<thinking_mode=medium>` | 1k–8k tokens |
| `<thinking_mode=high>` | 8k–16k tokens |
| `<thinking_mode=xtra-hi>` | >16k tokens |
| `<thinking_mode=auto>` | adaptive |

The tag goes at the **end** of the user message. Putting it in the system prompt has no effect.

**VRAM estimate:** ~18 GB weights + ~4.5 GB KV cache at 65k context ≈ 22.5 GB. Near the 93% ceiling — check VRAM after first inference.

---

### Vision Models

| Model ID | Base model | Quant | mmproj | Context | Notes |
|---|---|---|---|---|---|
| `qwen36-vision-agent` | Qwen3.6-35B-A3B | IQ4_XS | F16 | 102,400 | agent mode + vision |
| `thinking-primary-vision` | Qwen3.6-27B | Q4_K_XL | F16 | 94,208 | thinking + vision |
| `qwen3vl-30b` | Qwen3-VL-30B-A3B-Thinking | Q4_K_M | f16 | 65,536 | ~19 GB total |
| `qwen2vl-7b` | Qwen2-VL-7B-Instruct | Q6_K_L | f16 | 65,536 | compact, ~7.4 GB |

Vision models require `--mmproj` (multimodal projector) and `--n-gpu-layers 100,100`. The mmproj adds ~1–1.3 GB VRAM. MTP is not compatible with vision models.

Template note: Qwen3-VL uses the Qwen3.5 template family (`enable_thinking`). Qwen2-VL uses its own ChatML template. Neither is Qwen3.6.

---

## VRAM Budget Summary

| Scenario | Approx. VRAM | Status |
|---|---|---|
| 35B-A3B (non-MTP, IQ4_XS, 102k ctx) | ~20 GB | ✅ comfortable |
| 35B-A3B-MTP (IQ4_NL, 102k ctx) | ~22 GB | ✅ within ceiling |
| 27B dense (MTP or non-MTP, 94k ctx) | ~22–23 GB | ⚠ near ceiling — monitor |
| Mistral-3.2-24B (Q4_K_XL, 92k ctx) | ~20.5 GB | ✅ comfortable |
| GLM-4.7-Flash-23B (Q4_K_XL, 65k ctx) | ~15 GB | ✅ headroom for second model |
| Qwen3.5-9B (Q5_K_XL, 122k ctx) | ~14 GB | ✅ most VRAM-efficient |
| GRaPE-2-Pro (Q4_K_M, 65k ctx) | ~22.5 GB | ⚠ near ceiling — monitor |

Only one model above ~15 GB can be resident at a time. llama-swap evicts automatically; you don't manage this manually. The numbers above assume a single loaded model — no concurrent large models.

---

## Adding a Model

1. Verify GGUF exists in `/opt/llama/models/<model-dir>/`
2. Select a macro (or create one that references `${base}` only — no nested macros)
3. Add a model entry:
   ```yaml
   "my-model":
     name: "Display Name"
     cmd: >
       ${macro_name}
       --model /opt/llama/models/ModelDir/model.gguf
       --ctx-size NNNNN
     ttl: 28800
     metadata:
       role: developer  # or user
   ```
4. Validate: `python3 -c "import yaml; yaml.safe_load(open('config.yaml'))"`
5. Reload: `sudo systemctl restart llama-swap`
6. Verify: `curl -s http://localhost:12434/v1/models | python3 -m json.tool`
7. Check VRAM under load: `rocm-smi --showmeminfo vram`

## Removing a Model

1. Delete the model entry from `config.yaml`
2. Remove unused macros if nothing else references them
3. Validate and reload (same as above)
4. Optionally delete the GGUF to reclaim disk space

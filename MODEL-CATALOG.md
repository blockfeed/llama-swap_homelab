# Model Catalog

Per-model documentation: architecture, quantization, sampling parameters, VRAM estimates, and measured throughput.

Hardware: AMD Radeon RX 7900 XTX (24 GB VRAM, gfx1100, ROCm 7.2.1)

---

## Critical Constraints

These apply globally.

**VRAM ceiling = 93%.** Above ~22.3 GB on this card, ROCm spills into GTT (system RAM via PCIe). This tanks throughput and eventually causes segfaults in the HIP runtime. Always check `rocm-smi --showmeminfo vram` under load — not at idle, which is misleading.

**KV cache quantization.** All models use `--cache-type-k q4_0 --cache-type-v q4_0` via the `${base}` macro. This halves KV cache VRAM versus fp16. At 100k context the KV cache in fp16 would be 10–12 GB; q4_0 brings it under 4 GB. Quality impact is negligible for conversational contexts.

**Macro nesting.** llama-swap silently ignores `--ctx-size` when it appears in a macro that references another macro. All macros here expand `${base}` exactly one level deep. Context size always goes in the model entry. Exception: `gemma4_mtp` is self-contained and references no other macro — kept standalone because it carries a distinct full set of MTP flags and Gemma 4-specific sampling defaults.

**Flash Attention.** `--flash-attn on` requires the literal string `on`, not a boolean flag. Using `--flash-attn` alone without a value has no effect.

**Qwen3.6 vs Gemma 4 templates.** These model families look similar but use incompatible Jinja2 chat templates for thinking mode. See [Template Incompatibilities](#template-incompatibilities) below.

**Dense models + MTP deadlock.** ROCm flash attention and the MTP compute graph deadlock on the second load of a dense model after a swap. Sparse MoE models (35B-A3B) are unaffected. No dense models in the current config use MTP.

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
| `--jinja` | Use Jinja2 templates from GGUF metadata. Required for correct system prompt handling across model families. |
| `--flash-attn on` | Flash Attention 2. Faster inference and reduced memory bandwidth on RDNA3. |
| `-ub/-b 2048` | Default batch sizes; overridden per-macro where needed. |

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
  -b 512
  -ub 512
```

`preserve_thinking=true` retains prior-turn thinking context in multi-turn workflows even when the current response doesn't use thinking mode.

`-b 512 -ub 512` reduces compute scratch buffer allocation versus the base macro's 2048. Important for vision models (mmproj adds ~1.3 GB on top of base weights) and large-context sessions near the VRAM ceiling.

Parameters match Unsloth Qwen3.6 non-thinking general task recommendations.

---

### `qwen36_mtp_agent` — Qwen3.6 MTP agent (35B-A3B-MTP)

```yaml
"qwen36_mtp_agent": >
  ${base}
  --parallel 1
  --cache-ram 0
  --reasoning off
  --chat-template-kwargs '{"preserve_thinking":true}'
  --temp 0.7
  --top-p 0.8
  --top-k 20
  --min-p 0.0
  --presence-penalty 1.5
  -b 512
  -ub 512
  --spec-type draft-mtp
  --spec-draft-n-max 2
```

| Parameter | Why |
|---|---|
| `--spec-type draft-mtp` | Enables MTP speculative decoding. MTP is mainline since b9180 — no custom binary required for Qwen3.6. |
| `--spec-draft-n-max 2` | Draft up to 2 tokens per step. Unsloth recommendation for Qwen3.6 MTP — higher values have diminishing returns. |
| `--parallel 1` | MTP does not support concurrent request slots. Hard requirement. |
| `--cache-ram 0` | MTP draft heads use `llama_memory_recurrent`. The server's prompt-slot serialization invokes `llama_memory_recurrent::state_write`, which aborts unconditionally with "recurrent state read/write is not supported with partial rollback". `--cache-ram 0` nulls the server-side prompt cache, preventing `prompt_save` from ever being called. Trade-off: no KV prefix reuse across turns. MTP's speculative throughput gain is unaffected. |
| `-b/-ub 512` | Compute scratch buffer at b=2048 exhausts VRAM headroom with MTP draft graph overhead. |

---

### `gemma4_mtp` — Gemma 4 MTP (external drafter)

```yaml
"gemma4_mtp": >
  /usr/bin/llama-server
  --port ${PORT}
  --n-gpu-layers 999
  --cache-type-k q4_0
  --cache-type-v q4_0
  --host 0.0.0.0
  --jinja
  --flash-attn on
  --parallel 1
  --cache-ram 0
  --temp 1.0
  --top-p 0.95
  --top-k 64
  -b 512
  -ub 512
  --spec-type draft-mtp
  --spec-draft-n-max 3
  --chat-template-kwargs '{"enable_thinking":false}'
```

This macro is **self-contained** — it does not reference `${base}`. Kept standalone because it requires a complete set of MTP flags and Gemma 4-specific defaults (temp 1.0, top-k 64, `enable_thinking`) that differ structurally from `${base}`. PR #23398 merged into mainline; all models now use the standard `/usr/bin/llama-server`.

| Parameter | Why |
|---|---|
| `--spec-draft-n-max 3` | Gemma 4-specific: 3 draft tokens per step per the ironbcc HF card recommendation. |
| `--top-k 64` | Gemma 4 upstream sampling recommendation (vs 20 for Qwen3.6). |
| `--chat-template-kwargs '{"enable_thinking":false}'` | Gemma 4 uses `enable_thinking` exclusively. No `--reasoning` flag. No `preserve_thinking`. Default sets thinking off; `setParamsByID` overrides per alias. |
| `--cache-ram 0` | Same reason as `qwen36_mtp_agent` — prevents `prompt_save` abort with recurrent MTP memory. |

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

## Template Incompatibilities

Qwen3.6 and Gemma 4 both support a thinking mode but use incompatible mechanisms. Cross-applying either produces wrong behavior — thinking tokens leak into output or disappear completely.

| Family | Thinking activation | Notes |
|---|---|---|
| Qwen3.6 | `--reasoning on` + `preserve_thinking` | `preserve_thinking` retains prior-turn traces; use `enable_thinking` in `setParamsByID` to toggle per-alias |
| Gemma 4 | `enable_thinking` only | No `--reasoning` flag; no `preserve_thinking`; toggle entirely via `setParamsByID` |

Vision models (`--mmproj`) require `--n-gpu-layers 100,100` and cannot be used with MTP.

---

## Model Catalog

### qwen36-35b-a3b-mtp

| | |
|---|---|
| **Architecture** | Qwen3.6-35B-A3B, MoE (128 experts, 3 activated ≈ 3.5B active params) + internal MTP draft heads |
| **GGUF** | Qwen3.6-35B-A3B-UD-IQ4_NL.gguf (~18.9 GB) |
| **Context** | 262,144 |
| **Macro** | `qwen36_mtp_agent`; per-alias via `setParamsByID` |
| **Role** | developer |
| **Upstream** | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) |

MTP variant of the 35B-A3B. Draft head layers are baked into the GGUF — no separate draft model file. Runs on the standard system llama-server; MTP is mainline since b9180.

Single process serving two profiles without reload. `setParamsByID` injects sampling params and `chat_template_kwargs` per request.

| Alias | Thinking | Temp | Presence penalty | Use case |
|---|---|---|---|---|
| `qwen36-35b-a3b-mtp` | OFF | 0.7 | 1.5 | Base agent mode |
| `qwen36-code` | OFF | 0.6 | 0.0 | Precise coding |
| `qwen36-plan` | ON | 1.0 | 1.5 | Planning, design discussion |

`presence_penalty=0.0` for coding: code benefits from repetition (common patterns, boilerplate). The 1.5 presence penalty suppresses it unnecessarily.

**Measured throughput (RX 7900 XTX, warm model):**

| Metric | Value |
|---|---|
| Token generation | 75–103 t/s (typically 80–85 t/s sustained) |
| Prompt eval | ~900–1300 t/s |
| Draft acceptance | ~70% avg (range 54–95%) |
| Speculative | internal heads, n-max=2 |

Thinking mode runs slightly faster (+3–5%) because reasoning tokens are more predictable — draft heads achieve higher acceptance rates on chain-of-thought output than on direct answers.

**VRAM estimate:** ~18.9 GB weights (IQ4_NL) + ~2 GB KV cache at 262k context (q4_0) ≈ 21.6 GB (~88%). Within the 93% ceiling with headroom.

---

### gemma4-26b-a4b-mtp

| | |
|---|---|
| **Architecture** | Gemma 4 26B-A4B, MoE (~4B active per token) + external MTP draft model |
| **GGUF (main)** | gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf |
| **GGUF (draft)** | gemma-4-26B-A4B-it-assistant-Q2_K.gguf |
| **Context** | 262,144 |
| **Macro** | `gemma4_mtp` (self-contained); per-alias via `setParamsByID` |
| **Role** | developer |
| **Upstream** | [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) |

External-drafter MTP: draft heads live in a separate GGUF (`--model-draft`) rather than being embedded in the main model. This differs from Qwen3.6-35B-A3B-MTP where draft heads are baked into a single file. PR #23398 merged into mainline — runs on the standard system llama-server.

Template: Gemma 4 uses `enable_thinking` only. No `--reasoning` flag, no `preserve_thinking`. Thinking is controlled via `setParamsByID` per alias.

| Alias | Thinking | Temp | Use case |
|---|---|---|---|
| `gemma4-26b-a4b-mtp` | OFF | 1.0 | Base entry |
| `gemma4-code` | OFF | 1.0 | Agent/tool-calling |
| `gemma4-plan` | ON | 1.0 | Reasoning, planning |

**Measured throughput (RX 7900 XTX, fresh load):**

| Metric | Value |
|---|---|
| Token generation | ~65–70 t/s |
| Draft acceptance | ~50–55% |
| Speculative | external drafter, n-max=3 |

Lower draft acceptance than Qwen3.6-MTP (51% vs ~70%) because n-max=3 is harder to satisfy with an external drafter, and the external Q2_K draft model is more approximate than embedded draft heads.

**VRAM estimate:** ~19–22 GB (main + draft weights + KV at 262k). Verify with `rocm-smi --showmeminfo vram` after first inference.

---

### rocinante-x-12b

| | |
|---|---|
| **Architecture** | Rocinante-X-12B-v1b (Mistral-Nemo derivative) |
| **GGUF** | Rocinante-X-12B-v1b-Q8_0.gguf (~13 GB) |
| **Context** | 188,416 |
| **Macro** | `rocinante_general` |
| **Role** | user |
| **Upstream** | [TheDrummer/Rocinante-X-12B-v1](https://huggingface.co/TheDrummer/Rocinante-X-12B-v1) |

Q8_0 quantization for maximum quality at this model size. Context extended to 188,416 — with Q8_0 weights (~13 GB) and q4_0 KV cache, total VRAM stays well under the ceiling at this length. Voice and casual sessions rarely use the full window, but the headroom eliminates context pressure for extended sessions.

Uses Mistral v3 Tekken tokenizer — system prompts should be injected as user/assistant pairs, not as Nemo-style `[SYSTEM_PROMPT]` blocks.

**VRAM estimate:** ~13 GB weights + ~1 GB KV at 188k context ≈ 14 GB (~58%). Comfortable.

---

### qwen36-35b-a3b-vision

| | |
|---|---|
| **Architecture** | Qwen3.6-35B-A3B (MoE) + multimodal projector |
| **GGUF** | Qwen3.6-35B-A3B-UD-IQ4_XS.gguf (~17.5 GB) |
| **mmproj** | mmproj-F16.gguf (~1.3 GB) |
| **Context** | 262,144 |
| **Macro** | `qwen36_agent`; per-alias via `setParamsByID` |
| **Role** | developer |
| **Upstream** | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |

MTP is not applicable: `--mmproj` is incompatible with `--spec-type draft-mtp`. Standard single-token inference.

`-b 512 -ub 512` (via `qwen36_agent` macro) is required: at b=2048 the compute buffers plus mmproj overhead exhaust VRAM headroom.

| Alias | Thinking | Temp | Use case |
|---|---|---|---|
| `qwen36-35b-a3b-vision` | OFF | 0.7 | Base entry |
| `agent-primary-vision` | OFF | 0.7 | Agent/tool-calling with image understanding |
| `qwen36-chat-vision` | ON | 1.0 | Visual reasoning, document analysis |

**VRAM estimate:** ~17.5 GB weights + ~1.3 GB mmproj + ~2 GB KV at 262k context ≈ 22 GB (~90%). Monitor after first inference.

---

### huihui-35b-abliterated

| | |
|---|---|
| **Architecture** | Qwen3.6-35B-A3B MoE (Huihui abliterated merge) |
| **GGUF** | Huihui-Qwen3.6-35B-A3B-abliterated-Q4_K_M.gguf (~21.3 GB) |
| **Context** | 245,760 |
| **Macro** | `qwen36_agent` |
| **Role** | user |
| **Upstream** | [Huihui-AI/Huihui-Qwen3.6-35B-A3B-abliterated](https://huggingface.co/Huihui-AI/Huihui-Qwen3.6-35B-A3B-abliterated) |

Abliteration removes refusal vectors from the model's activation space. The model may produce confident output where the base model would hedge or decline — not for agent/infrastructure tasks; interactive use only.

Standard single-token inference (no MTP). Shares the `qwen36_agent` macro with the vision model; thinking is toggled via `setParamsByID`.

| Alias | Thinking | Temp | Use case |
|---|---|---|---|
| `huihui-35b-abliterated` | OFF | 0.7 | Non-thinking interactive |
| `huihui-35b-abliterated-think` | ON | 1.0 | Thinking interactive |

**VRAM estimate:** ~21.3 GB weights (Q4_K_M) + ~1 GB KV at 245k context ≈ 22.8 GB (~93%). At the ceiling — check VRAM after first inference.

---

## Context Sizing

### Checking the baked-in ceiling

Some GGUFs have a hard context ceiling encoded in the model metadata (`general.context_length` or `llama.context_length`). Setting `--ctx-size` above this value doesn't increase usable context — it generates garbage or errors. Check with:

```bash
llama-server --model /opt/llama/models/<dir>/<file>.gguf --model-info 2>&1 | grep -i context
```

### Finding the VRAM limit

Within the model ceiling, the practical limit is VRAM. The methodology:

1. Load the model at a candidate context size
2. Run an inference request to force KV cache allocation (idle VRAM readings are misleading)
3. Check `rocm-smi --showmeminfo vram`
4. If usage is above 93%, reduce context and retry

A binary-search script (`maximize_ctx.py`, not included in this repo) automates this as a two-phase process:

- **Phase 1 (expansion):** Starting from the current value, doubles the step size each successful iteration until OOM or the GGUF ceiling
- **Phase 2 (binary search):** Narrows to 2048-token precision between the last success and the failure point

For MTP models, each candidate also runs a post-swap reload test: the script loads a second model to evict the target, then reloads the target and re-runs inference. This catches the dense-model + MTP GPU deadlock pattern before committing a value.

The `--ctx-size` values in `config.yaml` are outputs of that process.

---

## VRAM Budget Summary

| Model | Approx. VRAM | Status |
|---|---|---|
| Qwen3.6-35B-A3B-MTP (IQ4_NL, 262k ctx) | ~21.6 GB | within ceiling |
| Gemma 4 26B-A4B-MTP (Q4_K_XL + Q2_K, 262k ctx) | ~22 GB est. | verify after load |
| Qwen3.6-35B-A3B Vision (IQ4_XS + mmproj, 262k ctx) | ~22 GB est. | verify after load |
| Huihui 35B Abliterated (Q4_K_M, 245k ctx) | ~22.8 GB | at ceiling — monitor |
| Rocinante X 12B (Q8_0, 188k ctx) | ~14 GB | comfortable |

Only one model above ~15 GB can be resident at a time. llama-swap evicts automatically on first request to a different model.

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

---

## Removed Models

Historical reference — these model IDs are no longer in `config.yaml`.

| Model ID | Reason removed |
|---|---|
| `qwen36-35b-a3b` (non-MTP) | Superseded by MTP variant |
| `agent-mistral` | Removed from roster |
| `qwen35-9b` | Removed from roster |
| `glm47-agent` / `glm47-thinking` | GLM-4.7 removed from roster |
| `qwen36-27b-agent` | Removed; dense 27B + MTP deadlock on gfx1100 |
| `qwen36-27b-q4km-reasoning` | Removed; dense 27B + MTP deadlock on gfx1100 |
| `qwen36-27b-reasoning` | Removed from roster |
| `qwen35-27b-abliterated` | Removed from roster |
| `grape2-thinking` | GRaPE-2 removed from roster |
| `qwen3vl-30b` | Qwen3-VL removed from roster |
| `qwen2vl-7b` | Qwen2-VL removed from roster |
| `thinking-primary-vision` | 27B vision removed from roster |
| `qwen36-plan-bartowski` | Bartowski 27B removed; dense 27B + MTP deadlock |
| `qwen36-plan-bartowski-mtp` | Bartowski 27B MTP removed; dense 27B deadlock |

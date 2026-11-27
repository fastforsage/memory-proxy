# memory-proxy v5: an append-only context architecture

> English translation of [`../../design/v5-architecture.md`](../../design/v5-architecture.md). If the two differ, the Russian original is authoritative.

Status: **implemented in v5.0.0** (deviations from the design are in §13, regression results in [`../audit/v5.0.0-regression.md`](../audit/v5.0.0-regression.md)). Based on [`v4.9-context.md`](v4.9-context.md) (P1–P13, §4.7, §5.9–§5.12).
Calculations: [`../../bench/sim_v5.py`](../../bench/sim_v5.py), [`../../bench/sim.json`](../../bench/sim.json) (keys `v5`, `v5_bg`). Check on a live llama.cpp: [`../../bench/validate_v5.py`](../../bench/validate_v5.py), [`../../bench/validate_v5.json`](../../bench/validate_v5.json).

---

## 0. Summary

All the problems found (P1–P13, latency spikes after compression, cache misses in the agent loop, 400 errors and silent truncation on overflow) follow from three architectural decisions in v4.8. v5 changes all three. The main rule: **the prompt sent to the model only grows**. The prefix does not change between turns except at rare epoch changes; the new epoch is prepared and warmed up in the background.

| | v4.8 | v4.9 + warm-up | **v5** |
|---|---|---|---|
| Recompute per turn, before overflow (tokens) | 1309 | 1309 | **277** |
| Recompute per turn, after overflow | 3376 | 1235 | **278** |
| Worst turn after overflow | 3453 | 1353 | **304** |
| TTFT, M2 Pro, after overflow | 11.0 s | 6.1 s | **1.4 s** |
| TTFT, laptop CPU, after overflow | 1.8 min | 65 s | **15 s** |
| Background over 60 turns (tokens) | 0 | 34,437 | 43,994 |

v5 targets **one user with any OpenAI-compatible clients** (§12); team mode is a separate design.

User-facing latency drops 4–8× on every hardware profile and does not grow on any turn. The price is background work (+28% over v4.9). Under the decision "load is fine if it is in the background" this is acceptable.

---

## 1. Three root causes

### R1. The prompt is rebuilt every time from volatile parts
On every request `forward` recomputes `<memory>` and inserts it into the **last** user message. The `<memory>` block consists of:
- facts (vector search on the current question);
- emotions (they decay over time, so they change every minute);
- style examples (`ORDER BY RANDOM()`);
- sometimes quotes from other chats.

As a result:
- `<memory>` disappears from the previous user message and a different one appears in the new one. The KV cache breaks at the second-to-last message, and every turn recomputes the previous answer and question (≈1300 tokens even without compression);
- in the agent loop the same question gets a new `<memory>` on every step (P8), and the cache breaks on every step;
- `compress()` rewrites the beginning of the history, after which everything is recomputed (spikes P5/P6).

**The violated principle:** the prompt is a function of time and randomness rather than of the chat history.

### R2. The budget is not tied to the real model
Decisions are made by message counts (`>10`, `KEEP_RECENT=8`) and a fixed `COMPRESS_AT=16000`. The proxy does not know the model's real `n_ctx` (P11–P13). As a result:
- a short chat of long messages gets 400 from llama.cpp;
- Ollama silently cuts the beginning, including system and the user's question in a tool loop.

### R3. Expensive, state-changing work runs in the hot path
Compression (and, in v4.9, summarisation) runs synchronously inside the user's request. Consequences:
- either the user waits for an LLM summary;
- or (the v4.9 variant without warm-up) the user gets a worst turn of 9163 recomputed tokens right after the window switches.

Rebuilding the context is an event that can be prepared in advance, not a side effect of answering.

---

## 2. v5 principles

1. **Append-only.**
   - `prompt(t+1)` starts byte-for-byte with `prompt(t)` up to the end of the previous answer.
   - Exceptions: epoch change (§4), editing or regenerating a message in the client.
2. **Determinism.**
   - Prompt assembly is a pure function of `(history, chat state)`.
   - No `RANDOM()`, `time.time()` or decay inside assembly. Everything volatile is frozen when an epoch is created or a delta is pinned.
3. **Budget in the real model's tokens.**
   - No message counters.
   - `B = min(COMPRESS_AT, 0.9·n_ctx − ANSWER_RESERVE − tools)` (§4.7 of v4.9).
4. **The hot path does not wait for the LLM.**
   - Everything that needs the LLM (summary, profile, warm-up) runs in the background and gives way to user requests (§5.12).
   - If the background did not finish, a cheap deterministic fallback without the LLM kicks in.
5. **Nothing is lost silently.**
   - What is cut off stays in the database and is available through recall (§5.9).
   - If even the current turn does not fit, that is an explicit reply in the chat (`OVERSIZE`), not truncation on the backend side.

---

## 3. Components

```
             ┌──────────────── hot path (≈1–3 ms) ───────────────┐
request ─▶ Router ──▶ ChatState.load ──▶ Assembler ──▶ Guard ──▶ Forwarder ──▶ backend
             │            ▲                   │          │ 400 → record n_ctx,
             │            │                   │          │ one retry
             │            │        delta = MemoryDelta(question) − already in window
             │            │                   │
             └── answer ─▶ Recorder ──▶ (facts, emotions, style: as in v4.8)
                          │
                          ▼ (if window > EPOCH_PREP_AT·B)
             ┌──────────── background, preemptible ───────────┐
             │ EpochPlanner: new boundary k, summary,        │
             │ profile → prefix warm-up → ready              │
             └───────────────────────────────────────────────┘
```

| Component | Responsibility | Does NOT |
|---|---|---|
| **Router** | `user_id`, `chat_id`, commands, OWUI service requests (`### Task:` → skip memory) | — |
| **ChatState** | epoch, pinned deltas, the ready next epoch; in SQLite | call the LLM |
| **Assembler** | pure function `(history, ChatState, delta) → messages` | touch the DB or know about HTTP |
| **MemoryDelta** | facts and recall for the current question minus what is already in the window | change past messages |
| **Guard** | invariants I1–I6 of v4.9, budget `B`, L1/L3 fallback, `OVERSIZE` | — |
| **Forwarder** | transport only: streaming, retry on 400, recording `n_ctx` | change content (today this is mixed into `forward`, ~200 lines) |
| **EpochPlanner** | background: boundary, summary (map-reduce §5.10), profile, warm-up | block the request |

The Assembler is a pure function, so the append-only property is checked by a unit test: assemble turns `t` and `t+1` and compare the common prefix.

---

## 4. Prompt layout

```
[system]  client's original system + MEMORY_POLICY            ← never changes
[user k]  <summary>…</summary>                                 ← only on epoch change
          <profile>core facts, emotions, style examples</profile>
          text of message k
[asst]    …
[user]    <memory>delta pinned to this message</memory> text
[asst]    …
  …                                                            ← all of this repeats byte-for-byte
[user t]  <memory>new delta</memory> text                      ← new: ≈280 tokens
```

### 4.1 Epoch
An epoch is an immutable record `(epoch_id, cut_hash, summary, profile, created)`. It stores:
- `cut_hash`: the hash of the first message in the window; cuts happen only at turn boundaries, following §4.3 of v4.9;
- `summary`: a summary of everything before `k` (incremental, using the hash chain of §5.3 of v4.9);
- `profile`: a snapshot of long-lived memory at the time of the epoch: the top-N facts by importance, emotions with decay already applied, 3 style examples picked deterministically (`ORDER BY hash(epoch_id, id)`). Up to `PROFILE_CH` = 1500 characters.

The `<profile>` block sits at the beginning of the window and does not change within an epoch. Therefore:
- emotion decay and style rotation work at epoch granularity rather than per request; a person will not notice the difference;
- a new fact extracted on turn `t` appears in the deltas right away (§4.2), and in the profile at the next epoch change.

### 4.2 Pinned delta
For every new user question:
1. Select facts for the question (as today) and recall from the cut-off part of the chat (§5.9).
2. **Subtract** what the model already sees: the profile and the deltas in the window (by `fact_id`). Trim to `DELTA_CH` = 300 characters.
3. Store: `pinned[(chat_id, msg_hash)] = delta`.
4. In all later requests, insert **the same string** into that message.

Consequences:
- **Agent loop (P8).** Steps with the same question find the delta by `msg_hash` and get the same string. The cache does not break and `embed_query` is not called on a step (−28 ms per step). A separate per-agent-turn memory cache (v4.9 §4.6) is no longer needed.
- **Regenerating an answer**: same `msg_hash`, same delta.
- **Editing a question**: new hash, new delta; after that the prompt is again append-only.
- **A fact said in the chat** is already in the window and does not need duplicating. The subtraction takes this into account: facts from messages inside the window are not returned.

`msg_hash = sha1(role, content_without_our_insert, turn_index)`. The index keeps two identical "ok"s from merging.

### 4.3 Epoch change (the only place where the prefix changes)

| Threshold | Default | Action |
|---|---|---|
| `EPOCH_PREP_AT` | 0.7·B | background: the EpochPlanner prepares the next epoch with a window of ≈ `EPOCH_TARGET` = 0.5·B (§5.11 of v4.9) |
| `EPOCH_AT` | 1.0·B | hot path: if the epoch is ready and warmed up, switch (costs the same as a normal turn) |
| > B, epoch not ready | — | fallback without the LLM: L1 elision of old tool outputs, then L3 (cut old turns + recall). Only in this rare case is a turn more expensive than normal |

`EPOCH_AT` = 1.0 instead of 0.85 halved the background work (5 changes instead of 8, 44k instead of 71k tokens). Recompute for the user did not change (mean 278, max 304).

Eliding old tool outputs (L1 of v4.9) rewrites the past. In v5 it **happens only on epoch change**, not on every request; otherwise it would break the prefix just as `<memory>` does today.

### 4.4 Warm-up and backend behaviour
The prepared prefix of the new epoch is sent to the backend with `max_tokens=1` (llama.cpp `cache_prompt`, Ollama `num_predict=1`) while the user is not waiting for an answer.
- In llama.cpp with `-np 1 --cache-ram N` the old prefix moves to the RAM cache and is not lost: if the user writes before the switch, their turn still hits the cache. This was verified back in v4.9.
- Open WebUI service requests (title, tags, follow-ups) are separate prompts that evict the slot with `-np 1`. README recommendation: llama.cpp `-np 2` or `--cache-ram`; for Ollama `OLLAMA_NUM_PARALLEL=2`.

### 4.5 Budget and overflow in short chats (P11–P13)
No "≤10 messages". The Guard counts the tokens of the assembled prompt and compares them with `B`:
- **`n_ctx`** is taken in this order: `CONTEXT_LIMIT` from env → llama.cpp `/props` → Ollama `/api/show` + `options.num_ctx`, which the proxy **sets itself** (otherwise Ollama silently keeps 2–4k) → learning from `400 exceed_context_size_error` (one transparent retry).
- **Order of sacrifice**: §4.7 of v4.9: old turns → profile/deltas → summary → (never) the current turn and system.
- **The current turn alone exceeds `B`**: `OVERSIZE=truncate` (the middle of long tool outputs and attachments, with a marker) or `reject` (a clear reply in the chat).

---

## 5. Storage (schema v4 instead of the v4.9 one)

```sql
CREATE TABLE chat_epochs(
  user_id TEXT, chat_id TEXT, epoch_id INTEGER, cut_hash TEXT,
  summary TEXT, profile TEXT, state TEXT,       -- 'active' | 'ready' | 'preparing'
  prefix_tokens INTEGER, created REAL,
  PRIMARY KEY(user_id, chat_id, epoch_id));
CREATE TABLE pinned_deltas(
  user_id TEXT, chat_id TEXT, msg_hash TEXT, delta TEXT, fact_ids TEXT, created REAL,
  PRIMARY KEY(user_id, chat_id, msg_hash));
CREATE TABLE chat_chain(                        -- binding a request to a chat by history prefix (§12.4)
  user_id TEXT, chat_id TEXT, pos INTEGER, msg_hash TEXT, parent_hash TEXT,
  PRIMARY KEY(user_id, chat_id, pos, msg_hash));
CREATE INDEX ix_chain_head ON chat_chain(user_id, msg_hash) WHERE pos <= 1;
```

- The `summaries` table of v4.9 (§5.4) becomes a field of the epoch. The hash chain stays for incremental summarisation.
- Cleanup: deltas older than the active epoch can be deleted (their messages are already past the boundary). `/forget chat` clears both tables.
- Migration: the tables start empty. A chat without an epoch starts with epoch 0 (`k=0`, no summary, the profile is built on the first request).

---

## 6. Why this does not slow down the answer

1. **The hot path is cheaper than in v4.8:**
   - reading the epoch and deltas is 2 key lookups;
   - vector search runs only for a **new** question, not on agent steps;
   - assembly is concatenation.
   The v4.9 proxy spent 0.6–1.9 ms (p95 3 ms); v5 spends no more.
2. **Model-side recompute drops 4.7× even before overflow:** the break at the second-to-last message (R1) disappears.
3. **An epoch change is free for the user:** the new prefix is already in the cache. Worst turn: 304 tokens versus 3453 in v4.8.
4. **The prompt is longer.** Pinned deltas accumulate (~95 tokens per turn), and the window averages 12k versus 3.6k in v4.8. This is accounted for in the speed model (pp and tg drop with depth):
   - a full 480-token answer is still faster on every profile (table below);
   - decoding on a 4090 is +0.1 s, within noise.

### 6.1 TTFT / full answer (480 tokens), mean

| Profile | Phase | v4.8 | v4.9 + warm-up | **v5** |
|---|---|---|---|---|
| Laptop CPU | before overflow | 60.7 s / 2.6 min | 60.7 s / 2.6 min | **13.3 s / 1.8 min** |
| Laptop CPU | after | 1.8 min / 2.9 min | 65.5 s / 2.9 min | **15.0 s / 2.1 min** |
| Apple M2 Pro | before overflow | 5.8 s / 19.8 s | 5.8 s / 19.8 s | **1.3 s / 15.4 s** |
| Apple M2 Pro | after | 11.0 s / 24.0 s | 6.1 s / 20.7 s | **1.4 s / 16.0 s** |
| RTX 3090 | before overflow | 0.4 s / 3.8 s | 0.4 s / 3.8 s | **0.1 s / 3.6 s** |
| RTX 3090 | after | 0.7 s / 3.9 s | 0.4 s / 4.0 s | **0.1 s / 3.7 s** |
| RTX 4090 | before overflow | 0.2 s / 3.1 s | 0.2 s / 3.1 s | **0.0 s / 3.0 s** |
| RTX 4090 | after | 0.3 s / 3.0 s | 0.2 s / 3.2 s | **0.0 s / 3.1 s** |

### 6.2 Background per epoch change (about every 8 turns after overflow)

| Profile | Summary (7k in → 700) | Warm-up (~8.8k) |
|---|---|---|
| Laptop CPU | 6.2 min | 5.5 min |
| M2 Pro | 45 s | 33 s |
| RTX 3090 | 6.5 s | 2.1 s |
| RTX 4090 | 4.9 s | 0.9 s |

On a CPU the background takes longer than the pause between messages. That is why `EPOCH_PREP_AT` = 0.7 leaves a margin of several turns, and the background gives way to user requests (§5.12). If it does not finish, the L3 fallback applies, as in v4.9.

### 6.3 Check on a live llama.cpp
`validate_v5.py`: Qwen2.5-0.5B, CPU, reduced scale (2000-token budget), 16 turns, 6 epoch changes with warm-up.
- The predicted recompute matched the server's `timings.prompt_n` in **16 of 16** requests.
- On turns with an epoch change the recompute was 158–178 tokens, less than a normal turn (364–411), so the epoch change is invisible to the user.

---

## 7. What gets fixed

| Problem | By |
|---|---|
| P1–P4 (cuts inside a turn, tool calls, role order) | the Guard and invariants I1–I6 from v4.9, unchanged |
| P5/P6 (spike after compression) | the epoch is prepared and warmed up in the background |
| P7 (summary loses detail) | summary + recall from the cut-off part (delta) |
| P8 (`<memory>` in the agent loop) | pinned delta by `msg_hash` |
| P9/P10 (huge tool outputs) | L1 on epoch change, `OVERSIZE` for the current turn |
| P11–P13 (overflow in short chats, model limit) | budget in tokens from the real `n_ctx`, learning, `num_ctx` for Ollama |
| `RANDOM()` and emotion decay break the cache | frozen in the epoch profile |

---

## 8. Configuration (additions to §6 of v4.9)

| Variable | Default | Meaning |
|---|---|---|
| `CONTEXT_MODE` | `v5` | `v5` / `legacy` (v4.8 behaviour, emergency fallback) |
| `USER_MODE`, `OWNER`, `CHAT_ID_HEADERS`, … | `single`, … | users and clients, see §12.6 |
| `EPOCH_PREP_AT` | 0.7 | fraction of `B` at which the background prepares an epoch |
| `EPOCH_AT` | 1.0 | fraction of `B` at which we switch to the ready epoch |
| `EPOCH_TARGET` | 0.5 | window size after the change |
| `PROFILE_CH` | 1500 | size of `<profile>` |
| `DELTA_CH` | 300 | delta size per turn |
| `SET_OLLAMA_NUM_CTX` | 1 | set `options.num_ctx` from the known limit |
| `CONTEXT_LIMIT`, `ANSWER_RESERVE`, `OVERSIZE`, `RECALL`, `SUMMARY_*`, `BACKGROUND_PREEMPT` | as in v4.9 | — |

---

## 9. Work plan (replaces §8 of v4.9)

**Decision (27.11.2026):** all PRs are in scope. Order: A → U → B → C → D.

| PR | Contents | LLM | Risk |
|---|---|---|---|
| **A** | Budget from `n_ctx` + learning + `OVERSIZE`; Guard with I1–I6; cutting by turns; Forwarder separated from assembly; **do not insert extra system messages** (template error on 6/32, §11.1); `ModelProfile` + tokens from the backend (§11.2) | no | low; fixes 400s, silent truncation and failures on Mistral/Gemma 2/Qwen3.5 |
| **B** | Assembler + `ChatState`; epoch 0 with `<profile>`; pinned deltas; deterministic style; `CONTEXT_MODE` | no | medium; the main speed gain (4.7× before overflow) |
| **C** | EpochPlanner: summary (map-reduce), warm-up, preemption; L1 only on epoch change | yes, in the background | medium |
| **D** | Recall from the cut-off part → into the delta | no | low |
| **U** | Single-user mode (§12): `USER_MODE=single`, `OWNER`, `manage.py merge`, chat by history prefix, service requests of any client, `/v1/*` passthrough. Can be done before B/C/D | no | low |

Tests, beyond those listed in v4.9:
- **prefix**: turns `t` and `t+1` without an epoch change, `prompt(t)` is a prefix of `prompt(t+1)` (a property test on random histories with tool calls);
- **agent**: 15 steps of one turn give the same delta and not a single `embed_query` call after the first;
- **regenerate and edit**: delta hash checks;
- **epoch change without a ready epoch**: the L3 fallback fits in `B`;
- **determinism**: two assemblies in a row give byte-identical results with a shifted `time.time()`.

---

## 10. Caveats

- The calculations use the model `pp(d) = pp0/(1+d/D)` and published measurements. Absolute seconds are indicative; ratios are more reliable. Predicted recompute matched the server in 59/60 (v4.9) and 16/16 (v5) requests.
- If the client changes the history itself (Open WebUI RAG in the last message, removing attachments), the prefix breaks on the client side. v5 survives this (new hash → new delta), but the gain on such turns is smaller.
- Cloud APIs with prefix caching (OpenAI, Anthropic, DeepSeek) benefit from the same principle in money as well: cached tokens are cheaper.

---

## 11. Universality: any models and chat templates

The proxy works at the level of OpenAI `messages`, and the template is applied by the backend (llama.cpp `--jinja`, vLLM, LM Studio; Ollama has its own Go templates). So the proxy must not know about `<|im_start|>`. Templates differ only in **what they do with the messages**.

### 11.1 Checking 32 real templates ([`../../bench/templates/`](../../bench/templates/))
Templates from `llama.cpp/models/templates`: Qwen 2.5/3/3.5/QwQ, Gemma 2/4, Llama 3.1/3.2/3.3, Mistral Nemo/Small 3.2/Ministral 3, DeepSeek R1/V3.1/V3.2, gpt-oss, GLM, Kimi, Phi, Granite, Hermes, GigaChat and others. Rendered with jinja2 (as HF does); the prompts of turn `t` and `t+1` were compared.

| Check | ok | only the tail (last answer) is lost | history breaks | template error |
|---|---|---|---|---|
| normal chat | 21 | 8 (reasoning: `<think>` is stripped from history) | 2 | 1 |
| agent step (+ tool result) | 22 | 7 | 1 | 2* |
| next turn after tool calls | 19 | 6 | 5 | 2* |
| date change | 29 | — | 2 | 1 |

\* 1 template (Kimi-K2-Thinking) is an artefact of the jinja2 sandbox (`list.append`), not a template error.

What breaks the prefix **on the template side**:
- **Mistral Nemo** moves system into the **last** user message → a break at the second-to-last question on every turn (like `<memory>` in v4.8).
- **Mistral Small 3.2 / Nemo** insert the tool list `[AVAILABLE_TOOLS]` before the last question → tool definitions are recomputed on every turn.
- **Llama 3.2, gpt-oss** write today's date (`strftime_now`) into system → a full recompute once a day.
- **Reasoning models** (Qwen3.5, QwQ, R1, DeepSeek V3.2, GLM-4.7, MiniMax-M2, Gemma 4, Nemotron) strip reasoning from past answers → the last answer is recomputed. This is cheap and built into the model; after a turn with tools Qwen3.5 also strips reasoning from agent steps.
- **Role restrictions:** Gemma 2 does not support system; Mistral requires strict user/assistant alternation; Ministral 3, Qwen3.5 and Apertus forbid system anywhere but first.

**A v4.8 bug found:** `compress()` inserts a second system (`[РАНЕЕ]`, "[EARLIER]") in the middle, and `inject_memory` adds a system if there is none. On **6 of 32** templates (Qwen3.5, Gemma 2, Mistral Nemo, Mistral Small 3.2, Ministral 3, Apertus) this causes a template error. After overflow the chat on these models stops answering ([`../../bench/templates/v48_out.txt`](../../bench/templates/v48_out.txt)).

### 11.2 v5 rules that make the proxy template-independent
1. **Do not add messages or change roles.** Profile, summary and deltas are inserted only into the `content` of existing user messages (v5 works this way). Then strict templates (alternation, system only first) do not break.
2. **`content` can be a list** (images, files): insert as a separate text part, not with an f-string. Today v4.8 simply skips memory for such messages.
3. **Model capability profile** (`ModelProfile`) **is detected automatically, not by model name:**
   - llama.cpp: `POST /apply-template` on synthetic dialogues (the same checks as in [`check.py`](../../bench/templates/check.py)). vLLM: `/tokenize` with `messages`. Detected: whether system is allowed; whether its position is stable; strictness of alternation; stability of tools; dependence on the date.
   - Strategies by result: `system_mode = native | first_user` (for Gemma 2 and Mistral Nemo the policy and the client's system are folded into the first user message, which is stable for any template); the date via `chat_template_kwargs.date_string`, pinned per epoch (Llama 3.x).
   - Backends without rendering (Ollama, clouds): a safe default strategy (`first_user` is not needed, nothing is added) + observing the cache in practice.
   - Override: `MODEL_PROFILES` (JSON: model-name regex → profile fields).
4. **Tokens** are counted by the backend's tokenizer: llama.cpp and vLLM `/tokenize`, otherwise a character-based estimate with **self-calibration** from `usage.prompt_tokens` per model and a 10% margin. For Cyrillic, tokenizers differ several-fold; one constant for all does not work.
5. **Context limit** (§4.5) per backend: llama.cpp `/props` (`n_ctx` divided by the number of slots `-np`), vLLM `/v1/models` `max_model_len`, Ollama `/api/show` + the actual `num_ctx` (it cannot be passed through `/v1`; `OLLAMA_CONTEXT_LENGTH` or a Modelfile is needed), clouds: a table or env. Plus learning from 400.
6. **Reasoning:** the answer is stored in the DB, facts and style examples **without** `<think>…</think>` / `reasoning_content`. Otherwise, on models without a reasoning parser, the reasoning ends up in memory.
7. **Cache monitor:** the expected reuse (common prefix with the previous request) is compared with the actual one (`timings.prompt_n` in llama.cpp, `usage.prompt_tokens_details.cached_tokens` in vLLM and clouds). A persistent mismatch → a warning in the log and `/stats`: "model X's template breaks the prefix at position …".
8. **Service LLM tasks** (fact extraction, summaries): `response_format` / `json_schema` where the backend supports it (llama.cpp, vLLM, Ollama, LM Studio), plus a tolerant parser. A separate `EXTRACT_MODEL` if the main model is weak.

### 11.3 What the proxy will not fix
- Moving tools before the last question (Mistral): only with your own template in the backend (`--chat-template-file`). The proxy can show this in the cache monitor.
- Reasoning models stripping their reasoning: a property of the model, costs about one answer per turn.

---

## 12. Users and clients: v5 is single-user

**Decision:** v5 is designed and tested for **one user** who works from **any OpenAI-compatible clients**: Open WebUI, LibreChat, SillyTavern, Cline/Roo/Continue, Aider, scripts and curl. Team mode (several people, roles, shared and personal memory, isolation) will be a separate design.

### 12.1 Why not keep the v4.8 behaviour
In v4.8 the user is identified by `X-OpenWebUI-User-Id` or the `user` field. For one person with several clients this is **harmful**:
- Open WebUI with headers writes memory under a UUID, while Cline without headers writes under `default`. One person gets two unrelated memories.
- Some clients fill the `user` field with arbitrary values (session name, character name, installation ID). Memory fragments without the user knowing.
- To get one shared memory today you need to know about `ENABLE_FORWARD_USER_INFO_HEADERS`, `reassign` and `TRUST_USER_HEADER`.

### 12.2 How v5 works
- **One owner.** Every request that passes the key check belongs to the user `OWNER` (default `default`).
  - The `X-OpenWebUI-User-*` headers and the `user` field are **ignored** for identification.
  - The `user` field is forwarded to the backend as is.
- **Access:** one `API_KEY`. Always recommended; without it a warning is logged at startup (as today).
  - `API_KEYS` is not needed in single-user mode.
  - If it is set, all keys lead to `OWNER`. This is handy for giving each client its own key and revoking them one by one. The number of keys is logged at startup.
- **The DB schema does not change.** The `user_id` column stays in all tables (including the new `chat_epochs`, `pinned_deltas`) and always equals `OWNER`. Team mode will later need no migration, only identification logic.
- **The v4.8 multi-user code** (`resolve_user` with headers, `TRUST_USER_HEADER`, key = user) **stays** behind `USER_MODE=multi` "as in v4.8", without further development until the separate team design. The README gets a "Multiple users" section: how to enable `USER_MODE=multi`, `ENABLE_FORWARD_USER_INFO_HEADERS` in Open WebUI, personal `API_KEYS`, the limits of trusting headers. **The default is always `USER_MODE=single`, `OWNER=default`.**
- Backward compatibility for personal keys is not required: an early MVP, nobody has `API_KEYS` configured.

### 12.3 Migrating from v4.8
The database may already contain data under several `user_id`s (`default` and UUIDs from Open WebUI).
- At startup in `single` the proxy counts the `user_id`s that have data. If there are others besides `OWNER`, a warning with the list and the command is logged. Nothing is merged automatically.
- A new command **`manage.py merge --into <owner>`** moves everything to the owner:
  - messages and chats as they are;
  - emotions and style examples are combined;
  - facts: conflicts of single-valued predicates ("lives in …") are resolved **by recency**: the latest `valid_from` stays current, the older one is closed with `valid_to` (history is kept);
  - vectors are recomputed (`reindex`).

  A backup is made with `conn.backup()` before merging.

### 12.4 Chat identification for any client
Without `X-OpenWebUI-Chat-Id`, v4.8 hashes the first user message. Problems:
- two different chats started with "hi" are merged;
- in agent clients the first message is a template with the environment, so different tasks with a similar start are merged too.

v5 binds a request to a chat **by history prefix**:
- **Hash chain.** For each known chat the chain of its message hashes is stored: the same `msg_hash` as for pinned deltas (§4.2); the root is the hash of system and the first question.
- **Choosing the chat.** A request belongs to the chat whose chain is the **longest prefix** of the history sent.
  - Two "hi" chats match only up to the first assistant answer. From the second turn each finds its own chat.
  - On the first turn (only system + "hi") a new chat is created unless there is a chat whose prefix **matches entirely** and has no continuation. Regenerating the first answer behaves the same way.
- **Branches.** If the client regenerated an answer or edited a message in the middle, the history matches up to the edit point, and from there a **branch of the same chat** begins. Facts, the epoch and deltas up to the edit point are reused.
- **Client headers** take priority when present: the list of names in `CHAT_ID_HEADERS` (default `X-OpenWebUI-Chat-Id`). Other clients work by prefix without configuration.
- **Cost:** an index lookup on `(user_id, hash of the first question)` plus a chain comparison, a fraction of a millisecond. No vector search is needed.

### 12.5 What else "any client" requires
- **Client service requests.** Title, tag and suggestion generation exist not only in Open WebUI. Handling:
  - `SERVICE_PATTERNS` becomes a setting, with known LibreChat and SillyTavern patterns added.
  - A heuristic is added: a request without history + small `max_tokens` + an instruction phrase.
  - Such requests pass without memory and without being stored.
- **Transparent passthrough.** The proxy serves only `/v1/chat/completions` and `/v1/models`. Clients that use the same address for `/v1/embeddings`, `/v1/completions`, `/tokenize` get 404 today. In v5 unknown `/v1/*` are forwarded to the backend unchanged (with the key check).
- **Agent client wrappers.** Cline, Roo and Aider wrap the question in their own tags and add the environment (`<environment_details>`, file lists).
  - Fact extraction and memory search work on the user's text without these blocks (`CLIENT_STRIP_PATTERNS`).
  - Such requests are more often classified as `technical`, and memory is not added. This is expected, as today.
- **Slash commands** are recognised only as an exact message. If a client wraps the text, the command will not fire and goes to the model. This is safe; the README will say so.
- **Not in v5:** the native Ollama API (`/api/chat`) and Anthropic (`/v1/messages`). The proxy speaks the OpenAI API, and such clients talk to the backend directly. A possible extension.

### 12.6 Settings
| Variable | Default | Meaning |
|---|---|---|
| `USER_MODE` | `single` | `single`: one owner; `multi`: v4.8 behaviour (frozen until the separate design) |
| `OWNER` | `default` | the owner's `user_id` in `single` |
| `CHAT_ID_HEADERS` | `X-OpenWebUI-Chat-Id` | chat ID headers, in priority order |
| `SERVICE_PATTERNS` | built-in list | extra service-request markers |
| `CLIENT_STRIP_PATTERNS` | built-in list | agent client wrappers stripped before analysis |
| `PASSTHROUGH` | `1` | forward unknown `/v1/*` to the backend |

### 12.7 Tests
- Requests with different `X-OpenWebUI-User-Id`, `user` field values or none give the same `OWNER` memory.
- Every key in `API_KEYS` leads to `OWNER`. A wrong key gives 401.
- Two chats started the same way diverge from the second turn. Regenerating and editing create a branch, not a new chat.
- `merge`: the conflict "lives in Kazan" (old) versus "lives in Samara" (new) — the new one is current, the old one is in history. A backup is created.
- Service requests (Open WebUI, LibreChat, heuristic) are not stored.
- `/v1/embeddings` is forwarded with the key check.

## 13. Implementation (v5.0.0)

Implemented following §9 (A → U → B → C → D) on the `release/v5.0.0` branch; results, tests per stage, the live run and defects found during regression are in [`../audit/v5.0.0-regression.md`](../audit/v5.0.0-regression.md). Deviations from the design:
- cold fact extraction also goes through the BackgroundGate and is bounded by `EXTRACT_MAX_TOKENS` (live run: without it a small model held the only slot for 300 s);
- recall searches only by words of the question that are not in the visible part (the window before the question, the summary, deltas) and does not repeat fragments already shown;
- the head reserve of a new epoch is computed from the actual summary/profile (+800/+300 characters), not from the maximums;
- the warm-up passes the same fields that affect rendering (`tools`, `tool_choice`, `chat_template_kwargs`, …);
- schema (§5): new tables without bumping `user_version` (created with `IF NOT EXISTS`, the schema stays v3). `chat_chain` stores one chain per chat (`pos`, `msg_hash`, `ts`, no `parent_hash`; an edit or regeneration overwrites the tail of the same chat's chain; index on `pos = 0`); `chat_epochs` has `cut_pos`, `elide_pos`, `profile_facts`, `memory`, `date_string` instead of `prefix_tokens`;
- configuration (§8): `DELTA_CH` defaults to 600; `SET_OLLAMA_NUM_CTX` and reading `/api/show` are not implemented: for Ollama the limit is set with `CONTEXT_LIMIT` (and `OLLAMA_CONTEXT_LENGTH` on the Ollama side), otherwise Ollama silently cuts the beginning; `CONTEXT_MODE=legacy` is a plain budget-based sliding window without epochs (not the v4.8 code; `compress()` was removed). Added: `EXTRACT_MAX_TOKENS`, `SUMMARY_MAX_CH`, `SUMMARY_INPUT_MAX`, `ELIDE_KEEP_TURNS`, `CHARS_PER_TOKEN`, `PROFILE_CH`.

The current list of variables is the "Configuration" table in the [README](../../../README.en.md#configuration).

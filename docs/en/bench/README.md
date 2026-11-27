# Benchmarks and calculations

> English translation of [`../../bench/README.md`](../../bench/README.md). The scripts and data live in [`docs/bench/`](../../bench/); paths below are relative to that directory.

The scripts behind the numbers in the [v4.9 design](../design/v4.9-context.md) (§10) and the [v5 design](../design/v5-architecture.md) (§0, §6), and the data from their runs. This is research code: it does not run in CI and is not included in the image.

## Requirements
- Python 3.11, `pip install httpx` (plus `jinja2` for `templates/`).
- llama.cpp `llama-server` with `--jinja` and a small model; the template and tokenizer come from it (`/apply-template`, `/tokenize`). Qwen2.5-0.5B-Instruct Q4_K_M on CPU was used:
  ```bash
  llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf -c 8192 --jinja --host 127.0.0.1 --port 8090 -np 1
  ```
  The address is set in `common.py` (`URL`).

## Scripts

| File | What it does | Output |
|---|---|---|
| [`common.py`](../../bench/common.py) | generator of a synthetic Russian dialogue, `tokens()` via `/apply-template` + `/tokenize`, `chat()` | — |
| [`sim.py`](../../bench/sim.py) | 60 turns: v4.8 compression (an exact copy of `compress()`) versus the v4.9 scheme; recompute per turn = prompt length − common prefix with the previous one | `sim.json` (`OUT=…`) |
| [`sim_v5.py`](../../bench/sim_v5.py) | the same for v5 (epochs, pinned deltas, background preparation) | keys `v5`, `v5_bg` in `sim.json` |
| [`validate.py`](../../bench/validate.py) / [`validate_v5.py`](../../bench/validate_v5.py) | a run on a live llama.cpp: the predicted recompute is compared with `timings.prompt_n` | `validate.json` (59/60 matches), `validate_v5.json` (16/16) |
| [`agent.py`](../../bench/agent.py) | an agent loop with tool calls: v4.8 versus v4.9 | `agent.json` |
| [`overhead.py`](../../bench/overhead.py) | proxy overhead for hashing and parsing long histories | console output |
| [`timing.py`](../../bench/timing.py) | tokens → seconds from published `llama-bench` figures (laptop CPU, M2 Pro, RTX 3090/4090) | [`timing.md`](../../bench/timing.md) (generated, Russian labels; the same tables are translated in §10 of the [v4.9 design](../design/v4.9-context.md#10-latency-v48-vs-v49-calculation--check-on-llamacpp)) |
| [`probe_compress_v48.py`](../../bench/probe_compress_v48.py) | reproduces the v4.8 compression problems (P1–P13); for **v4.8**, put it into `tests/` and run `pytest -s` | — |
| [`live_v5.py`](../../bench/live_v5.py) | live v5 run: proxy → llama.cpp, 22 turns, the full history as Open WebUI sends it; `prompt_n` / `cache_n` and `/health` on every turn | `live_v5.json` |
| [`templates/`](templates/README.md) | prefix stability check of 32 llama.cpp chat templates | `res.json`, `check_out.txt`, `v48_out.txt` |
| [`embeddings/`](embeddings/README.md) | embedding model bench: USER2-small vs granite-embedding-97m-multilingual-r2 (see the [study](../design/embedding-model.md)) | `emb_results.json` |

## Results
- Recompute per turn after overflow (tokens, 60 turns, `COMPRESS_AT=16000`): v4.8 — 3376, v4.9 + warm-up — 1235, **v5 — 278** (max 304), 5 epoch changes, background 43,994 tokens (`EPOCH_AT=0.85`: 8 changes, 70,638).
- `validate_v5.py`: budget 2000, 16 turns — the prediction matched `prompt_n` in 16/16.
- v4.8 overflow probe (`-c 2048`): 7 messages of 4000 characters — `compress` does nothing (8 → 8), llama.cpp returns 400 (7671 > 2048); 13 messages — 14 → 10, again 400 (9194 > 2048).
- `live_v5.py` (v5.0.0, `COMPRESS_AT=3000`, `-c 8192 --cache-ram 300`): a normal turn recomputes ~290 tokens, cache hit 0.816, 4 epoch changes, 0 synchronous shifts, 0 retries on 400. Details: [v5.0.0 regression](../audit/v5.0.0-regression.md).

## Running
```bash
cd docs/bench
python sim.py && python sim_v5.py          # needs llama-server on :8090
python validate_v5.py
python timing.py > timing.md
# live run: llama-server on :8080, the proxy on :8081 with COMPRESS_AT=3000
python live_v5.py live_v5.json
```

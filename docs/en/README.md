# memory-proxy documentation

[Русский](../README.md) | **English**

How the proxy works and why, for those who develop or review it. How to install and configure it is in the [README](../../README.en.md).

These are translations of the Russian documents in [`docs/`](../README.md). If a translation and the original differ, the Russian original is authoritative. Scripts and their raw outputs (`docs/bench/`, `docs/audit/poc/`) are not duplicated; the English documents link to them.

## Design
- [**v5: append-only context**](design/v5-architecture.md): the current architecture (v5.0.0).
  §1 three root causes of the v4.8 problems · §2–§5 principles, components, prompt layout, storage schema · §6 why the answer does not get slower · §8 configuration · §9 stage plan A → U → B → C → D · §11 universality across models and templates · §12 single-user mode and any clients · §13 how the implementation deviates from the design.
- [v4.9: context compression, summarisation and tool calls](design/v4.9-context.md): the predecessor of v5. An analysis of the v4.8 compression problems (P1–P13) and a v4.8 / v4.9 latency calculation on four hardware profiles. Never released separately; what went into v5 is listed in the document header.
- [Embedding model study: USER2-small vs granite-embedding-97m-multilingual-r2](design/embedding-model.md): is granite the ideal multilingual replacement for the default model? Bench on a domain memory set (14 predicates, 7 languages) and public MTEB sets, latency, memory, integration cost.

## Audits and checks
- [UX tests v5.0.0 → v5.0.1](audit/v5.0.1-ux.md): a live run with clients (Open WebUI, OpenAI SDK, SillyTavern, an agent), backend failures and `manage.py`; findings B1–B9 and R1–R7 and their fixes.
- [v5.0.0 regression](audit/v5.0.0-regression.md): tests per stage, 32 llama.cpp templates, a live run, defects found and fixed before the release.
- [v4.7.2 audit](audit/v4.7.2-audit.md): security, privacy, memory correctness, performance, Docker/CI, dependencies; a table of where each finding was closed.
- [v4.7.3 re-audit](audit/v4.7.3-recheck.md): checking the fixes, new findings (CSRF, DNS rebinding, command and classifier regressions), the patch that became v4.7.4.
- [`audit/poc/`](../audit/poc/): output of the PoC runs on v4.7.2, v4.7.3, v4.7.4 and v4.8 (modes `global1`, `global0`, `auth`).

The audits are historical documents: line numbers and code quotes refer to the version under review.

## Benchmarks
- [`bench/`](bench/README.md): calculation and measurement scripts (KV cache recompute simulation, checks on a live llama.cpp, conversion to seconds, the v5 live run) and their results.
- [`bench/templates/`](bench/templates/README.md): prefix stability check on 32 chat templates.
- [`bench/embeddings/`](bench/embeddings/README.md): the embedding model bench (domain memory set, harness, results JSON).

## Version history
[CHANGELOG](../../CHANGELOG.en.md).

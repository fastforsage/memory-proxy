# Prefix stability on chat templates

> English translation of [`../../../bench/templates/README.md`](../../../bench/templates/README.md). The scripts and outputs live in [`docs/bench/templates/`](../../../bench/templates/).

The templates come from [`llama.cpp/models/templates`](https://github.com/ggml-org/llama.cpp/tree/master/models/templates) (32 `.jinja` files). The scripts run in the directory with the templates:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp/models/templates && pip install jinja2
python /path/to/docs/bench/templates/check.py > check_out.txt   # → res.json
python /path/to/docs/bench/templates/v48.py   > v48_out.txt
```

- [`check.py`](../../../bench/templates/check.py) renders turns `t` and `t+1` (a normal chat, an agent step, the next turn after tool calls, a date change) and finds the first position where prompt `t` stops being a prefix of `t+1`; plus role checks (`nosys`, `sys`, `uu`). The results are in §11.1 of the [v5 design](../../design/v5-architecture.md#11-universality-any-models-and-chat-templates).
- [`v48.py`](../../../bench/templates/v48.py): the v4.8 prompt after compression (a second system `[РАНЕЕ]` in the middle): a template error on 6 of 32.
- Kimi-K2-Thinking fails in `ImmutableSandboxedEnvironment` (`list.append`), an artefact of the jinja2 sandbox; `tests/test_v5_templates.py` uses `SandboxedEnvironment`.

The proxy's own regression on these templates is `tests/test_v5_templates.py` (`TPL_DIR=… pytest`), 32/32.

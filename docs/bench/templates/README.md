# Стабильность префикса на шаблонах чата

Шаблоны — из [`llama.cpp/models/templates`](https://github.com/ggml-org/llama.cpp/tree/master/models/templates) (32 файла `.jinja`). Скрипты запускаются в каталоге с шаблонами:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp/models/templates && pip install jinja2
python /path/to/docs/bench/templates/check.py > check_out.txt   # → res.json
python /path/to/docs/bench/templates/v48.py   > v48_out.txt
```

- `check.py` рендерит ход `t` и `t+1` (обычный чат, шаг агента, следующий ход после tool calls, смена даты) и ищет первую позицию, где промпт `t` перестаёт быть префиксом `t+1`; плюс проверки ролей (`nosys`, `sys`, `uu`). Итог — в §11.1 [дизайна v5](../../design/v5-architecture.md).
- `v48.py` — промпт v4.8 после сжатия (второй system `[РАНЕЕ]` в середине): ошибка шаблона на 6 из 32.
- Kimi-K2-Thinking падает в `ImmutableSandboxedEnvironment` (`list.append`) — артефакт песочницы jinja2; в `tests/test_v5_templates.py` используется `SandboxedEnvironment`.

Регресс самого прокси по этим шаблонам — `tests/test_v5_templates.py` (`TPL_DIR=… pytest`), 32/32.

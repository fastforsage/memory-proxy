# Бенчмарки и расчёты

Скрипты, на которых построены цифры в [дизайне v4.9](../design/v4.9-context.md) (§10) и [дизайне v5](../design/v5-architecture.md) (§0, §6), и данные их прогонов. Это исследовательский код: в CI не запускается и в образ не попадает.

## Что нужно
- Python 3.11, `pip install httpx` (для `templates/` ещё `jinja2`).
- llama.cpp `llama-server` с `--jinja` и маленькой моделью — шаблон и токенизатор берутся у него (`/apply-template`, `/tokenize`). Использовалась Qwen2.5-0.5B-Instruct Q4_K_M, CPU:
  ```bash
  llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf -c 8192 --jinja --host 127.0.0.1 --port 8090 -np 1
  ```
  Адрес задан в `common.py` (`URL`).

## Скрипты

| Файл | Что делает | Результат |
|---|---|---|
| `common.py` | генератор синтетического русского диалога, `tokens()` через `/apply-template` + `/tokenize`, `chat()` | — |
| `sim.py` | 60 ходов: сжатие v4.8 (точная копия `compress()`) против схемы v4.9; пересчёт на ход = длина промпта − общий префикс с прошлым | `sim.json` (`OUT=…`) |
| `sim_v5.py` | то же для v5 (эпохи, закреплённые дельты, фоновая подготовка) | ключи `v5`, `v5_bg` в `sim.json` |
| `validate.py` / `validate_v5.py` | прогон на живом llama.cpp: предсказанный пересчёт сравнивается с `timings.prompt_n` | `validate.json` (59/60 совпадений), `validate_v5.json` (16/16) |
| `agent.py` | агентный цикл с tool calls: v4.8 против v4.9 | `agent.json` |
| `overhead.py` | накладные расходы прокси на хеширование и разбор длинных историй | вывод в консоль |
| `timing.py` | токены → секунды по опубликованным `llama-bench` (CPU ноутбука, M2 Pro, RTX 3090/4090) | `timing.md` |
| `probe_compress_v48.py` | воспроизведение проблем сжатия v4.8 (P1–P13); для **v4.8** — кладётся в `tests/` и запускается `pytest -s` | — |
| `live_v5.py` | живой прогон v5: прокси → llama.cpp, 22 хода, полная история как у Open WebUI; на каждом ходе `prompt_n` / `cache_n` и `/health` | `live_v5.json` |
| `templates/` | проверка 32 шаблонов чата llama.cpp на стабильность префикса (см. [`templates/README.md`](templates/README.md)) | `res.json`, `check_out.txt`, `v48_out.txt` |

## Итоги
- Пересчёт на ход после переполнения (токены, 60 ходов, `COMPRESS_AT=16000`): v4.8 — 3376, v4.9 + прогрев — 1235, **v5 — 278** (макс. 304), 5 смен эпохи, фон 43 994 токена (`EPOCH_AT=0.85`: 8 смен, 70 638).
- `validate_v5.py`: бюджет 2000, 16 ходов — предсказание совпало с `prompt_n` в 16/16.
- Проба переполнения v4.8 (`-c 2048`): 7 сообщений по 4000 символов — `compress` ничего не делает (8 → 8), llama.cpp отвечает 400 (7671 > 2048); 13 сообщений — 14 → 10, снова 400 (9194 > 2048).
- `live_v5.py` (v5.0.0, `COMPRESS_AT=3000`, `-c 8192 --cache-ram 300`): обычный ход — ~290 токенов пересчёта, cache hit 0.816, 4 смены эпохи, 0 синхронных сдвигов, 0 повторов по 400. Подробно — [регресс v5.0.0](../audit/v5.0.0-regression.md).

## Запуск
```bash
cd docs/bench
python sim.py && python sim_v5.py          # нужен llama-server на :8090
python validate_v5.py
python timing.py > timing.md
# живой прогон: llama-server на :8080, прокси на :8081 с COMPRESS_AT=3000
python live_v5.py live_v5.json
```

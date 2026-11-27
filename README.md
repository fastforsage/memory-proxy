# 🧠 memory-proxy

**Local-first memory layer for OpenAI-compatible clients (Open WebUI, SillyTavern, LibreChat) → local LLM (llama.cpp, Ollama, vLLM).**

[![Docker Image](https://img.shields.io/badge/docker-fastforsage%2Fmemory--proxy%3Av4.7.2-blue?logo=docker)](https://hub.docker.com/r/fastforsage/memory-proxy)
[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

`memory-proxy` — это FastAPI-прокси между вашим чат-клиентом и локальной LLM. Он даёт модели **долговременную память**, **семантический поиск**, **эмоциональные якоря**, **сквозной контекст между чатами** и **профиль стиля** — без облаков, без GPU, без утечек.

---

## ✅ Что реально работает (v4.7.2)

| Фича | Статус | Что делает |
| :--- | :--- | :--- |
| **Temporal Facts** | ✅ | История изменений (`works_at`, `lives_in`) с `valid_from` / `valid_to` |
| **Hybrid Search** | ✅ | FTS5 (BM25) + sqlite-vec (KNN) + RRF |
| **USER2-small embeddings** | ✅ | ONNX Runtime, FP32, ~5 мс на CPU, русский |
| **Emotional Anchors** | ✅ | `Jira: -1.0` с экспоненциальным затуханием |
| **Mood Engine** | ✅ | `Focus` / `Irritated` / `Relaxed` с TTL 30 мин |
| **Context Bridge** | ✅ | `/chats`, `/pull`, `/search` + auto-триггеры |
| **Style Profile** | ✅ | Few-shot из ответов, оценённых пользователем |
| **Semantic Classifier** | ✅ | Код/техника через zlib не засоряет память |
| **Proof-Check** | ✅ | Facts/emotions только с verbatim `evidence`/`quote` |
| **Compression** | ⚠️ | Наивная (усечение), не LLM-суммаризация |
| **API_KEY** | ✅ | Опциональная Bearer-аутентификация |
| **GLOBAL_MEMORY** | ✅ | Изоляция фактов по `chat_id` (опционально) |
| **Unit tests** | ✅ | `pytest` в `tests/test_proxy.py` |
| **CI** | ✅ | GitHub Actions → Docker Hub |

**Чего пока НЕТ:**
- LLM-суммаризация (только усечение контекста)
- Фоновая консолидация (dream phase)
- Multi-language embeddings (только RU + базовый EN)

---

## 📊 Производительность

| Метрика | Значение |
| :--- | :--- |
| Hot Path overhead | **< 10 мс** |
| Embedding (CPU) | **~5 мс** (ONNX, USER2-small) |
| RAM на инференс | **~50 MB** |
| Размер образа | **~500 MB** (без PyTorch/CUDA) |
| Первый старт | ~15–25 сек (ONNX + tokenizer) |

---

## 🚀 Быстрый старт

### Docker Compose

```yaml
services:
  memory-proxy:
    image: fastforsage/memory-proxy:v4.7.2
    container_name: memory-proxy
    ports:
      - "127.0.0.1:8081:8081"
    environment:
      - LLAMA_URL=http://host.docker.internal:8080/v1/chat/completions
      - DB_PATH=/data/memory.db
      - COMPRESS_AT=22000
      - KEEP_RECENT=8
    volumes:
      - ./data:/data
    extra_hosts:
      - "host.docker.internal:host-gateway"
    restart: unless-stopped
```

```bash
docker compose up -d
curl http://127.0.0.1:8081/health
# {"status": true, "db": "ok"}
```

### Подключение клиента

| Клиент | URL | API Key |
| :--- | :--- | :--- |
| **Open WebUI** | `http://host.docker.internal:8081/v1` | `sk-no-key` |
| **SillyTavern** | `http://127.0.0.1:8081/v1` | `sk-no-key` |
| **curl** | `http://127.0.0.1:8081/v1/chat/completions` | — |

---

## ⚙️ Конфигурация

| ENV | Default | Описание |
| :--- | :--- | :--- |
| `LLAMA_URL` | `http://127.0.0.1:8080/v1/chat/completions` | OpenAI-совместимый endpoint |
| `DB_PATH` | `/data/memory.db` | Путь к SQLite |
| `COMPRESS_AT` | `22000` | Порог токенов для компрессии |
| `KEEP_RECENT` | `8` | Последних сообщений при сжатии (min=1) |
| `MAX_INPUT_LEN` | `200000` | Жёсткий лимит длины сообщения |
| `API_KEY` | `""` | **Опционально.** Bearer-токен. Пусто = без авторизации |
| `CORS_ORIGINS` | `""` | Comma-separated. Пусто = `*` (только локально!) |
| `GLOBAL_MEMORY` | `1` | `1` = общие факты, `0` = изоляция по `chat_id` |
| `LOG_LEVEL` | `INFO` | Уровень логирования |
| `MODEL_DIR` | `/app/model` | ONNX-модель + tokenizer |

---

## 🔒 Безопасность

**По умолчанию:** без авторизации, слушает `127.0.0.1:8081` (не наружу).

**Для многопользовательского использования:**
```yaml
environment:
  - API_KEY=your-secret-key
  - CORS_ORIGINS=http://localhost:3000
  - GLOBAL_MEMORY=0   # изоляция фактов по чату
```

⚠️ **НЕ используйте** `network_mode: host` — это откроет сервис всей LAN.

⚠️ **НЕ выставляйте** наружу без reverse-proxy с HTTPS и rate-limiting.

---

## 🎮 Слэш-команды

| Команда | Действие |
| :--- | :--- |
| `/chats` | Список чатов с превью |
| `/pull <N>` | Контекст чата N (последние 15 сообщений) |
| `/pull <id>` | По первым 8 символам `chat_id` |
| `/search <query>` | FTS5 по всем чатам |
| `/help` | Справка |

**Auto-триггеры** (без команды): «помнишь», «обсуждали», «в другом чате», «в прошлый раз», «мы с тобой» и т.д.

---

## 🧪 Примеры

### Память с историей

```
Ты: Я живу в Москве
   → lives_in: Москве (valid_from=now)

Ты: Я переехал в Питер
   → Москве.valid_to = now, Питере (valid_from=now)

Ты: Где я жил раньше?
   → «В Москве» (из [ИСТОРИЯ])
```

### Семантический поиск

```
Ты: Я люблю программировать на Python и Rust

Ты: Что я люблю делать?
   → hybrid search (FTS5 + vector) находит Python и Rust
     (даже если в запросе нет слова «Python»)
```

### Эмоции

```
Ты: Ненавижу Jira, она бесит
   → emotion: Jira = -1.0

Ты: Что ты знаешь о Jira?
   → модель видит «Негатив: Jira» → отвечает сухо
```

### Context Bridge

```
[в чате A]
Ты: Мы обсуждали FastAPI + JWT

[в чате B]
Ты: Помнишь, что мы обсуждали про архитектуру?
   → [КОНТЕКСТ ИЗ ДРУГИХ ЧАТОВ] — модель подтягивает фрагмент из A
```

### Style Profile

```
[10 диалогов спустя]
Пользователь: «спасибо, отлично» → +1 к последнему примеру
Пользователь: «плохо, переделай» → -2 (удаляется при ≤ -2)
→ 3 случайных примера с score ≥ 0 подмешиваются в промпт
```

---

## 🏗 Архитектура

```
Client → FastAPI (uvloop, orjson)
  ├─ API_KEY check
  ├─ Validation (body, chat_id, content)
  ├─ preclassify (zlib) → personal / technical / mixed
  ├─ Memory assembly:
  │   ├─ get_smart_facts (FTS5 + vec_facts RRF)
  │   ├─ search_other_chats (auto-trigger)
  │   ├─ emotions + mood_state
  │   ├─ summaries
  │   └─ style_examples (few-shot)
  ├─ Compression (если > COMPRESS_AT)
  ├─ Streaming (SSE-buffer, status_code passthrough)
  └─ Cold Path (async):
      ├─ LLM extractor (facts + emotions + mood)
      ├─ Proof-check (verbatim)
      ├─ Embedding (batch ONNX)
      └─ Style scoring
```

### Схема БД

```
messages      — все сообщения (chat_id, role, content, content_type)
messages_fts  — FTS5 по messages (для cross-chat поиска)
facts         — факты с temporal validity (valid_from, valid_to)
facts_fts     — FTS5 по facts
vec_facts     — sqlite-vec эмбеддинги (384-dim, USER2-small)
emotions      — эмоциональные якоря (trigger, score, decay)
mood_state    — текущее настроение (TTL 30 мин)
summaries     — LLM-суммаризации (при overflow)
style_examples — few-shot примеры стиля с оценкой
```

Миграции применяются автоматически при старте (`ALTER TABLE IF NOT EXISTS`).

---

## 🔍 Диагностика

### Логи

```bash
docker logs -f memory-proxy | grep -E '\[(Cold|Compress|WARN|ERROR|Style)\]'
```

### Проверить БД

```bash
docker exec -it memory-proxy python -c "
import sqlite3, sqlite_vec
c = sqlite3.connect('/data/memory.db')
c.enable_load_extension(True); sqlite_vec.load(c)
print('Факты:', c.execute('SELECT predicate, object FROM facts WHERE valid_to IS NULL').fetchall())
print('История:', c.execute('SELECT predicate, object, valid_to FROM facts WHERE valid_to IS NOT NULL LIMIT 5').fetchall())
print('Векторов:', c.execute('SELECT COUNT(*) FROM vec_facts').fetchone()[0])
print('Стиль:', c.execute('SELECT context, score, substr(text,1,40) FROM style_examples ORDER BY score DESC LIMIT 5').fetchall())
"
```

### Проверить токенизатор

```bash
docker logs memory-proxy | grep -i token
# Ожидаемое: "✅ Tokenizer (fast) loaded, vocab=..."
```

### Тесты

```bash
docker run --rm -v ./:/app -w /app python:3.11-slim bash -c "
pip install -q pytest fastapi httpx orjson sqlite-vec onnxruntime transformers numpy
pytest tests/ -v
"
```

---

## 🛠 Сборка из исходников

```bash
git clone https://github.com/fastforsage/memory-proxy.git
cd memory-proxy
docker build -t fastforsage/memory-proxy:v4.7.2 .
docker push fastforsage/memory-proxy:v4.7.2
```

### Структура проекта

```
memory-proxy/
├── Dockerfile
├── requirements.txt
├── proxy.py              # FastAPI app (~600 строк)
├── download_model.py     # HF snapshot downloader
├── docker-compose.yml
├── LICENSE (MIT)
├── README.md
├── tests/
│   └── test_proxy.py     # pytest, 8 тестов
├── .github/workflows/
│   └── docker-publish.yml
└── data/                 # SQLite volume
```

---

## 🆚 Сравнение с аналогами

| Проект | Стек | Семантика | Context Bridge | Локальность | Размер |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **memory-proxy** | SQLite + sqlite-vec | Hybrid (FTS+vec) | ✅ | ✅ Full | **~500 MB** |
| Mem0 | Qdrant + KV | Vector | ⚠️ Partial | ❌ Cloud-first | > 2 GB |
| Letta (MemGPT) | Postgres + pgvector | Vector | ⚠️ Per-session | ❌ Server | > 1 GB |
| Zep / Graphiti | Neo4j | Graph+Vec | ⚠️ Episodes | ❌ Cloud | > 3 GB |

**Уникальное у memory-proxy:**
- 🎯 **Single-file SQLite** — вся память в одном `.db`, легко бэкапить
- 🇷🇺 **USER2-small** — специально обученная русскоязычная модель
- 🧬 **Temporal facts** — история изменений, не только «текущее»
- 🌉 **Context Bridge** — сквозной поиск по чатам из коробки
- ⚡ **~500 MB** — работает на любом CPU без GPU
- 🎨 **Style Profile** — few-shot эволюция стиля без переобучения

---

## 🐛 Известные ограничения

1. **Компрессия** — усечение по `KEEP_RECENT`, не LLM-суммаризация. Данные из старых сообщений (>8) теряются.
2. **Single-thread SQLite** — `check_same_thread=False` безопасен только для 1 uvicorn-воркера.
3. **Ollama** — `LLAMA_URL` должен указывать на полный `/v1/chat/completions` endpoint.
4. **Vision-контент** — пропускается без анализа (только текст).

---

## 📋 Roadmap

- [x] v4.0 — Temporal facts, эмоции, mood
- [x] v4.4 — Context Bridge (cross-chat)
- [x] v4.5 — Semantic search (PyTorch)
- [x] v4.6 — ONNX Runtime (×3 скорость, ×7 меньше)
- [x] v4.7 — Style Profile + все фиксы аудита
- [x] **v4.7.2 — Fast tokenizer (fix startup)**
- [ ] v4.8 — LLM-суммаризация вместо усечения
- [ ] v4.9 — Conflict resolution (один predicate — много значений)
- [ ] v5.0 — Мультиязычные эмбеддинги + авто-перевод

---

## 🤝 Contributing

PR welcome. Если нашли баг — создайте issue с:
1. Вывод `docker logs memory-proxy --tail 100`
2. Запрос, который сломался
3. Версия (`docker inspect memory-proxy | grep Image`)

---

## 📄 License

MIT © [fastforsage](LICENSE)

---

<p align="center">
  <i>Built with ❤️ for local-first AI</i>
</p>
```

# 🧠 Memory Proxy for Local LLMs (v4 Temporal)

**Умный слой памяти и эмоционального интеллекта для локальных LLM (llama.cpp / Ollama), совместимый с Open WebUI.**

Прокси-сервер перехватывает запросы в формате OpenAI API, незаметно для пользователя извлекает факты, эмоции и текущее настроение в фоновом режиме, и внедряет их в системный промпт. Это дает локальной модели «бесконечную» память, адаптивный тон общения и защиту от галлюцинаций, не замедляя генерацию ответа.

## 🎯 Функциональные требования (ТЗ v4)

### Основные функции:
1. **Прозрачный прокси:** Полная совместимость с OpenAI API (эндпоинты `/v1/chat/completions`, `/v1/models`).
2. **Temporal Facts (История):** Система не удаляет старые факты. При смене работы или города старый факт помечается как исторический (`valid_to`), а новый становится активным. Модель может ответить на вопрос «А где я жил раньше?», опираясь на хронологию.
3. **Безопасная инвалидация:** Предикаты разделены на `SINGLE_VALUED` (город, работа — перезаписываются) и `MULTI_VALUED` (навыки, хобби — накапливаются). Это защищает от потери данных, когда вы говорите «Я знаю Python и Rust».
4. **Эмоциональный интеллект:** Извлечение эмоционального фона к триггерам (темам/инструментам) с накоплением веса.
5. **Time Decay (Затухание):** Эмоции имеют свойство «остывать» со временем. Система применяет экспоненциальное затухание (полураспад ~30 дней). Если вы перестали ненавидеть Jira, её негативный вес снизится.
6. **Mood Engine:** Определяет текущее состояние пользователя (`Focus`, `Relaxed`, `Irritated` и др.) с TTL (30 минут) и мягко корректирует стиль ответа.
7. **Динамическое сжатие (Adaptive Context):** Порог сжатия контекста зависит от текущего настроения. Если вы в состоянии `Focus`, порог увеличивается (до 33k токенов), чтобы модель не теряла важные детали.
8. **Proof-Check (Анти-галлюцинации):** Факты и эмоции принимаются только если LLM приложила дословную цитату из текста пользователя.

### Нефункциональные требования:
* **Zero-Latency (Hot Path):** Ответ пользователю начинает стримиться мгновенно. Извлечение памяти работает в фоне через `asyncio.create_task`.
* **KV-Cache Optimization:** Динамический блок памяти вставляется как отдельное `system`-сообщение перед последней репликой пользователя. Это сохраняет статичный префикс промпта и позволяет llama.cpp переиспользовать KV-кэш.
* **Single-Pass Cold Path:** Один запрос к LLM извлекает сразу Факты, Эмоции и Настроение (Mood), возвращая комплексный JSON. Это снижает нагрузку на GPU в 2–3 раза.
* **Локальность:** Никаких внешних API (OpenAI, Anthropic). База данных — локальный SQLite.
* **Управление очередью GPU:** Использование `asyncio.Semaphore(1)` для предотвращения конфликтов, если llama.cpp запущен с `--parallel 1`.

---

## 🏗 Архитектура решения

```mermaid
sequenceDiagram
    participant U as Open WebUI
    participant P as Memory Proxy (FastAPI)
    participant DB as SQLite (Temporal DB)
    participant L as llama.cpp (Gemma 4)

    U->>P: POST /v1/chat/completions (User Text)
    activate P
    P->>DB: SELECT Active Facts + History + Mood
    DB-->>P: Context Data
    
    Note over P: Внедрение контекста перед последним сообщением (KV-Cache Friendly)
    P->>L: Forward Request (Stream)
    L-->>P: Stream Chunks
    P-->>U: Stream Chunks
    deactivate P

    Note over P, L: Фоновый процесс (Single-Pass Cold Path)
    P->>L: Prompt: "Extract JSON Facts + Emotions + Mood"
    L-->>P: JSON Response
    Note over P: Proof-Check (цитата есть в тексте?)
    P->>DB: UPSERT Facts (valid_from / valid_to)
    P->>DB: UPDATE Emotions (с учетом Decay)
    P->>DB: INSERT Mood (с TTL)

Компоненты:
Hot Path: Всё, что находится между получением запроса и отправкой его в llama.cpp. Работает <10 мс. Включает в себя сборку промпта из уже готовых данных в БД.

Cold Path (Фоновый): Запускается после ответа пользователю. Отправляет текст на анализ модели, проверяет галлюцинации (verbatim match), обновляет SQLite и рассчитывает исторические метки.

Математика эмоций: new_score = (old_score * exp(-0.023 * days_passed)) + valence.

🚀 Руководство по установке
Предварительные требования:
Docker и Docker Compose.

Запущенный сервер llama.cpp (в той же Docker-сети или доступный по IP).

1. Создание общей сети
bash
docker network create llm-net
2. Запуск через Docker Compose
Создайте файл docker-compose.yml:

yaml
version: '3.8'

services:
  memory-proxy:
    image: fastforsage/memory-proxy:v4
    container_name: memory-proxy
    environment:
      - LLAMA_URL=http://gemma4-hauhaucs:8080/v1/chat/completions
      - DB_PATH=/data/memory.db
      - COMPRESS_AT=22000
      - KEEP_RECENT=8
      - MOOD_TTL=1800
    volumes:
      - ./data:/data
    ports:
      - "8081:8081"
    networks:
      - llm-net
    restart: unless-stopped

networks:
  llm-net:
    external: true
Запустите:

bash
docker compose up -d
3. Настройка Open WebUI
Зайдите в настройки Open WebUI -> Подключения.

Укажите URL: http://memory-proxy:8081/v1 (если Open WebUI в той же сети) или http://<IP_хоста>:8081/v1.

Ключ API: любой (например, sk-no-key-required).

📖 Руководство пользователя
После установки прокси работает незаметно. Вам не нужно менять привычки общения.

Как работает память:
Факты и история: Если вы скажете "Я живу в Питере и работаю в Сбере", прокси в фоне извлечет это. В следующем чате модель уже будет знать ваш город и место работы. Если вы скажете "Я переехал в Москву", старый факт про Питер останется в истории, и на вопрос "Где я жил раньше?" модель ответит с учетом хронологии.

Эмоции и затухание: Если вы скажете "Я обожаю Python, но Jira — это просто кошмар", прокси запишет Python: +1.0, Jira: -1.0. В будущем, когда речь зайдет о Jira, модель будет отвечать более сухо. Если через месяц вы скажете "О, Jira починили", негативный вес начнет снижаться.

Настроение (Mood): Если вы напишете "Так, мне нужно срочно сосредоточиться", система на 30 минут включит режим Focus. В этом режиме модель будет отвечать кратко, а порог сжатия контекста увеличится, чтобы не потерять важные детали.

Просмотр базы данных (для администратора):
Активные факты:

bash
docker exec -it memory-proxy python -c "
import sqlite3
c = sqlite3.connect('/data/memory.db')
for row in c.execute('SELECT predicate, object FROM facts WHERE valid_to IS NULL'):
    print(f'{row[0]}: {row[1]}')
"
История (прошлые факты):

bash
docker exec -it memory-proxy python -c "
import sqlite3
c = sqlite3.connect('/data/memory.db')
for row in c.execute('SELECT predicate, object FROM facts WHERE valid_to IS NOT NULL LIMIT 10'):
    print(f'{row[0]}: {row[1]}')
"
Эмоциональный профиль:

bash
docker exec -it memory-proxy python -c "
import sqlite3
c = sqlite3.connect('/data/memory.db')
for row in c.execute('SELECT trigger, score, mention_count FROM emotions ORDER BY score ASC LIMIT 10'):
    print(f'{row[0]}: {row[1]:.2f} (упомянуто {row[2]} раз)')
"
Текущее настроение:

bash
docker exec -it memory-proxy python -c "
import sqlite3
c = sqlite3.connect('/data/memory.db')
print(c.execute('SELECT state, intensity FROM mood_state ORDER BY ts DESC LIMIT 1').fetchone())
"
⚙️ Описание настроек (Environment Variables)
Переменная	По умолчанию	Описание
LLAMA_URL	http://127.0.0.1:8080/v1/chat/completions	Полный URL до API вашего llama.cpp / Ollama.
DB_PATH	/data/memory.db	Путь к файлу SQLite внутри контейнера. Обязательно маунтить volume.
COMPRESS_AT	22000	Базовый лимит токенов для срабатывания умного сжатия.
KEEP_RECENT	8	Сколько последних сообщений оставить нетронутыми при сжатии.
MOOD_TTL	1800	Время жизни текущего настроения (в секундах). По умолчанию 30 минут.
🛠 Технологический стек
Python 3.11 + FastAPI (асинхронный сервер).

SQLite (WAL mode) — надежное и быстрое локальное хранилище с поддержкой временных меток.

httpx — асинхронные запросы к llama.cpp.

Docker — изоляция и легкое развертывание.

📄 Лицензия
MIT License. Используйте, модифицируйте и распространяйте свободно.

Сделано с любовью к локальным LLM и чистому коду.

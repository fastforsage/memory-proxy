"""Интеграционные тесты через HTTP (TestClient + mock LLM)."""
import json
import pytest


# ─── Базовое ───
def test_health(ctx):
    r = ctx.client.get("/health")
    assert r.status_code == 200 and r.json()["status"] is True and r.json()["version"] == ctx.proxy.VERSION


def test_models_proxied(ctx):
    r = ctx.client.get("/v1/models")
    assert r.status_code == 200 and r.json()["data"][0]["id"] == "test-model"


def test_invalid_messages_rejected(ctx):
    r = ctx.client.post("/v1/chat/completions", json={"messages": ["oops"]})
    assert r.status_code == 400


# ─── Безопасность ───
def test_csrf_text_plain_rejected(ctx):
    body = json.dumps({"messages": [{"role": "user", "content": "Я работаю в EvilCorp"}]})
    r = ctx.client.post("/v1/chat/completions", content=body, headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    assert ctx.db("SELECT COUNT(*) FROM messages")[0][0] == 0


def test_cross_site_browser_request_rejected(ctx):
    r = ctx.chat("привет", headers={"Sec-Fetch-Site": "cross-site", "Origin": "https://evil.example"})
    assert r.status_code == 403


def test_cross_site_allowed_for_configured_origin(make_ctx):
    c = make_ctx(CORS_ORIGINS="http://localhost:3000")
    r = c.chat("привет", headers={"Sec-Fetch-Site": "same-site", "Origin": "http://localhost:3000"})
    assert r.status_code == 200


def test_untrusted_host_rejected(ctx):
    r = ctx.client.get("/v1/models", headers={"Host": "attacker-rebind.example:8081"})
    assert r.status_code == 400


@pytest.mark.parametrize("host", ["localhost:8081", "127.0.0.1:8081", "host.docker.internal:8081", "memory-proxy:8081"])
def test_default_trusted_hosts(ctx, host):
    assert ctx.client.get("/health", headers={"Host": host}).status_code == 200


def test_custom_allowed_hosts(make_ctx):
    c = make_ctx(ALLOWED_HOSTS="localhost,my-proxy")
    assert c.client.get("/health", headers={"Host": "my-proxy:8081"}).status_code == 200
    assert c.client.get("/health", headers={"Host": "memory-proxy:8081"}).status_code == 400


def test_auth(make_ctx):
    c = make_ctx(API_KEY="s3cret")
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert c.client.post("/v1/chat/completions", json=body).status_code == 401
    assert c.client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.client.get("/v1/models").status_code == 401


def test_auth_non_ascii_header_is_401_not_500(make_ctx):
    c = make_ctx(API_KEY="s3cret")
    r = c.client.post("/v1/chat/completions", json={"messages": []},
                      headers={"Authorization": "Bearer s3cr\u00e9t".encode("latin-1")})
    assert r.status_code == 401


def test_cors_star_requires_api_key(make_ctx):
    with pytest.raises(RuntimeError):
        make_ctx(CORS_ORIGINS="*")


def test_cors_disabled_by_default(ctx):
    r = ctx.client.options("/v1/chat/completions", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


# ─── Слэш-команды ───
def test_commands_work(ctx):
    ctx.chat("Мой любимый фреймворк FastAPI", chat_id="chat-one")
    ctx.drain()
    r = ctx.chat("/chats", chat_id="chat-two")
    text = r.json()["choices"][0]["message"]["content"]
    assert "Твои чаты" in text and "chat-one" in text
    r = ctx.chat("/pull chat-one", chat_id="chat-two")
    assert "FastAPI" in r.json()["choices"][0]["message"]["content"]
    r = ctx.chat("/search фреймворк", chat_id="chat-two")
    assert "FastAPI" in r.json()["choices"][0]["message"]["content"]
    r = ctx.chat("/help")
    assert "/search" in r.json()["choices"][0]["message"]["content"]
    assert ctx.llm.chat_requests[-1]["messages"][-1]["content"] == "Мой любимый фреймворк FastAPI"   # команды не ушли в LLM


def test_commands_hidden_from_chats_and_pull(ctx):
    ctx.chat("Обычное сообщение про отпуск", chat_id="chat-one")
    ctx.chat("/help", chat_id="chat-one")
    ctx.chat("/chats", chat_id="cmd-only")
    listing = ctx.chat("/chats", chat_id="cmd-only").json()["choices"][0]["message"]["content"]
    assert "chat-one" in listing and "cmd-only" not in listing
    pulled = ctx.chat("/pull chat-one", chat_id="cmd-only").json()["choices"][0]["message"]["content"]
    assert "отпуск" in pulled and "/help" not in pulled and "Команды memory-proxy" not in pulled


def test_command_usage_without_argument(ctx):
    for cmd in ("/pull", "/search"):
        assert "Использование" in ctx.chat(cmd).json()["choices"][0]["message"]["content"]


def test_command_stream_format(ctx):
    r = ctx.chat("/help", stream=True)
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: {")]
    assert events[0]["object"] == "chat.completion.chunk" and events[0]["choices"][0]["delta"]["content"]
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert r.text.rstrip().endswith("data: [DONE]")


def test_command_non_stream_format(ctx):
    j = ctx.chat("/help").json()
    assert j["object"] == "chat.completion" and j["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.parametrize("text", ["/etc/nginx/nginx.conf — что за файл?", "/help me write a bash script"])
def test_slash_text_goes_to_llm(ctx, text):
    ctx.chat(text)
    assert ctx.llm.chat_requests and ctx.llm.chat_requests[-1]["messages"][-1]["content"] == text


# ─── Память ───
def test_memory_in_first_system_message(ctx):
    ctx.llm.cold(facts=[{"predicate": "lives_in", "object": "Казань", "evidence": "я живу в казани"}])
    ctx.chat("Я живу в Казани")
    ctx.drain()
    history = [{"role": "system", "content": "You are helpful"},
               {"role": "user", "content": "привет"}, {"role": "assistant", "content": "привет!"}]
    ctx.chat("Где я живу? Напомни, пожалуйста", history=history)
    msgs = ctx.llm.chat_requests[-1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"].startswith("You are helpful") and "lives_in: Казань" in msgs[0]["content"]


def test_technical_message_skips_memory(ctx):
    ctx.chat("```python\nprint('hello')\n```")
    assert all(m["role"] != "system" for m in ctx.llm.chat_requests[-1]["messages"])
    ctx.drain()
    assert not ctx.llm.cold_requests


def test_service_request_passthrough(ctx):
    ctx.chat("Расскажи коротко про Python")
    ctx.drain()
    before_msgs = ctx.db("SELECT COUNT(*) FROM messages")[0][0]
    ex_before = ctx.proxy.LAST_EXAMPLE_ID.get("chat-a")
    ctx.chat("### Task:\nGenerate 1-3 broad tags categorizing the main themes of the chat history")
    ctx.drain()
    assert ctx.db("SELECT COUNT(*) FROM messages")[0][0] == before_msgs
    assert ctx.proxy.LAST_EXAMPLE_ID.get("chat-a") == ex_before
    assert "system" not in [m["role"] for m in ctx.llm.chat_requests[-1]["messages"]]


def test_stream_answer_saved(ctx):
    r = ctx.chat("Привет, как дела?", stream=True)
    deltas = [json.loads(l[6:])["choices"][0]["delta"]["content"] for l in r.text.splitlines() if l.startswith("data: {")]
    assert "".join(deltas) == ctx.llm.ANSWER          # чанки переданы клиенту без изменений
    rows = ctx.db("SELECT content FROM messages WHERE role='assistant'")
    assert rows and rows[-1][0] == ctx.llm.ANSWER


def test_regenerate_does_not_duplicate(ctx):
    ctx.chat("Я люблю зелёный чай")
    ctx.chat("Я люблю зелёный чай")
    ctx.drain()
    assert ctx.db("SELECT COUNT(*) FROM messages WHERE role='user'")[0][0] == 1
    assert len(ctx.llm.cold_requests) == 1


def test_upstream_error_status_passthrough(ctx):
    import httpx
    ctx.proxy.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(503, json={"error": "busy"})))
    assert ctx.chat("привет").status_code == 503
    assert ctx.chat("привет ещё раз", stream=True).status_code == 503


# ─── Фоновый анализ ───
def test_cold_sends_model(ctx):
    ctx.chat("Я живу в Казани", model="qwen-7b")
    ctx.drain()
    assert ctx.llm.cold_requests[-1]["model"] == "qwen-7b"


def test_cold_uses_extract_model(make_ctx):
    c = make_ctx(EXTRACT_MODEL="small-extractor")
    c.chat("Я живу в Казани", model="big")
    c.drain()
    assert c.llm.cold_requests[-1]["model"] == "small-extractor"


def test_cold_saves_fact_with_vector_and_invalidates_old(ctx):
    ctx.llm.cold(facts=[{"predicate": "lives_in", "object": "Москва", "evidence": "я живу в москве"}])
    ctx.chat("Я живу в Москве")
    ctx.drain()
    ctx.llm.cold(facts=[{"predicate": "lives_in", "object": "Казань", "evidence": "я переехал в казань"}])
    ctx.chat("Я переехал в Казань")
    ctx.drain()
    facts = ctx.db("SELECT id, object, valid_to IS NULL FROM facts WHERE predicate='lives_in' ORDER BY id")
    assert [(o, live) for _, o, live in facts] == [("Москва", 0), ("Казань", 1)]
    vec_ids = {r[0] for r in ctx.db("SELECT rowid FROM vec_facts")}
    assert facts[0][0] not in vec_ids and facts[1][0] in vec_ids


def test_cold_rejects_hallucinations(ctx):
    ctx.llm.cold(facts=[{"predicate": "works_at", "object": "Google", "evidence": ""},
                        {"predicate": "works_at", "object": "Google", "evidence": "я работаю в яндексе"},
                        {"predicate": "hacked", "object": "x", "evidence": "я работаю в яндексе"}])
    ctx.chat("Я работаю в Яндексе, а в Google меня не взяли")
    ctx.drain()
    assert ctx.db("SELECT object FROM facts") == []


def test_cold_bad_item_does_not_drop_valid_ones(ctx):
    ctx.llm.cold(facts=[{"predicate": "likes", "object": "чай", "evidence": 12345},
                        "мусор",
                        {"predicate": "lives_in", "object": "Казань", "evidence": "я переехал в казань"}],
                 emotions=[None, {"trigger": "Jira", "valence": -1.0, "quote": "ненавижу jira"}])
    ctx.chat("Я переехал в Казань и ненавижу Jira")
    ctx.drain()
    assert ctx.db("SELECT object FROM facts") == [("Казань",)]
    assert ctx.db("SELECT trigger FROM emotions") == [("Jira",)]


def test_cold_garbage_json_is_ignored(ctx):
    ctx.llm.cold(raw="это не json")
    ctx.chat("Я живу в Казани")
    ctx.drain()
    ctx.llm.cold(raw='["list", "not", "object"]')
    ctx.chat("Я живу в Самаре")
    ctx.drain()
    assert ctx.db("SELECT COUNT(*) FROM facts")[0][0] == 0
    assert ctx.client.get("/health").status_code == 200


def test_emotion_inflected_trigger_and_decay(ctx):
    import time
    ctx.llm.cold(emotions=[{"trigger": "работа", "valence": -1.0, "quote": "ненавижу работу"}])
    ctx.chat("Ненавижу работу, всё надоело")
    ctx.drain()
    assert ctx.db("SELECT trigger FROM emotions") == [("работа",)]
    assert "Негатив: работа" in ctx.proxy.get_emotion_lines()
    assert ctx.proxy.get_emotion_lines(now=time.time() + 90 * 86400) == []    # через 90 дней затухло


def test_style_feedback_scoring(ctx):
    first = "Первый вариант ответа: " + "достаточно длинный текст для примера стиля. " * 3
    second = "Второй вариант ответа: " + "тоже достаточно длинный текст для примера стиля. " * 3
    ctx.llm.ANSWER = first
    ctx.chat("Расскажи про Python")
    ctx.llm.ANSWER = second
    ctx.chat("Не так, переделай")                 # критика → первый пример удаляется
    texts = [r[0] for r in ctx.db("SELECT text FROM style_examples")]
    assert first not in texts and second in texts
    ctx.llm.ANSWER = "Третий ответ: " + "ещё один длинный текст для примера стиля. " * 3
    ctx.chat("Спасибо, отлично!")                 # похвала → второй пример +1
    assert ctx.db("SELECT score FROM style_examples WHERE text=?", second)[0][0] == 1


# ─── Миграции ───
def test_embedding_migration_reembeds_legacy_vectors(make_ctx):
    c = make_ctx()
    c.llm.cold(facts=[{"predicate": "lives_in", "object": "Казань", "evidence": "я живу в казани"}])
    c.chat("Я живу в Казани")
    c.drain()
    fid = c.db("SELECT id FROM facts")[0][0]
    import numpy as np
    legacy = np.ones(384, dtype=np.float32).tobytes()
    with c.proxy._conn:
        c.proxy._conn.execute("UPDATE vec_facts SET embedding=? WHERE rowid=?", (legacy, fid))
    c.proxy._conn.execute("PRAGMA user_version=0"); c.proxy._conn.commit()
    c.client.__exit__(None, None, None)

    c2 = make_ctx()                                   # та же БД, новый старт
    assert c2.db("PRAGMA user_version")[0][0] == c2.proxy.EMB_SCHEMA_VERSION
    emb = c2.db("SELECT embedding FROM vec_facts WHERE rowid=?", fid)[0][0]
    assert emb != legacy


def test_global_memory_0_scopes_fact_search(make_ctx):
    c = make_ctx(GLOBAL_MEMORY="0")
    c.llm.cold(facts=[{"predicate": "likes", "object": "джаз", "evidence": "я люблю джаз"}])
    c.chat("Я люблю джаз", chat_id="chat-a")
    c.drain()
    c.chat("Какую музыку я люблю? джаз или рок", chat_id="chat-b")
    sys_b = " ".join(m["content"] for m in c.llm.chat_requests[-1]["messages"] if m["role"] == "system")
    assert "likes: джаз" not in sys_b
    c.chat("Какую музыку я люблю? джаз или рок", chat_id="chat-a")
    sys_a = " ".join(m["content"] for m in c.llm.chat_requests[-1]["messages"] if m["role"] == "system")
    assert "likes: джаз" in sys_a

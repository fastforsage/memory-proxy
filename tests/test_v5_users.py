"""v5, этап U: один пользователь с любыми клиентами; чат по префиксу истории; проброс; merge."""
import os, subprocess, sys, logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVES = lambda city, ev: {"predicate": "lives_in", "object": city, "evidence": ev}


def remember(c, text, facts=None, emotions=None, **kw):
    c.llm.cold(facts=facts, emotions=emotions)
    r = c.chat(text, **kw)
    c.drain()
    return r


def test_single_mode_ignores_user_headers_and_field(ctx):
    remember(ctx, "Я живу в Казани", [LIVES("Казань", "я живу в казани")], user="3f2a-owui-uuid")
    remember(ctx, "Я люблю джаз", [{"predicate": "likes", "object": "джаз", "evidence": "я люблю джаз"}],
             chat_id="cline-1", body_extra={"user": "some-session"})
    assert {r[0] for r in ctx.db("SELECT user_id FROM facts")} == {"default"}
    assert {r[0] for r in ctx.db("SELECT user_id FROM messages")} == {"default"}
    ctx.chat("Где я живу и что люблю?", chat_id="other-client", user="bob")
    mem = ctx.memory_text()
    assert "Казань" in mem and "джаз" in mem
    sent = ctx.llm.chat_requests[-1]
    assert "user" not in sent or sent.get("user") is None


def test_user_field_is_passed_to_backend(ctx):
    ctx.chat("Привет", body_extra={"user": "session-42"})
    assert ctx.llm.chat_requests[-1]["user"] == "session-42"


def test_single_mode_api_keys_all_map_to_owner(make_ctx):
    c = make_ctx(API_KEYS="laptop:key-1,phone:key-2")
    for k in ("key-1", "key-2"):
        assert c.chat(f"Привет с {k}", headers={"Authorization": f"Bearer {k}"}).status_code == 200
    assert c.chat("Чужой", headers={"Authorization": "Bearer wrong"}).status_code == 401
    c.drain()
    assert {r[0] for r in c.db("SELECT user_id FROM messages")} == {"default"}


def test_owner_name_configurable(make_ctx):
    c = make_ctx(OWNER="me")
    c.chat("Привет", user="alice")
    c.drain()
    assert {r[0] for r in c.db("SELECT user_id FROM messages")} == {"me"}


def ids(c):
    c.drain()
    return [r[0] for r in c.db("SELECT chat_id FROM messages WHERE role='user' ORDER BY id")]


def test_two_chats_with_same_greeting_are_separate(ctx):
    """Без заголовка v4.8 склеивал все чаты, начатые с «привет». v5 различает их со второго хода."""
    ctx.chat("привет", chat_id=None)                                        # чат A, ход 1
    ha = [{"role": "user", "content": "привет"}, {"role": "assistant", "content": "Привет! Чем помочь? (A)"}]
    ctx.chat("Расскажи про Байкал", chat_id=None, history=ha)               # A, ход 2
    ctx.chat("привет", chat_id=None)                                        # чат B, ход 1 — новый чат
    hb = [{"role": "user", "content": "привет"}, {"role": "assistant", "content": "Здравствуйте! (B)"}]
    ctx.chat("Как испечь хлеб?", chat_id=None, history=hb)                  # B, ход 2
    ha2 = ha + [{"role": "user", "content": "Расскажи про Байкал"}, {"role": "assistant", "content": "Байкал — озеро."}]
    ctx.chat("А глубина?", chat_id=None, history=ha2)                       # A, ход 3
    a1, a2, b1, b2, a3 = ids(ctx)
    assert a1 == a2 == a3 and b1 == b2 and a1 != b1


def test_regenerate_and_agent_steps_do_not_duplicate(ctx):
    h = [{"role": "user", "content": "Сколько будет 2+2?"}]
    body = {"model": "test-model", "messages": h}
    for _ in range(3):
        ctx.client.post("/v1/chat/completions", json=body)                 # перегенерация
    ctx.drain()
    assert ctx.db("SELECT COUNT(*) FROM messages WHERE role='user'")[0][0] == 1
    assert len(ctx.llm.cold_requests) == 1


def test_edit_in_the_middle_is_a_branch_of_same_chat(ctx):
    h = [{"role": "user", "content": "План поездки в Казань"}, {"role": "assistant", "content": "Вот план."},
         {"role": "user", "content": "Добавь музеи"}, {"role": "assistant", "content": "Добавил."}]
    ctx.chat("Сколько дней?", chat_id=None, history=h)
    edited = h[:2] + [{"role": "user", "content": "Добавь рестораны"}, {"role": "assistant", "content": "Добавил рестораны."}]
    ctx.chat("А бюджет?", chat_id=None, history=edited)
    a, b = ids(ctx)
    assert a == b
    assert len({r[0] for r in ctx.db("SELECT chat_id FROM chat_chain")}) == 1


def test_legacy_v48_chat_id_is_continued(ctx):
    """Разговор, начатый при v4.8 (без цепочки), продолжается под прежним chat_id."""
    first = "Помоги спланировать ремонт кухни"
    legacy = ctx.proxy.legacy_chat_id("default", [{"role": "user", "content": first}])
    h = [{"role": "user", "content": first}, {"role": "assistant", "content": "Давай."}]
    ctx.chat("С чего начать?", chat_id=None, history=h)
    assert ids(ctx) == [legacy]


def test_custom_chat_id_header(make_ctx):
    c = make_ctx(CHAT_ID_HEADERS="X-Session-Id,X-OpenWebUI-Chat-Id")
    c.chat("Привет", chat_id=None, headers={"X-Session-Id": "sess-1"})
    assert ids(c) == ["sess-1"]


def test_agent_client_wrappers_are_stripped_for_memory(ctx):
    text = "<task>\nЯ живу в Казани, почини тесты\n</task>\n<environment_details>\n# cwd /home/x\n</environment_details>"
    ctx.chat(text, chat_id="cline")
    ctx.drain()
    assert ctx.db("SELECT content FROM messages WHERE role='user'")[0][0] == "Я живу в Казани, почини тесты"


def test_passthrough_embeddings_requires_key(make_ctx):
    c = make_ctx(API_KEY="secret")
    r = c.client.post("/v1/embeddings", json={"input": "x", "model": "e"})
    assert r.status_code == 401
    r = c.client.post("/v1/embeddings", json={"input": "x", "model": "e"}, headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200 and r.json()["path"] == "/v1/embeddings"
    assert c.llm.other[-1][0] == "POST" and b'"input"' in c.llm.other[-1][2]


def test_passthrough_can_be_disabled(make_ctx):
    c = make_ctx(PASSTHROUGH="0")
    assert c.client.post("/v1/embeddings", json={"input": "x"}).status_code == 404


def test_service_request_heuristic_not_saved(ctx):
    ctx.chat("Write a short title for this conversation", body_extra={"max_tokens": 20})
    ctx.drain()
    assert ctx.db("SELECT COUNT(*) FROM messages")[0][0] == 0


def run_manage(db_path, *args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "manage.py"), *args], capture_output=True, text=True,
                       env={**os.environ, "DB_PATH": db_path}, timeout=60)
    return r.returncode, r.stdout + r.stderr


def test_startup_warns_about_other_users_and_merge(make_ctx, caplog):
    m = make_ctx(USER_MODE="multi")
    m.llm.cold(facts=[LIVES("Казань", "я живу в казани")])
    m.chat("Я живу в Казани")                                  # default, раньше
    m.drain()
    m.llm.cold(facts=[LIVES("Самара", "я живу в самаре")],
               emotions=[{"trigger": "джаз", "valence": 1.0, "quote": "люблю джаз"}])
    m.chat("Я живу в Самаре и люблю джаз", user="owui-uuid")     # тот же человек из Open WebUI, позже
    m.drain()
    db_path = m.proxy.DB_PATH
    with caplog.at_level(logging.WARNING, logger="memory-proxy"):
        make_ctx(USER_MODE="single")
    assert any("owui-uuid" in r.getMessage() and "merge" in r.getMessage() for r in caplog.records)

    code, out = run_manage(db_path, "merge", "--into", "default")
    assert code == 0 and "Резервная копия" in out, out
    c = make_ctx(USER_MODE="single")
    rows = c.db("SELECT object, valid_to IS NULL FROM facts WHERE predicate='lives_in' ORDER BY id")
    assert rows == [("Казань", 0), ("Самара", 1)]                # свежий факт актуален, старый — в истории
    assert {r[0] for r in c.db("SELECT user_id FROM messages")} == {"default"}
    assert c.db("SELECT trigger FROM emotions WHERE user_id='default'") == [("джаз",)]
    assert [p for p in os.listdir(os.path.dirname(db_path)) if ".bak-" in p]
    c.chat("Где я живу?", chat_id="new")
    assert "Самара" in c.memory_text() and "Казань" not in c.memory_text()


def test_merge_nothing_to_do(ctx):
    ctx.chat("Привет")
    ctx.drain()
    code, out = run_manage(ctx.proxy.DB_PATH, "merge", "--into", "default")
    assert code == 0 and "нечего" in out

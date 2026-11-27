"""v5.0.1: исправления по итогам UX-тестов — понятные ошибки, фон и служебные запросы, перегенерация,
обрыв стрима, опечатки в командах, извлечение фактов маленькими моделями, /health, manage.py."""
import asyncio, json, os, subprocess, sys
import httpx
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACT = {"predicate": "lives_in", "object": "Казань", "evidence": "я живу в Казани"}


def err(r):
    j = r.json()
    assert isinstance(j.get("error"), dict) and j["error"].get("message"), j
    return j["error"]


# ─── B1/B2/B3/B7: ошибки в формате OpenAI ───
def test_error_envelope_auth(make_ctx):
    c = make_ctx(API_KEY="sk-secret")
    r = c.client.get("/v1/models")
    assert r.status_code == 401 and err(r)
    r = c.client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401 and "API_KEY" in err(r)["message"]


def test_error_envelope_bad_requests(ctx):
    h = {"Content-Type": "application/json"}
    r = ctx.client.post("/v1/chat/completions", content=b"{oops", headers=h)
    assert r.status_code == 400 and err(r)["code"] == "invalid_json"
    r = ctx.client.post("/v1/chat/completions", json={"messages": "hi"})
    assert r.status_code == 400 and err(r)["code"] == "invalid_request"
    r = ctx.client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400 and "empty" in err(r)["message"]
    r = ctx.client.post("/v1/chat/completions", content=b'{"messages":[]}', headers={"Content-Type": "text/plain"})
    assert r.status_code == 415 and err(r)


def test_message_too_long(make_ctx):
    c = make_ctx(MAX_INPUT_LEN="100")
    r = c.chat("я" * 101)
    assert r.status_code == 413 and "100" in err(r)["message"] and err(r)["code"] == "message_too_long"


def test_host_rejection_is_json_with_hint(ctx):
    r = ctx.client.get("/health", headers={"Host": "evil.example"})
    assert r.status_code == 400 and "ALLOWED_HOSTS" in err(r)["message"]


def test_root_explains_what_this_is(ctx):
    j = ctx.client.get("/").json()
    assert j["name"] == "memory-proxy" and "/v1" in j["message"]


# ─── бэкенд недоступен / ломается ───
@pytest.mark.parametrize("exc,status,code", [(httpx.ConnectError("refused"), 502, "backend_unavailable"),
                                             (httpx.ReadTimeout("slow"), 504, "backend_timeout")])
def test_backend_errors_are_explained(ctx, exc, status, code):
    def boom(req): raise exc
    ctx.proxy.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    for stream in (False, True):
        r = ctx.chat("привет", stream=stream)
        assert r.status_code == status and err(r)["code"] == code
        if status == 502: assert "LLAMA_URL" in err(r)["message"]
    assert ctx.proxy.GATE.active == 0                  # фон не ждёт вечно после ошибки


def test_backend_broken_json_is_502_and_not_saved(ctx):
    ctx.proxy.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, content=b'{"choices":[{"message":{"content":"ok"', headers={"content-type": "application/json"})))
    r = ctx.chat("привет")
    assert r.status_code == 502 and err(r)["code"] == "backend_bad_response"
    assert ctx.db("SELECT COUNT(*) FROM messages WHERE role='assistant'")[0][0] == 0


class _BrokenStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"\xd0\x9d\xd0\xb0\xd1\x87\xd0\xb0\xd0\xbb\xd0\xbe "}}]}\n\n'
        raise httpx.RemoteProtocolError("peer closed connection")


def test_stream_break_sends_error_and_done_and_saves_nothing(ctx):
    ctx.proxy.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, stream=_BrokenStream(), headers={"content-type": "text/event-stream"})))
    r = ctx.chat("Расскажи историю", stream=True)
    assert r.status_code == 200
    events = [l[5:].strip() for l in r.text.splitlines() if l.startswith("data:")]
    assert events[-1] == "[DONE]"
    e = json.loads(events[-2])["error"]
    assert e["code"] == "backend_stream_error" and "incomplete" in e["message"]
    ctx.drain()
    assert ctx.db("SELECT COUNT(*) FROM messages WHERE role='assistant'")[0][0] == 0
    assert ctx.proxy.GATE.active == 0


# ─── /health ───
def test_health_reports_backend(ctx):
    j = ctx.client.get("/health").json()
    assert j["backend"]["reachable"] is True and j["background"]["waiting_for_user_requests"] == 0


def test_health_backend_down_keeps_status(ctx):
    def boom(req): raise httpx.ConnectError("refused")
    ctx.proxy.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    ctx.proxy.BACKEND_PROBE["ts"] = -1e9
    j = ctx.client.get("/health").json()
    assert j["status"] is True and j["backend"]["reachable"] is False and "ConnectError" in j["backend"]["error"]


# ─── B4: опечатки в командах ───
@pytest.mark.parametrize("typo,cmd", [("/fact", "/facts"), ("/forgt 3", "/forget"), ("/chat", "/chats"), ("/serch x", "/search")])
def test_near_command_hint(ctx, typo, cmd):
    r = ctx.chat(typo)
    t = r.json()["choices"][0]["message"]["content"]
    assert cmd in t and "/help" in t
    assert not ctx.llm.chat_requests                     # в модель не ушло


@pytest.mark.parametrize("text", ["/etc/hosts — что это за файл?", "/start", "/imagine кот", "/usr/bin"])
def test_paths_and_foreign_commands_go_to_model(ctx, text):
    ctx.chat(text)
    assert ctx.llm.chat_requests


# ─── B6: служебные запросы клиентов не пишутся в память ───
@pytest.mark.parametrize("text", [
    "Your task is to reflect the speaker's likely facial expression through a fitting emoji. Message: I'm happy",
    "You have been provided with a set of responses from various models to the latest user query: \"hi\"",
])
def test_owui_emoji_and_moa_not_saved(ctx, text):
    ctx.client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": text}]})
    ctx.drain()
    assert ctx.db("SELECT COUNT(*) FROM messages")[0][0] == 0 and not ctx.llm.cold_requests


def test_owui_tool_choice_request_not_saved(ctx):
    sys_prompt = ("Available Tools: [{\"name\": \"web_search\"}]\n\nYour task is to choose and return the correct "
                  "tool(s) from the list of available tools based on the query.")
    ctx.client.post("/v1/chat/completions", json={"model": "m", "messages": [
        {"role": "system", "content": sys_prompt}, {"role": "user", "content": "Query: Я живу в Казани"}]})
    ctx.drain()
    assert ctx.db("SELECT COUNT(*) FROM messages")[0][0] == 0


def test_service_request_preempts_background(ctx):
    """Служебный запрос (заголовок) тоже вытесняет фон — иначе ждал бы минутами за извлечением фактов."""
    calls = []
    orig = ctx.proxy.GATE.user_start
    ctx.proxy.GATE.user_start = lambda preempt=True: (calls.append(preempt), orig(preempt))
    ctx.client.post("/v1/chat/completions", json={"model": "m", "messages": [
        {"role": "user", "content": "### Task:\nGenerate a concise title"}]})
    assert calls == [True]


# ─── фон: вытеснение и повтор до COLD_RETRY_S ───
def test_gate_retries_until_deadline(ctx):
    """Вытеснили больше раз, чем attempts — с deadline задача всё равно выполнится (B5: факты терялись)."""
    import time
    G = ctx.proxy.BackgroundGate()
    n = {"i": 0}

    async def go():
        async def job():
            n["i"] += 1
            await asyncio.sleep(0.05)
            return "done"
        t = asyncio.ensure_future(G.run(job, attempts=1, deadline=time.monotonic() + 30))
        for _ in range(6):
            await asyncio.sleep(0.01)
            G.user_start(); G.user_end()
        return await t
    assert asyncio.run(go()) == "done" and n["i"] >= 3


def test_gate_gives_up_after_attempts_without_deadline(ctx):
    G = ctx.proxy.BackgroundGate()

    async def go():
        async def job():
            await asyncio.sleep(10)

        t = asyncio.ensure_future(G.run(job, attempts=2))
        for _ in range(3):
            await asyncio.sleep(0.02)
            G.user_start(); G.user_end()
        return await t
    assert asyncio.run(go()) is None


def test_commands_do_not_touch_gate(ctx):
    starts = []
    orig = ctx.proxy.GATE.user_start
    ctx.proxy.GATE.user_start = lambda preempt=True: (starts.append(preempt), orig(preempt))
    ctx.chat("/help"); ctx.chat("/facts")
    assert starts == [] and ctx.proxy.GATE.active == 0


# ─── B8: перегенерация заменяет ответ ───
def test_regenerate_replaces_answer_and_style_example(ctx):
    ctx.chat("Расскажи что-нибудь интересное про космос")
    ctx.drain()
    first = ctx.db("SELECT id, content FROM messages WHERE role='assistant'")
    ctx.llm.ANSWER = "Другой, тоже достаточно длинный ответ ассистента про космос и звёзды, чтобы стать примером."
    ctx.chat("Расскажи что-нибудь интересное про космос")          # «Регенерировать»
    ctx.drain()
    rows = ctx.db("SELECT id, content FROM messages WHERE role='assistant'")
    assert len(rows) == 1 and rows[0][0] == first[0][0] and rows[0][1] == ctx.llm.ANSWER
    ex = ctx.db("SELECT text FROM style_examples")
    assert all("Это развёрнутый ответ" not in t for (t,) in ex)


# ─── B5: извлечение фактов ───
def test_predicate_aliases_and_case(ctx):
    P = ctx.proxy
    assert P.norm_predicate("City") == "lives_in" and P.norm_predicate("works at") == "works_at"
    assert P.norm_predicate("pet") == "has_pet" and P.norm_predicate("lives_in") == "lives_in"
    assert P.norm_predicate("favourite_planet") == "" and P.norm_predicate(None) == ""


def test_match_evidence_fuzzy(ctx):
    P = ctx.proxy
    text = "Меня зовут Алексей. Я живу в Казани, работаю в Яндексе."
    assert P.match_evidence(text, "Я живу в Казани", "Казань")
    assert P.match_evidence(text, "я живу в казани.", "Казань")                   # регистр и пунктуация
    assert P.match_evidence(text, "Я живу в городе Казани", "Казань")             # модель чуть переписала цитату
    assert P.match_evidence(text, "Я живу в Москве", "Москва") is None            # выдумано
    assert P.match_evidence(text, "Я живу в Казани", "Москва") is None            # объект не из цитаты


def test_cold_accepts_alias_and_reworded_quote(ctx):
    ctx.llm.cold(facts=[{"predicate": "city", "object": "Казань", "evidence": "я живу в городе Казани"},
                        {"predicate": "name", "object": "Алексей", "evidence": "Меня зовут Алексей"}])
    ctx.chat("Меня зовут Алексей. Я живу в Казани.")
    ctx.drain()
    got = {p: o for p, o in ctx.db("SELECT predicate, object FROM facts WHERE valid_to IS NULL")}
    assert got == {"lives_in": "Казань", "name": "Алексей"}


def test_name_requires_self_introduction(ctx):
    ctx.llm.cold(facts=[{"predicate": "name", "object": "Маша", "evidence": "ходил с женой Машей"}])
    ctx.chat("Вчера ходил с женой Машей в кино")
    ctx.drain()
    assert ctx.db("SELECT COUNT(*) FROM facts")[0][0] == 0


def test_name_is_single_valued(ctx):
    for name in ("Оля", "Ира"):
        ctx.llm.cold(facts=[{"predicate": "name", "object": name, "evidence": f"меня зовут {name}"}])
        ctx.chat(f"Меня зовут {name}")
        ctx.drain()
    assert ctx.db("SELECT object FROM facts WHERE valid_to IS NULL") == [("Ира",)]


def test_negated_fact_rejected(ctx):
    ctx.llm.cold(facts=[{"predicate": "works_at", "object": "Яндекс", "evidence": "Я работаю в Яндексе"},
                        {"predicate": "works_at", "object": "Google", "evidence": "в Google меня не взяли"}])
    ctx.chat("Я работаю в Яндексе, а в Google меня не взяли.")
    ctx.drain()
    assert ctx.db("SELECT object FROM facts WHERE valid_to IS NULL") == [("Яндекс",)]


def test_dislikes_keeps_negation(ctx):
    ctx.llm.cold(facts=[{"predicate": "dislikes", "object": "кинза", "evidence": "Не люблю кинзу"}])
    ctx.chat("Не люблю кинзу.")
    ctx.drain()
    assert ctx.db("SELECT predicate FROM facts") == [("dislikes",)]


def test_llm_technical_does_not_drop_first_person_facts(ctx):
    ctx.llm.cold(type_="technical", facts=[{"predicate": "has_skill", "object": "программист", "evidence": "Я программист"},
                                           {"predicate": "uses_tool", "object": "Python", "evidence": "на Python"}])
    ctx.chat("Я программист, пишу на Python.")
    ctx.drain()
    assert ctx.db("SELECT predicate, object FROM facts") == [("has_skill", "программист")]


def test_code_message_still_skipped(ctx):
    ctx.llm.cold(facts=[FACT])
    ctx.chat("```python\nx = 1\n```\nя живу в Казани")
    ctx.drain()
    assert ctx.db("SELECT COUNT(*) FROM facts")[0][0] == 0


def test_rejections_are_logged(ctx, caplog):
    import logging
    ctx.llm.cold(facts=[{"predicate": "planet", "object": "Марс", "evidence": "Марс"},
                        {"predicate": "lives_in", "object": "Москва", "evidence": "я живу в Москве"}])
    with caplog.at_level(logging.INFO, logger=ctx.proxy.log.name):
        ctx.chat("Я живу в Казани")
        ctx.drain()
    assert any("facts rejected" in r.getMessage() for r in caplog.records)


def test_extract_prompt_lists_predicates_with_help(ctx):
    p = ctx.proxy.build_extract_payload("Я живу в Казани", "m")["messages"][0]["content"]
    assert "lives_in" in p and ctx.proxy.PREDICATE_HELP["lives_in"] in p and "Я живу в Казани" in p


# ─── manage.py ───
def _run(db, *args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "manage.py"), *args], capture_output=True, text=True,
                       env={**os.environ, "DB_PATH": db}, timeout=60)
    return r.returncode, r.stdout, r.stderr


def test_manage_facts_defaults_to_owner(ctx):
    ctx.llm.cold(facts=[FACT])
    ctx.chat("Я живу в Казани")
    ctx.drain()
    rc, out, _ = _run(ctx.proxy.DB_PATH, "facts")
    assert rc == 0 and "Казань" in out


def test_manage_unknown_command(ctx):
    ctx.chat("/help")
    rc, out, errout = _run(ctx.proxy.DB_PATH, "bogus")
    assert rc == 2 and "bogus" in errout and "Traceback" not in errout


def test_manage_single_valued_matches_proxy(ctx):
    sys.path.insert(0, ROOT)
    import manage
    assert manage.SINGLE_VALUED == ctx.proxy.SINGLE_VALUED


def test_object_must_match_a_word_not_a_prefix(ctx):
    P = ctx.proxy
    assert P._obj_in("Казань", "я живу в казани") and P._obj_in("Маша", "с женой машей")
    assert P._obj_in("бег", "обожаю бегать") and P._obj_in("пробки", "пробок не люблю")
    assert not P._obj_in("любил", "мой любимый цвет")       # 0.5B: has_pet: любил по цитате «любимый цвет»


# ─── примеры стиля: не обрезанные ответы и не копии примеров ───
def test_truncated_answer_is_not_a_style_example(ctx):
    ctx.llm.handler_orig = ctx.llm.handler
    long = "Очень длинный ответ, который модель оборвала по max_tokens, и он заканчивается на полусло"

    def h(req):
        r = ctx.llm.handler_orig(req)
        body = json.loads(req.content or b"{}")
        if req.url.path.endswith("/chat/completions") and not body.get("response_format") and not body.get("stream"):
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": long}, "finish_reason": "length"}]})
        return r
    ctx.proxy.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(h))
    ctx.chat("Расскажи длинную историю")
    ctx.drain()
    assert ctx.db("SELECT content FROM messages WHERE role='assistant'") == [(long,)]
    assert ctx.db("SELECT COUNT(*) FROM style_examples")[0][0] == 0


def test_parroted_style_example_is_removed(ctx):
    """0.5B копировала пример стиля вместо ответа, копия снова становилась примером — петля."""
    ctx.chat("Расскажи что-нибудь", chat_id="c1")
    ctx.drain()
    assert ctx.db("SELECT COUNT(*) FROM style_examples")[0][0] == 1
    ctx.chat("Как меня зовут?", chat_id="c2")              # тот же ответ дословно
    ctx.drain()
    assert ctx.db("SELECT COUNT(*) FROM style_examples")[0][0] == 0
    assert ctx.db("SELECT COUNT(*) FROM messages WHERE role='assistant'")[0][0] == 2

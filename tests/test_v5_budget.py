"""v5, этап A: бюджет от реального лимита модели, самообучение по 400, OVERSIZE, шаблоны."""
import json


def long_history(n_msgs, chars=4000, sys=False):
    msgs = [{"role": "system", "content": "Ты помощник."}] if sys else []
    for i in range(n_msgs):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"{role} {i}: " + ("слово " * (chars // 6))})
    if msgs and msgs[-1]["role"] == "user":
        msgs.append({"role": "assistant", "content": "ок"})
    return msgs


def sent_tokens(req):
    return sum(len(json.dumps(m.get("content"), ensure_ascii=False)) for m in req["messages"]) // 3


def test_short_chat_of_long_messages_fits_limit_from_props(ctx):
    """P11: 7 сообщений по 4000 символов — v4.8 не сжимал вовсе и получал 400 от llama.cpp -c 2048."""
    ctx.llm.props = {"default_generation_settings": {"n_ctx": 2048}}
    ctx.llm.ctx_limit = 2048
    r = ctx.chat("Что дальше?", history=long_history(6))
    assert r.status_code == 200
    req = ctx.llm.chat_requests[-1]
    assert sent_tokens(req) <= 2048 and req["messages"][-1]["content"].endswith("Что дальше?")
    assert req["messages"][0]["role"] == "user"                      # окно начинается с хода пользователя
    assert ctx.proxy.BACKENDS["test-model"]["n_ctx"] == 2048


def test_context_limit_learned_from_400_and_retried(ctx):
    """Лимит неизвестен (Ollama/облако без /props): 400 → запомнить n_ctx → один прозрачный повтор."""
    ctx.llm.ctx_limit = 2048
    r = ctx.chat("Продолжим?", history=long_history(6))
    assert r.status_code == 200
    rejected = [q for q in ctx.llm.requests if q.get("_rejected")]
    assert len(rejected) == 1 and sent_tokens(ctx.llm.chat_requests[-1]) <= 2048
    assert ctx.proxy.BACKENDS["test-model"]["n_ctx"] == 2048
    ctx.chat("И ещё вопрос", history=long_history(8))                # дальше — без отказов
    assert len([q for q in ctx.llm.requests if q.get("_rejected")]) == 1


def test_context_limit_learned_in_stream_mode(ctx):
    ctx.llm.ctx_limit = 2048
    r = ctx.chat("Продолжим в стриме?", history=long_history(6), stream=True)
    assert r.status_code == 200 and "data:" in r.text


def test_context_limit_env(make_ctx):
    c = make_ctx(CONTEXT_LIMIT="2048")
    c.llm.ctx_limit = 2048
    assert c.chat("Вопрос", history=long_history(10)).status_code == 200
    assert not [q for q in c.llm.requests if q.get("_rejected")]


def test_oversize_reject_answers_in_chat(make_ctx):
    c = make_ctx(CONTEXT_LIMIT="2048", OVERSIZE="reject")
    n_before = len(c.llm.chat_requests)
    r = c.chat("Проанализируй: " + "очень длинный текст " * 3000)
    assert r.status_code == 200 and "слишком большое" in r.json()["choices"][0]["message"]["content"]
    assert len(c.llm.chat_requests) == n_before


def test_oversize_truncate_current_turn_tool_output(make_ctx):
    c = make_ctx(CONTEXT_LIMIT="4096")
    c.llm.ctx_limit = 4096
    hist = [{"role": "user", "content": "Прочитай лог"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function",
                                                                "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "строка лога\n" * 8000}]
    body = {"model": "test-model", "messages": hist}
    r = c.client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    req = c.llm.chat_requests[-1]
    assert [m["role"] for m in req["messages"]] == ["user", "assistant", "tool"]
    assert "середина сокращена" in req["messages"][2]["content"] and sent_tokens(req) <= 4096


def test_template_without_system_support_folds_system(ctx):
    """Gemma 2: шаблон отвергает system → system клиента сворачивается в первое сообщение пользователя."""
    def gemma2(messages):
        if any(m["role"] == "system" for m in messages): raise ValueError("System role not supported")
        return "".join(f"<{m['role']}>{m['content']}" for m in messages)
    ctx.llm.props = {"chat_template": "{{ gemma }}", "default_generation_settings": {"n_ctx": 8192}}
    ctx.llm.templates = gemma2
    ctx.chat("Привет", history=[{"role": "system", "content": "Отвечай кратко."}])
    msgs = ctx.llm.chat_requests[-1]["messages"]
    assert [m["role"] for m in msgs] == ["user"] and msgs[0]["content"].startswith("Отвечай кратко.")
    assert ctx.proxy.BACKENDS["test-model"]["system_mode"] == "first_user"


def test_template_moving_system_detected(ctx):
    """Mistral Nemo: system в последнем сообщении пользователя → тоже first_user."""
    def nemo(messages):
        sys = next((m["content"] for m in messages if m["role"] == "system"), "")
        rest = [m for m in messages if m["role"] != "system"]
        last_u = max(i for i, m in enumerate(rest) if m["role"] == "user")
        return "".join(("[INST]" + (sys + "\n\n" if i == last_u else "") + m["content"] + "[/INST]") if m["role"] == "user"
                       else m["content"] for i, m in enumerate(rest))
    ctx.llm.props = {"chat_template": "{{ nemo }}"}
    ctx.llm.templates = nemo
    ctx.chat("Привет", history=[{"role": "system", "content": "Отвечай кратко."}])
    assert ctx.proxy.BACKENDS["test-model"]["system_mode"] == "first_user"


def test_stable_template_keeps_native_system(ctx):
    ctx.llm.props = {"chat_template": "{{ qwen }}"}
    ctx.llm.templates = lambda ms: "".join(f"<|{m['role']}|>{m['content']}" for m in ms)
    ctx.chat("Привет", history=[{"role": "system", "content": "Отвечай кратко."}])
    msgs = ctx.llm.chat_requests[-1]["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith("Отвечай кратко.")


def test_model_profiles_override(make_ctx):
    c = make_ctx(MODEL_PROFILES=json.dumps({"gemma-2": {"system_mode": "first_user", "context_limit": 4096}}))
    c.chat("Привет", history=[{"role": "system", "content": "Будь вежлив."}], model="gemma-2-9b")
    msgs = c.llm.chat_requests[-1]["messages"]
    assert msgs[0]["role"] == "user" and c.proxy.BACKENDS["gemma-2-9b"]["n_ctx"] == 4096


def test_date_in_template_is_pinned(ctx):
    """Llama 3.2: дата в шаблоне → прокси закрепляет date_string на эпоху (иначе пересчёт раз в сутки)."""
    ctx.llm.props = {"chat_template": "{% if not date_string is defined %}{% set date_string = strftime_now('%d %b %Y') %}{% endif %}"}
    ctx.chat("Привет")
    kw = ctx.llm.chat_requests[-1].get("chat_template_kwargs") or {}
    assert kw.get("date_string")
    ctx.chat("Ещё", history=[{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "ок"}])
    assert ctx.llm.chat_requests[-1]["chat_template_kwargs"]["date_string"] == kw["date_string"]


def test_unknown_backend_sends_no_extra_fields(ctx):
    ctx.chat("Привет")
    req = ctx.llm.chat_requests[-1]
    assert "chat_template_kwargs" not in req and len(ctx.llm.chat_requests) == 1


def test_tools_reduce_budget(make_ctx):
    c = make_ctx(CONTEXT_LIMIT="4096")
    tools = [{"type": "function", "function": {"name": f"tool{i}", "description": "описание " * 60,
                                               "parameters": {"type": "object", "properties": {}}}} for i in range(12)]
    r = c.chat("Вопрос", history=long_history(8, chars=1500), body_extra={"tools": tools})
    assert r.status_code == 200
    req = c.llm.chat_requests[-1]
    total = sent_tokens(req) + len(json.dumps(tools, ensure_ascii=False)) // 3
    assert total <= 4096


def test_invariant_violation_falls_back_to_original(ctx, monkeypatch):
    monkeypatch.setattr(ctx.proxy.cx, "validate", lambda msgs: ["I2: искусственно"])
    ctx.chat("Привет")
    assert ctx.llm.chat_requests[-1]["messages"] == [{"role": "user", "content": "Привет"}]
    assert ctx.proxy.STATS["invariant_fallback"] == 1


def test_health_reports_context(ctx):
    ctx.llm.props = {"default_generation_settings": {"n_ctx": 8192}}
    ctx.chat("Привет")
    h = ctx.client.get("/health").json()
    assert h["user_mode"] == "single" and h["context"]["backends"]["test-model"]["n_ctx"] == 8192


def test_budget_formula(make_ctx):
    c = make_ctx(COMPRESS_AT="0")
    b = c.proxy.budget
    assert b({"n_ctx": 0}, {}, "m") == 16000                       # лимит неизвестен и границы нет
    assert b({"n_ctx": 8192}, {}, "m") == int(8192 * 0.9) - 1024
    assert b({"n_ctx": 8192}, {"max_tokens": 100}, "m") == int(8192 * 0.9) - 100
    assert b({"n_ctx": 2048}, {"max_tokens": 4000}, "m") == int(2048 * 0.9) - 512   # резерв ≤ n/4
    c2 = make_ctx(COMPRESS_AT="3000")
    assert c2.proxy.budget({"n_ctx": 32768}, {}, "m") == 3000

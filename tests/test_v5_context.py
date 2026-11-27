"""v5, этапы B–D: неизменяемый префикс, закреплённые дельты, эпохи, фоновое саммари с прогревом, recall."""
import asyncio
import json

TOPICS = ["байкал", "кухня", "отпуск", "ремонт", "бюджет", "машина", "дача", "сад", "кот", "работа",
          "спорт", "книги", "кино", "музыка", "поход", "рыбалка", "театр", "выставка", "погода", "еда",
          "курсы", "язык", "здоровье", "планы", "друзья", "семья", "праздник", "подарки", "уборка", "итоги"]


class Conversation:
    """Клиент вроде Open WebUI: шлёт всю историю целиком, к истории добавляет ответ модели."""
    def __init__(self, ctx, chat_id="conv", words=110, system=None):
        self.ctx, self.chat_id, self.words = ctx, chat_id, words
        self.history = [{"role": "system", "content": system}] if system else []
        self.sent = []

    def turn(self, text=None, drain=True):
        i = len(self.sent)
        text = text or f"Ход {i}, тема {TOPICS[i % len(TOPICS)]}: " + " ".join(f"слово{i}_{k}" for k in range(self.words))
        r = self.ctx.chat(text, chat_id=self.chat_id, history=self.history)
        assert r.status_code == 200, r.text
        if drain: self.ctx.drain()
        self.sent.append(self.ctx.llm.chat_requests[-1]["messages"])
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": r.json()["choices"][0]["message"]["content"]}]
        return self.sent[-1]


def is_prefix(a, b):
    return b[:len(a)] == a


SMALL = dict(CONTEXT_LIMIT="4000", ANSWER_RESERVE="200")          # бюджет B = 3400 токенов


def test_prompt_is_append_only_before_threshold(make_ctx):
    c = make_ctx(**SMALL)
    conv = Conversation(c, system="Ты помощник.", words=60)
    for _ in range(6): conv.turn()
    for a, b in zip(conv.sent, conv.sent[1:]):
        assert is_prefix(a, b), "промпт хода t должен быть префиксом промпта хода t+1"
    assert not c.llm.summary_requests and c.proxy.STATS["epoch_switch"] == 0


def test_pinned_delta_stays_with_its_message(make_ctx):
    c = make_ctx(**SMALL)
    c.llm.cold(facts=[{"predicate": "lives_in", "object": "Казань", "evidence": "я живу в казани"}])
    c.chat("Я живу в Казани", chat_id="old")
    c.drain()
    conv = Conversation(c)
    first = conv.turn("Посоветуй, куда сходить вечером в городе")
    assert "Казань" in c.memory_text()
    for _ in range(3): conv.turn()
    assert all(is_prefix(first, s) for s in conv.sent[1:])
    assert c.db("SELECT COUNT(*) FROM pinned_deltas WHERE chat_id='conv'")[0][0] == 4
    # шаг агента / перегенерация не пересобирают дельту
    n_cold = len(c.llm.cold_requests)
    body = {"model": "test-model", "messages": conv.history[:-1]}
    c.client.post("/v1/chat/completions", json=body, headers={"X-OpenWebUI-Chat-Id": "conv"})
    assert c.llm.chat_requests[-1]["messages"] == conv.sent[-1]
    c.drain()
    assert len(c.llm.cold_requests) == n_cold


def test_new_fact_appears_only_in_new_delta(make_ctx):
    c = make_ctx(**SMALL)
    conv = Conversation(c)
    conv.turn("Привет, давай планировать неделю")
    c.llm.cold(facts=[{"predicate": "likes", "object": "плавание", "evidence": "люблю плавание"}])
    conv.turn("Я люблю плавание, учти это")
    after = conv.turn("Что посоветуешь на субботу? Может плавание?")
    assert is_prefix(conv.sent[1], after)
    assert "плавание" in c.memory_text() or "плавание" in json.dumps(after, ensure_ascii=False)


def test_background_epoch_summary_warmup_and_switch(make_ctx):
    c = make_ctx(**SMALL)
    c.llm.summary_reply = "- обсуждали байкал и кухню"
    conv = Conversation(c, system="Ты помощник.")
    while not c.llm.summary_requests and len(conv.sent) < 30:
        conv.turn()
    assert c.llm.summary_requests, "к 0.7·B должна начаться подготовка эпохи"
    assert c.llm.warmup_requests, "новый префикс прогревается в фоне"
    state = c.db("SELECT state FROM chat_epochs WHERE chat_id='conv' ORDER BY epoch_id DESC LIMIT 1")[0][0]
    assert state == "ready"
    warm = c.llm.warmup_requests[-1]["messages"]
    conv.turn()
    assert c.proxy.STATS["epoch_switch"] == 1 and c.proxy.STATS["epoch_sync"] == 0
    sw = conv.sent[-1]
    assert is_prefix(warm[:-1], sw), "переключение на прогретый префикс — кэш уже готов"
    assert sw[0]["role"] == "system"
    text = json.dumps(sw, ensure_ascii=False)
    assert "КРАТКОЕ СОДЕРЖАНИЕ НАЧАЛА ЭТОГО ЧАТА" in text and "обсуждали байкал и кухню" in text
    for _ in range(2): conv.turn()
    assert is_prefix(conv.sent[-3], conv.sent[-2]) and is_prefix(conv.sent[-2], conv.sent[-1])
    assert all(c.proxy.TOKENS.messages(s, "test-model") <= 3400 for s in conv.sent)


def test_sync_shift_when_jump_over_budget(make_ctx):
    """Нет готовой эпохи, а окно уже не влезает (клиент прислал длинную историю разом) — L1+L3 без LLM."""
    c = make_ctx(SUMMARY="0", WARMUP="0", **SMALL)
    hist = []
    for i in range(24):
        hist += [{"role": "user", "content": f"Вопрос {i} про {TOPICS[i]}: " + "текст " * 80},
                 {"role": "assistant", "content": f"Ответ {i}."}]
    r = c.chat("Итак, что решили?", chat_id="jump", history=hist)
    assert r.status_code == 200
    sent = c.llm.chat_requests[-1]["messages"]
    assert c.proxy.STATS["epoch_sync"] == 1
    assert c.proxy.TOKENS.messages(sent, "test-model") <= 3400
    assert "КРАТКОЕ СОДЕРЖАНИЕ НАЧАЛА ЭТОГО ЧАТА" in json.dumps(sent, ensure_ascii=False)
    assert sent[0]["role"] == "user" and sent[-1]["content"].endswith("Итак, что решили?")


def test_summary_failure_falls_back(make_ctx):
    c = make_ctx(**SMALL)
    c.llm.summary_reply = ""                              # пустой ответ саммаризатора
    conv = Conversation(c)
    while not c.llm.summary_requests and len(conv.sent) < 30:
        conv.turn()
    conv.turn()
    assert c.proxy.STATS["epoch_switch"] == 1
    assert "КРАТКОЕ СОДЕРЖАНИЕ НАЧАЛА ЭТОГО ЧАТА" in json.dumps(conv.sent[-1], ensure_ascii=False)


def test_recall_from_cut_part(make_ctx):
    c = make_ctx(**SMALL)
    conv = Conversation(c)
    conv.turn("Запомни: пароль от вайфая на даче — Зеленоглазово42, роутер в кладовке")
    while c.proxy.STATS["epoch_switch"] == 0 and len(conv.sent) < 40:
        conv.turn()
    assert c.proxy.STATS["epoch_switch"] == 1
    assert "Зеленоглазово42" not in json.dumps(conv.sent[-1], ensure_ascii=False)
    conv.turn("Какой там пароль Зеленоглазово42 был, напомни где роутер?")
    mem = c.memory_text()
    assert "[ИЗ НАЧАЛА ЭТОГО ЧАТА]" in mem and "кладовке" in mem


def test_recall_disabled(make_ctx):
    c = make_ctx(RECALL="0", **SMALL)
    conv = Conversation(c)
    conv.turn("Запомни: пароль от вайфая на даче — Зеленоглазово42")
    while c.proxy.STATS["epoch_switch"] == 0 and len(conv.sent) < 40:
        conv.turn()
    conv.turn("Какой пароль Зеленоглазово42?")
    assert "[ИЗ НАЧАЛА ЭТОГО ЧАТА]" not in c.memory_text()


def test_forget_invalidates_pinned_deltas(ctx):
    ctx.llm.cold(facts=[{"predicate": "lives_in", "object": "Казань", "evidence": "я живу в казани"}])
    ctx.chat("Я живу в Казани", chat_id="a")
    ctx.drain()
    conv = Conversation(ctx, chat_id="b")
    conv.turn("Куда сходить в городе вечером?")
    assert "Казань" in ctx.memory_text()
    ctx.chat("/forget Казань", chat_id="b")
    assert ctx.db("SELECT COUNT(*) FROM pinned_deltas")[0][0] == 0
    conv.turn("А днём?")
    assert "Казань" not in json.dumps(conv.sent[-1], ensure_ascii=False)


def test_legacy_context_mode_still_works(make_ctx):
    c = make_ctx(CONTEXT_MODE="legacy", **SMALL)
    conv = Conversation(c)
    for _ in range(14): conv.turn()
    assert all(c.proxy.TOKENS.messages(s, "test-model") <= 3400 for s in conv.sent)
    assert c.db("SELECT COUNT(*) FROM chat_epochs")[0][0] == 0


def test_reasoning_is_stripped_before_saving(ctx):
    ctx.llm.ANSWER = "<think>Пользователь спрашивает про погоду, надо подумать</think>Сегодня солнечно и тепло, около двадцати градусов."
    ctx.chat("Какая погода?")
    ctx.drain()
    saved = ctx.db("SELECT content FROM messages WHERE role='assistant'")[0][0]
    assert saved == "Сегодня солнечно и тепло, около двадцати градусов."


def test_cache_monitor_in_health(ctx):
    ctx.llm.usage = {"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 900}}
    ctx.chat("Привет")
    ctx.chat("Ещё", history=[{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "ок"}])
    st = ctx.client.get("/health").json()["context"]
    assert st["prompt_tokens"] == 2000 and st["cached_tokens"] == 1800 and st["cache_hit"] == 0.9


def test_background_gate_preemption(ctx):
    gate = ctx.proxy.BackgroundGate()
    calls = []

    async def scenario():
        async def work():
            calls.append("start")
            await asyncio.sleep(0.2)
            return "done"
        bg = asyncio.ensure_future(gate.run(work))
        await asyncio.sleep(0.05)
        gate.user_start()                               # пришёл пользователь — фон отменён
        await asyncio.sleep(0.05)
        assert len(calls) == 1 and not bg.done()
        gate.user_end()                                 # ушёл — фон повторяется
        return await asyncio.wait_for(bg, 2)

    assert asyncio.run(scenario()) == "done" and calls == ["start", "start"]


def test_background_waits_for_user_requests(ctx):
    gate = ctx.proxy.BackgroundGate()

    async def scenario():
        gate.user_start()
        bg = asyncio.ensure_future(gate.run(lambda: asyncio.sleep(0, result="ok")))
        await asyncio.sleep(0.05)
        assert not bg.done()
        gate.user_end()
        return await asyncio.wait_for(bg, 2)

    assert asyncio.run(scenario()) == "ok"


def test_warmup_renders_like_real_request(make_ctx):
    """Прогрев без tools дал бы другой промпт (инструменты рендерятся в начало) — кэш пропал бы зря."""
    c = make_ctx(**SMALL)
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}}]
    conv = Conversation(c)
    orig = c.chat
    c.chat = lambda *a, **k: orig(*a, **{**k, "body_extra": {"tools": tools, "tool_choice": "auto"}})
    while not c.llm.warmup_requests and len(conv.sent) < 30:
        conv.turn()
    w = c.llm.warmup_requests[-1]
    assert w["tools"] == tools and w["tool_choice"] == "auto" and w["max_tokens"] == 1


def test_cold_extraction_is_bounded_and_waits_for_user(ctx):
    """Извлечение фактов — фоновая задача: max_tokens и ожидание, пока пользовательский запрос не завершён
    (с одним слотом llama.cpp зацикленный JSON-ответ малой модели держал слот 300 с)."""
    ctx.llm.cold(facts=[{"predicate": "lives_in", "object": "Казань", "evidence": "я живу в казани"}])
    order = []
    orig = ctx.llm.handler

    def spy(request):
        body = json.loads(request.content or b"{}") if request.url.path.endswith("/chat/completions") else {}
        order.append("cold" if body.get("response_format") else "user" if body else "other")
        return orig(request)
    ctx.llm.handler = spy
    ctx.proxy.app.state.http = __import__("httpx").AsyncClient(
        transport=__import__("httpx").MockTransport(spy), base_url="http://llm")
    ctx.chat("Я живу в Казани", stream=True)
    ctx.drain()
    assert [o for o in order if o != "other"] == ["user", "cold"]
    assert ctx.llm.cold_requests[-1]["max_tokens"] == 512
    assert ctx.db("SELECT object FROM facts") == [("Казань",)]


def test_recall_not_repeated_for_every_question(make_ctx):
    """Живой прогон: общие слова тянули одни и те же цитаты из начала чата к каждому вопросу (+300 ток./ход)."""
    c = make_ctx(**SMALL)
    conv = Conversation(c)
    conv.turn("Запомни: пароль от вайфая на даче — Зеленоглазово42, роутер в кладовке")
    while c.proxy.STATS["epoch_switch"] == 0 and len(conv.sent) < 40:
        conv.turn()
    conv.turn("Какой там пароль Зеленоглазово42 был, напомни где роутер?")
    assert "кладовке" in c.memory_text()
    conv.turn("И ещё раз: пароль Зеленоглазово42 и роутер — точно?")
    assert "кладовке" not in c.memory_text()                  # уже виден в окне выше
    assert "кладовке" in json.dumps(conv.sent[-1], ensure_ascii=False)


def test_junk_summary_falls_back(make_ctx):
    c = make_ctx(**SMALL)
    c.llm.summary_reply = "0"                                  # так отвечала Qwen2.5-0.5B
    conv = Conversation(c)
    while c.proxy.STATS["epoch_switch"] == 0 and len(conv.sent) < 40:
        conv.turn()
    text = json.dumps(conv.sent[-1], ensure_ascii=False)
    assert "КРАТКОЕ СОДЕРЖАНИЕ НАЧАЛА ЭТОГО ЧАТА" in text and "Ход 0" in text
    assert c.proxy.STATS["summaries"] == 0

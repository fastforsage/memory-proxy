"""Регресс сборки v5 на реальных chat-шаблонах llama.cpp (models/templates/*.jinja).

Запуск:  TPL_DIR=/path/to/llama.cpp/models/templates pytest tests/test_v5_templates.py
(без TPL_DIR или jinja2 — пропускается; в CI не участвует).

Для каждого шаблона прокси проходит диалог: system + память + шаг агента с инструментами + новые ходы
до переключения эпохи. Каждый отправленный в модель запрос рендерится шаблоном. Проверяется:
  • шаблон не падает (v4.8 ронял 6/32 из-за добавленных system-сообщений);
  • префикс отрендеренного промпта не ломается между запросами сильнее, чем у того же диалога без прокси
    (собственные особенности шаблона — вырезание размышлений, перенос system — не вина прокси);
  • прогретый фоном префикс совпадает с промптом после переключения эпохи;
  • дата закреплена: смена суток не меняет промпт, если шаблон поддерживает date_string.
"""
import datetime
import glob
import json
import os

import pytest

TPL_DIR = os.getenv("TPL_DIR", "")
jinja2 = pytest.importorskip("jinja2") if TPL_DIR else None
FILES = sorted(glob.glob(os.path.join(TPL_DIR, "*.jinja"))) if TPL_DIR else []
pytestmark = pytest.mark.skipif(not FILES, reason="TPL_DIR с *.jinja не задан")

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Погода",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
TC = {"role": "assistant", "content": "", "tool_calls": [{"id": "abc123def", "type": "function",
      "function": {"name": "get_weather", "arguments": "{\"city\": \"Moscow\"}"}}]}
TR = {"role": "tool", "tool_call_id": "abc123def", "name": "get_weather", "content": "{\"t\": -3}"}


class Renderer:
    def __init__(self, src):
        from jinja2.sandbox import SandboxedEnvironment
        from jinja2.ext import loopcontrols
        self.day = 26
        env = SandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False, ensure_ascii=False: \
            json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        def raise_exception(m): raise ValueError(m)
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda f: datetime.datetime(2026, 11, self.day).strftime(f)
        self.t = env.from_string(src)

    def __call__(self, messages, tools=None, kwargs=None, gen=True):
        msgs = []
        for m in messages:                           # llama.cpp разбирает arguments в объект для шаблона
            if m.get("tool_calls"):
                m = {**m, "tool_calls": [{**tc, "function": {**tc["function"],
                     "arguments": json.loads(tc["function"]["arguments"])}} for tc in m["tool_calls"]]}
            msgs.append(m)
        kw = dict(messages=msgs, add_generation_prompt=gen, bos_token="<s>", eos_token="</s>", **(kwargs or {}))
        if tools: kw["tools"] = tools
        return self.t.render(**kw)


def lost(p0, p1):
    k = 0
    n = min(len(p0), len(p1))
    while k < n and p0[k] == p1[k]: k += 1
    return len(p0) - k


@pytest.mark.parametrize("path", FILES, ids=[os.path.basename(f)[:-6] for f in FILES])
def test_template(make_ctx, path):
    src = open(path, encoding="utf-8").read()
    R = Renderer(src)
    try:
        gen_tail = len(R([{"role": "user", "content": "x"}])) - len(R([{"role": "user", "content": "x"}], gen=False))
    except Exception:
        gen_tail = 0
    c = make_ctx(CONTEXT_LIMIT="4000", ANSWER_RESERVE="200")
    c.llm.props = {"chat_template": src, "default_generation_settings": {"n_ctx": 4000}}
    c.llm.templates = lambda msgs: R(msgs)
    c.llm.cold(facts=[{"predicate": "lives_in", "object": "Москва", "evidence": "я живу в москве"}])
    c.chat("Я живу в Москве", chat_id="seed")
    c.drain()

    hist = [{"role": "system", "content": "Ты помощник."}]
    raw_hist, sent = [], []

    def send(user=None, extra=None, tools=None):
        msgs = hist + ([{"role": "user", "content": user}] if user else []) + (extra or [])
        body = {"model": "test-model", "messages": msgs}
        if tools: body["tools"] = tools
        r = c.client.post("/v1/chat/completions", json=body, headers={"X-OpenWebUI-Chat-Id": "tpl"})
        assert r.status_code == 200, r.text
        c.drain()
        req = c.llm.chat_requests[-1]
        sent.append((req["messages"], req.get("chat_template_kwargs"), c.proxy.STATS["epoch_switch"]))
        raw_hist.append(msgs)
        return r.json()["choices"][0]["message"]["content"]

    a = send("Какая погода у меня в городе?", tools=TOOLS)
    send("Какая погода у меня в городе?", extra=[TC, TR], tools=TOOLS)              # шаг агента
    hist += [{"role": "user", "content": "Какая погода у меня в городе?"}, TC, TR, {"role": "assistant", "content": a}]
    i = 0
    while c.proxy.STATS["epoch_switch"] == 0 and i < 30:
        q = f"Вопрос {i} про тему {i}: " + " ".join(f"слово{i}_{k}" for k in range(110))
        a = send(q, tools=TOOLS)
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        i += 1
    q = "Последний вопрос после переключения"
    send(q, tools=TOOLS)
    assert c.proxy.STATS["epoch_switch"] == 1, "эпоха должна переключиться"

    info = c.proxy.BACKENDS["test-model"]
    def safe(f):
        try: return f()
        except Exception: return None
    raw = [safe(lambda m=m: R(m, TOOLS)) for m in raw_hist]
    # там, где шаблон рендерит исходный диалог клиента, он обязан рендерить и запрос прокси
    # (Gemma 2 сама не поддерживает роль tool — такие шаги пропускаются)
    rendered = [R(m, TOOLS, kw) if raw[k] is not None else safe(lambda m=m, kw=kw: R(m, TOOLS, kw))
                for k, (m, kw, _) in enumerate(sent)]
    bad = []
    for k in range(1, len(sent)):
        if sent[k][2] != sent[k - 1][2]: continue                    # переключение эпохи — проверяем ниже
        if None in (raw[k - 1], raw[k], rendered[k - 1], rendered[k]): continue
        d, d_raw = lost(rendered[k - 1], rendered[k]), lost(raw[k - 1], raw[k])
        if d <= max(d_raw, gen_tail + 2): continue
        # шаблон и сам ломает префикс здесь (Mistral переносит [AVAILABLE_TOOLS] к последнему вопросу):
        # прокси не виноват, если разрыв начинается в том же месте
        if d_raw > gen_tail + 2 and rendered[k - 1][len(rendered[k - 1]) - d:][:40] == raw[k - 1][len(raw[k - 1]) - d_raw:][:40]:
            continue
        bad.append((k, d, d_raw, rendered[k - 1][len(rendered[k - 1]) - d:][:80]))
    assert not bad, f"прокси ломает префикс: {bad[:3]} (system_mode={info['system_mode']})"

    warm = c.llm.warmup_requests[-1]
    sw = next(k for k in range(len(sent)) if sent[k][2] == 1)
    wp = R(warm["messages"], warm.get("tools"), warm.get("chat_template_kwargs"))
    d = lost(wp, rendered[sw])
    d_raw = lost(raw[sw - 1], raw[sw]) if None not in (raw[sw - 1], raw[sw]) else 0
    same_spot = d_raw > gen_tail + 2 and wp[len(wp) - d:][:30] == raw[sw - 1][len(raw[sw - 1]) - d_raw:][:30]
    assert d <= max(gen_tail + 2, d_raw) or same_spot, f"прогретый префикс не совпал: потеря {d} симв. «{wp[len(wp) - d:][:60]}»"

    if os.getenv("TPL_REPORT"):
        reuse = [1 - lost(rendered[k - 1], rendered[k]) / max(1, len(rendered[k - 1]))
                 for k in range(1, len(sent)) if rendered[k - 1] and rendered[k] and sent[k][2] == sent[k - 1][2]]
        with open(os.getenv("TPL_REPORT"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"template": os.path.basename(path)[:-6], "system_mode": info["system_mode"],
                                "date_kw": info["date_kw"], "requests": len(sent), "warm_lost": d, "warm_len": len(wp),
                                "min_prefix_reuse": round(min(reuse), 3)}, ensure_ascii=False) + "\n")

    if info["date_kw"]:
        m, kw, _ = sent[-1]
        R.day = 27
        assert R(m, TOOLS, kw) == rendered[-1], "дата не закреплена"

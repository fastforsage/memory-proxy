import json
def tc(i, name="read_file", args=None):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args or {"path": f"/src/f{i}.py"})}}

def agent_turn(n, out_len=3000, start=0):
    msgs = []
    for i in range(start, start + n):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [tc(i, args={"path": f"/src/f{i}.py", "patch": "x" * 4000})]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "y" * out_len})
    return msgs

def test_probe(ctx):
    P = ctx.proxy
    # 1. оценка токенов: аргументы tool_calls не учитываются
    m = [{"role": "assistant", "content": None, "tool_calls": [tc(0, args={"patch": "x" * 30000})]}]
    print("\n[1] total_tokens(assistant with 30k-char tool args) =", P.total_tokens(m))
    # 2. один длинный агентный ход: сжатие не происходит
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "почини баг"}] + agent_turn(20)
    out = P.compress(msgs)
    print("[2] one agentic turn 20 calls: tokens", P.total_tokens(msgs), "-> compress len", len(msgs), "->", len(out))
    # 3. обычная история: два system, текст пользователя в system
    hist = [{"role": "system", "content": "sys"}]
    for i in range(30):
        hist += [{"role": "user", "content": f"вопрос {i} " + "а" * 2000}, {"role": "assistant", "content": f"ответ {i} " + "б" * 2000}]
    hist.append({"role": "user", "content": "итог?"})
    out = P.compress(hist)
    print("[3] roles after compress:", [m["role"] for m in out][:4], "| systems:", sum(m["role"] == "system" for m in out),
          "| tokens", P.total_tokens(hist), "->", P.total_tokens(out), "| summary:", out[1]["content"][:120])
    # 4. окно начинается с assistant(tool_calls) из середины старого хода -> старый user отрезан
    msgs = [{"role": "system", "content": "s"}]
    for t in range(3):
        msgs += [{"role": "user", "content": f"задача {t} " + "з" * 3000}] + agent_turn(3, start=t * 10)
        msgs.append({"role": "assistant", "content": f"готово {t}"})
    msgs.append({"role": "user", "content": "дальше"})
    out = P.compress(msgs)
    print("[4] window starts with:", [m["role"] for m in out][:5])
    # 5. продолжение агентного цикла (последнее сообщение tool): повторное сохранение / cold
    ctx.llm.cold(facts=[])
    first = [{"role": "user", "content": "Я живу в Казани, найди погоду"}]
    ctx.client.post("/v1/chat/completions", json={"model": "m", "messages": first}, headers={"X-OpenWebUI-Chat-Id": "ag"})
    ctx.drain()
    cont = first + [{"role": "assistant", "content": "Сейчас посмотрю погоду в Казани.", "tool_calls": [tc(1, "weather", {"city": "Казань"})]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "+3, дождь"}]
    n_cold0 = len(ctx.llm.cold_requests)
    ctx.client.post("/v1/chat/completions", json={"model": "m", "messages": cont}, headers={"X-OpenWebUI-Chat-Id": "ag"})
    ctx.drain()
    rows = ctx.db("SELECT role, substr(content,1,40) FROM messages WHERE chat_id='ag' ORDER BY id")
    print("[5] stored:", rows, "| extra cold calls:", len(ctx.llm.cold_requests) - n_cold0)
    req = ctx.llm.chat_requests[-1]["messages"]
    print("    memory injected into:", [i for i, m in enumerate(req) if isinstance(m.get("content"), str) and "<memory>" in m["content"]], "of", len(req))

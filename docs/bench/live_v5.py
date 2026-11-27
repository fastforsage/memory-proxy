"""Живой прогон v5: прокси :8081 → llama-server :8080 (Qwen2.5-0.5B, -c 8192), COMPRESS_AT=3000.
Клиент шлёт полную историю (как Open WebUI). На каждом ходе берём timings llama.cpp: prompt_n (пересчитано)
и cache_n (из KV-кэша). Между ходами ждём, пока слот свободен (фоновые саммари/прогрев)."""
import json, sys, time
import httpx

P = "http://127.0.0.1:8081"
L = "http://127.0.0.1:8080"
c = httpx.Client(timeout=600)
TOPICS = ["байкал", "кухня", "отпуск", "ремонт", "бюджет", "машина", "дача", "сад", "кот", "работа", "спорт", "книги",
          "кино", "музыка", "поход", "рыбалка", "театр", "выставка", "погода", "еда", "курсы", "язык"]


def idle(maxwait=300):
    t0 = time.time()
    while time.time() - t0 < maxwait:
        try:
            if not any(s.get("is_processing") for s in c.get(L + "/slots").json()): return time.time() - t0
        except Exception: return 0
        time.sleep(0.5)
    return maxwait


def ask(chat, hist, text, stream=False, chat_id="live"):
    body = {"model": "qwen", "messages": hist + [{"role": "user", "content": text}], "max_tokens": 40,
            "temperature": 0, "stream": stream}
    t0 = time.time()
    if stream:
        content, tim = "", {}
        with c.stream("POST", P + "/v1/chat/completions", json=body, headers={"X-OpenWebUI-Chat-Id": chat_id}) as r:
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]": continue
                j = json.loads(line[6:])
                if j.get("timings"): tim = j["timings"]
                for ch in j.get("choices") or []:
                    content += (ch.get("delta") or {}).get("content") or ""
    else:
        r = c.post(P + "/v1/chat/completions", json=body, headers={"X-OpenWebUI-Chat-Id": chat_id})
        assert r.status_code == 200, r.text
        j = r.json(); content = j["choices"][0]["message"]["content"]; tim = j.get("timings") or {}
    return content, tim, time.time() - t0


def main():
    rows = []
    hist = [{"role": "system", "content": "Ты дружелюбный помощник. Отвечай кратко."}]
    first = "Запомни: ключ от гаража лежит в синей банке на третьей полке. Скажи «ок»."
    for i in range(22):
        if i == 0: q = first
        elif i == 21: q = "Напомни, где лежит ключ от гаража? Синяя банка — на какой полке?"
        else:
            q = f"Ход {i}. Тема: {TOPICS[i]}. " + " ".join(
                f"Предложение {k} о теме {TOPICS[i]}, где важны детали и подробности." for k in range(14))
        a, tim, dt = ask(c, hist, q, stream=(i % 2 == 1))
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        h = c.get(P + "/health").json()["context"]
        w = idle()
        rows.append({"turn": i, "prompt_n": tim.get("prompt_n"), "cache_n": tim.get("cache_n"), "sec": round(dt, 2),
                     "bg_wait": round(w, 1), "switch": h["epoch_switch"], "sync": h["epoch_sync"],
                     "summaries": h["summaries"], "warmup": h["warmup"], "answer": a[:60]})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    h = c.get(P + "/health").json()
    print(json.dumps(h, ensure_ascii=False, indent=1))
    json.dump({"rows": rows, "health": h}, open(sys.argv[1] if len(sys.argv) > 1 else "live_v5.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

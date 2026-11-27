"""Симуляция: один и тот же диалог, сжатие v4.8 (точная копия кода) vs схема v4.9.
Для каждого хода — токены промпта (шаблон Qwen2.5 через llama-server) и сколько из них пришлось бы
пересчитать: prompt_n = len(prompt) - LCP(prompt, состояние KV после прошлого хода)."""
import os, random, sys, json
from common import text, tokens

COMPRESS_AT = int(os.getenv("COMPRESS_AT", 16000)); KEEP_RECENT = 8
TARGET = int(0.6 * COMPRESS_AT)
USER_CH, ASSIST_CH, MEM_CH, SYS_CH, SUM_CH = (int(os.getenv(k, d)) for k, d in
    (("USER_CH", 500), ("ASSIST_CH", 1500), ("MEM_CH", 1500), ("SYS_CH", 400), ("SUM_CH", 1800)))
TURNS = int(os.getenv("TURNS", 60))
POLICY = "В сообщении пользователя может встретиться блок <memory>…</memory>. " * 4

# ── v4.8: дословно из proxy.py ──
def rough_tokens(t): return max(1, len(t) // 3) if t else 0
def total_tokens(msgs): return sum(rough_tokens(m["content"]) if isinstance(m.get("content"), str) else 10 for m in msgs)
def compress48(messages):
    system = [m for m in messages if m.get("role") == "system"]
    non_sys = [m for m in messages if m.get("role") != "system"]
    if len(non_sys) <= KEEP_RECENT + 2: return messages
    cut = len(non_sys) - KEEP_RECENT
    while cut > 0 and non_sys[cut].get("role") == "tool": cut -= 1
    old, recent = non_sys[:cut], non_sys[cut:]
    if not old: return messages
    frag = " | ".join(f"{m.get('role')}: {str(m.get('content',''))[:100]}" for m in old[-8:] if isinstance(m.get("content"), str))
    return system + [{"role": "system", "content": f"[РАНЕЕ]: {frag}"}] + recent

def inject(msgs, mem):
    msgs = [dict(m) for m in msgs]
    msgs[0]["content"] += "\n\n" + POLICY
    msgs[-1]["content"] = f"<memory>\n{mem}\n</memory>\n\n" + msgs[-1]["content"]
    return msgs

def v48(hist, mem, st):
    m = inject(hist, mem)
    return compress48(m) if total_tokens(m) > COMPRESS_AT else m

# ── v4.9: окно с границы хода k, саммари в первом user окна, k меняется ступенчато ──
def v49(hist, mem, st):
    sysm, body = hist[0], hist[1:]
    turns = [body[i:i + 2] for i in range(0, len(body) - 1, 2)]     # завершённые ходы (user, assistant)
    cur = body[-1]
    def build(k):
        win = [dict(m) for t in turns[k:] for m in t] + [dict(cur)]
        if k:
            rnd = random.Random(k)                                       # саммари меняется только вместе с k
            win[0]["content"] = f"<summary>\n{text(SUM_CH, rnd)}\n</summary>\n\n" + win[0]["content"]
        return inject([dict(sysm)] + win, mem)
    k = st.get("k", 0)
    if total_tokens(build(k)) > COMPRESS_AT:
        while k < len(turns) - KEEP_RECENT // 2 and total_tokens(build(k)) > TARGET: k += 1
        st["k"] = k; st["shifts"] = st.get("shifts", 0) + 1
    return build(k)

def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i

def run(strategy, reply_matches=True, seed=7):
    rnd = random.Random(seed)
    hist = [{"role": "system", "content": "Ты внимательный ассистент. " + text(SYS_CH, rnd)}]
    state, st, rows = [], {}, []
    for t in range(TURNS):
        hist.append({"role": "user", "content": f"Ход {t}. " + text(USER_CH, rnd)})
        mem = text(MEM_CH, random.Random(1000 + t))                     # память меняется каждый ход
        sent = strategy(hist, mem, st)
        toks = tokens(sent)
        reuse = lcp(toks, state)
        reply = {"role": "assistant", "content": text(ASSIST_CH, rnd)}
        state = tokens(sent + [reply]) if reply_matches else toks       # KV после генерации
        rows.append({"turn": t, "prompt": len(toks), "prefill": len(toks) - reuse, "msgs": len(sent),
                     "shift": st.pop("shifts", 0)})
        hist.append(reply)
    return rows

if __name__ == "__main__":
    out = {}
    for name, fn in (("v48", v48), ("v49", v49)):
        out[name] = run(fn, reply_matches=os.getenv("MATCH", "1") == "1")
        print(name, "done", file=sys.stderr)
    json.dump(out, open(os.getenv("OUT", "sim.json"), "w"))

# ── v4.9 + фоновый прогрев: новое окно (саммари + ходы) префиллится в фоне до переключения ──
def run_warm(seed=7, reply_matches=True):
    rnd = random.Random(seed)
    hist = [{"role": "system", "content": "Ты внимательный ассистент. " + text(SYS_CH, rnd)}]
    state, st, rows, bg = [], {}, [], 0
    for t in range(TURNS):
        hist.append({"role": "user", "content": f"Ход {t}. " + text(USER_CH, rnd)})
        mem = text(MEM_CH, random.Random(1000 + t))
        sent = v49(hist, mem, st)
        shifted = st.pop("shifts", 0)
        toks = tokens(sent)
        if shifted:
            # прогрев был сделан после прошлого ответа: префикс = всё, кроме последнего user-сообщения
            warm = tokens(sent[:-1])
            bg += len(warm)
            state = warm
        reuse = lcp(toks, state)
        reply = {"role": "assistant", "content": text(ASSIST_CH, rnd)}
        state = tokens(sent + [reply]) if reply_matches else toks
        rows.append({"turn": t, "prompt": len(toks), "prefill": len(toks) - reuse, "msgs": len(sent), "shift": shifted})
        hist.append(reply)
    return rows, bg

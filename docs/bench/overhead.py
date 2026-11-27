import hashlib, json, random, sqlite3, time, statistics, os, sys
from common import text
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__))); import sim
rnd = random.Random(5)
def chat_hist(turns):
    h = [{"role": "system", "content": text(400, rnd)}]
    for t in range(turns): h += [{"role": "user", "content": text(500, rnd)}, {"role": "assistant", "content": text(1500, rnd)}]
    return h + [{"role": "user", "content": text(500, rnd)}]
def agent_hist(steps):
    h = chat_hist(10)
    for s in range(steps):
        h += [{"role": "assistant", "content": None, "tool_calls": [{"id": f"c{s}", "type": "function", "function": {"name": "sh", "arguments": json.dumps({"cmd": "x" * 300})}}]},
              {"role": "tool", "tool_call_id": f"c{s}", "content": text(3000, rnd)}]
    return h

# ── прототип v4.9 hot path ──
def est(m):
    c = m.get("content"); n = len(c) // 3 if isinstance(c, str) else 0
    for tc in m.get("tool_calls") or []: n += (len(tc["function"]["name"]) + len(tc["function"]["arguments"])) // 3 + 10
    return n + 4
def segment(msgs):
    i = 0
    while i < len(msgs) and msgs[i]["role"] in ("system", "developer"): i += 1
    turns, cur = [], []
    for m in msgs[i:]:
        if m["role"] == "user" and cur: turns.append(cur); cur = []
        cur.append(m)
    if cur: turns.append(cur)
    return msgs[:i], turns
def canon(m):
    tc = [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []]
    return json.dumps([m["role"], m.get("content"), tc], ensure_ascii=False).encode()
def hash_chain(turns):
    h, out = b"", []
    for t in turns:
        s = hashlib.sha256(h)
        for m in t: s.update(canon(m))
        h = s.digest(); out.append(h.hex())
    return out
def elide(turns, keep=1500):
    out = []
    for t in turns[:-1]:
        nt = []
        for m in t:
            c = m.get("content")
            if m["role"] == "tool" and isinstance(c, str) and len(c) > keep:
                m = {**m, "content": c[:1000] + f"…[сокращено {len(c)}]…" + c[-300:]}
            nt.append(m)
        out.append(nt)
    return out + turns[-1:]
db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE chat_summaries(user_id, chat_id, prefix_hash, summary, UNIQUE(user_id, chat_id, prefix_hash))")
def lookup(hashes):
    ph = ",".join("?" * len(hashes))
    return db.execute(f"SELECT prefix_hash, summary FROM chat_summaries WHERE user_id=? AND chat_id=? AND prefix_hash IN ({ph})",
                      ["u", "c", *hashes]).fetchall()
def v49_hot(msgs):
    sysp, turns = segment(msgs)
    total = sum(est(m) for m in msgs)
    hs = hash_chain(turns)
    lookup(hs[-200:])
    t2 = elide(turns)
    return sysp, t2, total

def bench(fn, arg, n=200):
    xs = []
    for _ in range(n):
        t = time.perf_counter(); fn(arg); xs.append((time.perf_counter() - t) * 1000)
    return statistics.median(xs), sorted(xs)[int(n * .95)]
for name, h in (("чат 30 ходов", chat_hist(30)), ("чат 100 ходов", chat_hist(100)), ("агент 60 шагов", agent_hist(60))):
    kb = len(json.dumps(h, ensure_ascii=False).encode()) // 1024
    a = bench(lambda m: (sim.total_tokens(m), sim.compress48(m)), h)
    b = bench(v49_hot, h)
    print(f"{name:<16} {len(h):>4} сообщ, {kb:>4} KB | v4.8 compress: {a[0]:.2f} мс (p95 {a[1]:.2f}) | v4.9 hot path: {b[0]:.2f} мс (p95 {b[1]:.2f})")

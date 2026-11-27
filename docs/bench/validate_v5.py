import os, random, time, json
os.environ.setdefault("COMPRESS_AT", "2000")
import sim, sim_v5
from common import text, tokens, chat
from sim import lcp, total_tokens, POLICY
sim.USER_CH, sim.ASSIST_CH, sim.SYS_CH, sim.SUM_CH, sim.TURNS = 300, 700, 200, 600, 16
PROFILE_CH, DELTA_CH = 500, 150
def real(seed=7):
    rnd = random.Random(seed)
    hist = [{"role": "system", "content": "[v5] Ты внимательный ассистент. " + text(sim.SYS_CH, rnd)}]
    pinned, epoch, k, state, rows = {}, 0, 0, [], []
    def build(k, epoch, upto):
        win = []
        for j, m in enumerate(hist[1:upto][2 * k:], start=1 + 2 * k):
            m = dict(m)
            if m["role"] == "user" and j in pinned: m["content"] = f"<memory>\n{pinned[j]}\n</memory>\n\n" + m["content"]
            win.append(m)
        head = f"<profile>\n{text(PROFILE_CH, random.Random(5000 + epoch))}\n</profile>\n\n"
        if k: head = f"<summary>\n{text(sim.SUM_CH, random.Random(k))}\n</summary>\n\n" + head
        win[0] = {**win[0], "content": head + win[0]["content"]}
        return [{"role": "system", "content": hist[0]["content"] + "\n\n" + POLICY}] + win
    for t in range(sim.TURNS):
        hist.append({"role": "user", "content": f"Ход {t}. " + text(sim.USER_CH, rnd)})
        pinned[len(hist) - 1] = text(DELTA_CH, random.Random(1000 + t))
        sent = build(k, epoch, len(hist)); shifted = 0
        if total_tokens(sent) > sim.COMPRESS_AT:
            turns = (len(hist) - 2) // 2; nk = k
            while nk < turns - 2 and total_tokens(build(nk, epoch + 1, len(hist))) > sim.TARGET: nk += 1
            k, epoch, shifted = nk, epoch + 1, 1
            sent = build(k, epoch, len(hist))
            chat(sent[:-1], max_tokens=1); state = tokens(sent[:-1])          # фоновый прогрев
        toks = tokens(sent); pred = len(toks) - lcp(toks, state)
        r = chat(sent, max_tokens=4); tm = r["timings"]
        state = toks + [-1]
        rows.append((t, len(toks), pred, tm["prompt_n"], round(tm["prompt_ms"] / 1000, 2), shifted))
        hist.append({"role": "assistant", "content": text(sim.ASSIST_CH, rnd)})
    return rows
rows = real()
for x in rows: print(x)
print("совпадение:", sum(abs(a[2] - a[3]) <= 3 for a in rows), "/", len(rows), "| prefill после 8-го хода:", round(sum(a[4] for a in rows[8:]), 1), "с за", len(rows) - 8, "ходов")
json.dump(rows, open("validate_v5.json", "w"))

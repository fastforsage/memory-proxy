import os, random, json, time
os.environ.setdefault("COMPRESS_AT", "2000")
import sim
from common import text, tokens, chat
sim.USER_CH, sim.ASSIST_CH, sim.MEM_CH, sim.SYS_CH, sim.SUM_CH, sim.TURNS = 300, 700, 600, 200, 600, 20

def real(strategy, tag, warm=False, seed=7):
    rnd = random.Random(seed)
    hist = [{"role": "system", "content": f"[{tag}] Ты внимательный ассистент. " + text(sim.SYS_CH, rnd)}]
    st, rows, state = {}, [], []
    for t in range(sim.TURNS):
        hist.append({"role": "user", "content": f"Ход {t}. " + text(sim.USER_CH, rnd)})
        mem = text(sim.MEM_CH, random.Random(1000 + t))
        sent = strategy(hist, mem, st); shifted = st.pop("shifts", 0)
        toks = tokens(sent)
        if warm and shifted:
            chat(sent[:-1], max_tokens=1); state = tokens(sent[:-1])
        pred = len(toks) - sim.lcp(toks, state)
        t0 = time.time(); r = chat(sent, max_tokens=4); dt = time.time() - t0
        tm = r["timings"]
        state = toks + [-1]                        # сгенерированное не совпадает с историей (MATCH=0)
        rows.append((t, len(toks), pred, tm["prompt_n"], round(tm["prompt_ms"] / 1000, 2), shifted))
        hist.append({"role": "assistant", "content": text(sim.ASSIST_CH, rnd)})
    return rows

res = {}
for tag, fn, warm in (("v48", sim.v48, False), ("v49", sim.v49, False), ("v49w", sim.v49, True)):
    res[tag] = real(fn, tag, warm)
    r = res[tag]
    print(tag, "turn,prompt,pred_prefill,real_prompt_n,prefill_s,shift")
    for x in r: print("  ", x)
    print(f"  совпадение предсказания с llama.cpp: {sum(abs(a[2]-a[3]) <= 3 for a in r)}/{len(r)} (±3 ток); "
          f"prefill после сжатия: {sum(a[4] for a in r[8:]):.1f} с за {len(r)-8} ходов", flush=True)
json.dump(res, open("validate.json", "w"))

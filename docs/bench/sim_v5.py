"""v5: «промпт только дописывается».
- Профиль (факты, стиль, эмоции) — один блок на эпоху, в голове окна; меняется только при смене эпохи.
- На каждый ход — небольшая дельта памяти (только то, чего ещё нет в контексте), закрепляется
  в этом user-сообщении навсегда: при следующих запросах оно уходит в модель точно таким же.
- Смена эпохи (саммари + новый профиль + новая граница окна) готовится и прогревается в фоне."""
import json, os, random
import sim
from common import text, tokens
from sim import lcp, total_tokens, POLICY, COMPRESS_AT, TARGET, KEEP_RECENT
PROFILE_CH = int(os.getenv("PROFILE_CH", 1500)); DELTA_CH = int(os.getenv("DELTA_CH", 300))

def run_v5(seed=7):
    rnd = random.Random(seed)
    hist = [{"role": "system", "content": "Ты внимательный ассистент. " + text(sim.SYS_CH, rnd)}]
    pinned = {}                                   # индекс user-сообщения -> дельта, закреплённая при отправке
    epoch, k, state, rows, bg = 0, 0, [], [], 0
    def build(k, epoch, upto):
        sysm = {"role": "system", "content": hist[0]["content"] + "\n\n" + POLICY}
        body = hist[1:upto]
        win = []
        for j, m in enumerate(body[2 * k:], start=1 + 2 * k):
            m = dict(m)
            if m["role"] == "user" and j in pinned:
                m["content"] = f"<memory>\n{pinned[j]}\n</memory>\n\n" + m["content"]
            win.append(m)
        head = f"<profile>\n{text(PROFILE_CH, random.Random(5000 + epoch))}\n</profile>\n\n"
        if k:
            head = f"<summary>\n{text(sim.SUM_CH, random.Random(k))}\n</summary>\n\n" + head
        win[0] = {**win[0], "content": head + win[0]["content"]}
        return [sysm] + win
    for t in range(sim.TURNS):
        hist.append({"role": "user", "content": f"Ход {t}. " + text(sim.USER_CH, rnd)})
        idx = len(hist) - 1
        pinned[idx] = text(DELTA_CH, random.Random(1000 + t))            # дельта этого хода
        sent = build(k, epoch, len(hist))
        shifted = False
        if total_tokens(sent) > COMPRESS_AT * float(os.getenv("EPOCH_AT", "0.85")):                        # смена эпохи готовится заранее
            turns = (len(hist) - 2) // 2
            nk = k
            while nk < turns - KEEP_RECENT // 2 and total_tokens(build(nk, epoch + 1, len(hist))) > TARGET: nk += 1
            k, epoch, shifted = nk, epoch + 1, True
            sent = build(k, epoch, len(hist))
            warm = tokens(sent[:-1]); bg += len(warm); state = warm      # прогрев сделан в фоне
        toks = tokens(sent)
        reuse = lcp(toks, state)
        reply = {"role": "assistant", "content": text(sim.ASSIST_CH, rnd)}
        state = tokens(sent + [reply])
        rows.append({"turn": t, "prompt": len(toks), "prefill": len(toks) - reuse, "msgs": len(sent), "shift": int(shifted)})
        hist.append(reply)
    return rows, bg

if __name__ == "__main__":
    rows, bg = run_v5()
    d = json.load(open("sim.json")); d["v5"] = rows; d["v5_bg"] = bg
    json.dump(d, open("sim.json", "w"))
    for v in ("v48", "v49w", "v5"):
        r = d[v]
        pre = [x["prefill"] for x in r if 3 <= x["turn"] < 23]; post = [x for x in r if x["turn"] >= 23]
        print(f"{v:<5} до переполнения: пересчёт ср {sum(pre)/len(pre):.0f} | после: промпт ср {sum(x['prompt'] for x in post)/len(post):.0f}, "
              f"пересчёт ср {sum(x['prefill'] for x in post)/len(post):.0f}, макс {max(x['prefill'] for x in post)}, сдвигов {sum(x['shift'] for x in r)}")
    print("фон v49w:", d["v49w_bg"], "ток, v5:", bg, "ток")

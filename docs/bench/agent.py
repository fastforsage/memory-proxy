"""Агентный ход: 15 шагов инструментов. v4.8: <memory> собирается заново на каждом шаге (стиль ORDER BY RANDOM()).
v4.9: блок памяти из кеша — побайтно одинаковый."""
import json, random
from common import text, tokens
from sim import lcp, POLICY
rnd = random.Random(3)
sysm = {"role": "system", "content": "Ты агент-программист. " + text(400, rnd) + "\n\n" + POLICY}
prev = []
for t in range(6):
    prev += [{"role": "user", "content": text(500, rnd)}, {"role": "assistant", "content": text(1500, rnd)}]
task = "Почини падающие тесты в модуле оплаты. " + text(300, rnd)
facts = text(500, random.Random(11))
def style(seed): return text(1200, random.Random(seed))           # 3 случайных примера стиля
def run(cached):
    steps, state, res = [], [], []
    for s in range(15):
        mem = facts + "\n" + style(42 if cached else 100 + s)
        user = {"role": "user", "content": f"<memory>\n{mem}\n</memory>\n\n{task}"}
        msgs = [sysm] + prev + [user] + steps
        toks = tokens(msgs)
        res.append((s, len(toks), len(toks) - lcp(toks, state)))
        call = {"id": f"c{s}", "type": "function", "function": {"name": "run_shell", "arguments": json.dumps({"cmd": f"pytest tests/test_pay_{s}.py -x"})}}
        a = {"role": "assistant", "content": "", "tool_calls": [call]}
        steps += [a, {"role": "tool", "tool_call_id": f"c{s}", "content": text(2000, rnd)}]
        state = tokens(msgs + [a, {"role": "tool", "tool_call_id": f"c{s}", "content": "\u2400"}])
    return res
for name, cached in (("v48 (память заново)", False), ("v49 (кеш блока)", True)):
    r = run(cached)
    print(f"{name}: пересчёт по шагам {[x[2] for x in r]}\n   сумма {sum(x[2] for x in r)}, промпт на последнем шаге {r[-1][1]}")
import json as _j; _j.dump({"v48": run(False), "v49": run(True)}, open("agent.json", "w"))

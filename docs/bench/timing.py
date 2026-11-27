import json
# 7B Q4_0, llama.cpp: pp512 / tg128 (ток/с) на глубине 0 и «глубина половинного падения» D (pp(d)=pp0/(1+d/D))
P = {
 "CPU ноутбука (Ryzen AI, 8 потоков)": (36, 10.3, 12670, 8630),
 "Apple M2 Pro 19c":                   (341, 38.9, 17000, 65000),
 "RTX 3090":                           (5175, 158, 17000, 65000),
 "RTX 4090":                           (11993, 186, 17000, 65000),
}
REPLY = 480
def t_prefill(n, depth0, pp0, D): return (n + ((depth0 + n) ** 2 - depth0 ** 2) / (2 * D)) / pp0
def t_decode(m, depth, tg0, Dt): return m * (1 + depth / Dt) / tg0
def fmt(s): return f"{s:.1f} с" if s < 100 else f"{s/60:.1f} мин"

d = json.load(open("sim.json")); ag = json.load(open("agent.json"))
FIRST = 23
print("## Обычный чат, 60 ходов, ходы после переполнения (23–59)\n")
print("| Профиль | Версия | TTFT ср | TTFT макс | Полный ответ ср (480 ток) |")
print("|---|---|---|---|---|")
for pn, (pp, tg, Dp, Dt) in P.items():
    for v, label in (("v48", "v4.8"), ("v49", "v4.9 без прогрева"), ("v49w", "**v4.9 + прогрев**")):
        rows = [r for r in d[v] if r["turn"] >= FIRST]
        tt = [t_prefill(r["prefill"], r["prompt"] - r["prefill"], pp, Dp) for r in rows]
        full = [a + t_decode(REPLY, r["prompt"], tg, Dt) for a, r in zip(tt, rows)]
        print(f"| {pn} | {label} | {fmt(sum(tt)/len(tt))} | {fmt(max(tt))} | {fmt(sum(full)/len(full))} |")
print("\n## До переполнения (ходы 0–22) все версии одинаковы\n")
rows = [r for r in d["v48"] if 3 <= r["turn"] < FIRST]
for pn, (pp, tg, Dp, Dt) in P.items():
    tt = [t_prefill(r["prefill"], r["prompt"] - r["prefill"], pp, Dp) for r in rows]
    print(f"- {pn}: TTFT ср {fmt(sum(tt)/len(tt))}")
print("\n## Агентный ход, 15 шагов инструментов (суммарно на весь ход, без генерации)\n")
print("| Профиль | v4.8 | v4.9 (кеш блока памяти) |")
print("|---|---|---|")
for pn, (pp, tg, Dp, Dt) in P.items():
    s = {v: sum(t_prefill(p, tot - p, pp, Dp) for _, tot, p in ag[v]) for v in ("v48", "v49")}
    print(f"| {pn} | {fmt(s['v48'])} | {fmt(s['v49'])} |")
print("\n## Фоновая работа v4.9 (раз в ~10 ходов после переполнения)\n")
bg_warm = d["v49w_bg"] / 4
print("| Профиль | Суммаризация (~7k вход, 700 выход) | Прогрев (~%dk) | Итого слот занят |" % round(bg_warm / 1000))
print("|---|---|---|---|")
for pn, (pp, tg, Dp, Dt) in P.items():
    s = t_prefill(7000, 0, pp, Dp) + t_decode(700, 7000, tg, Dt)
    w = t_prefill(bg_warm, 0, pp, Dp)
    print(f"| {pn} | {fmt(s)} | {fmt(w)} | {fmt(s + w)} |")

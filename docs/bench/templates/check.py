import json, glob, os, datetime
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.ext import loopcontrols
class TplErr(Exception): pass
def raise_exception(m): raise TplErr(m)
DATE = ["26 Nov 2026"]
def mkenv():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False, ensure_ascii=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda f: datetime.datetime(2026, 11, 26 if DATE[0].startswith("26") else 27).strftime(f)
    return env
SYS = {"role": "system", "content": "Ты помощник."}
U = lambda i: {"role": "user", "content": f"Вопрос {i}: расскажи про тему номер {i}."}
A = lambda i: {"role": "assistant", "content": f"Ответ {i}: вот подробности по теме {i}."}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Погода", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
TC = {"role": "assistant", "content": "", "tool_calls": [{"id": "abc123def", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Moscow"}}}]}
TR = {"role": "tool", "tool_call_id": "abc123def", "name": "get_weather", "content": "{\"t\": -3}"}
def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i
res = {}
for f in sorted(glob.glob("*.jinja")):
    name = f[:-6]; t = mkenv().from_string(open(f).read())
    def R(msgs, tools=None, gen=True, date="26 Nov 2026"):
        DATE[0] = date
        kw = dict(messages=msgs, add_generation_prompt=gen, bos_token="<s>", eos_token="</s>")
        if tools: kw["tools"] = tools
        return t.render(**kw)
    r = {}
    def lost(p0, p1):
        k = lcp(p0, p1); return 0 if k >= len(p0) else len(p0) - k, p0[k:k+60]
    try:
        base = [SYS, U(1), A(1), U(2)]
        r["chat"] = lost(R(base), R(base + [A(2), U(3)]))
    except Exception as e: r["chat"] = ("ERR", str(e)[:80])
    try: R([U(1), A(1), U(2)]); r["nosys"] = "ok"
    except Exception as e: r["nosys"] = "ERR " + str(e)[:60]
    try: R(base); r["sys"] = "ok"
    except Exception as e: r["sys"] = "ERR " + str(e)[:60]
    try: R([SYS, U(1), U(2)]); r["uu"] = "ok"
    except Exception as e: r["uu"] = "ERR " + str(e)[:60]
    try:
        p0 = R(base, TOOLS); p1 = R(base + [TC, TR], TOOLS); p2 = R(base + [TC, TR, A(2), U(3)], TOOLS)
        r["tool_step"] = lost(p0, p1); r["tool_next_turn"] = lost(p1, p2)
    except Exception as e: r["tool_step"] = r["tool_next_turn"] = ("ERR", str(e)[:80])
    try: r["date"] = lost(R(base, date="26 Nov 2026"), R(base, date="27 Nov 2026"))
    except Exception as e: r["date"] = ("ERR", str(e)[:60])
    res[name] = r
json.dump(res, open("res.json", "w"), ensure_ascii=False, indent=1)
for n, r in res.items():
    def s(x): return x if isinstance(x, str) else ("ok" if x[0] == 0 else f"BREAK {x[0]} «{x[1]!r}»"[:90] if x[0] != "ERR" else "ERR " + x[1][:50])
    print(f"{n[:42]:42} chat={s(r['chat'])} | step={s(r['tool_step'])} | turn={s(r['tool_next_turn'])} | date={s(r['date'])} | nosys={r['nosys'][:30]} | sys={r['sys'][:40]} | uu={r['uu'][:40]}")
print("\n=== classification ===")
cnt = {}
for f in sorted(glob.glob("*.jinja")):
    name = f[:-6]; t = mkenv().from_string(open(f).read()); r = res[name]
    try:
        base = [SYS, U(1), A(1), U(2)]
        g = len(t.render(messages=base, add_generation_prompt=True, bos_token="<s>", eos_token="</s>")) - len(t.render(messages=base, add_generation_prompt=False, bos_token="<s>", eos_token="</s>"))
    except Exception: g = None
    out = []
    for k in ("chat", "tool_step", "tool_next_turn", "date"):
        x = r[k]
        if isinstance(x, tuple) and x[0] == "ERR": c = "err"
        elif x[0] == 0: c = "ok"
        elif g is not None and x[0] <= g + 2: c = "tail"
        else: c = f"BREAK{x[0]}"
        out.append(f"{k}={c}"); cnt.setdefault(k, {}).setdefault(c if not c.startswith("BREAK") else "break", 0); cnt[k][c if not c.startswith("BREAK") else "break"] += 1
    print(f"{name[:44]:44} gen={g} " + " ".join(out) + f" sys={r['sys'][:3]} uu={r['uu'][:3]}")
print(cnt)

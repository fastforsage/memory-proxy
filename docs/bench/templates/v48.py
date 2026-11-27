import glob
exec(open("check.py").read().split("res = {}")[0])
bad = []
for f in sorted(glob.glob("*.jinja")):
    t = mkenv().from_string(open(f).read())
    msgs = [SYS, {"role": "system", "content": "[РАНЕЕ]: user: ..."}] + sum([[U(i), A(i)] for i in range(4)], []) + [U(9)]
    try: t.render(messages=msgs, add_generation_prompt=True, bos_token="<s>", eos_token="</s>"); r = "ok"
    except Exception as e: r = "ERR " + str(e)[:70]; bad.append(f[:-6])
    print(f"{f[:-6][:44]:44} {r}")
print(len(bad), bad)

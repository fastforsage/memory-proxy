"""Парный бутстреп разницы MRR@10 между двумя вариантами (по обратным рангам `rr` из emb_results.json).

  python bootstrap.py emb_results.json user2 granite

Печатает разницу B − A и 95% доверительный интервал для доменного набора (всего и по языкам)
и для публичных наборов retrieval.
"""
import json, random, sys


def boot(a, b, n=10000, seed=0):
    rnd = random.Random(seed); k = len(a); d = [y - x for x, y in zip(a, b)]
    mean = sum(d) / k
    xs = sorted(sum(d[rnd.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return mean, xs[int(0.025 * n)], xs[int(0.975 * n)]


def main(path, A, B):
    r = json.load(open(path)); a, b = r[A], r[B]
    out = {}
    ma, mb = a["memory"], b["memory"]
    groups = {"all": None, "ru+ru2": {"ru", "ru2"}, "en": {"en"}, "uk": {"uk"},
              "de+kk+tr+es": {"de", "kk", "tr", "es"}, "mix": {"mix"}}
    for g, langs in groups.items():
        ii = [i for i, l in enumerate(ma["langs"]) if langs is None or l in langs]
        out[f"memory/{g}"] = (len(ii),) + boot([ma["rr"][i] for i in ii], [mb["rr"][i] for i in ii])
    for key in ("rubq", "wiki_bg", "wiki_de"):
        if key in a and key in b and "rr" in a[key] and "rr" in b[key]:
            out[key] = (len(a[key]["rr"]),) + boot(a[key]["rr"], b[key]["rr"])
    print(f"MRR@10: {B} − {A}, 95% CI (paired bootstrap, 10k)")
    for k, (n, m, lo, hi) in out.items():
        sig = "*" if lo > 0 or hi < 0 else " "
        print(f"  {k:<20} n={n:<4} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}] {sig}")
    return out


if __name__ == "__main__":
    main(*sys.argv[1:4])

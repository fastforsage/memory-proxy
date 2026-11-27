"""Сравнение моделей эмбеддингов для memory-proxy в тех же условиях, что в proxy.py:
onnxruntime CPU (intra_op=2), tokenizers, truncation 512, L2-нормировка, cosine.

  MODELS=user2,granite,granite_int8,granite_asis python eval.py out.json

Варианты:
  user2         — штатная: USER2-small fp32, mean pooling, префиксы search_query:/search_document:
  user2_int8    — то же, int8
  granite       — granite-97m-multilingual-r2 fp32, CLS pooling, без префиксов (как в карточке модели)
  granite_int8  — то же, int8 (onnx-community model_int8)
  granite_asis  — granite, подставленная в текущий proxy.py без правок кода: mean pooling + префиксы USER2
"""
import json, os, re, resource, sqlite3, sys, time, random
import numpy as np, onnxruntime as ort
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = os.getenv("EMB_ROOT", "/var/tmp/emb2")
DATA = os.path.join(ROOT, "data")
VARIANTS = {
    "user2":        dict(dir="user2",   onnx="onnx/model.onnx",      pool="mean", q="search_query: ", d="search_document: "),
    "user2_int8":   dict(dir="user2",   onnx="onnx/model_int8.onnx", pool="mean", q="search_query: ", d="search_document: "),
    "granite":      dict(dir="granite", onnx="onnx/model.onnx",      pool="cls",  q="", d=""),
    "granite_int8": dict(dir="granite", onnx="onnx/model_int8.onnx", pool="cls",  q="", d=""),
    "granite_asis": dict(dir="granite", onnx="onnx/model.onnx",      pool="mean", q="search_query: ", d="search_document: "),
}


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Emb:
    def __init__(self, name):
        v = VARIANTS[name]; self.v = v; self.name = name
        d = os.path.join(ROOT, v["dir"])
        r0, t0 = rss_mb(), time.perf_counter()
        o = ort.SessionOptions(); o.intra_op_num_threads = 2; o.inter_op_num_threads = 1
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; o.log_severity_level = 3
        self.s = ort.InferenceSession(os.path.join(d, v["onnx"]), sess_options=o, providers=["CPUExecutionProvider"])
        self.inp = {i.name for i in self.s.get_inputs()}
        t = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        t.enable_truncation(max_length=512)
        pad = (t.padding or {}).get("pad_token") or "[PAD]"; pid = t.token_to_id(pad)
        t.enable_padding(pad_id=pid if pid is not None else 0, pad_token=pad)
        self.t = t
        self.load_s = time.perf_counter() - t0
        self.onnx_mb = os.path.getsize(os.path.join(d, v["onnx"])) / 1e6
        self.tok_mb = os.path.getsize(os.path.join(d, "tokenizer.json")) / 1e6
        self._run(["warmup"])
        self.rss_after_load = rss_mb() - r0

    def _run(self, texts):
        enc = self.t.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64); am = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {k: v for k, v in {"input_ids": ids, "attention_mask": am,
                                  "token_type_ids": np.zeros_like(ids)}.items() if k in self.inp}
        h = self.s.run(None, feed)[0]
        if self.v["pool"] == "cls":
            e = h[:, 0]
        else:
            m = am[..., None].astype(np.float32); e = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None)

    def enc(self, texts, kind, bs=None):
        bs = bs or int(os.getenv("BS", "32"))
        pre = self.v["q"] if kind == "q" else self.v["d"]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            out[idx] = self._run([pre + texts[j] for j in idx])
        return out


def ranks(qv, dv, rel):
    """MRR@10, Recall@k, nDCG@10 по множествам релевантных."""
    sims = qv @ dv.T
    top = np.argsort(-sims, axis=1)[:, :100]
    mrr = r1 = r3 = r5 = r10 = nd = 0.0
    for i, rs in enumerate(rel):
        row = list(top[i])
        pos = [row.index(r) for r in rs if r in row]
        first = min(pos) if pos else 999
        mrr += 1 / (first + 1) if first < 10 else 0
        r1 += first < 1; r3 += first < 3; r5 += first < 5; r10 += first < 10
        dcg = sum(1 / np.log2(p + 2) for p in pos if p < 10)
        idcg = sum(1 / np.log2(p + 2) for p in range(min(len(rs), 10)))
        nd += dcg / idcg
    n = len(rel)
    return dict(mrr10=mrr / n, r1=r1 / n, r3=r3 / n, r5=r5 / n, r10=r10 / n, ndcg10=nd / n, n=n), top


def rr_list(top, rel):
    """Обратный ранг первого релевантного (0 за пределами топ-10) — по запросу, для парного бутстрепа."""
    out = []
    for i, rs in enumerate(rel):
        row = list(top[i][:10]); pos = [row.index(r) for r in rs if r in row]
        out.append(round(1 / (min(pos) + 1), 4) if pos else 0.0)
    return out


# ---------- 1. Доменный набор memory-proxy ----------
def search_words(text):          # = proxy._search_words
    return list({w for w in re.findall(r'\w+', text[:2000].lower()) if len(w) > 3})[:32]


def memory_eval(m):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import memory_set
    facts, qs = memory_set.build()
    dv = m.enc(facts, "d"); qv = m.enc([q for _, q, _, _ in qs], "q")
    res, top = ranks(qv, dv, [r for *_, r, _ in qs])
    by = {}
    for lang in sorted({l for l, *_ in qs}):
        ii = [i for i, (l, *_) in enumerate(qs) if l == lang]
        by[lang], _ = ranks(qv[ii], dv, [qs[i][2] for i in ii])
    # гибрид как в proxy.get_smart_facts: FTS5 (bm25, слова > 3 букв) + вектор, RRF k=61, топ-15 → в дельту
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE f USING fts5(predicate, object)")
    con.executemany("INSERT INTO f(rowid, predicate, object) VALUES (?,?,?)",
                    [(i, *x.split(": ", 1)) for i, x in enumerate(facts)])
    hyb = {"r1": 0, "r3": 0, "r5": 0, "mrr10": 0.0}; fts_only = {"r5": 0}
    misses = []
    for qi, (lang, q, rel, tid) in enumerate(qs):
        score = {}
        w = search_words(q)
        fts = []
        if w:
            try: fts = [r[0] for r in con.execute("SELECT rowid FROM f WHERE f MATCH ? ORDER BY bm25(f) LIMIT 20", [" OR ".join(f'"{x}"' for x in w)])]
            except Exception: fts = []
        for i, r in enumerate(fts): score[r] = score.get(r, 0) + 1 / (61 + i)
        for i, r in enumerate(top[qi][:20]): score[int(r)] = score.get(int(r), 0) + 1 / (61 + i)
        ranked = sorted(score, key=lambda f: (-score[f], f))[:15]
        pos = [ranked.index(r) for r in rel if r in ranked]; first = min(pos) if pos else 999
        hyb["r1"] += first < 1; hyb["r3"] += first < 3; hyb["r5"] += first < 5; hyb["mrr10"] += 1 / (first + 1) if first < 10 else 0
        fts_only["r5"] += any(r in fts[:5] for r in rel)
        vpos = [list(top[qi]).index(r) for r in rel]
        if min(vpos) >= 5: misses.append(dict(lang=lang, q=q, want=facts[tid - 1], rank=int(min(vpos)) + 1, got=facts[int(top[qi][0])]))
    n = len(qs)
    return dict(vector=res, by_lang=by, hybrid={k: v / n for k, v in hyb.items()}, fts_only_r5=fts_only["r5"] / n,
                rr=rr_list(top, [r for *_, r, _ in qs]), langs=[l for l, *_ in qs],
                vector_misses_top5=misses)


# ---------- 2. Публичные наборы retrieval (подвыборка корпуса) ----------
def load_retrieval(prefix, nq, ndist, seed=0):
    rnd = random.Random(seed)
    c = pq.read_table(os.path.join(DATA, f"{prefix}corpus.parquet")).to_pydict()
    qr = pq.read_table(os.path.join(DATA, f"{prefix}queries.parquet")).to_pydict()
    rl = pq.read_table(os.path.join(DATA, f"{prefix}qrels.parquet")).to_pydict()
    rels = {}
    for q, d, s in zip(rl["query-id"], rl["corpus-id"], rl["score"]):
        if int(s) > 0: rels.setdefault(str(q), set()).add(str(d))
    qtext = {str(i): t for i, t in zip(qr["_id"], qr["text"]) if str(i) in rels}
    qids = sorted(qtext); rnd.shuffle(qids); qids = qids[:nq]
    need = set().union(*[rels[q] for q in qids])
    doc = {str(i): ((ti + "\n") if ti else "") + tx for i, ti, tx in zip(c["_id"], c["title"], c["text"])}
    pool = [d for d in doc if d not in need]; rnd.shuffle(pool)
    dids = sorted(need & doc.keys()) + pool[:ndist]
    di = {d: n for n, d in enumerate(dids)}
    return [qtext[q] for q in qids], [doc[d] for d in dids], [{di[d] for d in rels[q] if d in di} for q in qids]


def retrieval_eval(m, prefix, nq, ndist):
    q, d, rel = load_retrieval(prefix, nq, ndist)
    t0 = time.perf_counter(); dv = m.enc(d, "d", bs=int(os.getenv("DOC_BS", "8"))); td = time.perf_counter() - t0
    qv = m.enc(q, "q")
    r, top = ranks(qv, dv, rel)
    r.update(docs=len(d), docs_per_s=len(d) / td, rr=rr_list(top, rel))
    return r


# ---------- 3. Кросс-языковое сопоставление (Tatoeba, X → en) ----------
def tatoeba_eval(m, pair):
    t = pq.read_table(os.path.join(DATA, f"tat_{pair}.parquet")).to_pydict()
    a = [s.strip() for s in t["sentence1"]]; b = [s.strip() for s in t["sentence2"]]
    va = m.enc(a, "q"); vb = m.enc(b, "d")
    acc = float((np.argmax(va @ vb.T, axis=1) == np.arange(len(a))).mean())
    return dict(acc=acc, n=len(a))


# ---------- 4. Скорость запроса (как aembed_query: один текст за вызов) ----------
def latency(m):
    base = ("Вчера мы с командой обсуждали миграцию базы данных и решили перенести её на следующую неделю, "
            "а ещё я обещал сестре помочь с переездом в Новосибирск. ")
    out = {}
    for n in (40, 300, 2000):
        txt = (base * (n // len(base) + 1))[:n]
        for _ in range(3): m.enc([txt], "q")
        xs = []
        for _ in range(15):
            t0 = time.perf_counter(); m.enc([txt], "q"); xs.append((time.perf_counter() - t0) * 1000)
        out[f"q{n}_ms"] = float(np.median(xs))
        out[f"q{n}_tokens"] = len(m.t.encode(m.v["q"] + txt).ids)
    return out


def tokens_per_1k(m):
    s = {"ru": "Мне нужно подготовить отчёт для руководителя, в котором будут цифры за третий квартал и прогноз на следующий год.",
         "en": "I need to prepare a report for my manager with the numbers for the third quarter and a forecast for next year.",
         "uk": "Мені потрібно підготувати звіт для керівника, в якому будуть цифри за третій квартал і прогноз на наступний рік.",
         "kk": "Маған басшыға үшінші тоқсанның көрсеткіштері мен келесі жылға болжамы бар есеп дайындау керек.",
         "de": "Ich muss einen Bericht für meinen Vorgesetzten mit den Zahlen für das dritte Quartal und einer Prognose vorbereiten.",
         "zh": "我需要为经理准备一份报告，其中包括第三季度的数据和明年的预测。"}
    return {k: round(len(m.t.encode(v, add_special_tokens=False).ids) / len(v) * 1000) for k, v in s.items()}


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "emb_results.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    PUB = os.getenv("PUB", "1") == "1"
    SKIP = set(filter(None, os.getenv("SKIP", "").split(",")))   # уже посчитанные разделы

    def save():
        json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    for name in os.getenv("MODELS", "user2,granite").split(","):
        m = Emb(name); r = res.setdefault(name, {})
        r["meta"] = dict(onnx_mb=round(m.onnx_mb, 1), tokenizer_mb=round(m.tok_mb, 1), load_s=round(m.load_s, 2),
                         rss_after_load_mb=round(m.rss_after_load), vocab=m.t.get_vocab_size(), pool=m.v["pool"], prefix=bool(m.v["q"]))
        r["tokens_per_1k_chars"] = tokens_per_1k(m)
        if "latency" not in SKIP: r["latency"] = latency(m); print(name, r["latency"], flush=True)
        r["memory"] = memory_eval(m); print(name, "memory", r["memory"]["vector"], r["memory"]["hybrid"], flush=True)
        r["meta"]["peak_rss_mb"] = round(rss_mb()); save()
        if PUB:
            if "tatoeba" not in SKIP:
                r["tatoeba"] = {p: tatoeba_eval(m, p) for p in ("rus-eng", "ukr-eng", "bel-eng", "kaz-eng", "uzb-eng", "tur-eng", "deu-eng", "fra-eng", "spa-eng", "cmn-eng")}
                print(name, "tatoeba", {k: round(v["acc"], 3) for k, v in r["tatoeba"].items()}, flush=True); save()
            for key, pre, nd in (("rubq", "rubq_", 4000), ("wiki_bg", "wiki_bg-", 2500), ("wiki_de", "wiki_de-", 2500)):
                if key in SKIP: continue
                r[key] = retrieval_eval(m, pre, 300, nd); print(name, key, {k: v for k, v in r[key].items() if k != "rr"}, flush=True)
                r["meta"]["peak_rss_mb"] = round(rss_mb()); save()
        del m

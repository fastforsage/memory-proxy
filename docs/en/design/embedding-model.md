# Embedding model: USER2-small vs granite-embedding-97m-multilingual-r2

[Русский](../../design/embedding-model.md) | **English**

> English translation of [`../../design/embedding-model.md`](../../design/embedding-model.md). If the two differ, the Russian original is authoritative.

Status: **study, 27.11.2026.** The proxy code was not changed. Scripts and raw results: [`../../bench/embeddings/`](../bench/embeddings/README.md).

**Hypothesis (FR "multilingual semantic search"):** for the memory-proxy architecture, `onnx-community/granite-embedding-97m-multilingual-r2-ONNX` is the ideal multilingual replacement for the default model `onnx-community/USER2-small-ONNX`.

---

## 0. Summary

**The hypothesis is not confirmed.** granite is clearly better in Ukrainian, Kazakh, German, Turkish and other languages, but in **Russian, the proxy's main language, it is statistically significantly worse**; it is slower on the hot path, takes three times the memory, and without code changes it does worse than USER2 in all the main languages. As an **option for users who write in languages other than Russian and English** it is a good candidate. As the default replacement, it is not.

| "Ideal replacement" criterion | Result | |
|---|---|---|
| Russian no worse | domain set, Russian queries: MRR@10 **−0.161** [−0.264; −0.058]; Tatoeba rus→eng 0.815 vs 0.938; RuBQ at parity (−0.008, not significant) | ❌ |
| Other languages better | uk +0.230, de/kk/tr/es +0.414 (significant); Wikipedia bg +0.050, de +0.135; Tatoeba de/fr/es/zh 0.88–0.95 vs 0.02–0.22 | ✅ |
| English no worse | domain set +0.003 (not significant) | ✅ |
| Hot-path latency does not grow | embedding a 300-char question: 19.6 → 31.5 ms (×1.6); 2000 chars: 159 → 292 ms | ❌ |
| Home-server resources | ONNX 138 → 390 MB, RSS after load 253 → 788 MB; on 2 GB of RAM, indexing with batch 32 was OOM-killed | ❌ |
| Works without code changes | dropped in as is (mean pooling + USER2 prefixes): MRR@10 0.511 vs 0.605 (−0.094, significant) | ❌ |
| Fits the schema | dimension 384 = `vec0 float[384]` | ✅ |

Recommendation (§7): keep USER2-small as the default; add granite as an **optional preset** `EMBED_MODEL` with the correct pooling, no prefixes and automatic reindexing when the model changes. A side finding: **USER2-small int8** is no worse than fp32 in Russian, four times smaller and 1.4–2× faster, which makes it a candidate for the default model in its own right (§5.5).

---

## 1. Where the proxy uses the model

| Place | What it does | What matters for a replacement |
|---|---|---|
| `load_tokenizer` (`proxy.py` ~L248) | `tokenizer.json` via `tokenizers`, truncation at 512 tokens | length in tokens depends on the tokenizer |
| `_embed` (~L257) | ONNX → the `sentence_embedding` output if present, otherwise **mean pooling** → L2 | for both models the ONNX export returns only `last_hidden_state`, so it is always mean pooling |
| `embed_query` / `embed_document` (~L274) | **hardcoded** USER2 prefixes `search_query: ` / `search_document: ` | granite was trained without prefixes |
| `vec_facts` (L309) | `vec0(user_id partition key, embedding float[384])` | both models are 384-dimensional |
| `backfill_vec` (~L435) | computes vectors only for facts **without** one | there is no model marker: after a model change the old vectors stay and are compared with new queries |
| hybrid search (~L1185) | FTS5 bm25 + vector, RRF 1/(61+i), top 15; the query is `user_text[:2000]` | the question embedding is computed on the hot path, before the LLM request |
| `download_model.py` | the USER2 repo and revision `4a3caf0c…` are hardcoded; `MODEL_REVISION` from env | the repo is not configurable |

In v5 vector search runs only for a **new** question (agent steps reuse the pinned delta), but for a new question the embedding sits directly in the TTFT.

## 2. Candidates

| | USER2-small (default) | granite-embedding-97m-multilingual-r2 |
|---|---|---|
| Base | RuModernBERT-small, 34M parameters | ModernBERT, 12 layers, ~97M |
| Dimension | 384 (Matryoshka 32–384) | 384 (no Matryoshka) |
| Context | 8192 | 32768 |
| Vocabulary | 50,368 | 180,000 |
| Pooling / prefixes | mean / `search_query: `, `search_document: ` | **CLS** / no prefixes |
| Languages | Russian and English | 200+, 52 of them "enhanced" (including ru, uk, kk) |
| Published metrics | MTEB-rus mean 58.32, retrieval 61.87 | MTEB multilingual retrieval 60.3, English retrieval 50.1; no MTEB-rus figure published |
| ONNX (onnx-community) | `model.onnx` 137.8 MB, int8 34.8 MB | `model.onnx` 390 MB, int8 97.9 MB; `tokenizer.json` 25.3 MB |

## 3. What "ideal replacement" means for this architecture

The criteria follow from how the proxy works and from earlier decisions:
1. **Russian no worse.** The proxy is Russian-first: `MEMORY_POLICY`, the fact extraction prompt, the Context Bridge triggers and style scoring are in Russian; the main user writes in Russian.
2. **Other languages better.** That is the goal of the FR.
3. **Answer latency does not grow**: a requirement of the v5 architecture. The embedding of a new question sits in the TTFT.
4. **A home server without a GPU:** the model runs on the CPU; image size and RAM within reason (test machine: 2 vCPU, 2 GB RAM).
5. **Minimal changes and migrations:** the same dimension, a predictable transition for existing databases.

## 4. Method

The harness [`eval.py`](../../bench/embeddings/eval.py) reproduces the conditions of `proxy.py`: onnxruntime 1.18.1 on the CPU, `intra_op_num_threads=2`, `tokenizers`, truncation at 512, L2 normalisation, cosine.

**Variants:** `user2` (default: mean + prefixes), `user2_int8`, `granite` (CLS, no prefixes, as in the model card), `granite_int8`, `granite_asis` (granite dropped into the current code without changes: mean + USER2 prefixes).

**Sets:**
1. **Domain** ([`memory_set.py`](../../bench/embeddings/memory_set.py)): what the proxy actually does, finding a fact about the user from their message. 30 facts covering all 14 predicates in the `vec_facts` format (`predicate: object`, values in Russian, as the proxy extracts them from a Russian conversation): 33 target records (the facts "which languages I know" and "what annoys me at work" have several correct records, any of them counts) and 42 distractors of the same kind, 75 records in total. 5 queries per fact: `ru`, `ru2` (indirect: «Куда сходить вечером в моём городе?», "Where to go tonight in my city?" → `lives_in`), `en`, `uk` and one of `de` (14), `kk` (5), `tr` (4), `es` (1), `mix` (6, Russian mixed with English). 150 queries. Metrics: MRR@10, Recall@k, nDCG@10; the FTS5 + vector + RRF hybrid as in `get_smart_facts`.
2. **Public (MTEB):** RuBQ (Russian retrieval; 300 queries, the corpus is the relevant documents + 4000 random ones), Wikipedia Retrieval Multilingual bg and de (monolingual retrieval; 300 queries, relevant + 2500; this set has no Russian), Tatoeba X→eng (cross-lingual translation search; 1000 pairs, kaz 575, uzb 428).
3. **Speed and resources:** median of 15 calls embedding a single question (like `aembed_query`) of 40, 300 and 2000 characters; tokens per 1000 characters by language; file sizes; RSS after load and peak; document indexing speed (batch 8).

Significance: a paired bootstrap of the per-query MRR@10 difference (10,000 resamples, 95% interval, [`bootstrap.py`](../../bench/embeddings/bootstrap.py)). "Significant" below means the interval does not contain 0.

## 5. Results

### 5.1 Domain set (150 queries, vector search)

| Variant | MRR@10 | R@1 | R@3 | R@5 | R@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| `user2` (default) | 0.605 | 0.527 | 0.660 | 0.733 | 0.780 | 0.644 |
| `user2_int8` | 0.612 | 0.533 | 0.687 | 0.713 | 0.767 | 0.646 |
| `granite` | **0.652** | **0.547** | **0.693** | **0.793** | **0.900** | **0.708** |
| `granite_int8` | 0.597 | 0.493 | 0.647 | 0.747 | 0.853 | 0.653 |
| `granite_asis` (no code changes) | 0.511 | 0.393 | 0.567 | 0.660 | 0.767 | 0.573 |

Across all queries granite leads by +0.048 MRR@10, but the interval [−0.025; +0.121] contains 0: **on average over the set the difference is not significant**. The picture is made of opposite shifts by language:

| MRR@10 by query language | n | `user2` | `granite` | `granite_int8` | `granite_asis` | granite − user2, 95% CI |
|---|---|---|---|---|---|---|
| `ru` | 30 | **0.759** | 0.666 | 0.638 | 0.601 | ru + ru2: **−0.161** [−0.264; −0.058] |
| `ru2` (indirect) | 30 | **0.844** | 0.616 | 0.518 | 0.432 | |
| `en` | 30 | 0.767 | 0.771 | 0.682 | 0.644 | +0.003 [−0.076; +0.085] |
| `uk` | 30 | 0.371 | **0.601** | 0.564 | 0.484 | **+0.230** [+0.035; +0.416] |
| `de` | 14 | 0.183 | **0.671** | 0.672 | 0.413 | de + kk + tr + es: **+0.414** [+0.248; +0.589] |
| `kk` | 5 | 0.400 | **0.629** | 0.500 | 0.342 | |
| `tr` | 4 | 0.059 | **0.550** | 0.354 | 0.319 | |
| `mix` | 6 | 0.441 | 0.417 | 0.514 | 0.333 | −0.024 [−0.097; +0.056] |

granite loses most on **indirect Russian** queries, where meaning matters rather than word overlap. Typical granite misses (fact not in the top 5) that USER2 finds:
- «Какую зелень я не ем?» ("Which greens don't I eat?") → first `likes: зелёный чай` ("green tea") instead of `dislikes: кинза` ("coriander");
- «Какой режиссёр мне нравится?» ("Which film director do I like?") → `dislikes: кинза` instead of `likes: фильмы Тарковского` ("Tarkovsky films");
- «Посоветуй телескоп для начинающего» ("Recommend a telescope for a beginner") → `dislikes: ранние подъёмы` ("early mornings") instead of `interested_in: астрономия` ("astronomy");
- «Когда пора менять резину на зимнюю на моём кроссовере?» ("When should I put winter tyres on my crossover?") → `likes: осень` ("autumn") instead of `owns: Toyota RAV4`.

The hybrid (FTS5 + vector, as in the proxy) barely differs from pure vector search (MRR 0.606 vs 0.605 for USER2, 0.652 for granite): FTS finds the relevant fact in the top 5 for only 2.7% of queries, because the words of the question and the fact almost never match, either across languages or across word forms. Multilinguality rests entirely on the embedding model.

### 5.2 Public retrieval sets

| MRR@10 / nDCG@10 | `user2` | `user2_int8` | `granite` | `granite_int8` | granite − user2 (MRR), 95% CI |
|---|---|---|---|---|---|
| RuBQ (ru) | **0.864** / 0.859 | 0.854 / 0.851 | 0.856 / 0.848 | 0.799 / 0.777 | −0.008 [−0.036; +0.019] |
| Wikipedia bg | 0.838 / 0.857 | 0.817 / 0.841 | **0.888** / 0.907 | 0.852 / 0.872 | **+0.050** [+0.018; +0.082] |
| Wikipedia de | 0.795 / 0.817 | 0.762 / 0.784 | **0.930** / 0.945 | 0.933 / 0.947 | **+0.135** [+0.098; +0.174] |

On Russian factoid questions (RuBQ) the models are at parity. On monolingual search in other languages granite is clearly stronger. USER2 holds up reasonably in Bulgarian and German when the query and the document are in the same language; USER2's weakness is specifically **cross-lingual** search.

### 5.3 Cross-lingual translation search (Tatoeba, X → eng, accuracy)

| | rus | ukr | bel | kaz | uzb | tur | deu | fra | spa | cmn |
|---|---|---|---|---|---|---|---|---|---|---|
| `user2` | **0.938** | 0.435 | 0.197 | 0.082 | 0.070 | 0.062 | 0.151 | 0.220 | 0.218 | 0.017 |
| `user2_int8` | 0.929 | 0.396 | 0.173 | 0.077 | 0.082 | 0.059 | 0.136 | 0.210 | 0.207 | 0.017 |
| `granite` | 0.815 | **0.691** | **0.416** | **0.230** | **0.136** | **0.649** | **0.949** | **0.879** | **0.894** | **0.889** |
| `granite_int8` | 0.758 | 0.625 | 0.341 | 0.179 | 0.112 | 0.560 | 0.924 | 0.828 | 0.870 | 0.857 |

USER2 is effectively a Russian and English model only. granite covers European languages and Chinese, but it is behind in Russian and stays weak in Kazakh and Uzbek despite the claimed support.

### 5.4 Speed, memory, size (2 vCPU, onnxruntime CPU)

| | `user2` | `user2_int8` | `granite` | `granite_int8` |
|---|---|---|---|---|
| Question embedding, 40 chars | 7.6 ms | 3.7 ms | 9.2 ms | 5.6 ms |
| 300 chars | 19.6 ms | 12.1 ms | 31.5 ms | 18.4 ms |
| 2000 chars (`user_text[:2000]`) | 158.7 ms (458 tokens) | 116.6 ms | 291.6 ms (512 tokens, truncated) | 196.4 ms |
| Indexing, docs/s (RuBQ / bg / de) | 24 / 22 / 19 | 34 / 31 / 29 | 13 / 15 / 20 | 21 / 24 / 34 |
| ONNX + `tokenizer.json` | 138 + 5 MB | 35 + 5 MB | 390 + 25 MB | 98 + 25 MB |
| RSS after load | 253 MB | 108 MB | 788 MB | 415 MB |
| Peak RSS (indexing long documents) | 612 MB | 461 MB | 1116 MB | 805 MB |
| Model load | 0.7 s | 0.4 s | 1.9 s | 1.4 s |

Tokens per 1000 characters (fewer means more text fits into 512 tokens and computation is cheaper):

| | ru | en | uk | kk | de | zh |
|---|---|---|---|---|---|---|
| USER2 | **195** | 209 | 429 | 681 | 440 | 3000 |
| granite | 230 | 209 | **312** | **372** | **207** | **581** |

granite's tokenizer splits Russian 18% more finely, but it is more economical in Ukrainian (1.4×), Kazakh (1.8×), German (2.1×) and Chinese (5.2×). That is why granite indexes the German corpus as fast as USER2 even though the model itself is three times larger. In Russian it is almost twice as slow.

On a machine with 2 GB of RAM, indexing the public corpora with granite fp32 at batch 32 was killed by the OOM killer; with batch 8 it passed. The proxy indexes short facts (`predicate: object`) in batches of 64, so this is not a problem there, but the memory headroom on small servers shrinks by ~0.5 GB.

### 5.5 int8 quantisation

- **USER2 int8** does not differ from fp32 on the domain set (+0.007 [−0.016; +0.031]), on Russian queries (+0.002) or on RuBQ (−0.011, not significant); it loses slightly in Bulgarian and German (−0.02…−0.03, significant). Meanwhile the file is four times smaller (35 MB), RSS is 108 MB and the question embedding is 1.4–2× faster.
- **granite int8** loses noticeably: domain set 0.597 (below USER2 fp32), Russian −0.224 relative to USER2, RuBQ 0.799 (−0.066, significant). German is preserved.

In other words, the fast granite (int8) is no longer better than USER2 on average, and the slow one (fp32) does not fit the latency and memory budget.

### 5.6 Drop-in without code changes (`granite_asis`)

If granite is put into `MODEL_DIR` instead of USER2, the proxy will use mean pooling and add the USER2 prefixes. The result is worse than the default model: MRR@10 0.511 vs 0.605 (−0.094 [−0.163; −0.023]), −0.286 in Russian, −0.123 in English; a gain remains only for German/Kazakh/Turkish (+0.165). In addition, `backfill_vec` will not recompute the USER2 vectors already stored: granite queries will be compared with vectors of another model until `manage.py reindex` is run and the proxy restarted.

## 6. Assessing the hypothesis

The summary is in §0. For each "ideal replacement" criterion from §3:
1. **Russian no worse: no.** A significant loss on the domain set (especially on indirect queries) and on Tatoeba; parity only on the factoid RuBQ.
2. **Other languages better: yes,** by a wide margin in Ukrainian, German, Turkish, European languages and Chinese.
3. **Latency does not grow: no:** the question embedding is 1.2–1.8× slower, the gap growing with question length (+12 ms on a typical 300-character question, +133 ms on a long one), which contradicts the v5 requirement. int8 fits the latency but loses quality.
4. **Resources: worse:** three times the file size and RSS.
5. **No code changes: no:** CLS pooling, disabling the prefixes, a configurable repo and reindexing are needed.

granite is a strong **multilingual** model, but for the memory-proxy architecture, where the main language is Russian, it is not ideal. The right framing is not "replace" but "give a choice".

## 7. Recommendation and integration plan

**Decision (proposed):** USER2-small stays the default; granite becomes an optional preset for those who write in Ukrainian, German, other European languages or mix languages.

1. **Model preset** `EMBED_MODEL` = `user2` (default) | `granite-multilingual`. The preset sets the repo, the pinned revision, the ONNX file, the pooling (`mean` | `cls`), and the query and document prefixes. Manual configuration for other models: `EMBED_REPO`, `EMBED_POOLING`, `EMBED_QUERY_PREFIX`, `EMBED_DOC_PREFIX` (the dimension must be 384, checked at startup).
2. **`_embed`:** pooling from the configuration; a `sentence_embedding` output, if the ONNX has one, still takes priority. `embed_query` / `embed_document` take the prefixes from the configuration.
3. **Model marker in the database:** store the model identifier (repo + revision + pooling + prefixes). On a mismatch at startup: `DELETE FROM vec_facts` and `backfill_vec()` with a warning in the log (the same as `manage.py reindex`, but automatic). Facts are short; reindexing thousands of facts takes seconds to minutes in the background at startup.
4. **Image:** `download_model.py` reads the preset; `ARG EMBED_MODEL=user2` in the Dockerfile. A separate image tag for the multilingual build (e.g. `vX.Y.Z-multilingual`): the model and tokenizer add ≈ 270 MB to the image.
5. **Bound the cost of the question embedding:** `EMBED_QUERY_MAX_CH` (default 2000, as today); for granite 1000 is reasonable, an estimated ~120–140 ms instead of ~290 ms on a long question (not measured).
6. **Consider USER2 int8 as the default separately** (§5.5): the same quality in Russian, an image ~100 MB lighter, faster question embedding. Before switching, the model marker from item 3 is needed (fp32 and int8 vectors are close but not identical).
7. **Tests:** pooling and prefixes per preset (ONNX stub), a marker change → reindexing, a wrong dimension → startup error; the bench in [`../../bench/embeddings/`](../bench/embeddings/README.md) stays manual, outside CI.

## 8. The model is not the only obstacle to multilinguality

Even with a multilingual model, part of the proxy stays Russian-only:
- the Context Bridge and history block triggers (`CROSS_CHAT_TRIGGERS`, `proxy.py` L129–131; the history trigger ~L590, ~L1275) are Russian words only;
- the praise and critique words for style scoring (`PRAISE` L134, `CRITIQUE` L135–136);
- `MEMORY_POLICY` (L139+) and the fact extraction prompt (`build_extract_payload`) are written in Russian; models understand them in any language, but fact values arrive in the language of the conversation;
- FTS5 has no stemming for other languages; in the hybrid it adds almost nothing anyway (§5.1).

Full multilinguality needs separate work: move the triggers and scoring words into configurable lists with sets for several languages, and write the policy and the prompt in English (neutral for models).

## 9. Limitations of the study

- The domain set is small: 150 queries, from 1 to 14 for `de`/`kk`/`tr`/`es`/`mix`. Conclusions for these languages also rely on the public sets and Tatoeba.
- Fact values in the domain set are in Russian (that is how the proxy extracts them from a Russian conversation). The "whole conversation in another language" scenario is covered by the monolingual Wikipedia bg/de, not by the domain set.
- The public corpora were cut down (relevant + 2500–4000 random documents) to fit the time budget on 2 vCPU; absolute values are higher than on the full MTEB corpora, but the comparison between models is valid.
- Truncation at 512 tokens, as in the proxy; granite's long context (up to 32k) was not tested, since the proxy does not use it.
- Speed was measured on 2 vCPU; on another CPU the absolute milliseconds will differ, the ratios should be close.
- Other candidates were not tested. From published data: `deepvk/USER2-base` is stronger than USER2-small in Russian (MTEB-rus 61.12 vs 58.32) but is also Russian and English only; `intfloat/multilingual-e5-small` is behind granite on multilingual retrieval in the granite model card (50.9 vs 60.3).

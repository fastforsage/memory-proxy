# Embedding model bench

[Русский](../../../bench/embeddings/README.md) | **English**

> The scripts and data live in [`docs/bench/embeddings/`](../../../bench/embeddings/); paths below are relative to that directory.

Scripts and data for the [embedding model study](../../design/embedding-model.md): is `onnx-community/granite-embedding-97m-multilingual-r2-ONNX` a suitable replacement for the default `onnx-community/USER2-small-ONNX`? Research code: it does not run in CI and is not included in the image.

## Files

| File | Contents |
|---|---|
| [`memory_set.py`](../../../bench/embeddings/memory_set.py) | The domain set: 30 facts about the user covering all 14 proxy predicates in the `vec_facts` format (`predicate: object`; 33 target records, since the facts "which languages I know" and "what annoys me at work" have several correct records) + 42 distractors; 5 queries per fact: `ru`, `ru2` (indirect, without the fact's words), `en`, `uk` and one of `de` / `kk` / `tr` / `es` / `mix` (Russian mixed with English). 75 records, 150 queries in total |
| [`eval.py`](../../../bench/embeddings/eval.py) | The harness: the same conditions as `proxy.py` (onnxruntime CPU, 2 threads, `tokenizers`, truncation at 512 tokens, L2 normalisation, cosine); hybrid FTS5 + vector + RRF as in `get_smart_facts`; public MTEB sets; query latency; tokens per 1000 characters; RSS |
| [`emb_results.json`](../../../bench/embeddings/emb_results.json) | Run results (per model; `rr` is the reciprocal rank of every query, for the paired bootstrap) |
| [`bootstrap.py`](../../../bench/embeddings/bootstrap.py) | Paired bootstrap of the MRR@10 difference between two variants from `emb_results.json` |

Variants in `eval.py`:

| Variant | Model | Pooling | Prefixes |
|---|---|---|---|
| `user2` | USER2-small fp32 (default) | mean | `search_query: ` / `search_document: ` |
| `user2_int8` | USER2-small int8 | mean | same |
| `granite` | granite-97m-multilingual-r2 fp32 | CLS (as in the model card) | none |
| `granite_int8` | same, int8 | CLS | none |
| `granite_asis` | granite dropped into the current `proxy.py` without code changes | mean | USER2 prefixes |

## Data
- Models: `onnx-community/USER2-small-ONNX` (`onnx/model.onnx`, `onnx/model_int8.onnx`, `tokenizer.json`), `onnx-community/granite-embedding-97m-multilingual-r2-ONNX` (the same files).
- Public MTEB sets as parquet (via the HF datasets-server): `RuBQRetrieval` (300 queries; the corpus is the relevant documents + 4000 random ones), `WikipediaRetrievalMultilingual` bg and de (300 queries, relevant + 2500), `tatoeba-bitext-mining` X → eng (1000 pairs; kaz 575, uzb 428). `WikipediaRetrievalMultilingual` has no Russian, so Russian retrieval uses RuBQ.

Directory layout (`EMB_ROOT`, default `/var/tmp/emb2`):
```
user2/{onnx/model.onnx, onnx/model_int8.onnx, tokenizer.json}
granite/{onnx/model.onnx, onnx/model_int8.onnx, tokenizer.json}
data/{rubq_,wiki_bg-,wiki_de-}{corpus,queries,qrels}.parquet, data/tat_<lang>-eng.parquet
```

## Running
```bash
pip install onnxruntime==1.18.1 tokenizers numpy "pyarrow<16"
DOC_BS=8 MODELS=user2,granite python eval.py emb_results.json      # PUB=0: domain set only
PUB=0 MODELS=granite_asis python eval.py emb_results.json
python bootstrap.py emb_results.json user2 granite
```
`DOC_BS` is the batch size for indexing the public corpora. With a batch of 32, granite fp32 gets killed by the OOM killer on long documents (512 tokens) on a machine with 2 GB of RAM; with 8 it passes.

Results are merged into the JSON per model, so variants can be computed one at a time (`SKIP=tatoeba,rubq` skips sections already computed).

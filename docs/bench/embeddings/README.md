# Бенчмарк моделей эмбеддингов

**Русский** | [English](../../en/bench/embeddings/README.md)

Скрипты и данные к [исследованию модели эмбеддингов](../../design/embedding-model.md): подходит ли `onnx-community/granite-embedding-97m-multilingual-r2-ONNX` на замену штатной `onnx-community/USER2-small-ONNX`. Исследовательский код: в CI не запускается и в образ не попадает.

## Файлы

| Файл | Что внутри |
|---|---|
| `memory_set.py` | Доменный набор: 30 фактов о пользователе по всем 14 предикатам прокси в формате `vec_facts` (`predicate: object`; 33 целевые записи — у фактов «какие языки я знаю» и «что раздражает на работе» правильных записей несколько) + 42 дистрактора; на каждый факт 5 запросов — `ru`, `ru2` (косвенный, без слов факта), `en`, `uk` и один из `de` / `kk` / `tr` / `es` / `mix` (смесь русского и английского). Итого 75 записей, 150 запросов |
| `eval.py` | Харнесс: те же условия, что в `proxy.py` (onnxruntime CPU, 2 потока, `tokenizers`, обрезка 512 токенов, L2-нормировка, косинус); гибрид FTS5 + вектор + RRF как в `get_smart_facts`; публичные наборы MTEB; задержка запроса; токены на 1000 символов; RSS |
| `emb_results.json` | Результаты прогона (по моделям; `rr` — обратный ранг по каждому запросу для парного бутстрепа) |
| `bootstrap.py` | Парный бутстреп разницы MRR@10 между двумя вариантами по `emb_results.json` |

Варианты в `eval.py`:

| Вариант | Модель | Пулинг | Префиксы |
|---|---|---|---|
| `user2` | USER2-small fp32 (штатная) | mean | `search_query: ` / `search_document: ` |
| `user2_int8` | USER2-small int8 | mean | те же |
| `granite` | granite-97m-multilingual-r2 fp32 | CLS (как в карточке модели) | нет |
| `granite_int8` | то же, int8 | CLS | нет |
| `granite_asis` | granite, подложенная в текущий `proxy.py` без правок кода | mean | префиксы USER2 |

## Данные
- Модели: `onnx-community/USER2-small-ONNX` (`onnx/model.onnx`, `onnx/model_int8.onnx`, `tokenizer.json`), `onnx-community/granite-embedding-97m-multilingual-r2-ONNX` (те же файлы).
- Публичные наборы MTEB в parquet (через HF datasets-server): `RuBQRetrieval` (300 запросов, корпус — релевантные + 4000 случайных документов), `WikipediaRetrievalMultilingual` bg и de (300 запросов, релевантные + 2500), `tatoeba-bitext-mining` X → eng (1000 пар; kaz 575, uzb 428). В `WikipediaRetrievalMultilingual` нет русского, поэтому русский retrieval — RuBQ.

Раскладка каталога (`EMB_ROOT`, по умолчанию `/var/tmp/emb2`):
```
user2/{onnx/model.onnx, onnx/model_int8.onnx, tokenizer.json}
granite/{onnx/model.onnx, onnx/model_int8.onnx, tokenizer.json}
data/{rubq_,wiki_bg-,wiki_de-}{corpus,queries,qrels}.parquet, data/tat_<lang>-eng.parquet
```

## Запуск
```bash
pip install onnxruntime==1.18.1 tokenizers numpy "pyarrow<16"
DOC_BS=8 MODELS=user2,granite python eval.py emb_results.json      # PUB=0 — только доменный набор
PUB=0 MODELS=granite_asis python eval.py emb_results.json
python bootstrap.py emb_results.json user2 granite
```
`DOC_BS` — размер батча при индексации публичных корпусов. С батчем 32 granite fp32 на машине с 2 GB RAM убивается OOM-killer'ом на длинных документах (512 токенов); с 8 — проходит.

Результаты добавляются в JSON по моделям, поэтому варианты можно считать по очереди (`SKIP=tatoeba,rubq` — пропустить уже посчитанные разделы).

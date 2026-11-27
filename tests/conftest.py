"""Общие фикстуры: заглушки ONNX/токенизатора, mock LLM, TestClient.

По умолчанию onnxruntime и tokenizers подменяются лёгкими заглушками, поэтому тесты не требуют
скачивания модели. Чтобы прогнать с настоящей моделью: MP_REAL_MODEL=/path/to/model pytest
"""
import os, sys, re, json, time, types, hashlib, importlib
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
REAL_MODEL = os.getenv("MP_REAL_MODEL")


def _install_fakes():
    ort = types.ModuleType("onnxruntime")

    class _IO:
        def __init__(self, name): self.name = name

    class SessionOptions: pass

    class GraphOptimizationLevel: ORT_ENABLE_ALL = 99

    class InferenceSession:
        def __init__(self, *a, **k): pass
        def get_inputs(self): return [_IO("input_ids"), _IO("attention_mask")]
        def get_outputs(self): return [_IO("last_hidden_state")]
        def run(self, _, feed):
            ids = feed["input_ids"]
            out = np.zeros((ids.shape[0], ids.shape[1], 384), dtype=np.float32)
            for b, row in enumerate(ids):
                for t, tok in enumerate(row):
                    if tok: out[b, t, int(tok) % 384] = 1.0
            return [out]

    ort.SessionOptions, ort.GraphOptimizationLevel, ort.InferenceSession = SessionOptions, GraphOptimizationLevel, InferenceSession

    tk = types.ModuleType("tokenizers")

    class _Enc:
        def __init__(self, ids):
            self.ids, self.attention_mask, self.type_ids = ids, [1 if i else 0 for i in ids], [0] * len(ids)

    class Tokenizer:
        padding = {"pad_token": "[PAD]"}
        @staticmethod
        def from_file(path): return Tokenizer()
        def enable_truncation(self, **k): pass
        def enable_padding(self, **k): pass
        def token_to_id(self, t): return 0
        def get_vocab_size(self): return 10000
        def encode_batch(self, texts):
            rows = [[int(hashlib.md5(w.encode()).hexdigest(), 16) % 9973 + 1
                     for w in re.findall(r"\w+", t.lower())][:512] or [0] for t in texts]
            n = max(len(r) for r in rows)
            return [_Enc(r + [0] * (n - len(r))) for r in rows]

    tk.Tokenizer = Tokenizer
    sys.modules["onnxruntime"] = ort
    sys.modules["tokenizers"] = tk


def _cache_real_session():
    """Каждый тест перезагружает proxy; одна сессия на весь прогон, чтобы не держать N копий модели в RAM."""
    import onnxruntime as ort
    orig, cache = ort.InferenceSession, {}

    def cached(path, *a, **k):
        if path not in cache:
            cache[path] = orig(path, *a, **k)
        return cache[path]
    ort.InferenceSession = cached


if REAL_MODEL:
    _cache_real_session()
else:
    _install_fakes()

os.environ.setdefault("MODEL_DIR", "/nonexistent")
os.environ.setdefault("LOG_LEVEL", "WARNING")


class FakeLLM:
    """Mock OpenAI-совместимого сервера. cold_replies — очередь JSON-ответов для фонового анализа."""
    ANSWER = "Это развёрнутый ответ ассистента, достаточно длинный, чтобы сохраниться как пример стиля в базе."

    def __init__(self):
        self.requests, self.cold_replies = [], []

    def cold(self, facts=None, emotions=None, type_="personal", raw=None):
        self.cold_replies.append(raw if raw is not None else
                                 json.dumps({"type": type_, "facts": facts or [], "emotions": emotions or []}))

    @property
    def chat_requests(self):
        return [r for r in self.requests if not r.get("response_format")]

    @property
    def cold_requests(self):
        return [r for r in self.requests if r.get("response_format")]

    def handler(self, request):
        import httpx
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "test-model"}]})
        body = json.loads(request.content or b"{}")
        self.requests.append(body)
        if body.get("response_format", {}).get("type") == "json_object":
            content = self.cold_replies.pop(0) if self.cold_replies else '{"type":"personal","facts":[],"emotions":[]}'
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        if body.get("stream"):
            half = len(self.ANSWER) // 2
            chunks = [json.dumps({"choices": [{"delta": {"content": p}}]}) for p in (self.ANSWER[:half], self.ANSWER[half:])]
            sse = "".join(f"data: {c}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": self.ANSWER}}]})


class Ctx:
    def __init__(self, client, proxy, llm):
        self.client, self.proxy, self.llm = client, proxy, llm

    def chat(self, text, chat_id="chat-a", stream=False, history=None, headers=None, model="test-model", user=None, body_extra=None):
        h = {"X-OpenWebUI-Chat-Id": chat_id} if chat_id else {}
        if user:
            h["X-OpenWebUI-User-Id"] = user
        if self.proxy.API_KEY:
            h["Authorization"] = f"Bearer {self.proxy.API_KEY}"
        h.update(headers or {})
        msgs = list(history or []) + [{"role": "user", "content": text}]
        body = {"model": model, "stream": stream, "messages": msgs, **(body_extra or {})}
        return self.client.post("/v1/chat/completions", json=body, headers=h)

    def drain(self, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            if not self.proxy.app.state.tasks:
                self.proxy._DB.submit(lambda: None).result(timeout=timeout)   # дождаться записей в потоке БД
                return
            time.sleep(0.02)
        raise AssertionError("cold tasks did not finish")

    def db(self, sql, *params):
        self.proxy._DB.submit(lambda: None).result(timeout=5)
        return self.proxy._DB.submit(lambda: self.proxy._conn.execute(sql, params).fetchall()).result(timeout=5)

    def system_text(self, idx=-1):
        return " ".join(m["content"] for m in self.llm.chat_requests[idx]["messages"] if m["role"] == "system")

    def last_user_text(self, idx=-1):
        return [m for m in self.llm.chat_requests[idx]["messages"] if m["role"] == "user"][-1]["content"]

    def memory_text(self, idx=-1):
        """Содержимое блока <memory> в последнем user-сообщении, отправленном в LLM ('' если блока нет)."""
        t = self.last_user_text(idx)
        return t.split("<memory>", 1)[1].split("</memory>", 1)[0] if t.startswith("<memory>") else ""


@pytest.fixture
def make_ctx(tmp_path, monkeypatch):
    """Фабрика: make_ctx(API_KEY="x", GLOBAL_MEMORY="0", ...) → Ctx с запущенным приложением."""
    import httpx
    from fastapi.testclient import TestClient
    opened = []

    def _make(**env):
        model_dir = REAL_MODEL or str(tmp_path / "model")
        if not REAL_MODEL:
            os.makedirs(model_dir, exist_ok=True)
            open(os.path.join(model_dir, "model.onnx"), "wb").close()
        base = {"MODEL_DIR": model_dir, "DB_PATH": str(tmp_path / "memory.db"),
                "API_KEY": "", "CORS_ORIGINS": "", "GLOBAL_MEMORY": "1"}
        base.update({k: str(v) for k, v in env.items()})
        for k, v in base.items():
            monkeypatch.setenv(k, v)
        sys.modules.pop("proxy", None)
        proxy = importlib.import_module("proxy")
        llm = FakeLLM()
        client = TestClient(proxy.app, base_url="http://localhost")
        client.__enter__()
        opened.append(client)
        proxy.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(llm.handler), base_url="http://llm")
        return Ctx(client, proxy, llm)

    yield _make
    for c in opened:
        c.__exit__(None, None, None)
    sys.modules.pop("proxy", None)


@pytest.fixture
def ctx(make_ctx):
    return make_ctx()

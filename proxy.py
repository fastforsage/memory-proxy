import os
import json
import re
import sqlite3
import time
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8080/v1/chat/completions")
DB_PATH = os.getenv("DB_PATH", "/data/memory.db")
COMPRESS_AT = int(os.getenv("COMPRESS_AT", "22000"))
KEEP_RECENT = int(os.getenv("KEEP_RECENT", "8"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.execute("PRAGMA journal_mode=WAL")
_conn.executescript("""
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, chat_id TEXT, role TEXT, content TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, chat_id TEXT, predicate TEXT, object TEXT, evidence TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS compressions(id INTEGER PRIMARY KEY, chat_id TEXT, before_tokens INTEGER, after_tokens INTEGER, ts REAL);
CREATE INDEX IF NOT EXISTS ix_msg_chat ON messages(chat_id, ts);
""")
_conn.commit()

PATTERNS = [
    (re.compile(r"\bя работаю в ([A-ZА-Я][\w\- ]{1,40})", re.I), "works_at"),
    (re.compile(r"\bя живу в ([A-ZА-Я][\w\- ]{1,40})", re.I), "lives_in"),
    (re.compile(r"\bя люблю (.{2,40})", re.I), "likes"),
    (re.compile(r"\b(?:я )?не люблю (.{2,40})", re.I), "dislikes"),
]

def rough_tokens(text: str) -> int: return max(1, len(text) // 3) if text else 0
def total_tokens(msgs) -> int: return sum(rough_tokens(str(m.get("content") or "")) for m in msgs)

def extract_facts(text: str):
    out = []
    if any(h in text.lower() for h in ("может", "вроде", "если бы", "представь")): return out
    for rx, pred in PATTERNS:
        m = rx.search(text)
        if m:
            obj = m.group(1).strip()
            if obj.lower() in m.group(0).lower():
                out.append({"predicate": pred, "object": obj, "evidence": m.group(0)})
    return out

def compress(messages):
    system = [m for m in messages if m.get("role") == "system"]
    non_sys = [m for m in messages if m.get("role") != "system"]
    if len(non_sys) <= KEEP_RECENT + 2: return messages
    old, recent = non_sys[:-KEEP_RECENT], non_sys[-KEEP_RECENT:]
    frag = " | ".join(f"{m.get('role')}: {str(m.get('content',''))[:100]}" for m in old[-8:])
    return system + [{"role": "system", "content": f"[РАНЕЕ В РАЗГОВОРЕ] {frag}"}] + recent

@app.get("/health")
async def health(): return {"status": True}

@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(LLAMA_URL.replace("/chat/completions", "/models"))
        return Response(content=r.content, media_type="application/json")

@app.post("/v1/chat/completions")
async def chat(request: Request):
    try: body = await request.json()
    except: return Response(content='{"error":"invalid json"}', status_code=400)

    chat_id = request.headers.get("X-OpenWebUI-Chat-Id") or body.get("user") or "default"
    messages = body.get("messages", [])

    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user and isinstance(last_user.get("content"), str):
        _conn.execute("INSERT INTO messages(chat_id,role,content,ts) VALUES (?,?,?,?)", (chat_id, "user", last_user["content"], time.time()))
        for f in extract_facts(last_user["content"]):
            _conn.execute("INSERT INTO facts(chat_id,predicate,object,evidence,ts) VALUES (?,?,?,?,?)", (chat_id, f["predicate"], f["object"], f["evidence"], time.time()))
        _conn.commit()

    before = total_tokens(messages)
    if before > COMPRESS_AT:
        body["messages"] = compress(messages)
        after = total_tokens(body["messages"])
        _conn.execute("INSERT INTO compressions(chat_id,before_tokens,after_tokens,ts) VALUES (?,?,?,?)", (chat_id, before, after, time.time()))
        _conn.commit()
        print(f"[memory] compressed {before}->{after} tokens")

    stream = bool(body.get("stream", False))
    async def relay_stream():
        acc = []
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", LLAMA_URL, json=body) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
                    for line in chunk.decode("utf-8", "ignore").split("\n"):
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            try:
                                delta = json.loads(line[6:]).get("choices", [{}])[0].get("delta", {})
                                acc.append(delta.get("content") or "")
                            except: pass
        if "".join(acc):
            _conn.execute("INSERT INTO messages(chat_id,role,content,ts) VALUES (?,?,?,?)", (chat_id, "assistant", "".join(acc), time.time()))
            _conn.commit()

    if stream: return StreamingResponse(relay_stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=None) as c:
        r = await c.post(LLAMA_URL, json=body)
        try:
            content = r.json()["choices"][0]["message"]["content"]
            _conn.execute("INSERT INTO messages(chat_id,role,content,ts) VALUES (?,?,?,?)", (chat_id, "assistant", content, time.time()))
            _conn.commit()
        except: pass
        return Response(content=r.content, media_type="application/json")

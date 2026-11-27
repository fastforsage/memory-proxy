"""manage.py: stats / facts / export / reassign / purge / reindex на базе, созданной прокси."""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(db_path, *args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "manage.py"), *args], capture_output=True, text=True,
                       env={**os.environ, "DB_PATH": db_path}, timeout=60)
    return r.returncode, r.stdout + r.stderr


def seed(ctx):
    ctx.llm.cold(facts=[{"predicate": "lives_in", "object": "Казань", "evidence": "я живу в казани"}])
    ctx.chat("Я живу в Казани")                                  # пользователь default
    ctx.llm.cold(facts=[{"predicate": "likes", "object": "джаз", "evidence": "я люблю джаз"}])
    ctx.chat("Я люблю джаз", user="alice")
    ctx.drain()
    return ctx.proxy.DB_PATH


def test_stats_facts_export(ctx, tmp_path):
    db = seed(ctx)
    code, out = run(db, "stats")
    assert code == 0 and "default" in out and "alice" in out
    code, out = run(db, "facts", "default")
    assert code == 0 and "lives_in: Казань" in out and "джаз" not in out
    code, out = run(db, "export", "alice")
    data = json.loads(out)
    assert data["user_id"] == "alice" and data["facts"][0]["object"] == "джаз" and data["messages"]
    code, _ = run(db, "export", "alice", str(tmp_path / "a.json"))
    assert code == 0 and json.load(open(tmp_path / "a.json"))["facts"]


def test_reassign_and_reembed_on_restart(make_ctx):
    c = make_ctx()
    db = seed(c)
    c.client.__exit__(None, None, None)
    code, out = run(db, "reassign", "default", "alice")
    assert code == 0, out
    c2 = make_ctx()                                              # рестарт → backfill векторов
    assert sorted(c2.db("SELECT user_id, object FROM facts")) == [("alice", "Казань"), ("alice", "джаз")]
    assert sorted(c2.db("SELECT user_id FROM vec_facts")) == [("alice",), ("alice",)]
    c2.chat("Напомни, где я живу?", user="alice")
    assert "Казань" in c2.memory_text()


def test_purge_requires_confirmation(ctx):
    db = seed(ctx)
    code, out = run(db, "purge", "alice")
    assert code != 0 and "--yes" in out
    code, out = run(db, "purge", "alice", "--yes")
    assert code == 0
    assert ctx.db("SELECT DISTINCT user_id FROM facts") == [("default",)]
    assert ctx.db("SELECT COUNT(*) FROM messages WHERE user_id='alice'")[0][0] == 0


def test_reindex(make_ctx):
    c = make_ctx()
    db = seed(c)
    c.client.__exit__(None, None, None)
    code, out = run(db, "reindex")
    assert code == 0 and "Удалено векторов: 2" in out
    c2 = make_ctx()
    assert c2.db("SELECT COUNT(*) FROM vec_facts")[0][0] == 2


def test_refuses_old_schema(tmp_path):
    import sqlite3
    p = str(tmp_path / "old.db")
    sqlite3.connect(p).execute("PRAGMA user_version=2").connection.commit()
    code, out = run(p, "stats")
    assert code != 0 and "миграцию" in out

#!/usr/bin/env python3
"""Управление базой memory-proxy (v4.8+).

    python manage.py stats                     # пользователи и объём данных
    python manage.py facts <user>              # актуальные факты пользователя
    python manage.py export <user> [file]      # всё о пользователе в JSON (stdout или файл)
    python manage.py reassign <from> <to>      # перенести данные (например, 'default' → id из Open WebUI)
    python manage.py purge <user> --yes        # удалить всё о пользователе
    python manage.py reindex                   # пересчитать все векторы при следующем старте

В контейнере:  docker exec memory-proxy python manage.py stats
Путь к базе — DB_PATH (по умолчанию /data/memory.db). Можно запускать при работающем прокси.
После reassign / reindex перезапустите прокси: векторы пересчитываются при старте.
"""
import json
import os
import sqlite3
import sys
import time

DB_PATH = os.getenv("DB_PATH", "/data/memory.db")
USER_TABLES = ("messages", "facts", "emotions", "style_examples")


def connect():
    if not os.path.exists(DB_PATH):
        sys.exit(f"❌ База не найдена: {DB_PATH}")
    c = sqlite3.connect(DB_PATH)
    try:
        import sqlite_vec
        c.enable_load_extension(True)
        sqlite_vec.load(c)
        c.enable_load_extension(False)
    except Exception as e:  # без sqlite_vec можно смотреть, но не трогать векторы
        print(f"⚠️  sqlite_vec недоступен ({e}); операции с векторами пропускаются", file=sys.stderr)
    c.execute("PRAGMA busy_timeout=5000")
    ver = c.execute("PRAGMA user_version").fetchone()[0]
    if ver < 3:
        sys.exit(f"❌ Схема v{ver}: запустите прокси v4.8+ один раз, чтобы выполнить миграцию.")
    return c


def _has_vec(c):
    try:
        c.execute("SELECT COUNT(*) FROM vec_facts").fetchone()
        return True
    except sqlite3.OperationalError:
        return False


def _day(ts):
    return time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "?"


def cmd_stats(c):
    users = sorted({r[0] for t in USER_TABLES for r in c.execute(f"SELECT DISTINCT user_id FROM {t}")})
    size = sum(os.path.getsize(p) for p in (DB_PATH, DB_PATH + "-wal") if os.path.exists(p))
    print(f"DB: {DB_PATH} ({size / 1024 / 1024:.1f} MB), schema v{c.execute('PRAGMA user_version').fetchone()[0]}")
    if not users:
        print("Пусто."); return
    print(f"{'user_id':<40} {'чатов':>6} {'сообщ.':>7} {'факты':>6} {'история':>8} {'эмоции':>7} {'стиль':>6}  последнее")
    for u in users:
        chats, msgs, last = c.execute("SELECT COUNT(DISTINCT chat_id), COUNT(*), MAX(ts) FROM messages WHERE user_id=?", (u,)).fetchone()
        live = c.execute("SELECT COUNT(*) FROM facts WHERE user_id=? AND valid_to IS NULL", (u,)).fetchone()[0]
        hist = c.execute("SELECT COUNT(*) FROM facts WHERE user_id=? AND valid_to IS NOT NULL", (u,)).fetchone()[0]
        emo = c.execute("SELECT COUNT(*) FROM emotions WHERE user_id=?", (u,)).fetchone()[0]
        sty = c.execute("SELECT COUNT(*) FROM style_examples WHERE user_id=?", (u,)).fetchone()[0]
        print(f"{u[:40]:<40} {chats:>6} {msgs:>7} {live:>6} {hist:>8} {emo:>7} {sty:>6}  {_day(last)}")
    if _has_vec(c):
        missing = c.execute("SELECT COUNT(*) FROM facts f LEFT JOIN vec_facts v ON v.rowid=f.id "
                            "WHERE f.valid_to IS NULL AND v.rowid IS NULL").fetchone()[0]
        if missing:
            print(f"ℹ️  {missing} фактов без вектора — пересчитаются при следующем старте прокси")


def cmd_facts(c, user):
    rows = c.execute("SELECT id, scope, predicate, object, evidence, valid_from FROM facts "
                     "WHERE user_id=? AND valid_to IS NULL ORDER BY predicate, id", (user,)).fetchall()
    if not rows:
        print("Фактов нет."); return
    for fid, scope, p, o, ev, vf in rows:
        where = f" [чат {scope[:8]}]" if scope else ""
        print(f"#{fid} {p}: {o} (с {_day(vf)}){where}\n     «{ev}»")


def cmd_export(c, user, out=None):
    def rows(sql):
        cur = c.execute(sql, (user,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    data = {
        "user_id": user, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "facts": rows("SELECT id, scope, chat_id, predicate, object, evidence, valid_from, valid_to FROM facts WHERE user_id=? ORDER BY id"),
        "emotions": rows("SELECT scope, trigger, score, mention_count, last_seen, last_quote FROM emotions WHERE user_id=?"),
        "style_examples": rows("SELECT id, context, score, text, ts FROM style_examples WHERE user_id=? ORDER BY id"),
        "messages": rows("SELECT id, chat_id, role, content, ts FROM messages WHERE user_id=? ORDER BY id"),
    }
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ {out}: фактов {len(data['facts'])}, сообщений {len(data['messages'])}", file=sys.stderr)
    else:
        print(text)


def cmd_reassign(c, src, dst):
    if src == dst:
        sys.exit("❌ Одинаковые user_id")
    vec = _has_vec(c)
    with c:
        moved_ids = [r[0] for r in c.execute("SELECT id FROM facts WHERE user_id=?", (src,))]
        n = {}
        for t in USER_TABLES:
            # Конфликты уникальности (у получателя уже есть такой факт/эмоция) — оставляем запись получателя
            n[t] = c.execute(f"UPDATE OR IGNORE {t} SET user_id=? WHERE user_id=?", (dst, src)).rowcount
            c.execute(f"DELETE FROM {t} WHERE user_id=?", (src,))
        if vec and moved_ids:
            ph = ",".join("?" * len(moved_ids))
            c.execute(f"DELETE FROM vec_facts WHERE rowid IN ({ph})", moved_ids)   # partition key → пересчёт при старте
    print(f"✅ {src} → {dst}: " + ", ".join(f"{t} {k}" for t, k in n.items()))
    print("Перезапустите прокси, чтобы пересчитать векторы перенесённых фактов.")


def cmd_purge(c, user, confirm):
    if confirm != "--yes":
        sys.exit(f"Удалит ВСЕ данные пользователя '{user}'. Повторите с --yes")
    with c:
        if _has_vec(c):
            c.execute("DELETE FROM vec_facts WHERE rowid IN (SELECT id FROM facts WHERE user_id=?)", (user,))
        n = {t: c.execute(f"DELETE FROM {t} WHERE user_id=?", (user,)).rowcount for t in USER_TABLES}
    print(f"🗑 {user}: " + ", ".join(f"{t} {k}" for t, k in n.items()))


def cmd_reindex(c):
    if not _has_vec(c):
        sys.exit("❌ Нужен sqlite_vec")
    with c:
        n = c.execute("DELETE FROM vec_facts").rowcount
    print(f"✅ Удалено векторов: {n}. Перезапустите прокси — они пересчитаются при старте.")


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help", "help"):
        print(__doc__); return 0
    cmd, args = argv[1], argv[2:]
    table = {"stats": (cmd_stats, 0, 0), "users": (cmd_stats, 0, 0), "facts": (cmd_facts, 1, 1),
             "export": (cmd_export, 1, 2), "reassign": (cmd_reassign, 2, 2), "purge": (cmd_purge, 1, 2),
             "reindex": (cmd_reindex, 0, 0)}
    if cmd not in table:
        print(__doc__); return 2
    fn, lo, hi = table[cmd]
    if not lo <= len(args) <= hi:
        print(__doc__); return 2
    if cmd == "purge" and len(args) == 1:
        args.append("")
    fn(connect(), *args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

import sqlite3, sqlite_vec
c = sqlite3.connect("/data/memory.db")
c.enable_load_extension(True); sqlite_vec.load(c); c.enable_load_extension(False)
row = c.execute("SELECT sql FROM sqlite_master WHERE name='vec_facts'").fetchone()
if not row:
    print("  ℹ️ vec_facts нет — создастся при старте"); raise SystemExit(0)
print("  схема:", row[0])
n = c.execute("SELECT COUNT(*) FROM vec_facts").fetchone()[0]
c.execute("DELETE FROM vec_facts"); c.commit()
print(f"  ✅ vec_facts: удалено {n} строк")

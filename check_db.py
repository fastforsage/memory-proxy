#!/usr/bin/env python3
"""Диагностика базы данных Memory Proxy v4."""
import os
import sqlite3
import sys
import time
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "/data/memory.db")

# ANSI цвета
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def fmt_ts(ts):
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def fmt_date(ts):
    if not ts:
        return "?"
    return datetime.fromtimestamp(ts).strftime("%Y-%m")


def color_score(score):
    if score > 0.5:
        return f"{GREEN}+{score:.2f}{RESET}"
    elif score < -0.5:
        return f"{RED}{score:.2f}{RESET}"
    return f"{GRAY}{score:+.2f}{RESET}"


def section(title):
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")


def check_table_exists(conn, table):
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def main():
    if not os.path.exists(DB_PATH):
        print(f"{RED}❌ База не найдена: {DB_PATH}{RESET}")
        print(f"{GRAY}Убедись, что контейнер был запущен хотя бы раз.{RESET}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print(f"{BOLD}🧠 Memory Proxy v4 — Diagnostics{RESET}")
    print(f"{GRAY}DB: {DB_PATH}{RESET}")
    print(f"{GRAY}Размер файла: {os.path.getsize(DB_PATH) / 1024:.1f} KB{RESET}")

    # ─── Статистика ───
    section("📊 Статистика")

    tables = ["messages", "facts", "emotions", "summaries", "mood_state"]
    for t in tables:
        if check_table_exists(conn, t):
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:<15} {count:>5} записей")
        else:
            print(f"  {t:<15} {RED}❌ таблица отсутствует{RESET}")

    # ─── Факты: активные ───
    section("✅ Активные факты (valid_to IS NULL)")
    if check_table_exists(conn, "facts"):
        rows = conn.execute(
            "SELECT predicate, object, evidence, valid_from, ts "
            "FROM facts WHERE valid_to IS NULL ORDER BY valid_from DESC"
        ).fetchall()
        if not rows:
            print(f"  {GRAY}(пусто){RESET}")
        for r in rows:
            since = fmt_date(r["valid_from"]) if r["valid_from"] else fmt_date(r["ts"])
            ev = (r["evidence"] or "")[:50].replace("\n", " ")
            print(f"  {BOLD}{r['predicate']}{RESET} = {r['object']}  {GRAY}(с {since}){RESET}")
            print(f"    {GRAY}↳ «{ev}»{RESET}")

    # ─── Факты: история ───
    section("📜 История фактов (прошлые, valid_to IS NOT NULL)")
    if check_table_exists(conn, "facts"):
        rows = conn.execute(
            "SELECT predicate, object, valid_from, valid_to "
            "FROM facts WHERE valid_to IS NOT NULL ORDER BY valid_to DESC LIMIT 20"
        ).fetchall()
        if not rows:
            print(f"  {GRAY}(нет истории){RESET}")
        for r in rows:
            vf = fmt_date(r["valid_from"]) if r["valid_from"] else "?"
            vt = fmt_date(r["valid_to"])
            print(f"  {GRAY}[история]{RESET} {r['predicate']} = {r['object']}  "
                  f"{GRAY}({vf} → {vt}){RESET}")

    # ─── Эмоции ───
    section("❤️ Эмоциональный профиль")
    if check_table_exists(conn, "emotions"):
        rows = conn.execute(
            "SELECT trigger, score, mention_count, last_seen, last_quote "
            "FROM emotions ORDER BY score ASC"
        ).fetchall()
        if not rows:
            print(f"  {GRAY}(пусто){RESET}")
        for r in rows:
            print(f"  {r['trigger']:<25} {color_score(r['score'])}  "
                  f"{GRAY}({r['mention_count']}×, "
                  f"последний раз: {fmt_ts(r['last_seen'])}){RESET}")
            if r["last_quote"]:
                print(f"    {GRAY}↳ «{r['last_quote'][:60]}»{RESET}")

    # ─── Настроение ───
    section("🎭 Текущее настроение")
    if check_table_exists(conn, "mood_state"):
        rows = conn.execute(
            "SELECT state, intensity, ts FROM mood_state ORDER BY ts DESC LIMIT 5"
        ).fetchall()
        if not rows:
            print(f"  {GRAY}(нет записей){RESET}")
        else:
            now = time.time()
            for i, r in enumerate(rows):
                age_min = (now - r["ts"]) / 60
                expired = " (истекло)" if age_min > 30 else ""
                marker = "🟢" if i == 0 and not expired else "⚪"
                print(f"  {marker} {r['state']:<12} intensity={r['intensity']:.2f}  "
                      f"{GRAY}{fmt_ts(r['ts'])} ({age_min:.0f} мин назад){expired}{RESET}")

    # ─── Саммари ───
    section("📝 Последние саммари (сжатия контекста)")
    if check_table_exists(conn, "summaries"):
        rows = conn.execute(
            "SELECT chat_id, text, ts FROM summaries ORDER BY ts DESC LIMIT 5"
        ).fetchall()
        if not rows:
            print(f"  {GRAY}(нет саммари){RESET}")
        for r in rows:
            print(f"  {GRAY}[{fmt_ts(r['ts'])}]{RESET} {r['text'][:150]}...")

    # ─── Последние сообщения ───
    section("💬 Последние сообщения")
    if check_table_exists(conn, "messages"):
        rows = conn.execute(
            "SELECT role, content, ts FROM messages ORDER BY ts DESC LIMIT 10"
        ).fetchall()
        if not rows:
            print(f"  {GRAY}(нет сообщений){RESET}")
        for r in reversed(rows):
            role_color = CYAN if r["role"] == "user" else GREEN
            content = (r["content"] or "")[:80].replace("\n", " ")
            print(f"  {role_color}[{r['role']}]{RESET} {content}")

    print(f"\n{BOLD}{GREEN}✅ Диагностика завершена.{RESET}\n")
    conn.close()


if __name__ == "__main__":
    main()

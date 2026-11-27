#!/usr/bin/env python3
"""Принудительная переиндексация векторов (необязательно).

С v4.7.4 миграция автоматическая: при старте прокси смотрит PRAGMA user_version и, если база
старая, сам удаляет векторы и пересчитывает их с префиксами USER2. Скрипт нужен, только если вы
хотите пересчитать всё ещё раз (например, после ручной правки фактов или смены модели).

Что делает: сбрасывает PRAGMA user_version в 0. При следующем старте прокси выполнит миграцию.
Запускать при ОСТАНОВЛЕННОМ прокси:
    docker compose stop memory-proxy
    docker compose run --rm --no-deps memory-proxy python cleanup_vec.py
    docker compose start memory-proxy
"""
import os
import sqlite3
import sys

DB_PATH = os.getenv("DB_PATH", "/data/memory.db")

if not os.path.exists(DB_PATH):
    print(f"❌ База не найдена: {DB_PATH}")
    sys.exit(1)

c = sqlite3.connect(DB_PATH)
ver = c.execute("PRAGMA user_version").fetchone()[0]
c.execute("PRAGMA user_version=0")
c.commit()
c.close()
print(f"✅ {DB_PATH}: user_version {ver} → 0. Векторы пересчитаются при следующем старте прокси.")

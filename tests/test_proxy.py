"""Юнит-тесты чистых функций."""
import pytest


def P():
    import proxy
    return proxy


def test_safe_float():
    sf = P().safe_float
    assert sf(0.5) == 0.5 and sf(2.0) == 1.0 and sf(-2.0) == -1.0
    assert sf("abc") == 0.0 and sf(None) == 0.0


def test_rough_tokens():
    rt = P().rough_tokens
    assert rt("") == 0 and rt("a" * 300) == 100 and rt(None) == 1


def test_total_tokens_vision():
    assert P().total_tokens([{"role": "user", "content": [{"type": "image_url"}]}]) < 100


PERSONAL = {
    "story_ru": "Вчера мы с женой ездили на дачу к её родителям, жарили шашлык, потом долго гуляли у реки и обсуждали, куда поехать летом. Я давно мечтаю о Байкале, а она хочет на море.",
    "emotional_ru": "Привет!!! Сегодня отличный день :) Сходили с Машей в кино — было супер! Завтра в 10:30 идём к врачу (наконец-то). Кто-то из друзей тоже придёт!",
    "todo_list_ru": "Планы на неделю:\n- купить продукты\n- позвонить маме\n- записаться к стоматологу\n- забрать посылку\n- полить цветы и погулять с собакой",
    "english": "I work for a small startup from Berlin and if everything goes well, for the next year I will be moving there while my wife stays here for a while.",
    "url_ru": "Посмотри, какую квартиру мы нашли: https://www.avito.ru/moskva/kvartiry/2-k._kvartira_54m_5-9et._123456789 — нравится? Я думаю брать.",
    "update_ru": "Update: мы всё-таки переехали в Казань, работаю теперь удалённо, а по выходным ездим к родителям жены за город.",
    "short": "hi",
}
TECHNICAL = {
    "short_block": "```python\nprint(1)\n```",
    "python_repeat": "def foo():\n    return 42\n" * 10,
    "python": "def fetch(url, retries=3):\n    for i in range(retries):\n        try:\n            r = httpx.get(url, timeout=5)\n            return r.json()\n        except Exception as e:\n            log.warning(e)\n",
    "sql_one_line": "SELECT u.id, u.name FROM users u JOIN orders o ON o.user_id = u.id WHERE o.total > 100 ORDER BY o.created_at DESC;",
    "json": '{"model": "llama", "messages": [{"role": "user", "content": "hi"}], "stream": true, "temperature": 0.7}',
    "js_short": "const x = await fetch('/api');\nif (x.ok) {\n  console.log(await x.json());\n}\n",
}


@pytest.mark.parametrize("name", PERSONAL)
def test_preclassify_personal(name):
    assert P().preclassify(PERSONAL[name]) == "personal"


@pytest.mark.parametrize("name", TECHNICAL)
def test_preclassify_technical(name):
    assert P().preclassify(TECHNICAL[name]) == "technical"


def test_evidence_ok():
    ok = P().evidence_ok
    assert ok("Я работаю в Google", "я работаю в Google", "Google") is True
    assert ok("Я живу в Москве", "я живу в москве", "Москва") is True                 # падеж
    assert ok("У меня есть кот по кличке Барсик", "есть кот по кличке барсик", "кот Барсик") is True
    assert ok("Я работаю в Google", "", "Google") is False                            # пустая цитата
    assert ok("Я работаю в Google", "12345", "Google") is False                       # короткая
    assert ok("погода", "я", "Google") is False
    assert ok("Я работаю в Яндексе, а в Google меня не взяли", "я работаю в яндексе", "Google") is False
    assert ok("Я работаю в Google", 12345, "Google") is False                         # не строка
    assert ok("Я работаю в Google", "работаю в амазоне", "Amazon") is False            # цитаты нет в тексте


@pytest.mark.parametrize("text,expected", [
    ("/chats", ("chats", None, None)),
    ("/help", ("help", None, None)),
    ("/pull 2", (None, "pull", "2")),
    ("/search jwt auth", (None, "search", "jwt auth")),
    ("/pull", (None, "pull", None)),
    ("/etc/nginx/nginx.conf что это?", None),
    ("/help me write a bash script", None),
    ("/chats are great", None),
    ("hello", None),
])
def test_command_regex(text, expected):
    m = P().COMMAND_RE.match(text)
    assert (m.groups() if m else None) == expected


@pytest.mark.parametrize("text,delta", [
    ("спасибо, отлично", 1), ("хорошо", 1), ("ошибка в коде", -2), ("бесполезно", -2),
    ("ужасно", -2), ("не хорошо", -2), ("спасибо, но не так", -2), ("продолжай", 0),
])
def test_classify_feedback(text, delta):
    assert P().classify_feedback(text) == delta


def test_is_service_request():
    s = P().is_service_request
    assert s("### Task: generate tags") is True and s("hello") is False


def test_keep_recent_validation():
    assert P().KEEP_RECENT >= 1


def test_compress_no_op_short():
    p = P()
    msgs = [{"role": "user", "content": "a"}] * (p.KEEP_RECENT + 1)
    assert p.compress(msgs) == msgs


def test_compress_keeps_system():
    p = P()
    msgs = [{"role": "system", "content": "sys"}] + [{"role": "user", "content": f"m{i}"} for i in range(p.KEEP_RECENT + 5)]
    result = p.compress(msgs)
    assert result[0]["role"] == "system" and len(result) == 2 + p.KEEP_RECENT


def test_compress_does_not_orphan_tool_messages():
    p = P()
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    msgs += [{"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
             {"role": "tool", "tool_call_id": "1", "content": "r1"},
             {"role": "tool", "tool_call_id": "1", "content": "r2"}]
    msgs += [{"role": "user", "content": f"n{i}"} for i in range(p.KEEP_RECENT - 2)]
    recent = [m for m in p.compress(msgs) if m["role"] != "system"]
    assert recent[0]["role"] != "tool"
    assert any(m.get("tool_calls") for m in recent)


def test_stems():
    p = P()
    assert p._all_stems_in("работа", "ненавижу работу")
    assert p._all_stems_in("Jira", "ненавижу jira")
    assert not p._all_stems_in("понедельники", "люблю пятницы")

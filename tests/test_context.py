"""v5: чистые функции сборки контекста (context.py)."""
import json, random
import context as cx


def conv(n_turns, tools_every=0, sys=True):
    msgs = [{"role": "system", "content": "Ты помощник."}] if sys else []
    for t in range(n_turns):
        msgs.append({"role": "user", "content": f"Вопрос {t}: " + "текст " * 20})
        if tools_every and t % tools_every == 0:
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{t}", "type": "function",
                         "function": {"name": "read", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{t}", "content": "вывод " * 400})
        msgs.append({"role": "assistant", "content": f"Ответ {t}: " + "слово " * 30})
    return msgs


def test_content_text_and_media():
    parts = [{"type": "text", "text": "что на фото?"}, {"type": "image_url", "image_url": {"url": "data:x"}}]
    assert cx.content_text(parts) == "что на фото?" and cx.has_media(parts)
    assert cx.content_text(None) == "" and not cx.has_media("строка")


def test_with_prefix_keeps_list_content():
    m = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}
    out = cx.with_prefix(m, "<memory>\nX\n</memory>\n\n")
    assert out["content"][0] == {"type": "text", "text": "<memory>\nX\n</memory>\n\n"}
    assert out["content"][1] == m["content"][0] and m["content"][0]["type"] == "image_url"     # оригинал не тронут


def test_strip_think():
    assert cx.strip_think("<think>рассуждаю</think>\n\nОтвет") == "Ответ"
    assert cx.strip_think("<think>не закончил") == ""
    assert cx.strip_think("обычный ответ ") == "обычный ответ "


def test_msg_hash_depends_on_position_and_content():
    a = {"role": "user", "content": "ок"}
    assert cx.msg_hash(a, 1) != cx.msg_hash(a, 3)
    assert cx.msg_hash(a, 1) == cx.msg_hash(dict(a), 1)
    assert cx.msg_hash({"role": "user", "content": "ок!"}, 1) != cx.msg_hash(a, 1)


def test_assemble_never_adds_messages_or_changes_roles():
    sys_msgs, non_sys = cx.split_system(conv(6, tools_every=2))
    hashes = cx.history_hashes(non_sys)
    deltas = {hashes[i]: f"дельта {i}" for i in cx.user_positions(non_sys)}
    for mode in ("native", "first_user"):
        for cut in cx.user_positions(non_sys):
            out = cx.assemble(sys_msgs, non_sys, cut, head="профиль", deltas=deltas, hashes=hashes,
                              policy="ПОЛИТИКА", system_mode=mode)
            roles = [m["role"] for m in out]
            expect = ([] if mode == "first_user" else ["system"]) + [m["role"] for m in non_sys[cut:]]
            assert roles == expect
            assert not cx.validate(out)


def test_assemble_without_system_puts_policy_into_first_user():
    _, non_sys = cx.split_system(conv(2, sys=False))
    out = cx.assemble([], non_sys, 0, head="", policy="ПОЛИТИКА")
    assert out[0]["role"] == "user" and out[0]["content"].startswith("[ПРАВИЛО ПАМЯТИ] ПОЛИТИКА")


def test_first_user_mode_folds_system():
    sys_msgs, non_sys = cx.split_system(conv(2))
    out = cx.assemble(sys_msgs, non_sys, 0, system_mode="first_user")
    assert out[0]["role"] == "user" and out[0]["content"].startswith("Ты помощник.")


def test_append_only_property_random_histories():
    """Главное свойство v5: промпт хода t — префикс промпта хода t+1 (в пределах эпохи)."""
    rnd = random.Random(1)
    for _ in range(30):
        full = conv(rnd.randint(3, 12), tools_every=rnd.choice([0, 2, 3]))
        sys_msgs, non_sys = cx.split_system(full)
        ups = cx.user_positions(non_sys)
        hashes = cx.history_hashes(non_sys)
        deltas = {hashes[i]: f"факт {i}" for i in ups if rnd.random() < 0.7}
        cut = rnd.choice(ups[: max(1, len(ups) // 2)])
        prev = None
        for k in ups[1:] + [len(non_sys)]:
            part = non_sys[:k]
            if cut >= len(part): continue
            out = cx.assemble(sys_msgs, part, cut, head="саммари+профиль", deltas=deltas,
                              hashes=hashes[:k], policy="P", elide_before=cut + 2)
            ser = [json.dumps(m, ensure_ascii=False, sort_keys=True) for m in out]
            if prev is not None:
                assert ser[:len(prev)] == prev
            prev = ser


def test_validate_catches_orphan_tool():
    bad = [{"role": "user", "content": "q"}, {"role": "tool", "tool_call_id": "x", "content": "r"}]
    assert cx.validate(bad)
    assert cx.validate([{"role": "tool", "tool_call_id": "x", "content": "r"}])
    ok = conv(3, tools_every=1)
    assert cx.validate(ok) == []


def test_elide_tool_outputs_only_before_position():
    _, non_sys = cx.split_system(conv(4, tools_every=1))
    tool_idx = [i for i, m in enumerate(non_sys) if m["role"] == "tool"]
    out = cx.elide_tool_outputs(non_sys, tool_idx[2])
    assert "сокращён прокси" in out[tool_idx[0]]["content"] and "сокращён прокси" in out[tool_idx[1]]["content"]
    assert out[tool_idx[2]] == non_sys[tool_idx[2]]


def test_shrink_turn_fits_budget_and_keeps_structure():
    tc = cx.TokenCounter(3.0)
    msgs = [{"role": "user", "content": "читай файл"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "r", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "x" * 60000}]
    out = cx.shrink_turn(msgs, 0, 3000, tc)
    assert tc.messages(out) <= 3000 and [m["role"] for m in out] == ["user", "assistant", "tool"]
    assert out[0]["content"] == "читай файл" and "середина сокращена" in out[2]["content"]


def test_choose_cut_turn_boundaries():
    _, non_sys = cx.split_system(conv(5, tools_every=1))
    ups = cx.user_positions(non_sys)
    c = cx.choose_cut(non_sys, 1, lambda c: c >= ups[2])
    assert c == ups[2] and non_sys[c]["role"] == "user"
    assert cx.choose_cut(non_sys, 1, lambda c: False) == ups[-1]          # не дальше последнего хода


def test_token_counter_calibration():
    tc = cx.TokenCounter(3.0)
    est = tc.text("a" * 3000, "m")
    for _ in range(20):
        tc.observe("m", tc.text("a" * 3000, "m"), 2000)        # реальный токенизатор вдвое «плотнее»
    assert abs(tc.text("a" * 3000, "m") - 2000) < 60 and est == 1000
    assert tc.text("a" * 3000, "other") == 1000                # коэффициент — на модель

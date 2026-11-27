"""Сборка контекста для модели (v5): чистые функции без БД и HTTP.

Главное правило: промпт к модели только дописывается. Всё, что прокси вставляет в историю,
детерминированно зависит от (история клиента, эпоха чата, закреплённые дельты) и не меняется
между ходами. Префикс меняется только при смене эпохи.

Прокси никогда не добавляет сообщений и не меняет роли: служебные блоки вставляются только
в content существующих сообщений user. Так строгие шаблоны (чередование ролей, system только
первым, Gemma без system) не ломаются.
"""
import hashlib
import json
import math
import re

ELIDE_MARK = "[… вывод инструмента сокращён прокси: было {n} символов …]"
TRUNC_MARK = "\n[… середина сокращена прокси: было {n} символов …]\n"
_THINK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
_THINK_OPEN = re.compile(r"^\s*<think>.*", re.S | re.I)


# ─── Содержимое сообщений ───
def content_text(c) -> str:
    """Текст сообщения: строка как есть, у списка частей (OpenAI vision) — только text-части."""
    if isinstance(c, str): return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
                         and isinstance(p.get("text"), str))
    return ""

def has_media(c) -> bool:
    return isinstance(c, list) and any(isinstance(p, dict) and p.get("type") != "text" for p in c)

def with_prefix(msg: dict, prefix: str) -> dict:
    """Копия сообщения с текстом перед содержимым. Список частей получает отдельную text-часть."""
    if not prefix: return msg
    c = msg.get("content")
    if isinstance(c, list):
        return {**msg, "content": [{"type": "text", "text": prefix}] + list(c)}
    return {**msg, "content": prefix + (c if isinstance(c, str) else "")}

def strip_think(text: str) -> str:
    """Ответ без рассуждений <think>…</think> (модели без парсера reasoning присылают их в content)."""
    if not isinstance(text, str): return ""
    if "<think" not in text.lower(): return text
    t = _THINK.sub("", text)
    if _THINK_OPEN.match(t): return ""          # рассуждение не закрыто — ответа нет
    return t.strip()

def truncate_middle(text: str, max_chars: int) -> str:
    if not isinstance(text, str) or len(text) <= max_chars: return text
    head = max(0, max_chars * 2 // 3)
    tail = max(0, max_chars - head)
    return text[:head] + TRUNC_MARK.format(n=len(text)) + (text[-tail:] if tail else "")


# ─── Хеши сообщений ───
def msg_hash(m: dict, pos: int) -> str:
    """Отпечаток сообщения клиента с позицией (два одинаковых «ок» в разных местах — разные)."""
    c = m.get("content")
    payload = [m.get("role"), c if isinstance(c, str) else json.dumps(c, ensure_ascii=False, sort_keys=True),
               json.dumps(m.get("tool_calls"), ensure_ascii=False, sort_keys=True) if m.get("tool_calls") else "",
               m.get("tool_call_id") or "", pos]
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]

def history_hashes(non_sys: list) -> list:
    return [msg_hash(m, i) for i, m in enumerate(non_sys)]


# ─── Токены ───
class TokenCounter:
    """Оценка токенов по символам с самокалибровкой по usage.prompt_tokens бэкенда.

    Токенизаторы моделей расходятся в разы (особенно на кириллице), поэтому одна константа не годится:
    для каждой модели хранится поправочный коэффициент (EMA), ограниченный разумными рамками.
    """
    def __init__(self, chars_per_token: float = 3.0, per_message: int = 6):
        self.cpt, self.per_message, self.factor = chars_per_token, per_message, {}

    def text(self, s: str, model: str = "") -> int:
        if not s: return 0
        return int(math.ceil(len(s) / self.cpt * self.factor.get(model, 1.0)))

    def message(self, m: dict, model: str = "") -> int:
        c = m.get("content")
        n = self.per_message + self.text(content_text(c), model)
        if isinstance(c, list):
            n += 300 * sum(1 for p in c if isinstance(p, dict) and p.get("type") != "text")
        if m.get("tool_calls"):
            n += self.text(json.dumps(m["tool_calls"], ensure_ascii=False), model)
        return n

    def messages(self, msgs: list, model: str = "") -> int:
        return sum(self.message(m, model) for m in msgs)

    def tools(self, tools, model: str = "") -> int:
        return self.text(json.dumps(tools, ensure_ascii=False), model) if tools else 0

    def observe(self, model: str, estimated: int, actual: int):
        """Подстроить коэффициент модели по реальному числу токенов промпта."""
        if not actual or not estimated or actual < 200: return
        cur = self.factor.get(model, 1.0)
        ratio = actual / (estimated / cur)                 # что было бы при factor=1
        self.factor[model] = max(0.4, min(3.0, cur * 0.7 + ratio * 0.3))


# ─── Структура истории ───
def user_positions(non_sys: list) -> list:
    """Индексы сообщений user — границы ходов. Резать окно можно только по ним."""
    return [i for i, m in enumerate(non_sys) if m.get("role") == "user"]

def last_user_index(non_sys: list):
    up = user_positions(non_sys)
    return up[-1] if up else None

def split_system(messages: list):
    """(ведущие system, остальное). System не в начале остаются на своих местах (их прислал клиент)."""
    i = 0
    while i < len(messages) and messages[i].get("role") in ("system", "developer"):
        i += 1
    return messages[:i], messages[i:]

def validate(messages: list) -> list:
    """Инварианты перед отправкой. Возвращает список нарушений (пусто — всё хорошо)."""
    errs = []
    _, non_sys = split_system(messages)
    if non_sys and non_sys[0].get("role") == "tool":
        errs.append("I1: окно начинается с результата инструмента")
    open_ids = set()
    for i, m in enumerate(non_sys):
        r = m.get("role")
        if r == "assistant" and m.get("tool_calls"):
            open_ids = {tc.get("id") for tc in m["tool_calls"] if isinstance(tc, dict)}
        elif r == "tool":
            tid = m.get("tool_call_id")
            if i == 0 or (tid and open_ids and tid not in open_ids):
                errs.append(f"I2: tool без вызова (позиция {i})")
            prev = non_sys[i - 1] if i else {}
            if prev.get("role") not in ("assistant", "tool"):
                errs.append(f"I3: tool не после assistant/tool (позиция {i})")
        elif r == "user":
            open_ids = set()
    return errs


# ─── Элизия и обрезка ───
def elide_tool_outputs(non_sys: list, before: int, keep_chars: int = 300, min_chars: int = 1200) -> list:
    """Длинные выводы инструментов до позиции before → начало + пометка (L1)."""
    out = []
    for i, m in enumerate(non_sys):
        c = m.get("content")
        if i < before and m.get("role") == "tool" and isinstance(c, str) and len(c) > min_chars:
            m = {**m, "content": c[:keep_chars] + "\n" + ELIDE_MARK.format(n=len(c))}
        out.append(m)
    return out

def shrink_turn(non_sys: list, start: int, budget_tokens: int, counter: TokenCounter, model: str = "") -> list:
    """Текущий ход не помещается (OVERSIZE=truncate): режем середину самых длинных текстов хода,
    начиная с выводов инструментов. Последнее сообщение пользователя режется последним."""
    msgs = list(non_sys)
    for _ in range(12):
        over = counter.messages(msgs, model) - budget_tokens
        if over <= 0: break
        cands = [(len(content_text(m.get("content"))), i) for i, m in enumerate(msgs)
                 if i >= start and isinstance(m.get("content"), str) and len(m["content"]) > 400]
        if not cands: break
        cands.sort(key=lambda x: (msgs[x[1]].get("role") == "user", -x[0]))
        n, i = cands[0]
        keep = max(400, n - int(over * counter.cpt * 1.15) - 200)
        if keep >= n: keep = n // 2
        msgs[i] = {**msgs[i], "content": truncate_middle(msgs[i]["content"], keep)}
    return msgs


# ─── Сборка ───
def memory_block(text: str) -> str:
    return f"<memory>\n{text}\n</memory>\n\n" if text else ""

def assemble(system_msgs: list, non_sys: list, cut: int, *, head: str = "", deltas: dict = None,
             hashes: list = None, policy: str = "", system_mode: str = "native", elide_before: int = 0) -> list:
    """Промпт = system (+политика) + окно истории non_sys[cut:].

    head — саммари и профиль эпохи (в начало первого сообщения окна), deltas — {msg_hash: текст}
    закреплённые за сообщениями user. system_mode='first_user' сворачивает system клиента в первое
    сообщение окна (шаблоны без system или переносящие его, напр. Gemma 2, Mistral Nemo).
    """
    deltas, hashes = deltas or {}, hashes or history_hashes(non_sys)
    window = elide_tool_outputs(non_sys, elide_before) if elide_before > cut else list(non_sys)
    out_win = []
    for i in range(cut, len(window)):
        m = window[i]
        if m.get("role") == "user":
            m = with_prefix(m, memory_block(deltas.get(hashes[i], "")))
        out_win.append(m)
    lead = ""
    sys_out = list(system_msgs)
    if system_mode == "first_user" and sys_out:
        lead = "\n\n".join(content_text(m.get("content")) for m in sys_out) + "\n\n"
        sys_out = []
    if policy:
        if sys_out and isinstance(sys_out[0].get("content"), str):
            sys_out[0] = {**sys_out[0], "content": sys_out[0]["content"] + "\n\n" + policy}
        else:
            lead += f"[ПРАВИЛО ПАМЯТИ] {policy}\n\n"
    lead += memory_block(head)
    if out_win and lead:
        j = next((k for k, m in enumerate(out_win) if m.get("role") == "user"), 0)
        out_win[j] = with_prefix(out_win[j], lead)
    return sys_out + out_win

def choose_cut(non_sys: list, start: int, fits) -> int:
    """Наименьшая граница хода ≥ start, при которой fits(cut) истинно. Не дальше последнего user."""
    ups = [i for i in user_positions(non_sys) if i >= start]
    if not ups: return start
    for c in ups:
        if fits(c): return c
    return ups[-1]

def lcp_len(a: list, b: list) -> int:
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i

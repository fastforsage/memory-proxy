"""Unit tests v4.7.3 (без ONNX)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["MODEL_DIR"] = "/nonexistent"

def test_safe_float():
    from proxy import safe_float
    assert safe_float(0.5) == 0.5
    assert safe_float(2.0) == 1.0
    assert safe_float(-2.0) == -1.0
    assert safe_float("abc") == 0.0
    assert safe_float(None) == 0.0

def test_rough_tokens():
    from proxy import rough_tokens
    assert rough_tokens("") == 0
    assert rough_tokens("a" * 300) == 100
    assert rough_tokens(None) == 1

def test_total_tokens_vision():
    from proxy import total_tokens
    msgs = [{"role": "user", "content": [{"type": "image_url"}]}]
    assert total_tokens(msgs) < 100

def test_preclassify_code():
    """H-3: код должен определяться без zlib."""
    from proxy import preclassify
    assert preclassify("hi") == "personal"
    assert preclassify("def foo():\n    return 42\n" * 10) == "technical"
    assert preclassify("```python\nprint(1)\n```") == "technical"
    # Русский текст не должен быть technical
    russian = ("Вчера я ездил на дачу к родителям. Погода была отличная, "
               "мы жарили шашлыки и долго гуляли по лесу. Вечером пили чай на веранде.") * 2
    assert preclassify(russian) == "personal", f"Russian wrongly classified: {preclassify(russian)}"

def test_evidence_ok():
    """H-4b: пустой evidence и object не в тексте — отклоняются."""
    from proxy import evidence_ok
    assert evidence_ok("Я работаю в Google", "я работаю в Google", "Google") is True
    assert evidence_ok("Я работаю в Google", "", "Google") is False
    assert evidence_ok("Я работаю в Google", "12345", "Google") is False
    assert evidence_ok("погода", "я", "Google") is False

def test_command_regex():
    """M-7: только whitelisted команды."""
    from proxy import COMMAND_RE
    assert COMMAND_RE.match("/chats") is not None
    assert COMMAND_RE.match("/pull 2") is not None
    assert COMMAND_RE.match("/etc/nginx/nginx.conf") is None
    assert COMMAND_RE.match("hello") is None

def test_is_service_request():
    from proxy import is_service_request
    assert is_service_request("### Task: generate tags") is True
    assert is_service_request("hello") is False

def test_keep_recent_validation():
    from proxy import KEEP_RECENT
    assert KEEP_RECENT >= 1

def test_compress_no_op_short():
    from proxy import compress, KEEP_RECENT
    msgs = [{"role": "user", "content": "a"}] * (KEEP_RECENT + 1)
    assert compress(msgs) == msgs

def test_compress_keeps_system():
    from proxy import compress, KEEP_RECENT
    msgs = ([{"role": "system", "content": "sys"}] +
            [{"role": "user", "content": f"m{i}"} for i in range(KEEP_RECENT + 5)])
    result = compress(msgs)
    assert result[0]["role"] == "system"
    assert len(result) == 2 + KEEP_RECENT

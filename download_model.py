import os, sys
from huggingface_hub import snapshot_download

MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
# Ревизия закреплена: смена весов/токенизатора меняет эмбеддинги и ломает совместимость с базой.
MODEL_REVISION = os.getenv("MODEL_REVISION", "4a3caf0c13a10330815441c2408b7b9447f23744")
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Downloading USER2-small ONNX @ {MODEL_REVISION[:10]} (only model.onnx + tokenizer)...")
snapshot_download(
    repo_id="onnx-community/USER2-small-ONNX",
    local_dir=MODEL_DIR,
    revision=MODEL_REVISION,
    allow_patterns=[
        "onnx/model.onnx",         # только fp32, не все 8 вариантов (M-9)
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    ],
)

onnx_path = os.path.join(MODEL_DIR, "onnx", "model.onnx")
if not os.path.exists(onnx_path):
    print(f"❌ Missing {onnx_path}"); sys.exit(1)
print(f"  ✅ {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"]:
    p = os.path.join(MODEL_DIR, f)
    if os.path.exists(p):
        print(f"  ✅ {f}")
    else:
        print(f"  ⚠️  {f} missing")

import os, time, json
os.environ["HF_HOME"] = "/workspace/nes_run_20261016_anchor_ablation/hf_home"
from huggingface_hub import snapshot_download

REPO = "stabilityai/stablelm-2-1_6b"
REV = "f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef"
DST = "/workspace/nes_run_20261016_anchor_ablation/ckpt/stablelm_fp16"

t = time.time()
p = snapshot_download(
    REPO, revision=REV, local_dir=DST,
    allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt"],
)
print(json.dumps({"status": "DONE", "path": p, "seconds": round(time.time() - t, 1)}), flush=True)

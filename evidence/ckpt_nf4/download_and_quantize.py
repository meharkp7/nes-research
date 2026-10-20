#!/usr/bin/env python3
"""Download and quantize a model to clean NF4 with pinned revision."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def core_files(folder: Path):
    names = {"config.json", "generation_config.json", "model.safetensors.index.json"}
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and (p.name in names or p.suffix == ".safetensors"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", required=True, help="HuggingFace model ID")
    ap.add_argument("--revision", required=True, help="Pinned revision/commit hash")
    ap.add_argument("--output", required=True, type=Path, help="NF4 output dir")
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--cache-dir", type=Path, default=None, help="HF cache dir")
    args = ap.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.uint8,
    )

    t0 = time.time()
    print(f"[clean-nf4] Loading {args.model_id}@{args.revision} with NF4 quantization", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        quantization_config=quant,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
        local_files_only=False,
        trust_remote_code=False,
        cache_dir=args.cache_dir,
    )
    model.eval()
    print("[clean-nf4] Saving clean NF4 checkpoint...", flush=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    del model
    gc.collect()

    files = core_files(args.output)
    hashes = {p.name: sha256_file(p) for p in files}
    report = {
        "status": "DONE",
        "model_id": args.model_id,
        "revision": args.revision,
        "output": str(args.output),
        "quantization": {
            "quant_type": "nf4", "blocksize": 64, "double_quant": True,
            "quant_storage": "uint8", "compute_dtype": "float16",
        },
        "output_core_files": hashes,
        "output_core_bytes": sum(p.stat().st_size for p in files),
        "elapsed_s": round(time.time() - t0, 1),
        "torch": torch.__version__,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "DONE", "output": str(args.output),
                      "files": len(files), "elapsed_s": report["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
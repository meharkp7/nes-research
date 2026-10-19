#!/usr/bin/env python3
"""Build a clean (unembedded) NF4 checkpoint from an fp16 source checkpoint.

Reproduces the same quantization used for the existing clean checkpoints:
BitsAndBytes NF4, blocksize 64, double (nested) quant, uint8 storage,
compute dtype float16. Loads the fp16 source, quantizes on load, and saves a
fresh NF4 checkpoint. New outputs only; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


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
    ap.add_argument("--source", required=True, type=Path, help="fp16 source checkpoint dir")
    ap.add_argument("--output", required=True, type=Path, help="new clean NF4 output dir")
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if not (args.source / "config.json").is_file():
        raise FileNotFoundError(f"No config.json in source {args.source}")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.uint8,
    )

    t0 = time.time()
    print(f"[clean-nf4] Loading fp16 source with NF4 quantization: {args.source}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.source), quantization_config=quant, device_map={"": "cpu"},
        torch_dtype=torch.float16, local_files_only=True, trust_remote_code=False,
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
        "source": str(args.source),
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

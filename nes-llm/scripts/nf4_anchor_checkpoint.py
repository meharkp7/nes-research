#!/usr/bin/env python3
"""Full-checkpoint development gate for the NF4 anchor experiment.

Every output is new; reports live outside model directories. Decoder receives
no clean checkpoint, expected payload, carrier map, or reference weights.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.contract_b_nf4_b14 import find_shard, read_tensor, DEFAULT_TENSOR_KEY
from scripts.contract_b_nf4_requantization import bypass_bnb4bit_reverse_conversion_after_dequantization
from scripts.contract_b_utility_eval_heldout import HELDOUT_TEXTS
from scripts.run_nf4_anchor_pilot import load_tensor, sha_file
from src.experiments.nf4_anchor_channel import METHODS, embed, extract, recovery_diagnostic


def key_for(seed):
    return hashlib.sha256(f"NES-development-{seed}".encode()).digest()


def core_files(folder):
    names = {"config.json", "generation_config.json", "model.safetensors.index.json"}
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and (p.name in names or p.suffix == ".safetensors"))


def new_directory(path, source=None):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if source and (path.resolve() == source.resolve() or source.resolve() in path.resolve().parents):
        raise ValueError("Outputs must be outside the source checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)


def materialize(args):
    if args.bits <= 0 or args.bits % 8:
        raise ValueError("Payload must be positive and byte aligned")
    new_directory(args.output, args.checkpoint)
    files = core_files(args.checkpoint)
    size = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(args.output.parent).free < size + 1024**3:
        raise OSError("Insufficient free disk for independent checkpoint copy")
    tensor, state = load_tensor(args.checkpoint, args.tensor)
    payload = np.random.default_rng(args.seed).bytes(args.bits // 8)
    raw, info = embed(tensor.numpy().tobytes(), payload, key_for(args.seed), args.tensor,
                      args.method, state.code.float().numpy())
    before = {p.name: sha_file(p) for p in files}
    args.output.mkdir()
    for p in files:
        shutil.copy2(p, args.output / p.name)
    shard = find_shard(args.output, args.tensor)
    offset, _, start, _, _ = read_tensor(shard, args.tensor)
    with shard.open("r+b") as f:
        f.seek(offset + start)
        f.write(raw)
    _, _, _, _, saved = read_tensor(shard, args.tensor)
    if saved != raw:
        raise RuntimeError("Serialized bytes differ from intended embedding")
    after = {p.name: sha_file(p) for p in files}
    if before != after:
        raise RuntimeError("Source checkpoint was modified")
    return {"status": "MATERIALIZED", **info,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "source_file_hashes": before, "source_unchanged": True,
            "output_file_hashes": {p.name: sha_file(p) for p in core_files(args.output)},
            "source_core_bytes": size,
            "output_core_bytes": sum(p.stat().st_size for p in core_files(args.output)),
            "serialized_tensor_sha256": hashlib.sha256(saved).hexdigest()}


def load_model(folder):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(folder), device_map={"": "cpu"}, trust_remote_code=False, local_files_only=True,
    )
    return model.eval()


def export_clean(args):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    new_directory(args.output, args.checkpoint)
    sources = core_files(args.checkpoint)
    size = sum(p.stat().st_size for p in sources)
    if shutil.disk_usage(args.output.parent).free < size + 1024**3:
        raise OSError("Insufficient space for clean export")
    hashes = {p.name: sha_file(p) for p in sources}
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.checkpoint), quantization_config=config, device_map={"": "cpu"},
        dtype=torch.bfloat16, trust_remote_code=False, local_files_only=True,
    )
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    return {"status": "PASS", "source_snapshot": str(args.checkpoint.resolve()),
            "source_file_hashes": hashes, "stored_dtype": "bfloat16", "nested_statistics": True,
            "output_file_hashes": {p.name: sha_file(p) for p in core_files(args.output)},
            "scope": "Clean NF4 export, no payload"}


def receive(args):
    # Extraction is completed before model/tokenizer loading or any scoring.
    digest = None
    if args.method != "clean":
        _, _, _, _, raw = read_tensor(find_shard(args.checkpoint, args.tensor), args.tensor)
        payload = extract(raw, key_for(args.seed), args.tensor, args.method)
        digest = hashlib.sha256(payload).hexdigest()
    model = load_model(args.checkpoint)
    selected = model.get_submodule(args.tensor.removesuffix(".weight")).weight
    if getattr(selected.quant_state, "quant_type", None) != "nf4":
        raise ValueError("Loaded selected layer is not NF4")
    loaded_digest = None
    if args.method != "clean":
        loaded_raw = selected.data.detach().cpu().contiguous().numpy().tobytes()
        loaded = extract(loaded_raw, key_for(args.seed), args.tensor, args.method)
        loaded_digest = hashlib.sha256(loaded).hexdigest()
        if loaded_digest != digest:
            raise ValueError("Reloaded model extraction differs from serialized artifact")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    results = []
    texts = HELDOUT_TEXTS if args.utility else HELDOUT_TEXTS[:1]
    with torch.inference_mode():
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=256)
            if len(ids) < 2:
                raise ValueError("Tokenizer produced fewer than two tokens")
            inputs = torch.tensor([ids], dtype=torch.long)
            loss = float(model(input_ids=inputs, labels=inputs).loss)
            if not math.isfinite(loss):
                raise ValueError("Non-finite inference loss")
            results.append({"text_index": i, "mean_nll": loss, "scored_tokens": len(ids)-1})
    n = sum(r["scored_tokens"] for r in results)
    loss = sum(r["mean_nll"] * r["scored_tokens"] for r in results) / n
    return {"status": "PASS", "raw_payload_sha256": digest, "loaded_payload_sha256": loaded_digest,
            "receiver_inputs": ["delivered checkpoint", "test key from seed", "fixed tensor/method/protocol"],
            "original_or_expected_payload_accessed": False,
            "inference_finite": True, "per_text": results, "token_count": n,
            "perplexity": math.exp(loss), "mean_nll": loss,
            "scope": "Development check; fixed project prompts, not external task benchmark",
            "prompt_sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest()}


def lifecycle(args):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    new_directory(args.output, args.checkpoint)
    new_directory(args.intermediate, args.checkpoint)
    if args.output.resolve() == args.intermediate.resolve() or args.output.resolve() in args.intermediate.resolve().parents or args.intermediate.resolve() in args.output.resolve().parents:
        raise ValueError("Output and intermediate directories must be separate")
    size = sum(p.stat().st_size for p in core_files(args.checkpoint))
    if shutil.disk_usage(args.output.parent).free < size * 5 + 1024**3:
        raise OSError("Insufficient free disk for floating intermediate and NF4 output")
    _, _, _, _, raw = read_tensor(find_shard(args.checkpoint, args.tensor), args.tensor)
    expected = extract(raw, key_for(args.seed), args.tensor, args.method)
    model = load_model(args.checkpoint)
    report = {}
    model.dequantize()
    bypass_bnb4bit_reverse_conversion_after_dequantization(model, report)
    weight = model.get_submodule(args.tensor.removesuffix(".weight")).weight
    if not weight.dtype.is_floating_point:
        raise ValueError("Dequantization did not produce floating weights")
    report["intermediate_dtype"] = str(weight.dtype)
    if hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None
    model.save_pretrained(args.intermediate, safe_serialization=True, max_shard_size="2GB")
    del weight, model
    gc.collect()
    quantizer = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.intermediate), quantization_config=quantizer, device_map={"": "cpu"},
        local_files_only=True, trust_remote_code=False,
    )
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    del model
    gc.collect()
    _, _, _, _, transformed = read_tensor(find_shard(args.output, args.tensor), args.tensor)
    recovery = recovery_diagnostic(transformed, expected, key_for(args.seed), args.tensor, args.method)
    return {**report, "status": "PASS" if recovery["exact_recovery"] else "RECOVERY_FAILED",
            "transformation_completed": True, "recovery": recovery,
            "output_tensor_sha256": hashlib.sha256(transformed).hexdigest(),
            "output_file_hashes": {p.name: sha_file(p) for p in core_files(args.output)},
            "transformation": "whole model NF4 -> floating checkpoint -> fresh nested NF4 checkpoint"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("export", "materialize", "receive", "lifecycle"))
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--intermediate", type=Path)
    p.add_argument("--method", choices=(*METHODS, "clean"), default="interior_nearest")
    p.add_argument("--tensor", default=DEFAULT_TENSOR_KEY)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--bits", type=int, default=10000)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B")
    p.add_argument("--utility", action="store_true")
    args = p.parse_args()
    if args.mode != "receive" and args.output is None:
        p.error("Export/materialize/lifecycle require an output")
    if args.mode in ("materialize", "lifecycle") and args.method == "clean":
        p.error("Materialize/lifecycle require an output and embedding method")
    if args.mode == "lifecycle" and args.intermediate is None:
        p.error("Lifecycle requires an intermediate path")
    report_path = args.report.resolve()
    for folder in (args.checkpoint, args.output, args.intermediate):
        if folder and (report_path == folder.resolve() or folder.resolve() in report_path.parents):
            p.error("Report must be outside all checkpoint and intermediate directories")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Reserve report before any model output mutation, and record failures too.
    with args.report.open("x") as f:
        started = time.monotonic()
        report = {"arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  "runner_sha256": sha_file(__file__),
                  "codec_sha256": sha_file(Path(__file__).parents[1] / "src/experiments/nf4_anchor_channel.py")}
        try:
            report.update({"export": export_clean, "materialize": materialize,
                           "receive": receive, "lifecycle": lifecycle}[args.mode](args))
        except Exception as exc:
            report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        report["duration_seconds"] = time.monotonic() - started
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": report["status"], "report": str(args.report), "error": report.get("error")}), flush=True)
    return 0 if report["status"] in ("PASS", "MATERIALIZED") else 2


if __name__ == "__main__":
    raise SystemExit(main())

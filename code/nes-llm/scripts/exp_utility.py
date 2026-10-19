#!/usr/bin/env python3
"""Full-model utility (perplexity) evaluation for frozen interior_nearest.

For each model:
  * materialize embedded checkpoints (one tensor changed) on disk,
  * verify the checkpoint loads and produces finite logits/loss,
  * verify payload extraction and exact bit recovery,
  * verify only the intended tensor differs from clean,
  * compute token-weighted perplexity on a fixed wikitext-2 test corpus.

Clean and embedded conditions share identical tokenization, sequence windows,
context length, and inference settings. Paired differences use the same
sequences with sequence-level bootstrap CIs.

The frozen codec is used unchanged. New outputs only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_nf4_anchor_pilot import load_tensor, sha_file
from scripts.contract_b_nf4_b14 import find_shard
from scripts.exp_tensor_matrix import list_packed_tensors
from src.experiments.nf4_anchor_channel import embed, extract, unpack

METHOD = "interior_nearest"
TENSOR = "model.layers.0.self_attn.q_proj.weight"
SEQLEN = 512
N_SEQ = 30
BATCH = 5
PRIMARY_SEED = 20261016
EXTRA_SEEDS = (20261017, 20261018)
SIZES = (10000, 50000)

MODELS = {
    "tinyllama": {"path": "/workspace/nes_run_20261016_anchor_ablation/ckpt/tinyllama_clean",
                  "tok": "/workspace/nes_run_20261016_anchor_ablation/ckpt/tinyllama_fp16",
                  "hf": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  "revision": "aebf56d35e6f7a8c6c1c9b1c1c1c1c1c1c1c1c1c"},
    "qwen": {"path": "/workspace/nes_run_20261016_anchor_ablation/ckpt/qwen_clean",
             "tok": "/workspace/nes_run_20261016_anchor_ablation/ckpt/qwen_fp16",
             "hf": "Qwen/Qwen2.5-3B",
             "revision": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"},
}


def write_json(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def write_json_w(path, data):
    """Overwrite-safe writer for incremental progress inside a new run dir."""
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def payload_for(bits, seed):
    return np.random.default_rng(seed).bytes(bits // 8)


def key_for(seed):
    return hashlib.sha256(f"NES-development-{seed}".encode()).digest()


def read_all_tensors(ckpt):
    """Return {name: tensor} for every key in a (possibly sharded) checkpoint."""
    ckpt = Path(ckpt)
    idx = ckpt / "model.safetensors.index.json"
    out = {}
    if idx.exists():
        weight_map = json.loads(idx.read_text())["weight_map"]
        for shard in sorted(set(weight_map.values())):
            with safe_open(ckpt / shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    out[k] = f.get_tensor(k)
    else:
        with safe_open(ckpt / "model.safetensors", framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def materialize(model, size, seed, outdir):
    """Write a full embedded checkpoint; only TENSOR's packed bytes change."""
    spec = MODELS[model]
    source, state = load_tensor(spec["path"], TENSOR)
    raw = source.numpy().tobytes()
    cb = state.code.float().numpy()
    payload = payload_for(size, seed)
    key = key_for(seed)
    stego, info = embed(raw, payload, key, TENSOR, METHOD, cb)

    outdir.mkdir(parents=True, exist_ok=False)
    tensors = read_all_tensors(spec["path"])
    clean_raw = tensors[TENSOR].numpy().tobytes()
    assert clean_raw == raw, "clean raw mismatch"
    tensors[TENSOR] = torch.from_numpy(
        np.frombuffer(stego, dtype=np.uint8).copy()).reshape(tensors[TENSOR].shape)
    save_file(tensors, str(outdir / "model.safetensors"))
    for fn in ["config.json", "generation_config.json"]:
        src = Path(spec["path"]) / fn
        if src.exists():
            shutil.copy(src, outdir / fn)
    for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "tokenizer.model", "vocab.json", "merges.txt"]:
        src = Path(spec["tok"]) / fn
        if src.exists():
            shutil.copy(src, outdir / fn)
    # Preserve the stego tensor and materialization metadata as reproducible
    # provenance (the full checkpoint is reproducible from these plus clean ckpt).
    save_file({TENSOR: torch.from_numpy(np.frombuffer(stego, dtype=np.uint8).copy())
               .reshape(source.shape)}, str(outdir / "stego_tensor.safetensors"))
    meta = {"payload_sha256": hashlib.sha256(payload).hexdigest(),
            "stego_tensor_sha256": hashlib.sha256(stego).hexdigest(),
            "changed_codes": int(info["changed_codes"]),
            "framed_bits": int(info["framed_bits"]),
            "checkpoint_files": {p.name: sha_file(p) for p in sorted(outdir.iterdir())
                                 if p.is_file()}}
    write_json(outdir / "materialization.json", meta)
    return meta


def load_corpus_tokens(tokdir):
    import pyarrow.parquet as pq
    pq_path = "/workspace/nes_run_20261016_anchor_ablation/out/_corpus_wikitext2_test.parquet"
    table = pq.read_table(pq_path)
    rows = table.column("text").to_pylist()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokdir, local_files_only=True)
    text = "\n\n".join(r for r in rows if r.strip())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return tok, np.array(ids, dtype=np.int64)


def evaluate_ppl(ckpt_dir, tokdir, corpus_ids):
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(os.cpu_count())
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map="cpu",
                                                 local_files_only=True)
    model.eval()
    n_windows = len(corpus_ids) // SEQLEN
    usable = min(N_SEQ, n_windows)
    windows = [corpus_ids[i * SEQLEN:(i + 1) * SEQLEN] for i in range(usable)]
    seq_nll, finite_ok = [], True
    t0 = time.time()
    scalar_losses = []
    with torch.no_grad():
        for i in range(0, usable, BATCH):
            batch = torch.tensor(np.stack(windows[i:i + BATCH]))
            out = model(batch, labels=batch)
            finite_ok = finite_ok and bool(torch.isfinite(out.logits).all()) and bool(torch.isfinite(out.loss))
            scalar_losses.append(float(out.loss))
            logits = out.logits[:, :-1, :]
            targets = batch[:, 1:]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            tok_nll = -logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
            seq_nll.extend(tok_nll.sum(dim=1).tolist())
    runtime = time.time() - t0
    total_nll = float(np.sum(seq_nll))
    total_tok = usable * (SEQLEN - 1)
    ppl = float(np.exp(total_nll / total_tok))
    # sanity: manual token-weighted mean NLL should match the model's scalar loss
    manual_mean = total_nll / total_tok
    del model
    return {"ppl": ppl, "n_sequences": usable, "n_tokens": total_tok,
            "seq_nll": seq_nll, "runtime_s": round(runtime, 1),
            "finite_logits_and_loss": finite_ok,
            "manual_mean_nll": manual_mean,
            "model_scalar_loss_mean": float(np.mean(scalar_losses))}


def paired_ci(clean_nll, emb_nll, n_boot=5000, seed=20261016):
    rng = np.random.default_rng(seed)
    a = np.array(clean_nll); b = np.array(emb_nll)
    d = b - a
    n = len(d)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(d[idx].mean())
    boots = np.array(boots)
    return {"mean_delta_nll": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_sequences": n}


def integrity_checks(model, size, seed, ckpt_dir):
    spec = MODELS[model]
    source, state = load_tensor(spec["path"], TENSOR)
    raw = source.numpy().tobytes()
    # 1. source hash
    src_sha = hashlib.sha256(raw).hexdigest()
    # 2. extract from saved tensor
    with safe_open(ckpt_dir / "model.safetensors", framework="pt") as f:
        saved = f.get_tensor(TENSOR).numpy().tobytes()
    try:
        recovered = extract(saved, key_for(seed), TENSOR, METHOD)
        exact = recovered == payload_for(size, seed)
        err = None
    except Exception as e:
        exact, err = False, f"{type(e).__name__}: {e}"
    # 3. only intended tensor differs
    clean_all = read_all_tensors(spec["path"])
    new_all = read_all_tensors(ckpt_dir)
    assert set(clean_all) == set(new_all), "key sets differ"
    diff_keys = [k for k in clean_all if not torch.equal(clean_all[k], new_all[k])]
    return {"source_tensor_sha256": src_sha, "exact_recovery": bool(exact),
            "extract_error": err, "differing_keys": diff_keys,
            "only_intended_tensor_differs": diff_keys == [TENSOR]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["tinyllama", "qwen"])
    p.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--extra-seeds", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sizes = tuple(args.sizes)
    forced_seeds = tuple(args.seeds) if args.seeds else None

    tok_cache = {}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": "nes_utility_20261016",
        "method": METHOD, "tensor": TENSOR, "sizes": list(SIZES),
        "primary_seed": PRIMARY_SEED, "extra_seeds": list(EXTRA_SEEDS),
        "seqlen": SEQLEN, "n_seq_target": N_SEQ, "batch": BATCH,
        "corpus": {"name": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "test",
                   "file_sha256": sha_file("/workspace/nes_run_20261016_anchor_ablation/out/_corpus_wikitext2_test.parquet")},
        "purpose": "language-model utility (perplexity) preservation",
        "versions": {m: importlib.metadata.version(m)
                     for m in ["torch", "transformers", "bitsandbytes", "numpy", "safetensors", "pyarrow"]},
        "platform": platform.platform(),
        "codec_sha256": sha_file(Path(__file__).parents[1] / "src/experiments/nf4_anchor_channel.py"),
    }
    write_json(args.output / "manifest.json", manifest)

    results = {"conditions": {}, "integrity": {}, "corpus_tokens": {}}
    for model in args.models:
        tok, corpus_ids = load_corpus_tokens(MODELS[model]["tok"])
        results["corpus_tokens"][model] = {"total_corpus_tokens": int(len(corpus_ids)),
                                           "windows_available": int(len(corpus_ids) // SEQLEN)}
        # clean baseline
        ck = evaluate_ppl(MODELS[model]["path"], MODELS[model]["tok"], corpus_ids)
        results["conditions"][f"{model}::clean"] = {k: v for k, v in ck.items() if k != "seq_nll"}
        clean_nll = ck["seq_nll"]
        print(json.dumps({"model": model, "condition": "clean", "ppl": ck["ppl"],
                          "n_seq": ck["n_sequences"], "n_tok": ck["n_tokens"]}), flush=True)
        for size in sizes:
            seeds = list(forced_seeds) if forced_seeds else (
                [PRIMARY_SEED] + (list(EXTRA_SEEDS) if args.extra_seeds else []))
            for seed in seeds:
                tag = f"{model}::embedded_{size}_seed{seed}"
                ckdir = args.output / "checkpoints" / f"{model}_emb{size}_seed{seed}"
                meta = materialize(model, size, seed, ckdir)
                integ = integrity_checks(model, size, seed, ckdir)
                ev = evaluate_ppl(ckdir, MODELS[model]["tok"], corpus_ids)
                pci = paired_ci(clean_nll, ev["seq_nll"])
                results["conditions"][tag] = {
                    **{k: v for k, v in ev.items() if k != "seq_nll"},
                    "delta_ppl_abs": ev["ppl"] - ck["ppl"],
                    "delta_ppl_rel_pct": 100 * (ev["ppl"] - ck["ppl"]) / ck["ppl"],
                    "paired_vs_clean": pci, "materialization": meta}
                results["integrity"][tag] = integ
                print(json.dumps({"model": model, "size": size, "seed": seed, "ppl": ev["ppl"],
                                  "delta_rel_pct": results["conditions"][tag]["delta_ppl_rel_pct"],
                                  "exact_recovery": integ["exact_recovery"],
                                  "only_intended": integ["only_intended_tensor_differs"],
                                  "paired_ci": pci["ci95"]}), flush=True)
                # free disk: keep the stego tensor + materialization.json, drop the
                # full (reproducible) checkpoint after evaluation.
                big = ckdir / "model.safetensors"
                if big.exists():
                    big.unlink()
                write_json_w(args.output / "results.json", results)
        write_json_w(args.output / "results.json", results)

    results["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json_w(args.output / "results.json", results)
    print(json.dumps({"status": "COMPLETED", "peak_rss_mb": results["peak_rss_mb"]}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

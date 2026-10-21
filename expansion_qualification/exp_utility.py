#!/usr/bin/env python3
"""Per-model utility (perplexity) evaluation for the frozen codec (V2 protocol).

Generalized from stablelm_qualification/exp_stablelm_utility.py. Paired
clean-vs-embedded evaluation on wikitext-2-raw-v1 test with the model's own
tokenizer, context 512, non-overlapping windows, token-weighted PPL.

Each embedded condition is materialized to a temporary checkpoint dir, loaded
fresh, evaluated, and deleted. Per-sequence token NLL is saved so paired CIs can
be recomputed without re-running. Predeclared margin: +/-0.5% relative PPL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = "/Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")
from scripts.run_nf4_anchor_pilot import load_tensor, sha_file  # noqa: E402
from src.experiments.nf4_anchor_channel import embed, unpack  # noqa: E402

ROOT = Path("/Users/meharkapoor7/nes-agent-workspace/nes-research")
CORPUS = ROOT / "out/_corpus_wikitext2_test.parquet"
METHOD = "interior_nearest"
SEQLEN = 512


def write_json(path, data):
    def sanitize(obj):
        if isinstance(obj, float):
            if not np.isfinite(obj):
                return None
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj
    Path(path).write_text(json.dumps(sanitize(data), indent=2, sort_keys=True, allow_nan=False) + "\n")


def payload_for(bits, seed):
    return np.random.default_rng(seed).bytes(bits // 8)


def key_for(seed):
    return hashlib.sha256(f"NES-development-{seed}".encode()).digest()


def corpus_windows(tokdir, n_seq):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    rows = pq.read_table(CORPUS).column("text").to_pylist()
    text = "\n\n".join(r for r in rows if r.strip())
    tok = AutoTokenizer.from_pretrained(tokdir, local_files_only=True)
    ids = np.array(tok(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
    n_win = len(ids) // SEQLEN
    usable = min(n_seq, n_win)
    return [ids[i * SEQLEN:(i + 1) * SEQLEN] for i in range(usable)], n_win


def materialize_embedded(dest: Path, ckpt_clean: Path, tensor: str, bits: int, seed: int):
    shutil.copytree(ckpt_clean, dest, dirs_exist_ok=False)
    source, state = load_tensor(ckpt_clean, tensor)
    raw = source.numpy().tobytes()
    cb = state.code.float().numpy()
    stego, _ = embed(raw, payload_for(bits, seed), key_for(seed), tensor, METHOD, cb)
    from safetensors.torch import load_file, save_file
    sd = load_file(dest / "model.safetensors")
    sd[tensor] = torch.from_numpy(np.frombuffer(stego, dtype=np.uint8).copy()).reshape(sd[tensor].shape)
    save_file(sd, dest / "model.safetensors", metadata={"format": "pt"})
    meta = {"stego_tensor_sha256": hashlib.sha256(stego).hexdigest(),
            "clean_tensor_sha256": hashlib.sha256(raw).hexdigest(),
            "size": bits, "seed": seed, "tensor": tensor}
    (dest.parent / f"{dest.name}.meta.json").write_text(json.dumps(meta, sort_keys=True) + "\n")
    return meta["stego_tensor_sha256"], meta["clean_tensor_sha256"]


def eval_nll(ckpt_dir, windows, batch, ckpt_path=None):
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(os.cpu_count())
    seq_nll, start = [], 0
    if ckpt_path is not None and Path(ckpt_path).exists():
        part = json.loads(Path(ckpt_path).read_text())
        seq_nll, start = part["seq_nll"], part["n_done"]
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map="cpu", local_files_only=True)
    model.eval()
    finite = True
    t0 = time.time()
    with torch.no_grad():
        for i in range(start, len(windows), batch):
            b = torch.tensor(np.stack(windows[i:i + batch]))
            out = model(b, labels=b)
            finite = finite and bool(torch.isfinite(out.logits).all()) and bool(torch.isfinite(out.loss))
            lg = out.logits[:, :-1, :]; tg = b[:, 1:]
            nll = -torch.log_softmax(lg.float(), -1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            seq_nll.extend(nll.sum(dim=1).tolist())
            if ckpt_path is not None:
                Path(ckpt_path).write_text(json.dumps(
                    {"seq_nll": seq_nll, "n_done": i + len(b)}, sort_keys=True) + "\n")
    del model
    seq_nll = np.array(seq_nll)
    # Filter out non-finite NLL values
    valid_nll = seq_nll[np.isfinite(seq_nll)]
    n_tok = len(valid_nll) * (SEQLEN - 1) if len(valid_nll) > 0 else 1
    if len(valid_nll) == 0:
        ppl = float('inf')
    else:
        mean_nll = valid_nll.sum() / n_tok
        # Cap PPL to avoid overflow
        if mean_nll > 100:
            ppl = float('inf')
        else:
            ppl = float(np.exp(mean_nll))
    return {"seq_nll": seq_nll.tolist(), "ppl": ppl,
            "n_sequences": len(windows), "n_tokens": n_tok,
            "runtime_s": round(time.time() - t0, 1), "finite_logits_and_loss": finite}


def paired_ci(clean_nll, emb_nll, n_boot=5000, seed=20261016):
    a = np.array(clean_nll); b = np.array(emb_nll)
    assert len(a) == len(b), "paired sequences must match"
    d = (b - a) / (SEQLEN - 1)
    n = len(d)
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    rel_lo = (np.exp(ci_lo) - 1) * 100
    rel_hi = (np.exp(ci_hi) - 1) * 100
    return {"mean_delta_per_token_nll": float(d.mean()),
            "ci95_nll": [ci_lo, ci_hi], "n_sequences": n,
            "relative_ppl_change_pct": float((np.exp(d.mean()) - 1) * 100),
            "relative_ppl_change_ci95_pct": [rel_lo, rel_hi],
            "within_margin_0p5pct": bool(rel_lo >= -0.5 and rel_hi <= 0.5)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-clean", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--tensor", required=True)
    p.add_argument("--n-seq", type=int, default=580)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--sizes", nargs="+", type=int, default=[10000, 50000])
    p.add_argument("--seeds", nargs="+", type=int, default=[20261016, 20261017, 20261018])
    p.add_argument("--conditions", nargs="+", default=["clean", "embedded"])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "nll").mkdir(exist_ok=True)

    windows, n_win = corpus_windows(args.tokenizer_dir, args.n_seq)
    manifest = {
        "run_id": args.run_id, "model": args.model_id, "revision": args.revision,
        "tensor": args.tensor, "method": METHOD, "seqlen": SEQLEN,
        "n_sequences": len(windows), "windows_available": n_win, "batch": args.batch,
        "corpus": {"name": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "test",
                   "file_sha256": sha_file(CORPUS)},
        "equivalence_margin_pct": 0.5,
        "codec_sha256": sha_file(Path(f"{REPO}/src/experiments/nf4_anchor_channel.py")),
    }
    write_json(args.output / "manifest.json", manifest)

    results = {}
    if "clean" in args.conditions:
        cp = args.output / "nll" / "clean.json"
        if cp.exists():
            results["clean"] = json.loads(cp.read_text())
        else:
            r = eval_nll(args.checkpoint_clean, windows, args.batch,
                         ckpt_path=args.output / "nll" / "clean.partial.json")
            r["checkpoint"] = str(args.checkpoint_clean)
            write_json(cp, r); results["clean"] = r
            (args.output / "nll" / "clean.partial.json").unlink(missing_ok=True)
        print(json.dumps({"cond": "clean", "ppl": results["clean"]["ppl"]}), flush=True)

    clean_nll = results.get("clean", {}).get("seq_nll")
    comparisons = []
    # Stage embedded checkpoints on the roomy overlay filesystem, not /workspace.
    staging = Path(os.environ.get("NES_UTILITY_STAGING", "/tmp/nes_stage/utility_staging")) / args.run_id
    staging.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        for seed in args.seeds:
            name = f"emb_{size}_{seed}"
            cp = args.output / "nll" / f"{name}.json"
            if cp.exists():
                r = json.loads(cp.read_text())
            else:
                dest = staging / name
                if not (dest / "model.safetensors").exists():
                    staging.mkdir(exist_ok=True)
                    if dest.exists():
                        shutil.rmtree(dest)
                    stego_sha, clean_sha = materialize_embedded(dest, args.checkpoint_clean, args.tensor, size, seed)
                else:
                    source, _ = load_tensor(args.checkpoint_clean, args.tensor)
                    clean_sha = hashlib.sha256(source.numpy().tobytes()).hexdigest()
                    stego_sha = json.loads((staging / f"{name}.meta.json").read_text())["stego_tensor_sha256"]
                r = eval_nll(dest, windows, args.batch,
                             ckpt_path=args.output / "nll" / f"{name}.partial.json")
                r.update({"size": size, "seed": seed, "stego_tensor_sha256": stego_sha,
                          "clean_tensor_sha256": clean_sha, "checkpoint": "staged_embedded"})
                write_json(cp, r)
                (args.output / "nll" / f"{name}.partial.json").unlink(missing_ok=True)
                shutil.rmtree(dest, ignore_errors=True)
            results[name] = r
            print(json.dumps({"cond": name, "ppl": r["ppl"]}), flush=True)
            if clean_nll is not None:
                pc = paired_ci(clean_nll, r["seq_nll"])
                comparisons.append({"condition": name, "size": size, "seed": seed, **pc,
                                    "clean_ppl": results["clean"]["ppl"], "embedded_ppl": r["ppl"]})
    write_json(args.output / "comparisons.json", {"comparisons": comparisons})
    print(json.dumps({"done": True, "comparisons": len(comparisons)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

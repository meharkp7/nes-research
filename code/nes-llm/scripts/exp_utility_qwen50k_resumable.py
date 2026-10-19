#!/usr/bin/env python3
"""Resumable Qwen 50k utility conditions.

The sandbox restarts periodically and kills long processes. This driver writes
each artifact independently and skips anything already done, so repeated
invocations converge:

  clean_baseline.json                 -> clean seq_nll + ppl (computed once)
  condition_<size>_<seed>.json        -> embedded condition + paired CI
  checkpoints/.../stego_tensor.safetensors + materialization.json

Safe to run many times; never overwrites a completed artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_utility as U

MODEL = "qwen"
SIZE = 50000
SEEDS = (20261016, 20261017, 20261018)


def load_or_none(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def write_once(p, data):
    p = Path(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tok, corpus_ids = U.load_corpus_tokens(U.MODELS[MODEL]["tok"])
    base_path = args.output / "clean_baseline.json"
    base = load_or_none(base_path)
    if base is None:
        ck = U.evaluate_ppl(U.MODELS[MODEL]["path"], U.MODELS[MODEL]["tok"], corpus_ids)
        base = {k: v for k, v in ck.items() if k != "seq_nll"}
        base["seq_nll"] = ck["seq_nll"]
        write_once(base_path, base)
        print(json.dumps({"done": "clean_baseline", "ppl": base["ppl"]}), flush=True)
    clean_nll = base["seq_nll"]

    for seed in SEEDS:
        cond_path = args.output / f"condition_{SIZE}_{seed}.json"
        if cond_path.exists():
            continue
        ckdir = args.output / "checkpoints" / f"{MODEL}_emb{SIZE}_seed{seed}"
        # A checkpoint dir without a completed condition_*.json is a partial
        # artifact from an interrupted run; re-materialize it (deterministic).
        if ckdir.exists() and not cond_path.exists():
            import shutil
            shutil.rmtree(ckdir)
        meta = U.materialize(MODEL, SIZE, seed, ckdir)
        integ = U.integrity_checks(MODEL, SIZE, seed, ckdir)
        ev = U.evaluate_ppl(ckdir, U.MODELS[MODEL]["tok"], corpus_ids)
        pci = U.paired_ci(clean_nll, ev["seq_nll"])
        write_once(cond_path, {
            **{k: v for k, v in ev.items() if k != "seq_nll"},
            "delta_ppl_abs": ev["ppl"] - base["ppl"],
            "delta_ppl_rel_pct": 100 * (ev["ppl"] - base["ppl"]) / base["ppl"],
            "paired_vs_clean": pci, "materialization": meta, "integrity": integ,
            "size": SIZE, "seed": seed, "model": MODEL})
        big = ckdir / "model.safetensors"
        if big.exists():
            big.unlink()
        print(json.dumps({"done": f"{SIZE}_{seed}", "ppl": ev["ppl"],
                          "rel_pct": 100 * (ev["ppl"] - base["ppl"]) / base["ppl"],
                          "exact": integ["exact_recovery"],
                          "only_intended": integ["only_intended_tensor_differs"]}), flush=True)

    done = all((args.output / f"condition_{SIZE}_{s}.json").exists() for s in SEEDS)
    print(json.dumps({"status": "COMPLETE" if done else "PARTIAL"}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

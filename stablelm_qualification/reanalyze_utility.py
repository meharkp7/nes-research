#!/usr/bin/env python3
"""Re-analyze StableLM utility results from saved per-sequence NLL files.

Reads utility_*/nll/*.json (clean + embedded conditions) and recomputes paired
confidence intervals in relative-PPL-change space, testing the predeclared
+/-0.5% equivalence margin. Independent of the run process; makes no model calls.

The margin is defined on relative PPL change: a condition PASSES only if the
paired 95% CI of (embedded/clean - 1) fits entirely inside [-0.5%, +0.5%].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEQLEN = 512


def paired(clean, emb, n_boot=5000, seed=20261016):
    a, b = np.array(clean), np.array(emb)
    assert len(a) == len(b)
    d = (b - a) / (SEQLEN - 1)  # per-token NLL delta
    n = len(d)
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    rel_lo, rel_hi = (np.exp(lo) - 1) * 100, (np.exp(hi) - 1) * 100
    return {
        "mean_delta_per_token_nll": float(d.mean()),
        "ci95_nll": [lo, hi],
        "relative_ppl_change_pct": float((np.exp(d.mean()) - 1) * 100),
        "relative_ppl_change_ci95_pct": [rel_lo, rel_hi],
        "n_sequences": n,
        "within_margin_0p5pct": bool(rel_lo >= -0.5 and rel_hi <= 0.5),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--utility-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    nll_dir = args.utility_dir / "nll"
    clean = json.loads((nll_dir / "clean.json").read_text())
    comparisons = []
    for f in sorted(nll_dir.glob("emb_*.json")):
        r = json.loads(f.read_text())
        pc = paired(clean["seq_nll"], r["seq_nll"])
        comparisons.append({
            "condition": f.stem, "size": r["size"], "seed": r["seed"],
            "clean_ppl": clean["ppl"], "embedded_ppl": r["ppl"],
            "stego_tensor_sha256": r.get("stego_tensor_sha256"), **pc})
    n_within = sum(c["within_margin_0p5pct"] for c in comparisons)
    out = {
        "clean_ppl": clean["ppl"], "n_sequences": clean["n_sequences"],
        "n_tokens_per_condition": clean["n_tokens"],
        "equivalence_margin_pct": 0.5,
        "n_conditions": len(comparisons),
        "n_within_margin": n_within,
        "all_within_margin": bool(comparisons) and n_within == len(comparisons),
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"clean_ppl": clean["ppl"], "n_conditions": len(comparisons),
                      "n_within_margin": n_within}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify utility (A5) results are matched to the correct immutable inputs and
apply the predeclared +/-0.5% equivalence rule exactly. Read-only.

For each model it:
  * checks the run manifest (codec hash, revision, tensor, corpus hash, margin,
    n_seq, batch, seqlen) against the frozen V2 protocol constants;
  * re-derives the stego-tensor sha256 for every embedded condition directly from
    the frozen codec + clean checkpoint and compares it to the value recorded in
    the result (ties the PPL to the exact embedded artifact);
  * checks payload size / seed against the predeclared grid and token counts;
  * applies: PASS iff the paired 95% CI fits entirely inside +/-0.5% relative PPL;
    otherwise INCONCLUSIVE (CI cannot decide) or FAIL (CI excludes 0 and exceeds
    the margin). A CI that merely contains 0 is NOT a pass.

Outputs expansion_qualification/<model>/utility_verification.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/workspace/nes_run_20261016_anchor_ablation")
EXP = ROOT / "expansion_qualification"
CODEC_SHA = "07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764"
CORPUS_SHA = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
SEQLEN, N_SEQ, BATCH, MARGIN = 512, 580, 4, 0.5
SIZES = [10000, 50000]
SEEDS = [20261016, 20261017, 20261018]

MODELS = {
    "smollm2": ("HuggingFaceTB/SmolLM2-1.7B", "effd688a12921b4cc83e3312b6feb579f70f9c71", "ckpt/smollm2_clean", "model.layers.0.self_attn.q_proj.weight"),
    "qwen15": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323", "ckpt/qwen15_clean", "model.layers.0.self_attn.q_proj.weight"),
    "pythia": ("EleutherAI/pythia-1.4b", "fedc38a16eea3bd36a96b906d78d11d2ce18ed79", "/tmp/nes_stage/pythia_clean", "gpt_neox.layers.0.attention.query_key_value.weight"),
    "falcon": ("tiiuae/Falcon3-1B-Base", "cb37ef3559b157b5c9d9226296ba01a5162da1f7", "/tmp/nes_stage/falcon_clean", "model.layers.0.self_attn.q_proj.weight"),
}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _payload_for(bits, seed):
    import numpy as np
    return np.random.default_rng(seed).bytes(bits // 8)


def _key_for(seed):
    return hashlib.sha256(f"NES-development-{seed}".encode()).digest()


def derive_stego_sha(ckpt, tensor, bits, seed):
    """Reproduce exactly what exp_utility.materialize_embedded embeds."""
    sys.path.insert(0, str(ROOT / "code/nes-llm"))
    sys.path.insert(0, str(ROOT / "code/nes-llm/scripts"))
    from scripts.run_nf4_anchor_pilot import load_tensor
    from src.experiments.nf4_anchor_channel import embed
    source, state = load_tensor(ckpt, tensor)
    raw = source.numpy().tobytes()
    cb = state.code.float().numpy()
    stego, _ = embed(raw, _payload_for(bits, seed), _key_for(seed), tensor, "interior_nearest", cb)
    return hashlib.sha256(stego).hexdigest()


def decide(ci):
    lo, hi = ci
    if lo >= -MARGIN and hi <= MARGIN:
        return "PASS"
    if lo <= 0 <= hi:
        return "INCONCLUSIVE"  # CI straddles 0 but is wider than the margin
    return "FAIL"


def verify_model(mk):
    mid, rev, ckpt, tensor = MODELS[mk]
    base = EXP / mk / "utility_v1"
    out = {"model": mid, "revision": rev, "tensor": tensor, "status": "UNVERIFIED",
           "checks": [], "conditions": []}
    man_p, comp_p = base / "manifest.json", base / "comparisons.json"
    if not man_p.exists() or not comp_p.exists():
        out["reason"] = "utility_v1 incomplete (manifest and/or comparisons.json absent)"
        return out
    man = json.loads(man_p.read_text())
    comp = json.loads(comp_p.read_text())

    def chk(name, ok, detail=""):
        out["checks"].append({"check": name, "pass": bool(ok), "detail": detail})

    chk("codec_sha256", man.get("codec_sha256") == CODEC_SHA, str(man.get("codec_sha256")))
    chk("revision", man.get("revision") == rev, str(man.get("revision")))
    chk("model", man.get("model") == mid, str(man.get("model")))
    chk("tensor", man.get("tensor") == tensor, str(man.get("tensor")))
    chk("corpus_sha256", man.get("corpus", {}).get("file_sha256") == CORPUS_SHA)
    chk("seqlen", man.get("seqlen") == SEQLEN, str(man.get("seqlen")))
    chk("n_sequences", man.get("n_sequences") == N_SEQ, str(man.get("n_sequences")))
    chk("batch", man.get("batch") == BATCH, str(man.get("batch")))
    chk("margin", man.get("equivalence_margin_pct") == MARGIN, str(man.get("equivalence_margin_pct")))

    expected = {(s, sd) for s in SIZES for sd in SEEDS}
    seen = set()
    for c in comp.get("comparisons", []):
        cond = {"condition": c["condition"], "size": c["size"], "seed": c["seed"],
                "n_sequences": c.get("n_sequences"), "n_tokens": c.get("n_tokens"),
                "clean_ppl": c["clean_ppl"], "embedded_ppl": c["embedded_ppl"],
                "rel_ppl_change_pct": c["relative_ppl_change_pct"],
                "ci95_pct": c["relative_ppl_change_ci95_pct"],
                "stego_tensor_sha256": c.get("stego_tensor_sha256"),
                "verdict": decide(c["relative_ppl_change_ci95_pct"])}
        seen.add((c["size"], c["seed"]))
        cond["size_seed_predeclared"] = (c["size"], c["seed"]) in expected
        # re-derive artifact hash from the frozen codec
        try:
            derived = derive_stego_sha(ckpt, tensor, c["size"], c["seed"])
            cond["stego_sha_matches_codec"] = (derived == c.get("stego_tensor_sha256"))
            cond["stego_sha_derived"] = derived
        except Exception as e:  # pragma: no cover
            cond["stego_sha_matches_codec"] = None
            cond["stego_sha_error"] = f"{type(e).__name__}: {e}"
        out["conditions"].append(cond)

    out["conditions_expected"] = sorted(expected - seen)
    out["all_conditions_present"] = not (expected - seen)
    out["n_conditions"] = len(out["conditions"])
    out["verdicts"] = {v: sum(1 for c in out["conditions"] if c["verdict"] == v)
                       for v in ("PASS", "INCONCLUSIVE", "FAIL")}
    out["all_artifacts_match_codec"] = all(c.get("stego_sha_matches_codec") for c in out["conditions"])
    out["status"] = "VERIFIED" if (all(c["pass"] for c in out["checks"]) and out["all_conditions_present"]) else "DISCREPANCY"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    summary = {}
    for mk in args.models:
        res = verify_model(mk)
        (EXP / mk).mkdir(parents=True, exist_ok=True)
        (EXP / mk / "utility_verification.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        summary[mk] = {"status": res["status"], "reason": res.get("reason"),
                       "verdicts": res.get("verdicts"),
                       "n_conditions": res.get("n_conditions"),
                       "all_artifacts_match_codec": res.get("all_artifacts_match_codec")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

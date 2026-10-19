#!/usr/bin/env python3
"""Aggregate per-model qualification evidence into an audit table and report.

Reads only existing artifacts produced by the expansion drivers:
  expansion_qualification/<model>/{roundtrip_stream_v1,detectability_v1,utility_v1}
Missing artifacts are reported as UNVERIFIED, never inferred.

Track A (technical generalization) and Track B (novelty/prior-art) stay separate.
Gate G remains PARTIAL unless the tracked evidence genuinely supports otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/workspace/nes_run_20261016_anchor_ablation")
EXP = ROOT / "expansion_qualification"
CODEC = ROOT / "code/nes-llm/src/experiments/nf4_anchor_channel.py"

MODELS = {
    "smollm2": ("HuggingFaceTB/SmolLM2-1.7B", "effd688a12921b4cc83e3312b6feb579f70f9c71", "ckpt/smollm2_clean"),
    "qwen15": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323", "ckpt/qwen15_clean"),
    "pythia": ("EleutherAI/pythia-1.4b", "fedc38a16eea3bd36a96b906d78d11d2ce18ed79", "/tmp/nes_stage/pythia_clean"),
    "falcon": ("tiiuae/Falcon3-1B-Base", "cb37ef3559b157b5c9d9226296ba01a5162da1f7", "/tmp/nes_stage/falcon_clean"),
}
METHODS = ("interior_nearest", "pair_parity", "nearest_parity")


def sha256_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            d.update(c)
    return d.hexdigest()


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def roundtrip_table(model_key):
    recs_dir = EXP / model_key / "roundtrip_stream_v1" / "records"
    if not recs_dir.exists():
        return None
    rows = []
    for f in sorted(recs_dir.glob("*.json")):
        r = json.loads(f.read_text())
        if r["status"] != "DONE":
            rows.append({"condition": r["condition"], "method": r["method"],
                         "tensor": r["tensor"], "bits": r["bits"], "seed": r["seed"],
                         "status": r["status"], "exact": None, "cycles": None,
                         "anchor_changes": None, "evidence": "record"})
            continue
        ex = sum(c["exact_recovery"] for c in r["cycles"])
        rows.append({
            "condition": r["condition"], "method": r["method"], "tensor": r["tensor"],
            "bits": r["bits"], "seed": r["seed"], "status": "DONE",
            "exact": f"{ex}/{len(r['cycles'])}", "cycles": len(r["cycles"]),
            "anchor_changes": sum(c["changed_anchor_codes_vs_source"] for c in r["cycles"]),
            "carrier_agreement": r.get("carrier_agreement"),
            "artifact_hashes": [c["artifact_sha256"][:12] for c in r["cycles"]],
            "evidence": "record"})
    return rows


def detectability_summary(model_key):
    d = load(EXP / model_key / "detectability_v1" / "detectability_results.json")
    if d is None:
        return None
    out = {"n_examples": d["n_examples"], "n_groups": d["n_groups"],
           "control_clean_vs_clean_auc": d["control_clean_vs_clean"]["auc"], "detectors": {}}
    for k, v in d["detectors"].items():
        out["detectors"][k] = {"auc": v["auc"], "auc_ci95": v["auc_ci95"],
                               "tpr_at_fpr_0.01": v["tpr_at_fpr_0.01"],
                               "tpr_at_fpr_0.05": v["tpr_at_fpr_0.05"],
                               "permutation_p": v["permutation_p"]}
    return out


def utility_summary(model_key):
    d = load(EXP / model_key / "utility_v1" / "comparisons.json")
    if d is None:
        return None
    comps = d.get("comparisons", [])
    return {"n_comparisons": len(comps),
            "within_margin_count": sum(c["within_margin_0p5pct"] for c in comps),
            "rows": [{"condition": c["condition"], "size": c["size"], "seed": c["seed"],
                      "clean_ppl": c["clean_ppl"], "embedded_ppl": c["embedded_ppl"],
                      "relative_ppl_change_pct": c["relative_ppl_change_pct"],
                      "ci95_pct": c["relative_ppl_change_ci95_pct"],
                      "within_margin_0p5pct": c["within_margin_0p5pct"]} for c in comps]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    codec_sha = sha256_file(CODEC)
    report = {"codec_sha256": codec_sha,
              "codec_expected": "07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764",
              "codec_frozen_ok": codec_sha == "07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764",
              "models": {}, "track_a": {}, "track_b_status": "PARTIAL (unchanged)",
              "gate_G": "PARTIAL"}

    for mk, (mid, rev, ckpt) in MODELS.items():
        rt = roundtrip_table(mk)
        det = detectability_summary(mk)
        util = utility_summary(mk)
        entry = {"model_id": mid, "revision": rev, "checkpoint": ckpt}
        if rt is None:
            entry["roundtrip"] = "UNVERIFIED (no records directory)"
        else:
            by = defaultdict(lambda: [0, 0, 0])  # exact, total, anchor_changes
            for r in rt:
                if r["status"] != "DONE":
                    by[r["method"]][1] += 0
                    continue
                e, t = map(int, r["exact"].split("/"))
                by[r["method"]][0] += e
                by[r["method"]][1] += t
                by[r["method"]][2] += r["anchor_changes"] or 0
            entry["roundtrip"] = {m: {"exact": f"{v[0]}/{v[1]}", "anchor_code_changes": v[2],
                                      "n_conditions": sum(1 for r in rt if r["method"] == m)}
                                  for m, v in sorted(by.items())}
            entry["roundtrip_rows"] = rt
        entry["detectability"] = det if det else "UNVERIFIED (no detectability_v1)"
        entry["utility"] = util if util else "UNVERIFIED (no utility_v1)"
        report["models"][mk] = entry

    (args.output / "expansion_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"codec_frozen_ok": report["codec_frozen_ok"],
                      "models": {k: v["roundtrip"] for k, v in report["models"].items()}}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

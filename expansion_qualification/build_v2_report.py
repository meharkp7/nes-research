#!/usr/bin/env python3
"""Build the per-model Track A gate matrix for EXPANSION_QUALIFICATION_REPORT_V2.

Reads only inspectable artifacts. Emits gate_matrix.json plus a markdown table.
A gate with no artifact is reported NOT COMPLETE (never inferred).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/workspace/nes_run_20261016_anchor_ablation")
EXP = ROOT / "expansion_qualification"
CODEC_SHA = "07ce008dcabebb7ac0d4fa10c88251bf10dc9f2a7ae27dfc79bf23b8cd75b764"

MODELS = [
    ("smollm2", "HuggingFaceTB/SmolLM2-1.7B", "effd688a12921b4cc83e3312b6feb579f70f9c71"),
    ("qwen15", "Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
    ("pythia", "EleutherAI/pythia-1.4b", "fedc38a16eea3bd36a96b906d78d11d2ce18ed79"),
    ("falcon", "tiiuae/Falcon3-1B-Base", "cb37ef3559b157b5c9d9226296ba01a5162da1f7"),
]


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def gate_matrix(mk, mid, rev):
    g = {"model": mid, "revision": rev, "gates": {}}

    rt = load(EXP / mk / "roundtrip_stream_v1" / "summary.json")
    if rt:
        ok = (rt.get("n_done") == rt.get("n_conditions") and rt.get("n_embed_failed") == 0
              and rt.get("codec_sha256") == CODEC_SHA)
        g["gates"]["A2_A3_roundtrip"] = {
            "status": "PASS" if ok else "FAIL",
            "detail": f"n_done={rt.get('n_done')}/{rt.get('n_conditions')} "
                      f"embed_failed={rt.get('n_embed_failed')} cycle_extractions={rt.get('n_cycle_extractions')}"}
    else:
        g["gates"]["A2_A3_roundtrip"] = {"status": "NOT COMPLETE"}

    integ = load(EXP / mk / "integrity.json")
    if integ:
        ok = (integ.get("only_intended_tensor_differs") and integ.get("shape_preserved")
              and integ.get("payload_exact_recovery") and integ.get("finite_logits_and_loss"))
        g["gates"]["A1_integrity"] = {"status": "PASS" if ok else "FAIL",
                                      "detail": f"n_tensors={integ.get('n_tensors')}"}
    else:
        g["gates"]["A1_integrity"] = {"status": "NOT COMPLETE"}

    reg = load(EXP / mk / "regressions.json")
    if reg:
        g["gates"]["A6_regressions"] = {
            "status": "PASS" if reg.get("n_pass") == reg.get("n_total") else "FAIL",
            "detail": f"{reg.get('n_pass')}/{reg.get('n_total')}"}
    else:
        g["gates"]["A6_regressions"] = {"status": "NOT COMPLETE"}

    det = load(EXP / mk / "detectability_v1" / "detectability_results.json")
    if det:
        aucs = {k: v["auc"] for k, v in det["detectors"].items()}
        tprs = [v["tpr_at_fpr_0.01"] for v in det["detectors"].values()]
        ok = all(a <= 0.55 for a in aucs.values()) and max(tprs) <= 0.02
        g["gates"]["A4_detectability"] = {
            "status": "PASS" if ok else "FAIL",
            "detail": {k: round(v, 4) for k, v in aucs.items()},
            "control_clean_vs_clean_auc": round(det["control_clean_vs_clean"]["auc"], 4),
            "max_tpr_at_1pct": round(max(tprs), 4)}
    else:
        g["gates"]["A4_detectability"] = {"status": "NOT COMPLETE"}

    ver = load(EXP / mk / "utility_verification.json")
    comp = load(EXP / mk / "utility_v1" / "comparisons.json")
    if ver and ver.get("status") == "VERIFIED":
        g["gates"]["A5_utility"] = {"status": "COMPLETE", "verdicts": ver.get("verdicts"),
                                    "n_conditions": ver.get("n_conditions")}
    elif comp:
        g["gates"]["A5_utility"] = {"status": "IN PROGRESS",
                                    "detail": f"{len(comp.get('comparisons', []))}/6 embedded conditions"}
    else:
        g["gates"]["A5_utility"] = {"status": "NOT COMPLETE"}

    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=EXP / "gate_matrix.json")
    args = ap.parse_args()
    matrix = [gate_matrix(mk, mid, rev) for mk, mid, rev in MODELS]
    args.output.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    hdr = ["model", "A1 integrity", "A2/A3 roundtrip", "A4 detectability",
           "A5 utility", "A6 regressions"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for g in matrix:
        row = [g["model"]] + [g["gates"][k]["status"] for k in
                              ("A1_integrity", "A2_A3_roundtrip", "A4_detectability",
                               "A5_utility", "A6_regressions")]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    raise SystemExit(main())

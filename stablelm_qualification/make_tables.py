#!/usr/bin/env python3
"""Emit manuscript-ready tables from frozen StableLM artifacts.

Reads the round-trip records and detectability results and writes CSV/JSON under
a tables directory. Read-only: never modifies source artifacts.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path


def table_exact_recovery(records, outdir):
    rows = []
    for r in records:
        for c in r["cycles"]:
            rows.append({
                "model": "StableLM-2-1.6B", "method": r["method"], "tensor": r["tensor"],
                "bits": r["bits"], "seed": r["seed"], "cycle": c["cycle"],
                "exact_recovery": c["exact_recovery"], "length_ok": c["length_ok"],
                "checksum_ok": c["checksum_ok"],
                "changed_anchor_codes": c["changed_anchor_codes_vs_source"],
                "anchor_codes_delta": c["anchor_codes_delta"],
                "blocks_without_endpoint": c["blocks_without_endpoint"],
                "carrier_agreement": r["carrier_agreement"],
                "artifact_sha256": c["artifact_sha256"],
                "fresh_process": c["fresh_process"],
            })
    path = outdir / "table1_exact_recovery_stablelm.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # aggregate
    agg = defaultdict(lambda: {"n": 0, "exact": 0, "anchor_changed": 0})
    for r in rows:
        k = (r["method"],)
        agg[k]["n"] += 1
        agg[k]["exact"] += int(r["exact_recovery"])
        agg[k]["anchor_changed"] += r["changed_anchor_codes"]
    summary = [{"method": k[0], "cycle_extractions": v["n"], "exact": v["exact"],
                "anchor_codes_changed": v["anchor_changed"]} for k, v in sorted(agg.items())]
    (outdir / "table2_method_aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
    return rows, summary


def table_detectability(detdir, outdir):
    r = json.loads((detdir / "detectability_results.json").read_text())
    rows = []
    for name, v in r["detectors"].items():
        rows.append({"detector": name, "auc": v["auc"],
                     "auc_ci95_lo": v["auc_ci95"][0], "auc_ci95_hi": v["auc_ci95"][1],
                     "tpr_at_fpr_1pct": v["tpr_at_fpr_0.01"],
                     "tpr_at_fpr_5pct": v["tpr_at_fpr_0.05"],
                     "permutation_p": v["permutation_p"], "permutation_n": v["permutation_n"]})
    path = outdir / "table3_detectability_stablelm.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    extra = {"n_examples": r["n_examples"], "n_groups": r["n_groups"],
             "clean_vs_clean_auc": r["control_clean_vs_clean"]["auc"],
             "clean_vs_clean_ci95": r["control_clean_vs_clean"]["auc_ci95"]}
    (outdir / "table3_detectability_meta.json").write_text(json.dumps(extra, indent=2) + "\n")
    return rows, extra


def table_utility(utildir, outdir):
    r = json.loads((utildir / "gate_e_reanalysis.json").read_text())
    rows = []
    for c in r["comparisons"]:
        rows.append({
            "model": "StableLM-2-1.6B", "method": "interior_nearest",
            "tensor": "model.layers.0.self_attn.q_proj.weight",
            "bits": c["size"], "seed": c["seed"], "condition": c["condition"],
            "clean_ppl": c["clean_ppl"], "embedded_ppl": c["embedded_ppl"],
            "relative_ppl_change_pct": c["relative_ppl_change_pct"],
            "ci95_lo_pct": c["relative_ppl_change_ci95_pct"][0],
            "ci95_hi_pct": c["relative_ppl_change_ci95_pct"][1],
            "margin_pct": r["equivalence_margin_pct"],
            "within_margin": c["within_margin_0p5pct"],
            "stego_tensor_sha256": c["stego_tensor_sha256"],
        })
    path = outdir / "table4_utility_stablelm.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    meta = {"clean_ppl": r["clean_ppl"], "n_conditions": r["n_conditions"],
            "n_within_margin": r["n_within_margin"],
            "equivalence_margin_pct": r["equivalence_margin_pct"],
            "n_sequences": r["n_sequences"], "n_tokens_per_condition": r["n_tokens_per_condition"],
            "all_within_margin": r["all_within_margin"]}
    (outdir / "table4_utility_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return rows, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qual-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records = [json.loads(Path(f).read_text())
               for f in glob.glob(str(args.qual_dir / "roundtrip_stream_v1/records/*.json"))]
    rows, summary = table_exact_recovery(records, args.output)
    det_rows, det_meta = table_detectability(args.qual_dir / "detectability_v1", args.output)
    util_rows, util_meta = table_utility(args.qual_dir / "utility_v2", args.output)
    print(json.dumps({"exact_rows": len(rows), "method_summary": summary,
                      "detectors": len(det_rows), "det_meta": det_meta,
                      "utility_rows": len(util_rows), "utility_meta": util_meta}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

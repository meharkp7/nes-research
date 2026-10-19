#!/usr/bin/env python3
"""Emit the per-cycle evidence audit table for the expansion pilot as CSV.

One row per method x model x tensor x payload size x seed x requantization cycle,
matching the operating-contract deliverable. Values come from the on-disk
condition records; nothing is inferred from summary notes. Missing conditions are
not emitted (they would be UNVERIFIED), and the count of expected vs present
conditions is reported to stderr.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXP = Path("/workspace/nes_run_20261016_anchor_ablation/expansion_qualification")
MODELS = ["smollm2", "qwen15", "pythia", "falcon"]
METHODS = ["interior_nearest", "pair_parity", "nearest_parity"]
SIZES = [1000, 5000, 10000, 20000, 50000]
SEEDS = [20261016, 20261017, 20261018]
CYCLES = 4  # cycles 0..3
TENSORS_PER_MODEL = 6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    present = 0
    expected = 0
    for mk in MODELS:
        rdir = EXP / mk / "roundtrip_stream_v1" / "records"
        expected += len(METHODS) * TENSORS_PER_MODEL * len(SIZES) * len(SEEDS)
        if not rdir.exists():
            continue
        for f in sorted(rdir.glob("*.json")):
            r = json.loads(f.read_text())
            present += 1
            if r["status"] != "DONE":
                rows.append({"model": mk, "method": r["method"], "tensor": r["tensor"],
                             "payload_bits": r["bits"], "seed": r["seed"], "cycle": "",
                             "status": r["status"], "exact_recovery": "", "length_ok": "",
                             "checksum_ok": "", "artifact_sha256": "",
                             "changed_anchor_codes": "", "source_revision": r.get("revision", ""),
                             "evidence": "record", "verified": "UNVERIFIED"})
                continue
            for c in r["cycles"]:
                rows.append({
                    "model": mk, "method": r["method"], "tensor": r["tensor"],
                    "payload_bits": r["bits"], "seed": r["seed"], "cycle": c["cycle"],
                    "status": "DONE",
                    "exact_recovery": int(c["exact_recovery"]),
                    "length_ok": int(c["length_ok"]), "checksum_ok": int(c["checksum_ok"]),
                    "artifact_sha256": c["artifact_sha256"],
                    "changed_anchor_codes": c["changed_anchor_codes_vs_source"],
                    "source_revision": "",
                    "evidence": "record", "verified": "VERIFIED",
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "method", "tensor", "payload_bits", "seed", "cycle", "status",
              "exact_recovery", "length_ok", "checksum_ok", "artifact_sha256",
              "changed_anchor_codes", "evidence", "verified"]
    with args.output.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(json.dumps({"rows": len(rows), "conditions_present": present,
                      "conditions_expected": expected,
                      "missing_conditions": expected - present}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

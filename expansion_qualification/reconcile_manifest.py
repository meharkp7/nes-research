#!/usr/bin/env python3
"""Reconcile FROZEN_ALL_HASHES.txt against the current tree, without modifying
anything. Classifies each hashed entry as raw artifact vs narrative/metadata and
emits a machine-readable reconciliation. Read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

ROOT = Path("/workspace/nes_run_20261016_anchor_ablation")
MANIFEST = ROOT / "FROZEN_ALL_HASHES.txt"

RAW_PREFIXES = ("out/",)
RAW_SUBSTR = ("/conditions/", "/checkpoints/", "/nll/")
RAW_EXTS = {".json", ".jsonl", ".npz", ".safetensors", ".pt", ".bin", ".csv", ".log"}
NARRATIVE_EXTS = {".md", ".bib", ".txt", ".rst"}


def classify(p: str) -> str:
    if p.startswith(RAW_PREFIXES) or any(s in p for s in RAW_SUBSTR):
        return "raw_artifact"
    if Path(p).suffix.lower() in NARRATIVE_EXTS:
        return "narrative_metadata"
    if Path(p).suffix.lower() in RAW_EXTS:
        return "raw_artifact"
    return "other"


def main():
    entries, annotations = [], []
    for line in MANIFEST.read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            entries.append((parts[0], parts[1].strip()))
        elif line.strip():
            annotations.append(line.strip())

    rows = []
    for recorded, path in entries:
        f = ROOT / path
        kind = classify(path)
        if not f.exists():
            rows.append({"path": path, "kind": kind, "status": "MISSING",
                         "recorded_sha256": recorded, "current_sha256": None,
                         "mtime": None})
            continue
        cur = hashlib.sha256(f.read_bytes()).hexdigest()
        rows.append({"path": path, "kind": kind,
                     "status": "MATCH" if cur == recorded else "CHANGED",
                     "recorded_sha256": recorded, "current_sha256": cur,
                     "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.stat(f).st_mtime))})

    changed = [r for r in rows if r["status"] == "CHANGED"]
    out = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_self_sha256_recorded": next((e[0] for e in entries if e[1] == "FROZEN_ALL_HASHES.txt"), None),
        "manifest_self_sha256_current": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "n_hashed_entries": len(rows),
        "n_match": sum(r["status"] == "MATCH" for r in rows),
        "n_changed": len(changed),
        "n_missing": sum(r["status"] == "MISSING" for r in rows),
        "annotation_lines": annotations,
        "changed_by_kind": {
            k: sorted(r["path"] for r in changed if r["kind"] == k)
            for k in ("raw_artifact", "narrative_metadata", "other")},
        "changed_raw_artifact_count": sum(r["kind"] == "raw_artifact" for r in changed),
        "rows": rows,
    }
    (ROOT / "expansion_qualification" / "manifest_reconciliation.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in (
        "manifest_self_sha256_recorded", "manifest_self_sha256_current",
        "n_hashed_entries", "n_match", "n_changed", "n_missing",
        "changed_raw_artifact_count", "changed_by_kind")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

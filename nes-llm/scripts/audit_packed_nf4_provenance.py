#!/usr/bin/env python3
"""Audit packed-NF4 feature CSV provenance and matched clean/embedded pairs.

This is a preflight validator, not a detector. It verifies row metadata,
within-pair model/tensor consistency, identical block-index coverage, distinct
artifact identities for clean versus embedded, and enough independent source
groups for a grouped split. Because feature CSVs do not include a verified immutable
model revision, the independent-group count is conservatively based on model_id,
not source_id or tensor_key. This still cannot prove independence from IDs alone.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REQUIRED = (
    "artifact_id", "artifact_sha256", "run_id", "source_id", "model_id",
    "tensor_key", "role", "label", "block_index",
)
ROLES = {"clean": "0", "embedded": "1"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(paths: list[Path], min_sources: int) -> dict:
    pairs: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    source_signatures: dict[str, set[tuple[str, str]]] = defaultdict(set)
    input_info = []
    feature_columns = None
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            missing = sorted(set(REQUIRED) - set(fields))
            if missing:
                raise ValueError(f"{path}: missing required columns: {missing}")
            features = [f for f in fields if f not in REQUIRED]
            if feature_columns is None:
                feature_columns = features
            elif features != feature_columns:
                raise ValueError(f"{path}: feature columns/order mismatch")
            rows = list(reader)
        if not rows:
            raise ValueError(f"{path}: no data rows")
        for line_no, row in enumerate(rows, start=2):
            for field in REQUIRED:
                if not (row.get(field) or "").strip():
                    raise ValueError(f"{path}:{line_no}: empty {field}")
            role = row["role"]
            if role not in ROLES:
                raise ValueError(f"{path}:{line_no}: unexpected role {role!r}")
            if row["label"] != ROLES[role]:
                raise ValueError(f"{path}:{line_no}: role/label mismatch")
            try:
                block_index = int(row["block_index"])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: block_index must be integer") from exc
            if block_index < 0:
                raise ValueError(f"{path}:{line_no}: block_index must be >= 0")
            key = (row["source_id"], row["run_id"])
            role_data = pairs[key].setdefault(role, {
                "model_id": row["model_id"],
                "tensor_key": row["tensor_key"],
                "artifact_ids": set(),
                "artifact_hashes": set(),
                "block_indices": set(),
                "row_count": 0,
            })
            if (role_data["model_id"], role_data["tensor_key"]) != (
                row["model_id"], row["tensor_key"]
            ):
                raise ValueError(f"{key}/{role}: model_id or tensor_key changes within artifact")
            role_data["artifact_ids"].add(row["artifact_id"])
            role_data["artifact_hashes"].add(row["artifact_sha256"])
            role_data["block_indices"].add(block_index)
            role_data["row_count"] += 1
            source_signatures[row["source_id"]].add((row["model_id"], row["tensor_key"]))
        input_info.append({
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "rows": len(rows),
        })

    incomplete = [(key, sorted(data)) for key, data in pairs.items() if set(data) != set(ROLES)]
    if incomplete:
        raise ValueError(f"Missing clean/embedded counterpart; first: {incomplete[0]}")

    pair_summaries = []
    for key, roles in sorted(pairs.items()):
        clean, embedded = roles["clean"], roles["embedded"]
        for role, data in roles.items():
            if len(data["artifact_ids"]) != 1 or len(data["artifact_hashes"]) != 1:
                raise ValueError(f"{key}/{role}: rows refer to multiple artifact identities")
            if len(data["block_indices"]) != data["row_count"]:
                raise ValueError(f"{key}/{role}: duplicate block indices")
        if clean["model_id"] != embedded["model_id"] or clean["tensor_key"] != embedded["tensor_key"]:
            raise ValueError(f"{key}: clean/embedded model_id or tensor_key mismatch")
        if clean["artifact_hashes"] == embedded["artifact_hashes"]:
            raise ValueError(f"{key}: clean and embedded artifact hashes are identical")
        if clean["block_indices"] != embedded["block_indices"]:
            raise ValueError(f"{key}: clean/embedded block-index coverage mismatch")
        pair_summaries.append({
            "source_id": key[0],
            "run_id": key[1],
            "model_id": clean["model_id"],
            "tensor_key": clean["tensor_key"],
            "block_count_per_role": len(clean["block_indices"]),
            "clean_artifact_sha256": next(iter(clean["artifact_hashes"])),
            "embedded_artifact_sha256": next(iter(embedded["artifact_hashes"])),
        })

    # A source_id denotes one source tensor/model identity across all runs.
    inconsistent_sources = {
        source_id: sorted(signatures)
        for source_id, signatures in source_signatures.items()
        if len(signatures) != 1
    }
    if inconsistent_sources:
        first_source = sorted(inconsistent_sources)[0]
        raise ValueError(
            f"source_id {first_source!r} maps to multiple model_id/tensor_key identities: "
            f"{inconsistent_sources[first_source]}"
        )

    source_ids = sorted({key[0] for key in pairs})
    model_ids = sorted({roles["clean"]["model_id"] for roles in pairs.values()})
    if len(model_ids) < min_sources:
        raise ValueError(
            f"Need >= {min_sources} distinct model_id groups for grouped train/validation/test; "
            f"found {len(model_ids)}. Q/K/V tensors, nested/plain conditions, blocks, and repeated runs "
            "from one model_id do not increase the independent group count."
        )
    return {
        "schema": "nes.packed_nf4_provenance_audit.v1",
        "status": "passed",
        "model_group_count": len(model_ids),
        "grouping_key": "model_id",
        "pair_count": len(pair_summaries),
        "min_sources_required": min_sources,
        "source_ids": source_ids,
        "pairs": pair_summaries,
        "input_files": input_info,
        "feature_columns": feature_columns or [],
        "limitations": [
            "Source independence is not inferable from identifiers; provenance must be verified externally.",
            "Passing preflight does not establish detector performance or statistical power.",
            "Use model_id as the conservative split grouping key because immutable model revision is not recorded in feature rows.",
            "Distinct model IDs are necessary but not sufficient proof of independent sources; verify model/revision provenance externally.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="Feature CSV; repeat for every clean/embedded feature artifact")
    parser.add_argument("--min-sources", type=int, default=3)
    parser.add_argument("--output-json", type=Path,
                        help="Optional new JSON report path; refuses overwrite")
    args = parser.parse_args()
    if args.min_sources < 3:
        raise ValueError("--min-sources must be >= 3 for train/validation/test")
    paths = [p.resolve() for p in args.input]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    report = audit(paths, args.min_sources)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        output = args.output_json.resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate artifact/run-level split integrity for packed-NF4 paired datasets.

This checks split bookkeeping and clean/embedded pairing only. It does not
certify detector validity, statistical independence, or absence of all leakage.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REQUIRED = {
    "artifact_sha256", "run_id", "source_id", "tensor_key",
    "role", "split", "block_index",
}
PAIRED_ROLES = {"clean", "embedded"}


def validate_rows(rows: Iterable[dict[str, str]]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("Dataset has no rows")
    missing = REQUIRED - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    errors: list[str] = []
    artifact_splits: dict[str, set[str]] = defaultdict(set)
    run_splits: dict[str, set[str]] = defaultdict(set)
    pair_groups: dict[tuple[str, str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    seen_blocks: set[tuple[str, str, str, str, str, str]] = set()
    pair_artifacts: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)

    for line_no, row in enumerate(rows, start=2):
        artifact = row["artifact_sha256"].strip()
        run = row["run_id"].strip()
        source = row["source_id"].strip()
        tensor = row["tensor_key"].strip()
        role = row["role"].strip().lower()
        split = row["split"].strip().lower()
        block = row["block_index"].strip()
        if not all((artifact, run, source, tensor, role, split, block)):
            errors.append(f"line {line_no}: empty required value")
            continue
        if role not in PAIRED_ROLES:
            errors.append(f"line {line_no}: unexpected role {role!r}")
            continue

        block_key = (source, run, tensor, split, role, block)
        if block_key in seen_blocks:
            errors.append(
                f"line {line_no}: duplicate block row for "
                f"{source}/{run}/{tensor}/{split}/{role}/{block}"
            )
        seen_blocks.add(block_key)

        artifact_splits[artifact].add(split)
        run_splits[run].add(split)
        group_key = (source, run, tensor, split)
        pair_groups[group_key][role].add(block)
        pair_artifacts[group_key + (role,)].add(artifact)

    for artifact, splits in sorted(artifact_splits.items()):
        if len(splits) > 1:
            errors.append(f"artifact {artifact} appears in multiple splits: {sorted(splits)}")
    for run, splits in sorted(run_splits.items()):
        if len(splits) > 1:
            errors.append(f"run {run} appears in multiple splits: {sorted(splits)}")

    for key, roles in sorted(pair_groups.items()):
        clean = roles.get("clean", set())
        embedded = roles.get("embedded", set())
        label = "/".join(key)
        if not clean or not embedded:
            errors.append(f"pair group {label}: missing clean or embedded rows")
        elif clean != embedded:
            only_clean = sorted(clean - embedded)[:5]
            only_embedded = sorted(embedded - clean)[:5]
            errors.append(
                f"pair group {label}: block indices differ; "
                f"clean_only={only_clean}, embedded_only={only_embedded}"
            )
        for role in sorted(PAIRED_ROLES):
            artifacts = pair_artifacts.get(key + (role,), set())
            if len(artifacts) > 1:
                errors.append(
                    f"pair group {label}: role {role} maps to multiple artifacts: "
                    f"{len(artifacts)}"
                )

    return {
        "schema": "nes.packed_nf4_split_validation.v1",
        "rows": len(rows),
        "unique_artifacts": len(artifact_splits),
        "unique_runs": len(run_splits),
        "pair_groups": len(pair_groups),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "scope_note": (
            "Validates artifact/run split exclusivity, duplicate block rows, one artifact "
            "per role/group, and clean/embedded block-index pairing only; it does not "
            "establish statistical independence or detector validity."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Packed-NF4 paired feature CSV")
    parser.add_argument("--output", default="", help="Optional JSON report path")
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    with input_path.open(newline="", encoding="utf-8") as f:
        report = validate_rows(csv.DictReader(f))
    report["input_path"] = str(input_path)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["output_path"] = str(output_path)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

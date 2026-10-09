#!/usr/bin/env python3
"""Assemble block feature CSVs and assign leakage-resistant source-group splits.

This prepares a dataset only; it does not train/evaluate a detector. All rows from one model_id stay in one partition. This is deliberately
conservative because the feature schema does not carry a verified immutable
model-revision identifier: Q/K/V tensors and nested/plain runs from the same
model ID are not independent sources. At least three distinct model IDs are
required; this is a necessary but not sufficient condition for independence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

METADATA = {
    "artifact_id", "artifact_sha256", "run_id", "source_id", "model_id",
    "tensor_key", "role", "label", "block_index",
}
SPLIT_NAMES = ("train", "validation", "test")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        feature_columns = [name for name in reader.fieldnames if name not in METADATA]
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows, feature_columns


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_group_splits(groups: list[str], seed: int) -> dict[str, str]:
    unique = sorted(set(groups))
    if len(unique) < 3:
        raise ValueError(
            f"Need >=3 distinct model_id groups for train/validation/test; "
            f"found {len(unique)}. Blocks and tensors from one model are not independent groups."
        )
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    if n_test + n_val >= n:
        n_test, n_val = 1, 1
    n_train = n - n_test - n_val
    ordered = (
        [(g, "train") for g in unique[:n_train]]
        + [(g, "validation") for g in unique[n_train:n_train+n_val]]
        + [(g, "test") for g in unique[n_train+n_val:]]
    )
    return dict(ordered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="Feature CSV; repeat once per artifact")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-splits", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    output_csv = args.output_csv.resolve()
    output_splits = args.output_splits.resolve()
    if output_csv.exists() or output_splits.exists():
        raise FileExistsError("Refusing to overwrite an existing output")
    if output_csv == output_splits:
        raise ValueError("CSV and split manifest outputs must be different paths")

    all_rows = []
    common_features = None
    input_info = []
    for path_arg in args.input:
        path = path_arg.resolve()
        rows, features = read_csv(path)
        if common_features is None:
            common_features = features
        elif features != common_features:
            raise ValueError(f"Feature columns/order mismatch in {path}")
        all_rows.extend(rows)
        input_info.append({"path": str(path), "sha256": sha256(path), "rows": len(rows)})

    assert common_features is not None
    by_source_role = defaultdict(set)
    by_source_run = defaultdict(set)
    seen_blocks = set()
    for row in all_rows:
        for field in ("source_id", "run_id", "model_id", "tensor_key", "role", "label", "block_index", "artifact_sha256"):
            if not row.get(field):
                raise ValueError(f"Row missing required metadata field {field!r}")
        if row["role"] not in {"clean", "embedded"}:
            raise ValueError(f"Unexpected role: {row['role']!r}")
        expected_label = "0" if row["role"] == "clean" else "1"
        if row["label"] != expected_label:
            raise ValueError(f"Role/label mismatch for {row['role']!r}")
        identity = (row["source_id"], row["run_id"], row["role"], row["block_index"])
        if identity in seen_blocks:
            raise ValueError(f"Duplicate block row: {identity}")
        seen_blocks.add(identity)
        by_source_role[(row["source_id"], row["run_id"])].add(row["role"])
        by_source_run[(row["source_id"], row["run_id"], row["model_id"], row["tensor_key"])].add(row["role"])

    incomplete = [key for key, roles in by_source_role.items() if roles != {"clean", "embedded"}]
    if incomplete:
        raise ValueError(
            "Every source_id/run_id pair must have both clean and embedded rows; "
            f"first incomplete group: {incomplete[0]}"
        )

    model_ids = sorted({row["model_id"] for row in all_rows})
    split_for_model = assign_group_splits(model_ids, args.seed)
    # Defensive check: all rows for a source have the same split by construction.
    seen_split = {}
    for row in all_rows:
        model_id, split = row["model_id"], split_for_model[row["model_id"]]
        if model_id in seen_split and seen_split[model_id] != split:
            raise AssertionError("model group crossed splits")
        seen_split[model_id] = split
        row["split"] = split

    metadata_columns = [
        "artifact_id", "artifact_sha256", "run_id", "source_id", "model_id",
        "tensor_key", "role", "label", "block_index", "split",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metadata_columns + common_features)
        writer.writeheader()
        writer.writerows(all_rows)

    split_manifest = {
        "schema": "nes.packed_nf4_group_splits.v1",
        "grouping_key": "model_id",
        "seed": args.seed,
        "model_group_count": len(model_ids),
        "group_to_split": dict(sorted(split_for_model.items())),
        "row_counts": {
            split: sum(row["split"] == split for row in all_rows)
            for split in SPLIT_NAMES
        },
        "groups_per_split": {
            split: sum(value == split for value in split_for_model.values())
            for split in SPLIT_NAMES
        },
        "input_files": input_info,
        "combined_csv": str(output_csv),
        "feature_columns": common_features,
        "metadata_columns_excluded_from_features": metadata_columns,
        "guardrails": [
            "All rows sharing a model_id stay in one split, including Q/K/V tensors and nested/plain runs.",
            "Blocks and tensors from one model checkpoint are not independent source groups.",
            "model_id grouping is conservative because immutable model revision is not present in this feature schema.",
            "Three distinct model IDs are necessary but not sufficient evidence of source independence.",
        ],
    }
    output_splits.parent.mkdir(parents=True, exist_ok=True)
    with output_splits.open("x", encoding="utf-8") as stream:
        json.dump(split_manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(json.dumps({
        "status": "grouped_dataset_prepared",
        "output_csv": str(output_csv),
        "output_splits": str(output_splits),
        "model_group_count": len(model_ids),
        "groups_per_split": split_manifest["groups_per_split"],
        "row_counts": split_manifest["row_counts"],
        "warning": "Dataset preparation only; no detector trained or evaluated. Groups are model_id, not individual tensors.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

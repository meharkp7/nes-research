import csv
import json
from pathlib import Path

import pytest

from scripts.validate_packed_nf4_detector_readiness import audit_dataset


FIELDS = [
    "artifact_id", "artifact_sha256", "run_id", "source_id", "model_id",
    "tensor_key", "role", "label", "block_index", "split", "entropy",
]


def write_fixture(tmp_path: Path, model_ids=("m1", "m2", "m3", "m4", "m5", "m6", "m7")):
    csv_path = tmp_path / "dataset.csv"
    split_path = tmp_path / "splits.json"
    assignments = {
        model_id: ("train" if i < 3 else "validation" if i == 3 else "test")
        for i, model_id in enumerate(model_ids)
    }
    rows = []
    for model_id, split in assignments.items():
        for role, label in (("clean", "0"), ("embedded", "1")):
            sha = f"{model_id}-{role}-sha"
            for block in ("0", "1"):
                rows.append({
                    "artifact_id": f"{model_id}-{role}",
                    "artifact_sha256": sha,
                    "run_id": f"{model_id}-run",
                    "source_id": f"{model_id}-source",
                    "model_id": model_id,
                    "tensor_key": "layer.q_proj.weight",
                    "role": role,
                    "label": label,
                    "block_index": block,
                    "split": split,
                    "entropy": "3.5",
                })
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    split_path.write_text(json.dumps({
        "schema": "nes.packed_nf4_group_splits.v1",
        "grouping_key": "model_id",
        "group_to_split": assignments,
    }), encoding="utf-8")
    return csv_path, split_path


def test_passes_matched_dataset_with_sufficient_groups(tmp_path):
    dataset, splits = write_fixture(tmp_path)
    report = audit_dataset(dataset, splits)
    assert report["status"] == "PASS"
    assert report["model_groups_per_split"] == {"test": 3, "train": 3, "validation": 1}


def test_blocks_too_few_independent_test_models(tmp_path):
    dataset, splits = write_fixture(tmp_path, model_ids=("m1", "m2", "m3"))
    report = audit_dataset(dataset, splits)
    assert report["status"] == "BLOCKED"
    assert any("insufficient independent model groups in test" in e for e in report["errors"])


def test_blocks_model_crossing_splits(tmp_path):
    dataset, splits = write_fixture(tmp_path)
    rows = list(csv.DictReader(dataset.open()))
    rows[0]["split"] = "test"
    with dataset.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = audit_dataset(dataset, splits)
    assert report["status"] == "BLOCKED"
    assert any("disagrees with manifest" in e for e in report["errors"])


def test_blocks_unpaired_or_mismatched_block_rows(tmp_path):
    dataset, splits = write_fixture(tmp_path)
    rows = list(csv.DictReader(dataset.open()))
    rows = [row for row in rows if not (
        row["model_id"] == "m1" and row["role"] == "embedded" and row["block_index"] == "1"
    )]
    with dataset.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = audit_dataset(dataset, splits)
    assert report["status"] == "BLOCKED"
    assert any("block-index mismatch" in e for e in report["errors"])


def test_blocks_artifact_sha_reused_for_both_classes(tmp_path):
    dataset, splits = write_fixture(tmp_path)
    rows = list(csv.DictReader(dataset.open()))
    for row in rows:
        if row["model_id"] == "m1" and row["role"] == "embedded":
            row["artifact_sha256"] = "m1-clean-sha"
    with dataset.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = audit_dataset(dataset, splits)
    assert report["status"] == "BLOCKED"
    assert any("same artifact SHA appears as clean and embedded" in e for e in report["errors"])

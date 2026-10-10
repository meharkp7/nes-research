import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_packed_nf4_grouped_dataset.py"


def write_feature_csv(path: Path, source_id: str, role: str, run_id: str = "run-1"):
    label = "0" if role == "clean" else "1"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "artifact_id", "artifact_sha256", "run_id", "source_id",
                "model_id", "tensor_key", "role", "label", "block_index",
                "entropy_bits",
            ],
        )
        writer.writeheader()
        for index in range(2):
            writer.writerow({
                "artifact_id": f"{source_id}-{role}",
                "artifact_sha256": (source_id[0] * 64),
                "run_id": run_id,
                "source_id": source_id,
                "model_id": f"test-model-{source_id}",
                "tensor_key": "layer.weight",
                "role": role,
                "label": label,
                "block_index": index,
                "entropy_bits": index + (role == "embedded"),
            })


def invoke(tmp_path: Path, inputs: list[Path]):
    out_csv = tmp_path / "dataset.csv"
    out_splits = tmp_path / "splits.json"
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-csv", str(out_csv),
        "--output-splits", str(out_splits),
        "--seed", "7",
    ]
    for item in inputs:
        cmd.extend(["--input", str(item)])
    return subprocess.run(cmd, text=True, capture_output=True), out_csv, out_splits


def test_grouped_dataset_keeps_source_groups_intact(tmp_path: Path):
    inputs = []
    for source in ("A", "B", "C", "D", "E", "F"):
        for role in ("clean", "embedded"):
            path = tmp_path / f"{source}_{role}.csv"
            write_feature_csv(path, source, role)
            inputs.append(path)

    result, out_csv, out_splits = invoke(tmp_path, inputs)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(out_splits.read_text())
    assert manifest["model_group_count"] == 6
    assert set(manifest["groups_per_split"].values()) >= {1}
    assert sum(manifest["row_counts"].values()) == 24

    with out_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_source.values())


def test_requires_three_independent_source_groups(tmp_path: Path):
    inputs = []
    for source in ("A", "B"):
        for role in ("clean", "embedded"):
            path = tmp_path / f"{source}_{role}.csv"
            write_feature_csv(path, source, role)
            inputs.append(path)
    result, _, _ = invoke(tmp_path, inputs)
    assert result.returncode != 0
    assert "Need >=3 distinct model_id groups" in result.stderr


def test_missing_paired_role_is_rejected(tmp_path: Path):
    inputs = []
    for source in ("A", "B", "C"):
        path = tmp_path / f"{source}_clean.csv"
        write_feature_csv(path, source, "clean")
        inputs.append(path)
    result, _, _ = invoke(tmp_path, inputs)
    assert result.returncode != 0
    assert "must have both clean and embedded rows" in result.stderr



def test_mismatched_clean_embedded_block_indices_are_rejected(tmp_path: Path):
    inputs = []
    for source in ("A", "B", "C"):
        for role in ("clean", "embedded"):
            path = tmp_path / f"{source}_{role}.csv"
            write_feature_csv(path, source, role)
            if source == "A" and role == "embedded":
                with path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                rows.pop()
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            inputs.append(path)
    result, _, _ = invoke(tmp_path, inputs)
    assert result.returncode != 0
    assert "Clean/embedded block-index mismatch" in result.stderr


def test_multiple_artifact_hashes_per_pair_role_are_rejected(tmp_path: Path):
    inputs = []
    for source in ("A", "B", "C"):
        for role in ("clean", "embedded"):
            path = tmp_path / f"{source}_{role}.csv"
            write_feature_csv(path, source, role)
            if source == "A" and role == "clean":
                with path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                rows[1]["artifact_sha256"] = "f" * 64
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            inputs.append(path)
    result, _, _ = invoke(tmp_path, inputs)
    assert result.returncode != 0
    assert "Expected exactly one artifact_sha256" in result.stderr

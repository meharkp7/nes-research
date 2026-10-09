import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_packed_nf4_provenance.py"
FIELDS = [
    "artifact_id", "artifact_sha256", "run_id", "source_id", "model_id",
    "tensor_key", "role", "label", "block_index", "entropy_bits",
]


def write_csv(path: Path, source: str, role: str, *, run: str = "run-1",
              blocks=(0, 1), model: str = "model-a", tensor: str = "layer.weight",
              digest: str | None = None):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index in blocks:
            writer.writerow({
                "artifact_id": f"{source}-{role}",
                "artifact_sha256": digest or (("a" if role == "clean" else "b") * 64),
                "run_id": run,
                "source_id": source,
                "model_id": model,
                "tensor_key": tensor,
                "role": role,
                "label": "0" if role == "clean" else "1",
                "block_index": index,
                "entropy_bits": index,
            })


def invoke(tmp_path: Path, inputs: list[Path], *extra: str):
    command = [sys.executable, str(SCRIPT), "--min-sources", "3"]
    for path in inputs:
        command.extend(["--input", str(path)])
    command.extend(extra)
    return subprocess.run(command, text=True, capture_output=True)


def test_audit_accepts_complete_matched_pairs(tmp_path: Path):
    paths = []
    for source in ("source-a", "source-b", "source-c"):
        for role in ("clean", "embedded"):
            path = tmp_path / f"{source}-{role}.csv"
            write_csv(path, source, role)
            paths.append(path)
    result = invoke(tmp_path, paths)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "passed"
    assert report["source_group_count"] == 3
    assert report["pair_count"] == 3
    assert all(pair["block_count_per_role"] == 2 for pair in report["pairs"])


def test_audit_rejects_missing_counterpart(tmp_path: Path):
    paths = []
    for source in ("a", "b", "c"):
        path = tmp_path / f"{source}-clean.csv"
        write_csv(path, source, "clean")
        paths.append(path)
    result = invoke(tmp_path, paths)
    assert result.returncode != 0
    assert "Missing clean/embedded counterpart" in result.stderr


def test_audit_rejects_block_coverage_mismatch(tmp_path: Path):
    paths = []
    for source in ("a", "b", "c"):
        clean = tmp_path / f"{source}-clean.csv"
        embedded = tmp_path / f"{source}-embedded.csv"
        write_csv(clean, source, "clean", blocks=(0, 1))
        write_csv(embedded, source, "embedded", blocks=(0, 2))
        paths.extend([clean, embedded])
    result = invoke(tmp_path, paths)
    assert result.returncode != 0
    assert "block-index coverage mismatch" in result.stderr


def test_audit_rejects_same_artifact_hash_for_pair(tmp_path: Path):
    paths = []
    for source in ("a", "b", "c"):
        for role in ("clean", "embedded"):
            path = tmp_path / f"{source}-{role}.csv"
            write_csv(path, source, role, digest="c" * 64)
            paths.append(path)
    result = invoke(tmp_path, paths)
    assert result.returncode != 0
    assert "artifact hashes are identical" in result.stderr


def test_audit_rejects_too_few_source_groups(tmp_path: Path):
    paths = []
    for source in ("a", "b"):
        for role in ("clean", "embedded"):
            path = tmp_path / f"{source}-{role}.csv"
            write_csv(path, source, role)
            paths.append(path)
    result = invoke(tmp_path, paths)
    assert result.returncode != 0
    assert "Need >= 3 independent source_id groups" in result.stderr



def test_audit_rejects_source_id_reused_for_different_tensor(tmp_path: Path):
    paths = []
    for source in ("a", "b", "c"):
        for role in ("clean", "embedded"):
            path = tmp_path / f"{source}-{role}.csv"
            tensor = "layer.weight" if source != "a" else (
                "layer.weight" if role == "clean" else "other.weight"
            )
            write_csv(path, source, role, tensor=tensor)
            paths.append(path)
    result = invoke(tmp_path, paths)
    assert result.returncode != 0
    assert "clean/embedded model_id or tensor_key mismatch" in result.stderr


def test_audit_rejects_source_id_reused_across_runs_for_different_tensor(tmp_path: Path):
    paths = []
    for source in ("a", "b", "c"):
        for role in ("clean", "embedded"):
            path = tmp_path / f"{source}-{role}.csv"
            write_csv(path, source, role)
            paths.append(path)
    for role in ("clean", "embedded"):
        path = tmp_path / f"a-run2-{role}.csv"
        write_csv(path, "a", role, run="run-2", tensor="other.weight")
        paths.append(path)
    result = invoke(tmp_path, paths)
    assert result.returncode != 0
    assert "maps to multiple model_id/tensor_key identities" in result.stderr

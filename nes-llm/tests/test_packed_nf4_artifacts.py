from pathlib import Path
import hashlib
import pytest

from src.steganalysis.packed_nf4_artifacts import (
    ArtifactRecord,
    sha256_file,
    validate_matched_pair,
    validate_run_splits,
    write_manifest,
)


def record(**overrides):
    raw = {
        "method_id": "clean" if overrides.get("method_id") == "clean" else "method_a",
        "run_id": "run-001",
        "model_id": "Qwen/Qwen2.5-3B",
        "tensor_key": "model.layers.0.self_attn.q_proj.weight",
        "tensor_shape": [4, 4],
        "quantizer": "bitsandbytes-nf4",
        "quantizer_config": {"blocksize": 64, "double_quant": True},
        "source_id": "source-001",
        "artifact_path": "fixture.bin",
        "artifact_sha256": "a" * 64,
        "receiver_contract": "artifact_only",
        "payload_bits": 8,
        "status": "MEASURED",
    }
    raw.update(overrides)
    return ArtifactRecord.from_dict(raw)


def test_valid_manifest_record_round_trip():
    item = record()
    assert ArtifactRecord.from_dict(item.to_dict()) == item


def test_missing_required_field_is_rejected():
    raw = record().to_dict()
    del raw["tensor_key"]
    with pytest.raises(ValueError, match="Missing artifact fields"):
        ArtifactRecord.from_dict(raw)


def test_run_id_cannot_leak_across_splits():
    with pytest.raises(ValueError, match="appears in both"):
        validate_run_splits({
            "train": ["run-1", "run-2"],
            "validation": ["run-3"],
            "test": ["run-2", "run-4"],
        })


def test_empty_or_duplicate_split_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        validate_run_splits({"train": ["a"], "validation": [], "test": ["c"]})
    with pytest.raises(ValueError, match="Duplicate run ID"):
        validate_run_splits({"train": ["a", "a"], "validation": ["b"], "test": ["c"]})


def test_clean_and_embedded_pair_must_be_matched():
    clean = record(method_id="clean")
    embedded = record(method_id="method_a")
    validate_matched_pair(clean, embedded)
    bad = record(method_id="method_a", tensor_shape=[8, 8])
    with pytest.raises(ValueError, match="tensor_shape"):
        validate_matched_pair(clean, bad)


def test_sha256_streaming(tmp_path: Path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"packed nf4 fixture")
    assert sha256_file(sample) == hashlib.sha256(b"packed nf4 fixture").hexdigest()


def test_manifest_refuses_overwrite(tmp_path: Path):
    target = tmp_path / "manifest.json"
    write_manifest(target, [record()])
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_manifest(target, [record()])

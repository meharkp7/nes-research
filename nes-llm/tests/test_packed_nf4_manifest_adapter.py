import json
from pathlib import Path

import pytest

from scripts.build_packed_nf4_manifest import build_record


def fixtures(tmp_path: Path):
    artifact = tmp_path / "embedded_artifact.pt"
    metadata = tmp_path / "embedded_artifact.json"
    extraction = tmp_path / "extract.json"
    artifact.write_bytes(b"fixture artifact bytes")
    metadata.write_text(json.dumps({
        "schema": "nes.seven_method_nf4_artifact.v1",
        "method": "qse",
        "model_id": "Qwen/Qwen2.5-3B",
        "quantizer": "bitsandbytes NF4",
        "blocksize_requested": 64,
        "payload_bits": 8,
        "receiver_contract": "reference-assisted QSE",
        "tensor_reports": {
            "model.layers.0.self_attn.q_proj.weight": {"num_values": 16}
        },
    }), encoding="utf-8")
    extraction.write_text(json.dumps({
        "schema": "nes.seven_method_nf4_extract.v1",
        "method": "qse",
        "exact_match": True,
        "bit_errors": 0,
        "embedded_manifest_digest_match": True,
    }), encoding="utf-8")
    return artifact, metadata, extraction


def test_builds_measured_reference_assisted_record_and_marks_unknown_revision(tmp_path):
    artifact, metadata, extraction = fixtures(tmp_path)
    record = build_record(
        artifact, metadata, extraction,
        run_id="run-001", tensor_shape=(4, 4),
    )
    assert record.status == "MEASURED"
    assert record.receiver_contract == "reference_assisted"
    assert record.source_id == "Qwen/Qwen2.5-3B@revision-unrecorded"
    assert record.quantizer_config["blocksize_requested"] == 64
    assert record.quantizer_config["double_quant"] is None
    assert len(record.artifact_sha256) == 64


def test_rejects_shape_that_does_not_match_flattened_value_count(tmp_path):
    artifact, metadata, extraction = fixtures(tmp_path)
    with pytest.raises(ValueError, match="has 15 values"):
        build_record(
            artifact, metadata, extraction,
            run_id="run-001", tensor_shape=(3, 5),
        )


def test_rejects_non_exact_extraction(tmp_path):
    artifact, metadata, extraction = fixtures(tmp_path)
    report = json.loads(extraction.read_text())
    report["exact_match"] = False
    extraction.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="exact bit recovery"):
        build_record(
            artifact, metadata, extraction,
            run_id="run-001", tensor_shape=(4, 4),
        )


def test_records_artifact_only_contract_when_reported(tmp_path):
    artifact, metadata, extraction = fixtures(tmp_path)
    report = json.loads(metadata.read_text())
    report["receiver_contract"] = "packed-code parity"
    metadata.write_text(json.dumps(report), encoding="utf-8")
    record = build_record(
        artifact, metadata, extraction,
        run_id="run-001", tensor_shape=(4, 4),
    )
    assert record.receiver_contract == "artifact_only"

#!/usr/bin/env python3
"""Build a provenance-conscious manifest for a measured packed-NF4 artifact.

This adapter intentionally requires the native tensor shape as an explicit
input because the artifact stores flattened values and does not preserve it.
It never reads or emits payload text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.steganalysis.packed_nf4_artifacts import (  # noqa: E402
    ArtifactRecord,
    sha256_file,
    write_manifest,
)


def build_record(
    artifact_path: Path,
    metadata_path: Path,
    extract_path: Path,
    *,
    run_id: str,
    tensor_shape: tuple[int, ...],
    source_revision: str | None = None,
) -> ArtifactRecord:
    """Validate reports and construct one measured record without guessing provenance."""
    artifact_path = artifact_path.resolve()
    metadata_path = metadata_path.resolve()
    extract_path = extract_path.resolve()

    for path in (artifact_path, metadata_path, extract_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    extraction = json.loads(extract_path.read_text(encoding="utf-8"))

    if metadata.get("schema") != "nes.seven_method_nf4_artifact.v1":
        raise ValueError("Unsupported artifact metadata schema")
    if extraction.get("schema") != "nes.seven_method_nf4_extract.v1":
        raise ValueError("Unsupported extraction report schema")
    if metadata.get("method") != extraction.get("method"):
        raise ValueError("Embedding and extraction reports disagree on method")
    if metadata.get("model_id") is None:
        raise ValueError("Artifact metadata is missing model_id")
    if extraction.get("exact_match") is not True or extraction.get("bit_errors") != 0:
        raise ValueError("Extraction report does not establish exact bit recovery")
    if extraction.get("embedded_manifest_digest_match") is not True:
        raise ValueError("Recovered corpus digest does not match embedded metadata")

    reports = metadata.get("tensor_reports")
    if not isinstance(reports, dict) or len(reports) != 1:
        raise ValueError("This adapter requires exactly one reported tensor")
    tensor_key, tensor_report = next(iter(reports.items()))
    num_values = tensor_report.get("num_values")
    shape_product = 1
    for dimension in tensor_shape:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Every tensor-shape dimension must be a positive integer")
        shape_product *= dimension
    if shape_product != num_values:
        raise ValueError(
            f"Provided tensor shape {tensor_shape} has {shape_product} values, "
            f"but artifact metadata reports {num_values}"
        )

    receiver = metadata.get("receiver_contract", "")
    if receiver == "reference-assisted QSE":
        receiver_contract = "reference_assisted"
    elif receiver == "packed-code parity":
        receiver_contract = "artifact_only"
    else:
        raise ValueError(f"Unrecognized receiver contract: {receiver!r}")

    blocksize = metadata.get("blocksize_requested")
    if not isinstance(blocksize, int) or blocksize <= 0:
        raise ValueError("Missing or invalid blocksize_requested")

    # Revision provenance is explicitly marked unknown unless supplied by the caller.
    source_id = (
        f"{metadata['model_id']}@{source_revision}"
        if source_revision
        else f"{metadata['model_id']}@revision-unrecorded"
    )
    return ArtifactRecord.from_dict({
        "method_id": str(metadata["method"]),
        "run_id": run_id,
        "model_id": str(metadata["model_id"]),
        "tensor_key": tensor_key,
        "tensor_shape": list(tensor_shape),
        "quantizer": str(metadata.get("quantizer", "unknown")),
        "quantizer_config": {
            "blocksize_requested": blocksize,
            "double_quant": None,
            "config_completeness": "partial; only blocksize_requested was recorded",
        },
        "source_id": source_id,
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "receiver_contract": receiver_contract,
        "payload_bits": int(metadata["payload_bits"]),
        "status": "MEASURED",
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="embedded_artifact.pt")
    parser.add_argument("--metadata", type=Path, required=True, help="embedded_artifact.json")
    parser.add_argument("--extract", type=Path, required=True, help="extract.json")
    parser.add_argument("--output", type=Path, required=True, help="New manifest JSON path")
    parser.add_argument("--run-id", required=True, help="Independent run identifier")
    parser.add_argument(
        "--tensor-shape", type=int, nargs="+", required=True,
        help="Verified original tensor shape; product must equal num_values",
    )
    parser.add_argument(
        "--source-revision", default=None,
        help="Exact model revision/commit if independently verified; otherwise left unknown",
    )
    args = parser.parse_args()

    record = build_record(
        args.artifact, args.metadata, args.extract,
        run_id=args.run_id,
        tensor_shape=tuple(args.tensor_shape),
        source_revision=args.source_revision,
    )
    write_manifest(args.output, [record])
    print(json.dumps({
        "status": "manifest_written",
        "output": str(args.output.resolve()),
        "records": 1,
        "method_id": record.method_id,
        "run_id": record.run_id,
        "model_id": record.model_id,
        "tensor_key": record.tensor_key,
        "tensor_shape": list(record.tensor_shape),
        "payload_bits": record.payload_bits,
        "receiver_contract": record.receiver_contract,
        "artifact_sha256": record.artifact_sha256,
        "source_id": record.source_id,
        "quantizer_config": dict(record.quantizer_config),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

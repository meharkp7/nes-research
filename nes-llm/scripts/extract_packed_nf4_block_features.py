#!/usr/bin/env python3
"""Extract per-block packed-NF4 features from one explicitly labelled artifact.

Example (run from nes-llm):
../.venv/bin/python scripts/extract_packed_nf4_block_features.py \
  --artifact ../cache/matched_qse_nf4_20261010_r2/clean_nf4.pt \
  --role clean --run-id matched_r2 --source-id qwen25_3b_qproj_l16 \
  --output ../cache/matched_qse_nf4_20261010_r2/clean_block_features.csv

Labels and grouping identifiers are explicit metadata, never inferred from paths
and never included in the numeric feature vector. The script does not train a
detector. Split train/validation/test by independent source/run, not by block.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.steganalysis.packed_nf4_block_features import (  # noqa: E402
    packed_block_feature_rows,
)

QSE_TENSOR_KEY = "model.layers.16.self_attn.q_proj.weight"
ROLE_LABEL = {"clean": 0, "embedded": 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_packed_tensor(path: Path) -> tuple[dict, str, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError("Artifact root must be a dictionary")
    schema = obj.get("schema")
    metadata = obj.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Artifact is missing a metadata dictionary")

    if schema == "nes.clean_nf4_control.v1":
        tensor = obj.get("tensor")
        if not isinstance(tensor, dict):
            raise ValueError("Clean NF4 artifact is missing its tensor dictionary")
        tensor_key = metadata.get("tensor_key")
    elif schema == "nes.seven_method_nf4_artifact.v1":
        tensors = obj.get("tensors")
        if not isinstance(tensors, dict) or len(tensors) != 1:
            raise ValueError("Embedded artifact must contain exactly one tensor")
        tensor_key, tensor = next(iter(tensors.items()))
    else:
        raise ValueError(f"Unsupported artifact schema: {schema!r}")

    for key in ("packed_codes", "num_values", "blocksize"):
        if key not in tensor:
            raise ValueError(f"Artifact tensor is missing {key!r}")
    return tensor, str(tensor_key or ""), metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--role", choices=sorted(ROLE_LABEL), required=True)
    parser.add_argument("--run-id", required=True,
                        help="Independent experimental run/group identifier")
    parser.add_argument("--source-id", required=True,
                        help="Shared source tensor identifier for paired controls")
    parser.add_argument("--output", type=Path, required=True,
                        help="New CSV path; existing files are never overwritten")
    parser.add_argument("--blocksize", type=int, default=64)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    output = args.output.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if not args.run_id.strip() or not args.source_id.strip():
        raise ValueError("run-id and source-id must be non-empty")

    tensor, tensor_key, metadata = load_packed_tensor(artifact)
    declared_blocksize = int(tensor["blocksize"])
    if args.blocksize != declared_blocksize:
        raise ValueError(
            f"Requested blocksize {args.blocksize} differs from artifact "
            f"blocksize {declared_blocksize}"
        )
    rows = packed_block_feature_rows(
        tensor["packed_codes"], int(tensor["num_values"]), args.blocksize
    )
    artifact_digest = sha256_file(artifact)
    model_id = str(metadata.get("model_id") or metadata.get("model") or "unknown")
    feature_names = list(rows[0].keys())
    metadata_fields = [
        "artifact_id", "artifact_sha256", "run_id", "source_id", "model_id",
        "tensor_key", "role", "label", "block_index",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metadata_fields + feature_names)
        writer.writeheader()
        for block_index, features in enumerate(rows):
            writer.writerow({
                "artifact_id": artifact.stem,
                "artifact_sha256": artifact_digest,
                "run_id": args.run_id,
                "source_id": args.source_id,
                "model_id": model_id,
                "tensor_key": tensor_key,
                "role": args.role,
                "label": ROLE_LABEL[args.role],
                "block_index": block_index,
                **features,
            })

    summary = {
        "schema": "nes.packed_nf4_block_features.v1",
        "output": str(output),
        "artifact_sha256": artifact_digest,
        "artifact_schema": metadata.get("schema", ""),
        "model_id": model_id,
        "tensor_key": tensor_key,
        "run_id": args.run_id,
        "source_id": args.source_id,
        "role": args.role,
        "label": ROLE_LABEL[args.role],
        "logical_code_count": int(tensor["num_values"]),
        "blocksize": args.blocksize,
        "block_rows": len(rows),
        "feature_columns": feature_names,
        "metadata_columns_excluded_from_features": metadata_fields,
        "warning": (
            "Block rows are clustered within a source tensor/run and are not "
            "independent samples. Hold out entire source_id/run_id groups."
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Manifest validation primitives for packed-NF4 detectability experiments.

This module deliberately does not load model checkpoints or infer labels from
paths. Artifact discovery/format-specific decoding belongs in method adapters;
this module validates the metadata contract shared by all adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import json


VALID_STATES = {"MEASURED", "NOT_RUN", "NOT_AVAILABLE", "NOT_APPLICABLE"}
VALID_CONTRACTS = {"artifact_only", "reference_assisted"}


@dataclass(frozen=True)
class ArtifactRecord:
    method_id: str
    run_id: str
    model_id: str
    tensor_key: str
    tensor_shape: tuple[int, ...]
    quantizer: str
    quantizer_config: Mapping[str, Any]
    source_id: str
    artifact_path: str
    artifact_sha256: str
    receiver_contract: str
    payload_bits: int | None = None
    status: str = "MEASURED"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ArtifactRecord":
        required = (
            "method_id", "run_id", "model_id", "tensor_key", "tensor_shape",
            "quantizer", "quantizer_config", "source_id", "artifact_path",
            "artifact_sha256", "receiver_contract", "status",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Missing artifact fields: {', '.join(missing)}")
        shape = raw["tensor_shape"]
        if not isinstance(shape, (list, tuple)) or not shape or any(
            not isinstance(n, int) or n <= 0 for n in shape
        ):
            raise ValueError("tensor_shape must be a non-empty sequence of positive integers")
        record = cls(
            method_id=str(raw["method_id"]),
            run_id=str(raw["run_id"]),
            model_id=str(raw["model_id"]),
            tensor_key=str(raw["tensor_key"]),
            tensor_shape=tuple(shape),
            quantizer=str(raw["quantizer"]),
            quantizer_config=dict(raw["quantizer_config"]),
            source_id=str(raw["source_id"]),
            artifact_path=str(raw["artifact_path"]),
            artifact_sha256=str(raw["artifact_sha256"]),
            receiver_contract=str(raw["receiver_contract"]),
            payload_bits=raw.get("payload_bits"),
            status=str(raw["status"]),
        )
        record.validate()
        return record

    def validate(self) -> None:
        for field in (
            "method_id", "run_id", "model_id", "tensor_key", "quantizer",
            "source_id", "artifact_path", "artifact_sha256",
        ):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must be non-empty")
        if self.receiver_contract not in VALID_CONTRACTS:
            raise ValueError(f"receiver_contract must be one of {sorted(VALID_CONTRACTS)}")
        if self.status not in VALID_STATES:
            raise ValueError(f"status must be one of {sorted(VALID_STATES)}")
        if self.payload_bits is not None and self.payload_bits < 0:
            raise ValueError("payload_bits must be non-negative or null")
        if self.status == "MEASURED" and len(self.artifact_sha256) != 64:
            raise ValueError("measured artifacts require a 64-character SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["tensor_shape"] = list(self.tensor_shape)
        result["quantizer_config"] = dict(self.quantizer_config)
        return result


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally to avoid loading large artifacts into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_splits(
    split_to_run_ids: Mapping[str, Iterable[str]],
) -> None:
    """Reject duplicate/empty run IDs across train/validation/test partitions."""
    required = {"train", "validation", "test"}
    missing = required - set(split_to_run_ids)
    if missing:
        raise ValueError(f"Missing split partitions: {', '.join(sorted(missing))}")

    seen: dict[str, str] = {}
    for split in sorted(required):
        run_ids = list(split_to_run_ids[split])
        if not run_ids:
            raise ValueError(f"Split '{split}' must contain at least one independent run")
        if len(run_ids) != len(set(run_ids)):
            raise ValueError(f"Duplicate run ID within split '{split}'")
        for run_id in run_ids:
            if not isinstance(run_id, str) or not run_id.strip():
                raise ValueError(f"Split '{split}' contains an empty/invalid run ID")
            if run_id in seen:
                raise ValueError(
                    f"Run ID '{run_id}' appears in both '{seen[run_id]}' and '{split}'"
                )
            seen[run_id] = split


def validate_matched_pair(clean: ArtifactRecord, embedded: ArtifactRecord) -> None:
    """Ensure a clean/stego comparison does not change nuisance variables."""
    fields = (
        "model_id", "tensor_key", "tensor_shape", "quantizer",
        "quantizer_config", "source_id",
    )
    mismatches = [
        field for field in fields
        if getattr(clean, field) != getattr(embedded, field)
    ]
    if mismatches:
        raise ValueError(f"Clean/embedded pair mismatch: {', '.join(mismatches)}")
    if clean.run_id != embedded.run_id:
        raise ValueError("Clean/embedded pair must share the same independent run_id")
    if clean.method_id == embedded.method_id:
        raise ValueError("Clean and embedded records must have distinct method labels")


def write_manifest(path: str | Path, records: Iterable[ArtifactRecord]) -> None:
    """Write a deterministic JSON manifest; caller chooses a new output path."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {target}")
    rows = [record.to_dict() for record in records]
    rows.sort(key=lambda row: (row["run_id"], row["method_id"], row["tensor_key"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"schema": "nes.packed_nf4_manifest.v1", "records": rows},
                                 indent=2, sort_keys=True) + "\n", encoding="utf-8")

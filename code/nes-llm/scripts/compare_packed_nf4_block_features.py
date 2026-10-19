#!/usr/bin/env python3
"""Descriptive comparison of paired packed-NF4 per-block feature CSVs.

This is not a detector, hypothesis test, or security/stealth claim. It requires
clean and embedded CSVs from the same declared source/run, compares feature
columns only, and writes a new JSON report without overwriting existing files.
Blocks are correlated observations; summaries are descriptive only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


METADATA_COLUMNS = {
    "artifact_id", "artifact_sha256", "run_id", "source_id", "model_id",
    "tensor_key", "role", "label", "block_index",
}


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        features = [name for name in reader.fieldnames if name not in METADATA_COLUMNS]
        if not features:
            raise ValueError(f"No feature columns found in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows, features


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def summarize(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "mean": mean,
        "std_population": statistics.pstdev(values),
        "median": statistics.median(values),
        "q05": quantile(values, 0.05),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
        "q95": quantile(values, 0.95),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--embedded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    clean_rows, clean_features = read_rows(args.clean)
    embedded_rows, embedded_features = read_rows(args.embedded)
    if clean_features != embedded_features:
        raise ValueError("Clean/embedded feature columns or ordering differ")
    if len(clean_rows) != len(embedded_rows):
        raise ValueError("Clean/embedded block row counts differ")

    for label, rows, expected_role, expected_label in (
        ("clean", clean_rows, "clean", "0"),
        ("embedded", embedded_rows, "embedded", "1"),
    ):
        roles = {row.get("role") for row in rows}
        labels = {row.get("label") for row in rows}
        if roles != {expected_role} or labels != {expected_label}:
            raise ValueError(f"{label} CSV role/label mismatch: {roles}, {labels}")

    # Pairing is permitted only for descriptive same-source comparisons.
    for field in ("run_id", "source_id", "model_id", "tensor_key", "block_index"):
        if [r.get(field) for r in clean_rows] != [r.get(field) for r in embedded_rows]:
            raise ValueError(f"Clean/embedded rows are not aligned on {field}")

    results = {}
    for feature in clean_features:
        c = [float(row[feature]) for row in clean_rows]
        e = [float(row[feature]) for row in embedded_rows]
        delta = [b - a for a, b in zip(c, e)]
        c_summary = summarize(c)
        e_summary = summarize(e)
        pooled = math.sqrt((statistics.pvariance(c) + statistics.pvariance(e)) / 2)
        results[feature] = {
            "clean": c_summary,
            "embedded": e_summary,
            "embedded_minus_clean_mean": e_summary["mean"] - c_summary["mean"],
            "paired_delta_median": statistics.median(delta),
            "paired_delta_mean": statistics.fmean(delta),
            "paired_delta_std_population": statistics.pstdev(delta),
            "paired_blocks_with_nonzero_delta": sum(value != 0 for value in delta),
            "standardized_mean_difference_descriptive_only": (
                (e_summary["mean"] - c_summary["mean"]) / pooled
                if pooled > 0 else None
            ),
        }

    artifact = {
        "schema": "nes.packed_nf4_block_feature_comparison.v1",
        "purpose": "Descriptive paired feature summaries only; no trained detector or inferential test",
        "inputs": {
            "clean_csv_sha256": __import__("hashlib").sha256(args.clean.read_bytes()).hexdigest(),
            "embedded_csv_sha256": __import__("hashlib").sha256(args.embedded.read_bytes()).hexdigest(),
            "clean_artifact_sha256": clean_rows[0]["artifact_sha256"],
            "embedded_artifact_sha256": embedded_rows[0]["artifact_sha256"],
            "run_id": clean_rows[0]["run_id"],
            "source_id": clean_rows[0]["source_id"],
            "model_id": clean_rows[0]["model_id"],
            "tensor_key": clean_rows[0]["tensor_key"],
        },
        "block_rows_per_artifact": len(clean_rows),
        "feature_count": len(clean_features),
        "features": results,
        "interpretation_guardrails": [
            "Blocks from one tensor are correlated and are not independent samples.",
            "Paired deltas and standardized differences are descriptive, not p-values or evidence of generalizable detection.",
            "The independent grouping unit remains source tensor/run; do not split blocks across train, validation, and test.",
            "Only one matched clean/embedded pair is compared here.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2, sort_keys=True)
        stream.write("\n")

    ranked = sorted(
        results.items(),
        key=lambda item: abs(item[1]["standardized_mean_difference_descriptive_only"] or 0),
        reverse=True,
    )
    print(json.dumps({
        "status": "descriptive_report_written",
        "output": str(output),
        "block_rows_per_artifact": len(clean_rows),
        "feature_count": len(clean_features),
        "largest_descriptive_standardized_differences": [
            {
                "feature": name,
                "mean_delta": data["embedded_minus_clean_mean"],
                "standardized_mean_difference_descriptive_only":
                    data["standardized_mean_difference_descriptive_only"],
                "blocks_with_nonzero_delta": data["paired_blocks_with_nonzero_delta"],
            }
            for name, data in ranked[:10]
        ],
        "warning": "Not a detector evaluation; block rows are not independent samples.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build an evidence-aware 7-method × 7-model inventory from existing matrix runs.

Read-only with respect to source artifacts: writes only the new inventory JSON/CSV.
Historical matrix statuses are preserved in separate fields from recovery evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RUNS_ROOT = REPO_ROOT / "cache" / "seven_method_long_matrix"
OUTPUT_ROOT = REPO_ROOT / "cache"

MODELS = [
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.3",
    "google/gemma-2-9b",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-4k-instruct",
]
METHODS = [
    "sign", "magnitude_aware", "qae", "lwe_grid_parity",
    "split_sign_parity", "qse", "dce",
]
STATUSES = {"PASS", "BER_FAIL", "EMBED_FAILED", "EXTRACT_FAILED", "BLOCKED"}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidate = (REPO_ROOT / path).resolve()
    return candidate


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_bits(report: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        report.get("payload_bits", report.get("expected_bits")),
        report.get("ber"),
        report.get("exact_match"),
    )


def main() -> int:
    if not RUNS_ROOT.is_dir():
        raise SystemExit(f"Matrix cache not found: {RUNS_ROOT}")

    summaries: list[tuple[str, Path, dict[str, Any]]] = []
    for path in sorted(RUNS_ROOT.glob("*/matrix_summary.json")):
        data = read_json(path)
        if data:
            run_id = str(data.get("run_id_utc") or path.parent.name)
            summaries.append((run_id, path, data))

    # Group every recorded attempt by intended model/method. Corpus variants and
    # repeated runs remain evidence paths, not independent replication counts.
    attempts: dict[tuple[str, str], list[dict[str, Any]]] = {
        (model, method): [] for model in MODELS for method in METHODS
    }
    extraction_evidence: dict[tuple[str, str], list[dict[str, Any]]] = {
        key: [] for key in attempts
    }
    observed_models: set[str] = set()

    for run_id, summary_path, summary in summaries:
        for row in summary.get("results", []):
            model = str(row.get("model", ""))
            method = str(row.get("method", ""))
            if not model or not method:
                continue
            observed_models.add(model)
            key = (model, method)
            if key not in attempts:
                continue
            artifact_path = resolve_path(row.get("artifact"))
            extract_path = resolve_path(row.get("extract_report"))
            if extract_path is None or not extract_path.is_file():
                # Common runner layout: model-slug/corpus/method/extract.json
                if artifact_path:
                    fallback = artifact_path.parent / "extract.json"
                    extract_path = fallback if fallback.is_file() else extract_path
            report = read_json(extract_path) if extract_path else None
            attempts[key].append({
                "run_id": run_id,
                "summary_path": str(summary_path),
                "corpus": row.get("corpus"),
                "matrix_status": row.get("status", "UNKNOWN"),
                "matrix_ber": row.get("ber"),
                "matrix_exact_match": row.get("exact_match"),
                "payload_bits": row.get("payload_bits"),
                "artifact_path": str(artifact_path) if artifact_path else row.get("artifact"),
                "artifact_sha256": sha256_file(artifact_path),
                "extract_report_path": str(extract_path) if extract_path else row.get("extract_report"),
                "extract_report_sha256": sha256_file(extract_path),
                "extract_report": report,
                "reason": row.get("reason"),
            })
            if report:
                bits, ber, exact = report_bits(report)
                extraction_evidence[key].append({
                    "run_id": run_id,
                    "corpus": row.get("corpus"),
                    "report_path": str(extract_path),
                    "report_sha256": sha256_file(extract_path),
                    "payload_bits": bits,
                    "ber": ber,
                    "exact_match": exact,
                    "embedded_manifest_digest_match": report.get("embedded_manifest_digest_match"),
                    "expected_bits": report.get("expected_bits"),
                    "recovered_bits": report.get("recovered_bits"),
                })

    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for method in METHODS:
            key = (model, method)
            found = sorted(attempts[key], key=lambda x: (x["run_id"], str(x.get("corpus"))))
            evidence = extraction_evidence[key]
            latest = found[-1] if found else None
            exact_reports = [item for item in evidence if item.get("exact_match") is True and item.get("ber") == 0]
            failed_reports = [item for item in evidence if item.get("exact_match") is False or (
                isinstance(item.get("ber"), (int, float)) and item["ber"] > 0
            )]
            if exact_reports:
                recovery_status = "VERIFIED_EXACT_RECOVERY"
            elif failed_reports:
                recovery_status = "RECOVERY_FAILURE_REPORTED"
            elif found:
                recovery_status = "UNKNOWN_REPORT_INSUFFICIENT"
            else:
                recovery_status = "NOT_RUN_IN_DISCOVERED_MATRIX_CACHE"

            rows.append({
                "model": model,
                "method": method,
                "matrix_attempt_count": len(found),
                "latest_run_id": latest.get("run_id") if latest else None,
                "latest_corpus": latest.get("corpus") if latest else None,
                "latest_matrix_status": latest.get("matrix_status") if latest else "NOT RUN",
                "latest_matrix_ber": latest.get("matrix_ber") if latest else None,
                "latest_matrix_exact_match": latest.get("matrix_exact_match") if latest else None,
                "recovery_evidence_status": recovery_status,
                "exact_recovery_report_count": len(exact_reports),
                "failure_report_count": len(failed_reports),
                "recovery_payload_bits": exact_reports[-1].get("payload_bits") if exact_reports else None,
                "recovery_ber": exact_reports[-1].get("ber") if exact_reports else (
                    failed_reports[-1].get("ber") if failed_reports else None
                ),
                "artifact_sha256_latest": latest.get("artifact_sha256") if latest else None,
                "latest_artifact_path": latest.get("artifact_path") if latest else None,
                "latest_extract_report_path": latest.get("extract_report_path") if latest else None,
                "evidence_paths": " ; ".join(dict.fromkeys(
                    item["summary_path"] for item in found
                )),
                "note": (
                    "Exact recovery report exists; preserve matrix status separately."
                    if exact_reports and latest and latest.get("matrix_status") != "PASS"
                    else "Repeated attempts are not independent replications."
                ) if found else "No attempt found in discovered long-matrix summaries; check other evidence sources before final NOT RUN classification.",
            })

    stamp = "seven_method_evidence_inventory"
    json_path = OUTPUT_ROOT / f"{stamp}.json"
    csv_path = OUTPUT_ROOT / f"{stamp}.csv"
    payload = {
        "schema": "nes.seven_method_evidence_inventory.v1",
        "purpose": "Inventory of existing long-matrix artifacts only; not a claim of independent replication.",
        "models_intended": MODELS,
        "methods_intended": METHODS,
        "summary_files_found": len(summaries),
        "models_observed_in_summaries": sorted(observed_models),
        "planned_cells": len(MODELS) * len(METHODS),
        "rows": rows,
        "limitations": [
            "This script inventories only cache/seven_method_long_matrix and its referenced reports.",
            "Absence from this cache is not proof that no evidence exists elsewhere.",
            "Historical matrix statuses are never rewritten.",
            "A recovery report is evidence for that artifact/protocol, not independent replication.",
            "Detector, utility, transformation, source revision, and independence status require separate evidence joins/audits.",
            "Artifact hashes are calculated only when the referenced artifact exists at the recorded path.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summaries inventoried: {len(summaries)}")
    print(f"Intended cells: {len(rows)}")
    print(f"Observed model IDs in summaries: {len(observed_models)}")
    print(f"Exact-recovery evidence cells: {sum(r['recovery_evidence_status'] == 'VERIFIED_EXACT_RECOVERY' for r in rows)}")
    print(f"Matrix-status/recovery discrepancies: {sum(r['recovery_evidence_status'] == 'VERIFIED_EXACT_RECOVERY' and r['latest_matrix_status'] != 'PASS' for r in rows)}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

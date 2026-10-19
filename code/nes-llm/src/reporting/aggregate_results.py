"""
Aggregation: turn the manifest into the deliverable tables.

Produces the model x experiment matrix and the six-gate cross-model
table. Every cell keeps its real status; nothing is normalized toward
PASS, and a missing cell is reported as missing rather than omitted.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List

from src.experiments import manifest as manifest_mod
from src.experiments.experiment_registry import TARGET_MODELS
from src.experiments.paths import RESULTS_DIR

MATRIX_JSON = RESULTS_DIR / "experiment_matrix.json"
MATRIX_CSV = RESULTS_DIR / "experiment_matrix.csv"

CROSS_MODEL_JSON = RESULTS_DIR / "cross_model_table_real.json"
CROSS_MODEL_CSV = RESULTS_DIR / "cross_model_table_real.csv"

EXPERIMENT_COLUMNS = [
    "exp1", "exp2", "exp3", "exp4", "exp5",
    "exp6", "exp7", "exp7_neural", "exp8", "exp9",
]

GATE_COLUMNS = [
    "g2_qaci", "g3_ber", "g4_ppl",
    "g5_robustness", "g6_statistical", "g6_neural_detector",
]


def build_matrix(
    manifest: Dict[str, Any],
) -> Dict[str, Dict[str, str]]:
    return manifest_mod.matrix(manifest)


def write_matrix(manifest: Dict[str, Any]) -> Dict[str, Path]:
    """Write the model x experiment matrix to JSON and CSV."""
    matrix = build_matrix(manifest)

    MATRIX_JSON.write_text(
        __import__("json").dumps(
            {"matrix": matrix, "coverage": manifest_mod.coverage(manifest)},
            indent=2,
        ),
        encoding="utf-8",
    )

    experiments = [
        name
        for name in EXPERIMENT_COLUMNS
        if any(name in row for row in matrix.values())
    ]

    with MATRIX_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_id", *experiments])
        for model_id, row in matrix.items():
            writer.writerow(
                [model_id, *[row.get(e, "NOT_RUN") for e in experiments]]
            )

    return {"json": MATRIX_JSON, "csv": MATRIX_CSV}


def read_cross_model() -> Dict[str, Any]:
    """Read the Exp8 table, or an empty dict when absent."""
    if not CROSS_MODEL_JSON.exists():
        return {}

    try:
        return __import__("json").loads(
            CROSS_MODEL_JSON.read_text(encoding="utf-8")
        )
    except Exception:
        return {}


def write_cross_model_csv() -> Path:
    """Flatten the Exp8 JSON table into the deliverable CSV."""
    table = read_cross_model()

    with CROSS_MODEL_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model_id", "family", "expected_layers",
                "actual_layers", "layer_count_matches_expected",
                "overall_status", *GATE_COLUMNS,
            ]
        )

        for model_id, row in table.items():
            writer.writerow(
                [
                    model_id,
                    row.get("family"),
                    row.get("expected_layers"),
                    row.get("actual_layers"),
                    (
                        row.get("actual_layers")
                        == row.get("expected_layers")
                    ),
                    row.get("overall_status"),
                    *[
                        (row.get(gate) or {}).get("status", "NOT_RUN")
                        for gate in GATE_COLUMNS
                    ],
                ]
            )

    return CROSS_MODEL_CSV


def missing_cells(
    manifest: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Which registry targets have no cell for an experiment.

    Used to identify genuinely missing work rather than inferring it from
    the absence of a file.
    """
    matrix = build_matrix(manifest)
    missing: List[Dict[str, str]] = []

    for spec in TARGET_MODELS:
        model_id = spec["model_id"]
        row = matrix.get(model_id, {})

        if not row:
            missing.append(
                {"model_id": model_id, "experiment": "all", "status": "NOT_RUN"}
            )
            continue

        for experiment in EXPERIMENT_COLUMNS:
            if experiment not in row:
                missing.append(
                    {
                        "model_id": model_id,
                        "experiment": experiment,
                        "status": manifest_mod.NOT_RUN,
                    }
                )

    return missing


def aggregate() -> Dict[str, Any]:
    """Build every deliverable table from the current manifest."""
    manifest = manifest_mod.load()

    return {
        "matrix": build_matrix(manifest),
        "coverage": manifest_mod.coverage(manifest),
        "missing_cells": missing_cells(manifest),
        "cross_model": read_cross_model(),
    }
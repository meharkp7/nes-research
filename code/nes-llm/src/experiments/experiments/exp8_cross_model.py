"""
Experiment 8 — Real cross-model table.

This is an orchestration layer, not an independent measurement. It reads
the artifacts produced by Exp1-Exp7 and assembles the cross-model view.
It deliberately does not invent its own version of an earlier experiment
(§24), and it never uses the synthetic ``NESBenchmark`` residuals.

The authoritative orchestrator for real residuals remains
``src/model/exp8_real_cross_model_table.py``. This module runs it and
normalizes its output into the suite's manifest/artifact schema, so the
two paths cannot disagree about what ran.
"""

import json
from typing import Any, Dict, List

from src.experiments.experiment_registry import (
    TARGET_MODELS,
    gate_for,
)
from src.experiments.model_context import ModelContext
from src.experiments.paths import RESULTS_DIR

EXPERIMENT = "exp8"

TABLE_JSON = RESULTS_DIR / "cross_model_table_real.json"
TABLE_CSV = RESULTS_DIR / "cross_model_table_real.csv"

GATE_ORDER = [
    ("g2_qaci", "QACI allocation", "exp1"),
    ("g3_ber", "Clean BER", "exp3"),
    ("g4_ppl", "Real PPL", "exp5"),
    ("g5_robustness", "Robustness", "exp6"),
    ("g6_statistical", "Statistical security", "exp7"),
    ("g6_neural_detector", "Neural detector", "exp7_neural"),
]


def read_table() -> Dict[str, Any]:
    if not TABLE_JSON.exists():
        return {}
    try:
        return json.loads(TABLE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run(
    contexts: List[ModelContext] = None,
    selected_models: List[str] = None,
) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)
    table = read_table()

    if selected_models:
        wanted = set(selected_models)
        table = {
            model: row
            for model, row in table.items()
            if model in wanted
        }

    rows: Dict[str, Any] = {}

    for model_id, raw_row in table.items():
        rows[model_id] = _normalize_row(model_id, raw_row, gate)

    if contexts:
        for context in contexts:
            if context.model_id in rows:
                rows[context.model_id].update(
                    {
                        "actual_layers": context.actual_layers,
                        "layer_count_matches_expected": (
                            context.layer_count_matches_expected
                        ),
                    }
                )

    missing_models = [
        spec["model_id"]
        for spec in TARGET_MODELS
        if spec["model_id"] not in rows
    ]

    all_statuses = [
        row["overall_status"] for row in rows.values()
    ]

    if not rows:
        overall = "NOT_RUN"
    elif "FAIL" in all_statuses:
        overall = "FAIL"
    elif "INCOMPLETE" in all_statuses or missing_models:
        overall = "INCOMPLETE"
    else:
        overall = "PASS"

    metrics = {
        "models_in_table": sorted(rows.keys()),
        "models_missing_from_table": missing_models,
        "overall_status": overall,
        "rows": rows,
        "target_model_count": len(TARGET_MODELS),
        "table_artifact": str(TABLE_JSON),
        "table_csv": str(TABLE_CSV),
    }

    if overall == "FAIL":
        notes = (
            "At least one gate failed on a real completed experiment. "
            "A FAIL here reflects genuine evidence, not a pipeline "
            "error."
        )
    elif overall == "INCOMPLETE":
        notes = (
            "Not every target model has a complete row. Missing cells "
            "are reported as NOT_RUN/MISSING_ARTIFACT and are never "
            "promoted to PASS."
        )
    else:
        notes = "All target models have a complete six-gate row."

    return {
        "experiment": EXPERIMENT,
        "title": "Real Cross-Model Table",
        "configuration": {
            "gates": [key for key, _, _ in GATE_ORDER],
            "residual_source": "real NF4 residuals",
            "synthetic_benchmark_used": False,
            "target_models": [
                spec["model_id"] for spec in TARGET_MODELS
            ],
        },
        "metrics": metrics,
        "thresholds": gate,
        "status": overall,
        "gate_status": overall,
        "reproducibility": {
            "aggregation_only": True,
            "note": (
                "Exp8 aggregates artifacts from Exp1-Exp7; it performs "
                "no independent measurement."
            ),
            "table_generated": TABLE_JSON.stat().st_mtime
            if TABLE_JSON.exists()
            else None,
        },
        "notes": notes,
        "source": "aggregation",
    }


def _normalize_row(
    model_id: str,
    raw_row: Dict[str, Any],
    gate: Dict,
) -> Dict[str, Any]:
    """Turn one raw Exp8 row into the suite's schema.

    Preserves the distinction between a FAIL and a gate that never ran.
    Collapsing NOT_RUN into PASS here would be the exact error the
    handoff calls out as the original problem (§17).
    """
    per_gate = {}

    for key, label, source_experiment in GATE_ORDER:
        cell = raw_row.get(key) or {}
        per_gate[key] = {
            "label": label,
            "source_experiment": source_experiment,
            "status": cell.get("status", "NOT_RUN"),
            "metrics": {
                k: v
                for k, v in cell.items()
                if k != "status"
            },
        }

    statuses = [c["status"] for c in per_gate.values()]

    if "FAIL" in statuses:
        overall = "FAIL"
    elif "NOT_RUN" in statuses:
        overall = "INCOMPLETE"
    else:
        overall = raw_row.get("overall_status", "INCOMPLETE")

    return {
        "model_id": model_id,
        "family": raw_row.get("family"),
        "expected_layers": raw_row.get("expected_layers"),
        "actual_layers": raw_row.get("actual_layers"),
        "layer_count_matches_expected": (
            raw_row.get("actual_layers") == raw_row.get("expected_layers")
            if raw_row.get("actual_layers") is not None
            else None
        ),
        "gates": per_gate,
        "overall_status": overall,
    }
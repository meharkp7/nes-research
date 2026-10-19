"""
Experiment manifest.

The manifest is the single source of truth for what has actually been
run, what passed, what failed, and what is still missing. Every runner
step updates it, and the audit matrix in §3 of the handoff is derived
from it rather than maintained by hand.

Status vocabulary is deliberately narrow and never collapsed (§21,
§25 rules 3-4):

    PASS              ran and met its gate
    FAIL              ran and failed its gate (valid evidence, preserved)
    NOT_RUN           code exists / required artifact genuinely absent
    MISSING_ARTIFACT  expected artifact file does not exist
    IMPLEMENTED_ONLY  code path exists, no experimental evidence
    ERROR             the experiment raised before producing a verdict
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.experiments.artifact_manager import load_json, save_json, utc_now
from src.experiments.paths import RESULTS_DIR

MANIFEST_PATH = RESULTS_DIR / "experiment_manifest.json"

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
MISSING_ARTIFACT = "MISSING_ARTIFACT"
IMPLEMENTED_ONLY = "IMPLEMENTED_ONLY"
ERROR = "ERROR"

# Statuses that represent real experimental evidence.
COMPLETED_STATUSES = {PASS, FAIL}

# Statuses that mean "no verdict was produced".
INCOMPLETE_STATUSES = {
    NOT_RUN,
    MISSING_ARTIFACT,
    IMPLEMENTED_ONLY,
    ERROR,
}


def _key(experiment: str, model_id: str) -> str:
    return f"{experiment}::{model_id}"


def empty() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "description": (
            "NES multi-model experiment manifest. One entry per "
            "experiment x model cell. Status is never inferred from "
            "implementation alone."
        ),
        "records": {},
    }


def load() -> Dict[str, Any]:
    data = load_json(MANIFEST_PATH)

    if data is None or "records" not in data:
        return empty()

    return data


def save(manifest: Dict[str, Any]) -> Path:
    manifest["generated_at"] = utc_now()
    manifest["records"] = manifest.get("records", {})

    # Stable ordering makes diffs reviewable across runs.
    manifest["records"] = dict(
        sorted(manifest["records"].items())
    )

    return save_json(
        MANIFEST_PATH,
        manifest,
        archive_previous=True,
        reason="manifest_update",
    )


def record(
    manifest: Dict[str, Any],
    experiment: str,
    model_id: str,
    family: str,
    status: str,
    gate_status: Optional[str] = None,
    artifact_path: Optional[str] = None,
    configuration: Optional[Dict[str, Any]] = None,
    source: str = "",
    notes: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Insert or update one experiment x model record in the manifest."""
    if status not in (
        COMPLETED_STATUSES | INCOMPLETE_STATUSES
    ):
        raise ValueError(f"Unknown status: {status!r}")

    entry: Dict[str, Any] = {
        "experiment": experiment,
        "model_id": model_id,
        "family": family,
        "status": status,
        "gate_status": gate_status if gate_status is not None else status,
        "artifact_path": artifact_path,
        "artifact_exists": (
            Path(artifact_path).exists() if artifact_path else False
        ),
        "configuration": configuration or {},
        "source": source,
        "notes": notes,
        "updated_at": utc_now(),
    }

    if metrics:
        entry["metrics"] = metrics

    if extra:
        entry.update(extra)

    manifest.setdefault("records", {})[_key(experiment, model_id)] = entry
    return entry


def get(
    manifest: Dict[str, Any],
    experiment: str,
    model_id: str,
) -> Optional[Dict[str, Any]]:
    return manifest.get("records", {}).get(_key(experiment, model_id))


def status_of(
    manifest: Dict[str, Any],
    experiment: str,
    model_id: str,
) -> Optional[str]:
    entry = get(manifest, experiment, model_id)
    return entry["status"] if entry else None


def is_completed(
    manifest: Dict[str, Any],
    experiment: str,
    model_id: str,
) -> bool:
    """True only for PASS or FAIL.

    A FAIL cell is completed work. Re-running it by default would
    overwrite a real measurement, and turning a FAIL into a PASS by
    rerunning until it passes is exactly the behaviour the safety rules
    forbid (§25 rules 3-4).
    """
    return status_of(manifest, experiment, model_id) in COMPLETED_STATUSES


def all_models(manifest: Dict[str, Any]) -> List[str]:
    models = {
        entry["model_id"]
        for entry in manifest.get("records", {}).values()
    }
    return sorted(models)


def all_experiments(manifest: Dict[str, Any]) -> List[str]:
    experiments = {
        entry["experiment"]
        for entry in manifest.get("records", {}).values()
    }
    return sorted(experiments)


def matrix(manifest: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Build the model x experiment status matrix (§22 step 3)."""
    result: Dict[str, Dict[str, str]] = {}

    for entry in manifest.get("records", {}).values():
        model = entry["model_id"]
        experiment = entry["experiment"]
        result.setdefault(model, {})[experiment] = entry["status"]

    return dict(sorted(result.items()))


def coverage(manifest: Dict[str, Any]) -> Dict[str, int]:
    """Counts by status, for the summary report."""
    counts: Dict[str, int] = {
        PASS: 0,
        FAIL: 0,
        NOT_RUN: 0,
        MISSING_ARTIFACT: 0,
        IMPLEMENTED_ONLY: 0,
        ERROR: 0,
    }

    for entry in manifest.get("records", {}).values():
        status = entry.get("status", ERROR)
        counts[status] = counts.get(status, 0) + 1

    return counts
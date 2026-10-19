"""
Experiment 2 — Residual Fingerprint.

Profiles the residual distribution of every layer and preserves the
profile permanently, because it informs later alpha/gamma/payload
decisions and the architecture section of the write-up.

Guide criterion:
    mean mag_mean > 0.002 for at least 80% of layers
"""

import json
from pathlib import Path
from typing import Any, Dict

from src.carrier_intelligence.layer_profiler import LayerProfiler
from src.experiments.artifact_manager import utc_now
from src.experiments.experiment_registry import gate_for
from src.experiments.model_context import ModelContext
from src.experiments.paths import model_slug

EXPERIMENT = "exp2"


def run(context: ModelContext) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)

    if not context.has_residuals:
        return {
            "experiment": EXPERIMENT,
            "title": "Residual Fingerprint",
            "configuration": {},
            "metrics": {},
            "thresholds": gate,
            "status": "NOT_RUN",
            "gate_status": "NOT_RUN",
            "reproducibility": context.reproducibility(),
            "notes": "Residuals unavailable; cannot profile.",
            "source": "run",
        }

    total_layers = context.actual_layers or context.expected_layers
    profiler = LayerProfiler()

    per_layer = []
    for layer_id in sorted(context.residuals.keys()):
        per_layer.append(
            profiler.profile(
                residual=context.residuals[layer_id],
                layer_id=layer_id,
                module_name="mlp.down_proj",
                total_layers=total_layers,
            )
        )

    min_mag = gate["min_mean_magnitude"]
    above = [p for p in per_layer if p["mag_mean"] > min_mag]
    fraction = len(above) / max(len(per_layer), 1)

    metrics: Dict[str, Any] = {
        "num_layers": len(per_layer),
        "min_magnitude_threshold": min_mag,
        "layers_above_threshold": len(above),
        "fraction_above_threshold": fraction,
        "required_fraction": gate["min_fraction_layers_above"],
        "mean_quality": sum(
            p["quality_score"] for p in per_layer
        ) / max(len(per_layer), 1),
        "mean_mag_mean": sum(
            p["mag_mean"] for p in per_layer
        ) / max(len(per_layer), 1),
        "mean_entropy": sum(
            p["entropy"] for p in per_layer
        ) / max(len(per_layer), 1),
        "min_mag_mean": min(p["mag_mean"] for p in per_layer),
        "max_mag_mean": max(p["mag_mean"] for p in per_layer),
    }

    passed = fraction >= gate["min_fraction_layers_above"]

    if passed:
        notes = (
            f"{len(above)}/{len(per_layer)} layers exceed "
            f"mag_mean {min_mag}."
        )
    else:
        notes = (
            f"Only {len(above)}/{len(per_layer)} layers exceed "
            f"mag_mean {min_mag} "
            f"({fraction * 100:.1f}% < "
            f"{gate['min_fraction_layers_above'] * 100:.0f}%). "
            "A FAIL here is a real property of this model's NF4 "
            "residual scale, not a pipeline error."
        )

    return {
        "experiment": EXPERIMENT,
        "title": "Residual Fingerprint",
        "model": context.model_id,
        "family": context.family,
        "num_layers": len(per_layer),
        "mean_quality": metrics["mean_quality"],
        "mean_mag_mean": metrics["mean_mag_mean"],
        "mean_entropy": metrics["mean_entropy"],
        "per_layer": per_layer,
        "configuration": {
            "profiler": "LayerProfiler",
            "num_bins": LayerProfiler().num_bins,
            "module": "mlp.down_proj",
        },
        "metrics": metrics,
        "thresholds": gate,
        "status": "PASS" if passed else "FAIL",
        "gate_status": "PASS" if passed else "FAIL",
        "reproducibility": context.reproducibility(),
        "notes": notes,
        "source": "run",
    }


def legacy_profile_path(context: ModelContext) -> Path:
    """Path of the original top-level ``residual_profile_*.json`` files.

    The handoff §11 names that artifact explicitly. The suite writes the
    standardized per-experiment artifact instead, so this helper lets the
    runner mirror the profile to the historical location without
    maintaining two implementations.
    """
    from src.experiments.paths import REPO_ROOT

    return (
        REPO_ROOT
        / f"residual_profile_{context.family}_{model_slug(context.model_id).split('__')[-1]}.json"
    )


def write_legacy_profile(context: ModelContext, artifact: Dict[str, Any]) -> Path:
    """Mirror the validated profile into the historical filename."""
    profile = {
        "model": artifact.get("model", context.model_id),
        "family": context.family,
        "num_layers": artifact.get("num_layers"),
        "mean_quality": artifact.get("mean_quality"),
        "mean_mag_mean": artifact.get("mean_mag_mean"),
        "mean_entropy": artifact.get("mean_entropy"),
        "per_layer": artifact.get("per_layer"),
        "generated_at": utc_now(),
        "source_artifact": (
            f"results/{EXPERIMENT}_"
            f"{model_slug(context.model_id)}.json"
        ),
    }

    path = legacy_profile_path(context)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return path
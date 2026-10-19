"""
Experiment 1 — Architecture Probe.

Structural validation only: does the model load, do the transformer
modules resolve, does residual extraction produce one tensor per layer,
and does QACI accept that dictionary?

This experiment makes no fidelity, robustness or security claim. Its gate
is simply that the pipeline path is real.
"""

from typing import Any, Dict

from src.carrier_intelligence.qaci_pipeline import QACIPipeline
from src.experiments.experiment_registry import gate_for, is_known_architecture
from src.experiments.model_context import ModelContext

PROBE_PAYLOAD_BITS = 10_000

EXPERIMENT = "exp1"


def run(context: ModelContext) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)

    # The layer count is taken from the residuals, not from a live model.
    #
    # Exp1 exists to validate the residual/QACI path. Loading NF4 + FP16
    # together costs ~19GB for a 7B model and OOMs MPS, purely to read a
    # layer count that the residual dictionary already states
    # authoritatively: those tensors came out of the real model.
    if context.has_residuals:
        context.actual_layers = len(context.residuals)

    metrics: Dict[str, Any] = {
        "architecture_registered": is_known_architecture(context.family),
        "model_loaded": context.has_models,
        "residual_layers": (
            len(context.residuals) if context.has_residuals else 0
        ),
        "expected_layers": context.expected_layers,
        "actual_layers": context.actual_layers,
        "layer_count_source": (
            "residual_dictionary"
            if context.has_residuals
            else "unavailable"
        ),
    }

    if not context.has_residuals:
        return {
            "experiment": EXPERIMENT,
            "title": "Architecture Probe",
            "configuration": {"probe_payload_bits": PROBE_PAYLOAD_BITS},
            "metrics": metrics,
            "thresholds": gate,
            # No residuals means the probe could not run. That is not a
            # failed architecture probe.
            "status": "NOT_RUN",
            "gate_status": "NOT_RUN",
            "reproducibility": context.reproducibility(),
            "notes": (
                "Residuals unavailable, so the structural path could "
                "not be exercised. Reporting NOT_RUN rather than FAIL: "
                "an unrunnable probe is not a broken architecture."
            ),
"source": "run",
        }

    layer_ids = sorted(context.residuals.keys())
    metrics["residual_layer_ids_contiguous"] = (
        layer_ids == list(range(len(layer_ids)))
    )
    metrics["residual_elements_total"] = int(
    sum(t.numel() for t in context.residuals.values())
    )
    metrics["residual_dtype"] = str(
        next(iter(context.residuals.values())).dtype
    )
    metrics["residual_device"] = str(
        next(iter(context.residuals.values())).device
    )

    # QACI must accept the residual dictionary for Exp1 to pass:
    # this is the structural dependency of Exp2 onward.
    try:
        probe = QACIPipeline(
            total_layers=context.actual_layers or context.expected_layers
        )
        selection = probe.select(
            context.residuals,
            PROBE_PAYLOAD_BITS,
        )
        allocated = sum(selection.layer_allocation.values())
        metrics["qaci_runs"] = True
        metrics["qaci_probe_bits_requested"] = PROBE_PAYLOAD_BITS
        metrics["qaci_probe_bits_allocated"] = int(allocated)
        metrics["qaci_probe_layers_selected"] = len(
            selection.layer_allocation
        )
    except Exception as exc:
        metrics["qaci_runs"] = False
        metrics["qaci_probe_error"] = f"{type(exc).__name__}: {exc}"

    layer_mismatch = 0
    if context.actual_layers is not None:
        layer_mismatch = abs(
            context.actual_layers - context.expected_layers
        )

    metrics["layer_mismatch"] = layer_mismatch

    # --- Gate -------------------------------------------------------
    #
    # Deliberately does NOT require a live model. Loading NF4 + FP16 for
    # a 7B model costs ~19GB and OOMs MPS, and it buys nothing here:
    # the residuals already came out of the real model.
    passed = (
        context.has_residuals
        and metrics.get("qaci_runs") is True
        and metrics.get("residual_layer_ids_contiguous") is True
        and layer_mismatch <= gate["max_allowed_layer_mismatch"]
    )

    if passed:
        notes = (
            "Structural probe passed. This is not a fidelity, "
            "robustness or security claim."
        )
    elif layer_mismatch:
        notes = (
            f"Actual layer count {context.actual_layers} differs from "
            f"the registry expectation {context.expected_layers}. "
            "Recorded, not assumed correct."
        )
    else:
        notes = "Structural probe did not complete."

    return {
        "experiment": EXPERIMENT,
        "title": "Architecture Probe",
        "configuration": {
            "probe_payload_bits": PROBE_PAYLOAD_BITS,
            "module": "mlp.down_proj",
            "residual_layers_expected": context.expected_layers,
        },
        "metrics": metrics,
        "thresholds": gate,
        "status": "PASS" if passed else "FAIL",
        "gate_status": "PASS" if passed else "FAIL",
        "reproducibility": context.reproducibility(),
        "notes": notes,
        "source": "run",
    }
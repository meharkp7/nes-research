"""
Experiment 7 — Security.

Deliberately split into two independent sub-experiments that must never
be merged into one number:

    exp7_statistical   KL divergence, sign bias, moment shift, simple
                       statistical detector. Run here.

    exp7_neural        trained neural steganalysis detector. NOT run
                       here.

The split is not bookkeeping. §16 requires statistical security and
neural-adversary security to stay separate, because they answer different
questions and have different failure modes: the statistical gate can pass
while a carrier-centered neural detector still recovers the signal at
70.5% accuracy on this pipeline. Averaging those into a single "security
score" would hide exactly the finding that matters.

The neural sub-experiment is a dataset-and-training cost, so it is a
separate module with its own artifact. This module records it as
NOT_RUN with the reason, rather than silently omitting it.
"""

from typing import Any, Dict

from src.experiments.experiment_registry import gate_for
from src.experiments.experiments.exp3_clean_ber import (
    PAYLOAD_BITS,
    embed_payload,
)
from src.experiments.model_context import ModelContext, seed_everything
from src.steganalysis.security_validator import SecurityValidator

EXPERIMENT = "exp7"

STATISTICAL_EXPERIMENT = "exp7_statistical"
NEURAL_EXPERIMENT = "exp7_neural"

MESSAGE = "A" * 6_000


def run(context: ModelContext) -> Dict[str, Any]:
    gate = gate_for(STATISTICAL_EXPERIMENT)

    if not context.has_residuals:
        return {
            "experiment": EXPERIMENT,
            "title": "Security (statistical)",
            "configuration": {},
            "metrics": {},
            "thresholds": gate,
            "status": "NOT_RUN",
            "gate_status": "NOT_RUN",
            "reproducibility": context.reproducibility(),
            "notes": "Residuals unavailable; cannot measure security.",
            "source": "run",
            "sub_experiments": {
                NEURAL_EXPERIMENT: {
                    "status": "NOT_RUN",
                    "reason": (
                        "Independent neural-adversary experiment; "
                        "never run implicitly by the statistical gate."
                    ),
                }
            },
        }

    seed_everything()

    result = embed_payload(context, PAYLOAD_BITS, MESSAGE)

    validator = SecurityValidator(
        max_kl_divergence=gate["max_kl_divergence"],
        max_detector_accuracy=gate["max_detector_accuracy"],
    )

    outcome = validator.validate(
        original_residuals=context.residuals,
        embedded_residuals=result.embedded_residuals,
        carrier_indices=result.carrier_indices,
    )

    metrics: Dict[str, Any] = {
        "kl_divergence": outcome.kl_divergence,
        "sign_bias": outcome.sign_bias,
        "mean_shift": outcome.moment_shift["mean_shift"],
        "std_shift": outcome.moment_shift["std_shift"],
        "statistical_detector_accuracy": outcome.detector_accuracy,
        "max_kl_divergence": gate["max_kl_divergence"],
        "max_detector_accuracy": gate["max_detector_accuracy"],
        "kl_gate_passed": (
            outcome.kl_divergence <= gate["max_kl_divergence"]
        ),
        "detector_gate_passed": (
            outcome.detector_accuracy <= gate["max_detector_accuracy"]
        ),
        "embedding_distortion": _measure_distortion(
            context.residuals, result.embedded_residuals
        ),
    }

    passed = outcome.passed

    return {
        "experiment": EXPERIMENT,
        "title": "Security (statistical)",
        "configuration": {
            "payload_bits": PAYLOAD_BITS,
            "validator": "SecurityValidator",
            "num_bins": SecurityValidator().num_bins,
            "adversary": (
                "threshold classifier on mean absolute value of "
                "10,000-value samples"
            ),
            "scope": "statistical undetectability only",
        },
        "metrics": metrics,
        "thresholds": gate,
        "status": "PASS" if passed else "FAIL",
        "gate_status": "PASS" if passed else "FAIL",
        "reproducibility": context.reproducibility(),
        "notes": (
            "Statistical undetectability gates hold. This says nothing "
            "about a trained neural adversary; see "
            f"{NEURAL_EXPERIMENT}."
            if passed
            else "Statistical undetectability gates were exceeded."
        ),
        "source": "run",
        "sub_experiments": {
            NEURAL_EXPERIMENT: {
                "status": "NOT_RUN",
                "reason": (
                    "Neural-adversary security is a separate experiment "
                    "with its own dataset and training cost. Passing the "
                    "statistical gate must not be read as passing the "
                    "neural gate."
                ),
            }
        },
    }


def _measure_distortion(
    original: Dict[int, Any],
    embedded: Dict[int, Any],
) -> Dict[str, Any]:
    total_changed = 0
    total_values = 0
    sum_diff = 0.0

    for layer_id, orig in original.items():
        diff = (embedded[layer_id] - orig).abs()
        total_values += diff.numel()
        total_changed += int((diff > 0).sum().item())
        sum_diff += float(diff.sum().item())

    return {
        "total_values": total_values,
        "changed_values": total_changed,
        "changed_ratio": (
            total_changed / total_values if total_values else 0.0
        ),
        "mean_abs_difference_over_changed": (
            sum_diff / total_changed if total_changed else 0.0
        ),
    }
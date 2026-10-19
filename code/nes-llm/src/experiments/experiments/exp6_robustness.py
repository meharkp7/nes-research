"""
Experiment 6 — Robustness under additive Gaussian noise.

Measures BER across the guide's sigma grid and evaluates the two
operating-point gates. Stochastic, so the seed and trial count are
recorded in every artifact (§27).
"""

from typing import Any, Dict, List

from src.evaluation.robustness_validator import RobustnessValidator
from src.experiments.experiment_registry import gate_for
from src.experiments.experiments.exp3_clean_ber import embed_payload
from src.experiments.model_context import ModelContext, seed_everything

EXPERIMENT = "exp6"

SIGMAS: List[float] = [
    0.0, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.020
]

PAYLOAD_BITS = 10_000
MESSAGE = "A" * 1_250
TRIALS = 5


def run(
    context: ModelContext,
    sigmas: List[float] = None,
    trials: int = TRIALS,
    seed: int = 42,
) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)
    sigmas = list(sigmas or SIGMAS)

    if not context.has_residuals:
        return {
            "experiment": EXPERIMENT,
            "title": "Robustness",
            "configuration": {"sigmas": sigmas, "trials": trials},
            "metrics": {},
            "thresholds": gate,
            "status": "NOT_RUN",
            "gate_status": "NOT_RUN",
            "reproducibility": context.reproducibility(),
            "notes": "Residuals unavailable; cannot measure robustness.",
            "source": "run",
        }

    seed_everything(seed)

    result = embed_payload(context, PAYLOAD_BITS, MESSAGE)

    validator = RobustnessValidator(
        max_ber_at_001=gate["max_ber_at_sigma_0_001"],
        max_ber_at_002=gate["max_ber_at_sigma_0_002"],
        num_trials=trials,
    )

    outcome = validator.validate(
        result.embedded_residuals,
        result.carrier_indices,
        result.embedded_bits,
        sigmas,
    )

    metrics: Dict[str, Any] = {
        "ber_curve": {str(k): v for k, v in outcome.ber_curve.items()},
        "ber_at_sigma_0_001": outcome.ber_at_001,
        "ber_at_sigma_0_002": outcome.ber_at_002,
        "max_ber_at_sigma_0_001": gate["max_ber_at_sigma_0_001"],
        "max_ber_at_sigma_0_002": gate["max_ber_at_sigma_0_002"],
        "total_bits": outcome.total_bits,
        "trials": trials,
        "payload_bits": PAYLOAD_BITS,
        "gate_0_001_passed": (
            outcome.ber_at_001 is not None
            and outcome.ber_at_001 <= gate["max_ber_at_sigma_0_001"]
        ),
        "gate_0_002_passed": (
            outcome.ber_at_002 is not None
            and outcome.ber_at_002 <= gate["max_ber_at_sigma_0_002"]
        ),
    }

    passed = outcome.passed

    return {
        "experiment": EXPERIMENT,
        "title": "Robustness",
        "configuration": {
            "sigmas": sigmas,
            "trials_per_sigma": trials,
            "payload_bits": PAYLOAD_BITS,
            "noise": "additive i.i.d. Gaussian on all residual tensors",
            "seed": seed,
        },
        "metrics": metrics,
        "thresholds": gate,
        "status": "PASS" if passed else "FAIL",
        "gate_status": "PASS" if passed else "FAIL",
        "reproducibility": context.reproducibility(),
        "notes": (
            "Both operating-point gates hold. Non-gate sigma values are "
            "stochastic and vary slightly across reruns."
            if passed
            else "At least one operating-point gate was exceeded."
        ),
        "source": "run",
    }
"""
Shared context object passed to every experiment.

Holds the model/residual state so that a suite run loads each model once
instead of once per experiment, and records the reproducibility metadata
required by §27 into every artifact.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from src.experiments.artifact_manager import utc_now
from src.experiments.environment_check import describe
from src.experiments.experiment_registry import family_of, get_model


@dataclass
class ModelContext:
    """One model plus its residuals, loaded at most once per run."""

    model_id: str
    family: str
    expected_layers: int
    actual_layers: Optional[int] = None

    models: Optional[Tuple] = None
    residuals: Optional[Dict[int, torch.Tensor]] = None

    residual_provenance: Dict[str, Any] = field(default_factory=dict)
    model_error: Optional[str] = None

    @property
    def has_models(self) -> bool:
        return self.models is not None

    @property
    def has_residuals(self) -> bool:
        return self.residuals is not None

    @property
    def layer_count_matches_expected(self) -> Optional[bool]:
        if self.actual_layers is None:
            return None
        return self.actual_layers == self.expected_layers

    def note(self, **kwargs: Any) -> Dict[str, Any]:
        self.residual_provenance.update(kwargs)
        return self.residual_provenance

    def reproducibility(self) -> Dict[str, Any]:
        """Metadata every artifact must carry (§27)."""
        env = describe()

        return {
            "model_id": self.model_id,
            "family": self.family,
            "expected_layers": self.expected_layers,
            "actual_layers": self.actual_layers,
            "layer_count_matches_expected": (
                self.layer_count_matches_expected
            ),
            "random_seed": SEED,
            "device": env["device"],
            "software_versions": env["software_versions"],
            "platform_notes": env["platform_notes"],
            "residual_source": self.residual_provenance.get("source"),
            "residual_definition": (
                "R = W_FP16 - dequantize(W_NF4)"
            ),
            "timestamp": utc_now(),
        }

    def release(self) -> None:
        """Drop heavy references and free accelerator memory."""
        self.residuals = None
        self.models = None

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Suite-wide deterministic seed. Individual experiments may override it
# for stochastic work (Exp6, Exp7) but must record what they used.
SEED = 42


def seed_everything(seed: int = SEED) -> int:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def make_context(model_id: str) -> ModelContext:
    spec = get_model(model_id)

    return ModelContext(
        model_id=model_id,
        family=spec["family"] or family_of(model_id),
        expected_layers=int(spec["expected_layers"]),
    )
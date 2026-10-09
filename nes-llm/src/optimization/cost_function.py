"""Cost function for the first Distribution-Constrained Embedding prototype.

The objective separates payload correctness, post-quantization correctness,
a supplied distribution-cost surrogate, and squared perturbation. The
distribution_cost term is deliberately supplied by the caller: a per-candidate
proxy is not the same thing as a measured batch-level KL divergence, and must
not be reported as such.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CostWeights:
    """Non-negative weights for the DCE objective."""

    payload_error: float = 1_000.0
    post_quantization_error: float = 1_000.0
    distribution: float = 1.0
    perturbation: float = 1.0

    def __post_init__(self):
        for name, value in (
            ("payload_error", self.payload_error),
            ("post_quantization_error", self.post_quantization_error),
            ("distribution", self.distribution),
            ("perturbation", self.perturbation),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} weight must be finite and >= 0")


@dataclass(frozen=True)
class EmbeddingCost:
    """Observable cost components for one candidate carrier value."""

    payload_error: int
    post_quantization_error: int
    distribution_cost: float
    squared_perturbation: float

    def __post_init__(self):
        if self.payload_error not in (0, 1):
            raise ValueError("payload_error must be 0 or 1")
        if self.post_quantization_error not in (0, 1):
            raise ValueError("post_quantization_error must be 0 or 1")
        for name, value in (
            ("distribution_cost", self.distribution_cost),
            ("squared_perturbation", self.squared_perturbation),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")

    def total(self, weights: CostWeights = CostWeights()) -> float:
        """Return weighted scalar cost; component values remain inspectable."""
        return (
            weights.payload_error * self.payload_error
            + weights.post_quantization_error * self.post_quantization_error
            + weights.distribution * self.distribution_cost
            + weights.perturbation * self.squared_perturbation
        )

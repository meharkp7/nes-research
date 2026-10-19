"""Deterministic candidate generation for the DCE prototype.

A caller supplies the allowed representation/codebook and the actual
quantizer/decoder pair for the experiment. This avoids silently pretending
that a generic rounding function models BitsAndBytes NF4. Candidate-level
distribution costs are optional proxies; global distribution divergence
must be measured separately on the completed tensor/artifact.
"""

from dataclasses import dataclass
import math
from typing import Callable, Iterable, List, Optional

from src.optimization.cost_function import CostWeights, EmbeddingCost


@dataclass(frozen=True)
class Candidate:
    """One possible value for a carrier and its measured objective parts."""

    value: float
    quantized_value: float
    decoded_bit: int
    post_quantized_bit: int
    cost: EmbeddingCost

    def score(self, weights: CostWeights = CostWeights()) -> float:
        return self.cost.total(weights)


class CandidateGenerator:
    """Enumerate allowed values and evaluate them using caller-supplied rules."""

    def __init__(self, max_candidates: Optional[int] = None):
        if max_candidates is not None and max_candidates < 1:
            raise ValueError("max_candidates must be >= 1 or None")
        self.max_candidates = max_candidates

    def generate(
        self,
        cover_value: float,
        target_bit: int,
        candidate_values: Iterable[float],
        quantize: Callable[[float], float],
        decode: Callable[[float], int],
        distribution_cost: Optional[Callable[[float], float]] = None,
    ) -> List[Candidate]:
        """Generate candidates, ordered by distance from the cover value.

        decode(value) must implement the experiment's actual bit-decoding rule.
        quantize(value) must implement the quantizer being evaluated. Both
        functions are invoked for each candidate, before and after quantization.
        """
        if target_bit not in (0, 1):
            raise ValueError("target_bit must be 0 or 1")
        if not math.isfinite(cover_value):
            raise ValueError("cover_value must be finite")

        unique_values = set()
        for value in candidate_values:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("candidate values must be finite")
            unique_values.add(value)
        ordered = sorted(unique_values, key=lambda value: (abs(value - cover_value), value))
        if self.max_candidates is not None:
            ordered = ordered[: self.max_candidates]

        candidates = []
        for value in ordered:
            quantized_value = float(quantize(value))
            if not math.isfinite(quantized_value):
                raise ValueError("quantizer returned a non-finite value")
            decoded_bit = decode(value)
            post_bit = decode(quantized_value)
            if decoded_bit not in (0, 1) or post_bit not in (0, 1):
                raise ValueError("decoder must return bit 0 or 1")
            dist_cost = float(distribution_cost(value)) if distribution_cost else 0.0
            if not math.isfinite(dist_cost) or dist_cost < 0:
                raise ValueError("distribution_cost must return finite value >= 0")
            candidates.append(
                Candidate(
                    value=value,
                    quantized_value=quantized_value,
                    decoded_bit=decoded_bit,
                    post_quantized_bit=post_bit,
                    cost=EmbeddingCost(
                        payload_error=int(decoded_bit != target_bit),
                        post_quantization_error=int(post_bit != target_bit),
                        distribution_cost=dist_cost,
                        squared_perturbation=(value - cover_value) ** 2,
                    ),
                )
            )
        return candidates

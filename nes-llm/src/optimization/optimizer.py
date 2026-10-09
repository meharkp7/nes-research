"""Minimum-cost selection for the DCE candidate prototype."""

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence

from src.optimization.candidate_generator import Candidate
from src.optimization.cost_function import CostWeights


@dataclass(frozen=True)
class OptimizationResult:
    """Selection and auditable objective breakdown."""

    selected: Candidate
    total_cost: float
    candidate_count: int
    weights: CostWeights

    def as_dict(self) -> dict:
        return {
            "selected_value": self.selected.value,
            "selected_quantized_value": self.selected.quantized_value,
            "decoded_bit": self.selected.decoded_bit,
            "post_quantized_bit": self.selected.post_quantized_bit,
            "total_cost": self.total_cost,
            "candidate_count": self.candidate_count,
            "cost_components": {
                "payload_error": self.selected.cost.payload_error,
                "post_quantization_error": self.selected.cost.post_quantization_error,
                "distribution_cost_proxy": self.selected.cost.distribution_cost,
                "squared_perturbation": self.selected.cost.squared_perturbation,
            },
            "weights": {
                "payload_error": self.weights.payload_error,
                "post_quantization_error": self.weights.post_quantization_error,
                "distribution": self.weights.distribution,
                "perturbation": self.weights.perturbation,
            },
        }


class CandidateOptimizer:
    """Choose the minimum weighted-cost candidate with deterministic ties."""

    def __init__(self, weights: CostWeights = CostWeights()):
        self.weights = weights

    def optimize(self, candidates: Sequence[Candidate]) -> OptimizationResult:
        if not candidates:
            raise ValueError("cannot optimize an empty candidate set")
        scored = [(candidate.score(self.weights), candidate.value, candidate) for candidate in candidates]
        score, _, selected = min(scored, key=lambda item: (item[0], item[1]))
        if not math.isfinite(score):
            raise ValueError("candidate objective is not finite")
        return OptimizationResult(
            selected=selected,
            total_cost=score,
            candidate_count=len(candidates),
            weights=self.weights,
        )

    def optimize_many(
        self, candidate_sets: Iterable[Sequence[Candidate]]
    ) -> List[OptimizationResult]:
        """Optimize each carrier independently; does not claim global KL optimality."""
        results = []
        for index, candidates in enumerate(candidate_sets):
            try:
                results.append(self.optimize(candidates))
            except ValueError as exc:
                raise ValueError(f"candidate set {index}: {exc}") from exc
        return results

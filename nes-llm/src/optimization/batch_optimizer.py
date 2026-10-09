"""Greedy batch-level histogram-constrained candidate selection.

Research prototype only. It greedily tracks aggregate quantized-bin counts
while selecting one payload-feasible candidate per carrier. It is not a
globally optimal solver and does not model BitsAndBytes NF4 unless the caller
supplies an appropriate candidate set and quantizer/decoder.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from src.optimization.candidate_generator import Candidate


@dataclass(frozen=True)
class BatchSelection:
    selected: tuple[Candidate, ...]
    target_histogram: dict[float, int]
    selected_histogram: dict[float, int]
    objective_trace: tuple[float, ...]

    @property
    def mean_squared_perturbation(self) -> float:
        if not self.selected:
            return 0.0
        return sum(c.cost.squared_perturbation for c in self.selected) / len(self.selected)

    @property
    def histogram_total_variation_distance(self) -> float:
        if not self.selected:
            return 0.0
        support = set(self.target_histogram) | set(self.selected_histogram)
        n_target = sum(self.target_histogram.values())
        n_selected = len(self.selected)
        if n_target == 0:
            return 0.0
        return 0.5 * sum(
            abs(self.target_histogram.get(k, 0) / n_target
                - self.selected_histogram.get(k, 0) / n_selected)
            for k in support
        )


class BatchDistributionOptimizer:
    """Greedily minimize partial-batch histogram mismatch plus perturbation."""

    def __init__(self, distribution_weight: float = 1.0, perturbation_weight: float = 1.0):
        for name, value in (
            ("distribution_weight", distribution_weight),
            ("perturbation_weight", perturbation_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        self.distribution_weight = distribution_weight
        self.perturbation_weight = perturbation_weight

    def optimize(
        self,
        candidate_sets: Sequence[Sequence[Candidate]],
        target_histogram: Mapping[float, int],
        target_bits: Sequence[int],
    ) -> BatchSelection:
        if len(candidate_sets) != len(target_bits):
            raise ValueError("candidate_sets and target_bits must have equal length")
        if any(bit not in (0, 1) for bit in target_bits):
            raise ValueError("target bits must be 0 or 1")
        if any(not isinstance(count, int) or count < 0 for count in target_histogram.values()):
            raise ValueError("target histogram counts must be non-negative integers")
        total_target = sum(target_histogram.values())
        if candidate_sets and total_target == 0:
            raise ValueError("target histogram must contain at least one carrier")
        if total_target and total_target != len(candidate_sets):
            raise ValueError("target histogram total must equal carrier count")

        target_probs = {
            key: count / total_target for key, count in target_histogram.items()
        } if total_target else {}
        counts: Counter[float] = Counter()
        selected = []
        trace = []
        for index, (candidates, bit) in enumerate(zip(candidate_sets, target_bits), start=1):
            feasible = [
                c for c in candidates
                if c.decoded_bit == bit and c.post_quantized_bit == bit
            ]
            if not feasible:
                raise ValueError(f"candidate set {index - 1} has no payload-feasible candidate")
            support = set(target_probs)
            support.update(c.quantized_value for c in feasible)
            best = None
            best_score = None
            for candidate in feasible:
                projected = counts.copy()
                projected[candidate.quantized_value] += 1
                # Match the cover distribution in expectation for the prefix
                # already assigned; square-count loss is an aggregate objective.
                distribution_loss = sum(
                    (projected.get(key, 0) - index * target_probs.get(key, 0.0)) ** 2
                    for key in support
                )
                score = (
                    self.distribution_weight * distribution_loss
                    + self.perturbation_weight * candidate.cost.squared_perturbation
                )
                tie_key = (score, abs(candidate.value), candidate.value)
                if best_score is None or tie_key < best_score:
                    best_score = tie_key
                    best = candidate
            assert best is not None and best_score is not None
            selected.append(best)
            counts[best.quantized_value] += 1
            trace.append(best_score)
        return BatchSelection(
            selected=tuple(selected),
            target_histogram=dict(target_histogram),
            selected_histogram=dict(counts),
            objective_trace=tuple(trace),
        )

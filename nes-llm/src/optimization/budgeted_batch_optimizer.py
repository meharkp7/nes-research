"""Distortion-budgeted greedy batch histogram matching.

The hard perturbation budget is relative to the nearest-feasible per-carrier
baseline. Candidate selection may improve the aggregate histogram only while
remaining inside that budget; infeasibility is reported, never silently relaxed.
Synthetic prototype only; caller must provide the actual representation,
quantizer, and decoder for any real-model experiment.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from src.optimization.candidate_generator import Candidate


@dataclass(frozen=True)
class BudgetedBatchSelection:
    selected: tuple[Candidate, ...]
    target_histogram: dict[float, int]
    selected_histogram: dict[float, int]
    objective_trace: tuple[float, ...]
    baseline_mean_squared_perturbation: float
    perturbation_budget_total: float
    actual_total_perturbation: float
    max_relative_perturbation: float

    @property
    def mean_squared_perturbation(self) -> float:
        return self.actual_total_perturbation / len(self.selected) if self.selected else 0.0

    @property
    def histogram_total_variation_distance(self) -> float:
        if not self.selected or not self.target_histogram:
            return 0.0
        support = set(self.target_histogram) | set(self.selected_histogram)
        n_target = sum(self.target_histogram.values())
        n_selected = len(self.selected)
        return 0.5 * sum(
            abs(self.target_histogram.get(k, 0) / n_target
                - self.selected_histogram.get(k, 0) / n_selected)
            for k in support
        )


class DistortionBudgetedBatchOptimizer:
    """Greedy histogram matching under a hard baseline-relative distortion cap."""

    def __init__(self, max_relative_perturbation: float = 0.10):
        if not math.isfinite(max_relative_perturbation) or max_relative_perturbation < 0:
            raise ValueError("max_relative_perturbation must be finite and >= 0")
        self.max_relative_perturbation = max_relative_perturbation

    def optimize(
        self,
        candidate_sets: Sequence[Sequence[Candidate]],
        target_histogram: Mapping[float, int],
        target_bits: Sequence[int],
    ) -> BudgetedBatchSelection:
        if len(candidate_sets) != len(target_bits):
            raise ValueError("candidate_sets and target_bits must have equal length")
        if any(bit not in (0, 1) for bit in target_bits):
            raise ValueError("target bits must be 0 or 1")
        if any(not isinstance(count, int) or count < 0 for count in target_histogram.values()):
            raise ValueError("target histogram counts must be non-negative integers")
        if sum(target_histogram.values()) != len(candidate_sets):
            raise ValueError("target histogram total must equal carrier count")

        feasible_sets = []
        for index, (candidates, bit) in enumerate(zip(candidate_sets, target_bits)):
            feasible = [c for c in candidates
                        if c.decoded_bit == bit and c.post_quantized_bit == bit]
            if not feasible:
                raise ValueError(f"candidate set {index} has no payload-feasible candidate")
            feasible_sets.append(feasible)

        minimum_costs = [min(c.cost.squared_perturbation for c in group) for group in feasible_sets]
        baseline_total = sum(minimum_costs)
        budget_total = baseline_total * (1.0 + self.max_relative_perturbation)
        suffix_minimum = [0.0] * (len(feasible_sets) + 1)
        for i in range(len(feasible_sets) - 1, -1, -1):
            suffix_minimum[i] = suffix_minimum[i + 1] + minimum_costs[i]

        target_total = sum(target_histogram.values())
        target_probs = {key: count / target_total for key, count in target_histogram.items()} if target_total else {}
        counts: Counter[float] = Counter()
        selected = []
        trace = []
        spent = 0.0
        tolerance = max(1e-12, abs(budget_total) * 1e-12)

        for i, feasible in enumerate(feasible_sets):
            remaining_minimum = suffix_minimum[i + 1]
            allowed = [
                c for c in feasible
                if spent + c.cost.squared_perturbation + remaining_minimum <= budget_total + tolerance
            ]
            if not allowed:
                raise RuntimeError(
                    f"distortion budget infeasible at carrier {i}; "
                    "baseline feasibility invariant was violated"
                )
            support = set(target_probs) | set(counts)
            support.update(c.quantized_value for c in allowed)
            best = None
            best_key = None
            for candidate in allowed:
                projected = counts.copy()
                projected[candidate.quantized_value] += 1
                prefix_len = i + 1
                loss = sum(
                    (projected.get(key, 0) - prefix_len * target_probs.get(key, 0.0)) ** 2
                    for key in support
                )
                key = (loss, candidate.cost.squared_perturbation, abs(candidate.value), candidate.value)
                if best_key is None or key < best_key:
                    best, best_key = candidate, key
            assert best is not None and best_key is not None
            selected.append(best)
            counts[best.quantized_value] += 1
            spent += best.cost.squared_perturbation
            trace.append(best_key[0])

        return BudgetedBatchSelection(
            selected=tuple(selected),
            target_histogram=dict(target_histogram),
            selected_histogram=dict(counts),
            objective_trace=tuple(trace),
            baseline_mean_squared_perturbation=baseline_total / len(selected) if selected else 0.0,
            perturbation_budget_total=budget_total,
            actual_total_perturbation=spent,
            max_relative_perturbation=self.max_relative_perturbation,
        )

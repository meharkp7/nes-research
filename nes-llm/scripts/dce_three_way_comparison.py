#!/usr/bin/env python3
"""Matched synthetic comparison of nearest, independent DCE, and batch DCE.

All three methods receive identical seeded cover values, payload bits, candidate
sets, and toy quantizer. This is synthetic algorithm-mechanics evidence only;
it is not a BitsAndBytes NF4 or model-level evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

from scripts.dce_candidate_benchmark import (
    baseline_nearest,
    decode,
    distribution_metrics,
    quantize,
    summarize_method,
)
from src.optimization import CandidateGenerator, CandidateOptimizer, CostWeights
from src.optimization.batch_optimizer import BatchDistributionOptimizer
from src.optimization.budgeted_batch_optimizer import DistortionBudgetedBatchOptimizer
from src.optimization.candidate_generator import Candidate


def run_comparison(
    carriers: int = 2000,
    seed: int = 20261009,
    step: float = 0.25,
    radius_codes: int = 8,
    smoothing_alpha: float = 0.5,
    distribution_weight: float = 0.5,
    perturbation_weight: float = 1.0,
    include_budgeted: bool = False,
    distortion_budget_fraction: float = 0.10,
) -> dict:
    if carriers < 1:
        raise ValueError("carriers must be >= 1")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and > 0")
    if radius_codes < 1:
        raise ValueError("radius_codes must be >= 1")
    if not math.isfinite(smoothing_alpha) or smoothing_alpha <= 0:
        raise ValueError("smoothing_alpha must be finite and > 0")
    if not math.isfinite(distribution_weight) or distribution_weight < 0:
        raise ValueError("distribution_weight must be finite and >= 0")
    if not math.isfinite(perturbation_weight) or perturbation_weight < 0:
        raise ValueError("perturbation_weight must be finite and >= 0")
    if not math.isfinite(distortion_budget_fraction) or distortion_budget_fraction < 0:
        raise ValueError("distortion_budget_fraction must be finite and >= 0")

    rng = random.Random(seed)
    cover = [rng.gauss(0.0, 1.0) for _ in range(carriers)]
    bits = [rng.randrange(2) for _ in range(carriers)]
    cover_codes = [int(round(value / step)) for value in cover]
    cover_histogram = Counter(cover_codes)
    support = range(
        min(cover_codes) - radius_codes,
        max(cover_codes) + radius_codes + 1,
    )
    denom = carriers + smoothing_alpha * len(support)
    probabilities = {
        code: (cover_histogram.get(code, 0) + smoothing_alpha) / denom
        for code in support
    }

    generator = CandidateGenerator()
    weights = CostWeights(
        payload_error=1000.0,
        post_quantization_error=1000.0,
        distribution=distribution_weight,
        perturbation=perturbation_weight,
    )
    independent_optimizer = CandidateOptimizer(weights)
    candidate_sets: list[list[Candidate]] = []
    baseline_choices: list[Candidate] = []
    independent_choices: list[Candidate] = []

    for original, target, nearest_code in zip(cover, bits, cover_codes):
        values = [
            code * step
            for code in range(nearest_code - radius_codes, nearest_code + radius_codes + 1)
        ]

        def distribution_cost(value: float) -> float:
            code = int(round(quantize(value, step) / step))
            return -math.log(probabilities.get(code, smoothing_alpha / denom))

        candidates = generator.generate(
            cover_value=original,
            target_bit=target,
            candidate_values=values,
            quantize=lambda value: quantize(value, step),
            decode=lambda value: decode(value, step),
            distribution_cost=distribution_cost,
        )
        candidate_sets.append(candidates)
        baseline_choices.append(baseline_nearest(candidates, target))
        independent_choices.append(independent_optimizer.optimize(candidates).selected)

    batch_result = BatchDistributionOptimizer(
        distribution_weight=distribution_weight,
        perturbation_weight=perturbation_weight,
    ).optimize(
        candidate_sets=candidate_sets,
        target_histogram={code * step: count for code, count in cover_histogram.items()},
        target_bits=bits,
    )

    methods = {
        "nearest_feasible_baseline": summarize_method(
            cover, bits, baseline_choices, step, smoothing_alpha
        ),
        "independent_dce": summarize_method(
            cover, bits, independent_choices, step, smoothing_alpha
        ),
        "batch_dce_greedy": summarize_method(
            cover, bits, list(batch_result.selected), step, smoothing_alpha
        ),
    }
    budgeted_result = None
    if include_budgeted:
        budgeted_result = DistortionBudgetedBatchOptimizer(
            max_relative_perturbation=distortion_budget_fraction
        ).optimize(
            candidate_sets=candidate_sets,
            target_histogram={code * step: count for code, count in cover_histogram.items()},
            target_bits=bits,
        )
        methods["batch_dce_distortion_budgeted"] = summarize_method(
            cover, bits, list(budgeted_result.selected), step, smoothing_alpha
        )
    baseline = methods["nearest_feasible_baseline"]
    deltas = {}
    for method_name, report in methods.items():
        if method_name == "nearest_feasible_baseline":
            continue
        deltas[method_name] = {
            "ber_after_quantization": report["ber_after_quantization"] - baseline["ber_after_quantization"],
            "mean_squared_perturbation": report["mean_squared_perturbation"] - baseline["mean_squared_perturbation"],
            "histogram_total_variation_distance": (
                report["distribution"]["histogram_total_variation_distance"]
                - baseline["distribution"]["histogram_total_variation_distance"]
            ),
            "histogram_kl_cover_to_embedded_nats": (
                report["distribution"]["histogram_kl_cover_to_embedded_nats"]
                - baseline["distribution"]["histogram_kl_cover_to_embedded_nats"]
            ),
        }
    return {
        "experiment": "matched-nearest-independent-DCE-batch-DCE-synthetic-comparison",
        "status": "SYNTHETIC_MECHANICS_ONLY",
        "seed": seed,
        "parameters": {
            "carriers": carriers,
            "payload_bits": carriers,
            "toy_quantizer_step": step,
            "candidate_radius_codes": radius_codes,
            "smoothing_alpha": smoothing_alpha,
            "distribution_weight": distribution_weight,
            "perturbation_weight": perturbation_weight,
            "include_budgeted": include_budgeted,
            "distortion_budget_fraction": distortion_budget_fraction,
            "candidate_count_per_carrier": 2 * radius_codes + 1,
        },
        "methods": methods,
        "deltas_vs_nearest_feasible": deltas,
        "batch_optimizer": {
            "algorithm": "greedy prefix squared-count mismatch plus perturbation",
            "objective_trace_length": len(batch_result.objective_trace),
            "final_histogram_tv": batch_result.histogram_total_variation_distance,
            "mean_squared_perturbation": batch_result.mean_squared_perturbation,
        },
        "budgeted_batch_optimizer": ({
            "max_relative_perturbation": budgeted_result.max_relative_perturbation,
            "baseline_mean_squared_perturbation": budgeted_result.baseline_mean_squared_perturbation,
            "mean_squared_perturbation": budgeted_result.mean_squared_perturbation,
            "actual_total_perturbation": budgeted_result.actual_total_perturbation,
            "perturbation_budget_total": budgeted_result.perturbation_budget_total,
            "within_budget": budgeted_result.actual_total_perturbation <= budgeted_result.perturbation_budget_total + max(1e-12, budgeted_result.perturbation_budget_total * 1e-12),
            "final_histogram_tv": budgeted_result.histogram_total_variation_distance,
        } if budgeted_result is not None else None),
        "interpretation_limit": (
            "Synthetic scalar quantizer only. Does not establish NF4 compatibility, "
            "model utility preservation, steganographic undetectability, cryptographic "
            "security, or robustness under real checkpoint transformations."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carriers", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--radius-codes", type=int, default=8)
    parser.add_argument("--smoothing-alpha", type=float, default=0.5)
    parser.add_argument("--distribution-weight", type=float, default=0.5)
    parser.add_argument("--perturbation-weight", type=float, default=1.0)
    parser.add_argument("--include-budgeted", action="store_true")
    parser.add_argument("--distortion-budget-fraction", type=float, default=0.10)
    parser.add_argument("--output", default="", help="Optional JSON path; refuses to overwrite.")
    args = parser.parse_args()
    report = run_comparison(
        carriers=args.carriers,
        seed=args.seed,
        step=args.step,
        radius_codes=args.radius_codes,
        smoothing_alpha=args.smoothing_alpha,
        distribution_weight=args.distribution_weight,
        perturbation_weight=args.perturbation_weight,
        include_budgeted=args.include_budgeted,
        distortion_budget_fraction=args.distortion_budget_fraction,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).expanduser()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Report: {output.resolve()}")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

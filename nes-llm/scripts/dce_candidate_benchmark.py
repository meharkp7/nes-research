#!/usr/bin/env python3
"""Controlled synthetic benchmark: nearest feasible candidate vs DCE.

This is an algorithm-mechanics benchmark with a toy scalar quantizer. It is
not a BitsAndBytes NF4 experiment and must not be presented as model-level
evidence. Both methods see the same cover values, payload bits and candidate
sets. The report measures post-quantization BER, perturbation, and empirical
distribution shifts on the complete synthetic carrier vector.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from src.optimization import CandidateGenerator, CandidateOptimizer, CostWeights
from src.optimization.candidate_generator import Candidate


def quantize(value: float, step: float) -> float:
    """Toy uniform scalar quantizer, explicitly not an NF4 implementation."""
    return round(value / step) * step


def decode(value: float, step: float) -> int:
    """Encode a bit as parity of the quantized integer code index."""
    return int(round(value / step)) % 2


def smoothed_distribution(values: Sequence[float], step: float, support: Sequence[int], alpha: float) -> List[float]:
    counts = Counter(int(round(value / step)) for value in values)
    denom = len(values) + alpha * len(support)
    return [(counts.get(index, 0) + alpha) / denom for index in support]


def distribution_metrics(cover: Sequence[float], embedded: Sequence[float], step: float, alpha: float) -> dict:
    all_indices = [int(round(value / step)) for value in cover + embedded]
    low, high = min(all_indices), max(all_indices)
    support = list(range(low, high + 1))
    p = smoothed_distribution(cover, step, support, alpha)
    q = smoothed_distribution(embedded, step, support, alpha)
    tv = 0.5 * sum(abs(a - b) for a, b in zip(p, q))
    kl_pq = sum(a * math.log(a / b) for a, b in zip(p, q))
    kl_qp = sum(b * math.log(b / a) for a, b in zip(p, q))
    return {
        "histogram_total_variation_distance": tv,
        "histogram_kl_cover_to_embedded_nats": kl_pq,
        "histogram_kl_embedded_to_cover_nats": kl_qp,
        "histogram_support_bins": len(support),
        "dirichlet_smoothing_alpha": alpha,
    }


def baseline_nearest(candidates: Sequence[Candidate], target_bit: int) -> Candidate:
    feasible = [
        candidate for candidate in candidates
        if candidate.decoded_bit == target_bit
        and candidate.post_quantized_bit == target_bit
    ]
    if not feasible:
        raise ValueError("candidate set contains no payload-correct, post-quantization-correct candidate")
    return min(feasible, key=lambda candidate: (candidate.cost.squared_perturbation, candidate.value))


def summarize_method(cover: Sequence[float], bits: Sequence[int], chosen: Sequence[Candidate], step: float, alpha: float) -> dict:
    embedded = [candidate.value for candidate in chosen]
    quantized = [quantize(value, step) for value in embedded]
    decoded = [decode(value, step) for value in quantized]
    errors = sum(actual != expected for actual, expected in zip(decoded, bits))
    perturbation_sq = sum((after - before) ** 2 for before, after in zip(cover, embedded))
    return {
        "carrier_count": len(cover),
        "payload_bits": len(bits),
        "bit_errors_after_quantization": errors,
        "ber_after_quantization": errors / len(bits) if bits else 0.0,
        "mean_squared_perturbation": perturbation_sq / len(cover) if cover else 0.0,
        "rmse_perturbation": math.sqrt(perturbation_sq / len(cover)) if cover else 0.0,
        "changed_carrier_fraction": sum(a != b for a, b in zip(cover, embedded)) / len(cover) if cover else 0.0,
        "distribution": distribution_metrics(list(cover), quantized, step, alpha),
    }


def run_benchmark(
    carriers: int = 2000,
    seed: int = 20261009,
    step: float = 0.25,
    radius_codes: int = 8,
    smoothing_alpha: float = 0.5,
    distribution_weight: float = 0.5,
    perturbation_weight: float = 1.0,
) -> dict:
    if carriers < 1:
        raise ValueError("carriers must be >= 1")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and > 0")
    if radius_codes < 1:
        raise ValueError("radius_codes must be >= 1")
    if not math.isfinite(smoothing_alpha) or smoothing_alpha <= 0:
        raise ValueError("smoothing_alpha must be finite and > 0")

    rng = random.Random(seed)
    cover = [rng.gauss(0.0, 1.0) for _ in range(carriers)]
    bits = [rng.randrange(2) for _ in range(carriers)]
    cover_codes = [int(round(value / step)) for value in cover]
    histogram = Counter(cover_codes)
    support = range(min(cover_codes) - radius_codes, max(cover_codes) + radius_codes + 1)
    denom = len(cover_codes) + smoothing_alpha * len(support)
    probabilities = {
        index: (histogram.get(index, 0) + smoothing_alpha) / denom
        for index in support
    }

    generator = CandidateGenerator()
    weights = CostWeights(
        payload_error=1_000.0,
        post_quantization_error=1_000.0,
        distribution=distribution_weight,
        perturbation=perturbation_weight,
    )
    optimizer = CandidateOptimizer(weights)
    baseline_choices: List[Candidate] = []
    dce_choices: List[Candidate] = []
    candidate_counts = []

    for original, target, nearest_code in zip(cover, bits, cover_codes):
        candidate_values = [
            index * step
            for index in range(nearest_code - radius_codes, nearest_code + radius_codes + 1)
        ]

        def distribution_cost(value: float) -> float:
            index = int(round(quantize(value, step) / step))
            return -math.log(probabilities.get(index, smoothing_alpha / denom))

        candidates = generator.generate(
            cover_value=original,
            target_bit=target,
            candidate_values=candidate_values,
            quantize=lambda value: quantize(value, step),
            decode=lambda value: decode(value, step),
            distribution_cost=distribution_cost,
        )
        candidate_counts.append(len(candidates))
        baseline_choices.append(baseline_nearest(candidates, target))
        dce_choices.append(optimizer.optimize(candidates).selected)

    baseline = summarize_method(cover, bits, baseline_choices, step, smoothing_alpha)
    dce = summarize_method(cover, bits, dce_choices, step, smoothing_alpha)
    return {
        "experiment": "DCE-vs-nearest-feasible-synthetic-scalar-quantizer",
        "status": "SYNTHETIC_MECHANICS_ONLY",
        "seed": seed,
        "parameters": {
            "carriers": carriers,
            "payload_bits": len(bits),
            "toy_quantizer_step": step,
            "candidate_radius_codes": radius_codes,
            "smoothing_alpha": smoothing_alpha,
            "cost_weights": {
                "payload_error": weights.payload_error,
                "post_quantization_error": weights.post_quantization_error,
                "distribution_proxy": weights.distribution,
                "squared_perturbation": weights.perturbation,
            },
            "candidate_count_min": min(candidate_counts),
            "candidate_count_mean": sum(candidate_counts) / len(candidate_counts),
            "candidate_count_max": max(candidate_counts),
        },
        "nearest_feasible_baseline": baseline,
        "dce": dce,
        "dce_minus_baseline": {
            "ber_after_quantization": dce["ber_after_quantization"] - baseline["ber_after_quantization"],
            "mean_squared_perturbation": dce["mean_squared_perturbation"] - baseline["mean_squared_perturbation"],
            "histogram_total_variation_distance": (
                dce["distribution"]["histogram_total_variation_distance"]
                - baseline["distribution"]["histogram_total_variation_distance"]
            ),
            "histogram_kl_cover_to_embedded_nats": (
                dce["distribution"]["histogram_kl_cover_to_embedded_nats"]
                - baseline["distribution"]["histogram_kl_cover_to_embedded_nats"]
            ),
        },
        "interpretation_limit": (
            "Synthetic scalar quantizer only. These results do not establish NF4 compatibility, "
            "model utility preservation, steganographic undetectability, cryptographic security, "
            "or robustness under real checkpoint transformations."
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
    parser.add_argument("--output", default="", help="Optional JSON report path; refuses to overwrite.")
    args = parser.parse_args()
    report = run_benchmark(
        carriers=args.carriers,
        seed=args.seed,
        step=args.step,
        radius_codes=args.radius_codes,
        smoothing_alpha=args.smoothing_alpha,
        distribution_weight=args.distribution_weight,
        perturbation_weight=args.perturbation_weight,
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

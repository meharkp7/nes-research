#!/usr/bin/env python3
"""Repeated-seed summary for the matched synthetic three-way DCE benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.dce_three_way_comparison import run_comparison


METRICS = (
    ("ber_after_quantization", lambda report: report["ber_after_quantization"]),
    ("mean_squared_perturbation", lambda report: report["mean_squared_perturbation"]),
    ("histogram_total_variation_distance", lambda report: report["distribution"]["histogram_total_variation_distance"]),
    ("histogram_kl_cover_to_embedded_nats", lambda report: report["distribution"]["histogram_kl_cover_to_embedded_nats"]),
)


def summarize(values: list[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize an empty list")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "n": len(values),
        "mean": mean,
        "population_std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def run_multiseed(
    carriers: int = 2000,
    seeds: tuple[int, ...] = (20261009, 20261010, 20261011, 20261012, 20261013),
    step: float = 0.25,
    radius_codes: int = 8,
    smoothing_alpha: float = 0.5,
    distribution_weight: float = 0.5,
    perturbation_weight: float = 1.0,
) -> dict:
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    runs = [
        run_comparison(
            carriers=carriers,
            seed=seed,
            step=step,
            radius_codes=radius_codes,
            smoothing_alpha=smoothing_alpha,
            distribution_weight=distribution_weight,
            perturbation_weight=perturbation_weight,
        )
        for seed in seeds
    ]
    method_names = list(runs[0]["methods"])
    aggregate = {}
    win_counts = {}
    for method_name in method_names:
        aggregate[method_name] = {}
        win_counts[method_name] = {}
        for metric_name, getter in METRICS:
            values = [getter(run["methods"][method_name]) for run in runs]
            aggregate[method_name][metric_name] = summarize(values)
            if method_name == "nearest_feasible_baseline":
                win_counts[method_name][metric_name] = len(values)
            else:
                baseline_values = [
                    getter(run["methods"]["nearest_feasible_baseline"]) for run in runs
                ]
                # Lower is better; count strict wins, not ties.
                win_counts[method_name][metric_name] = sum(
                    candidate < baseline for candidate, baseline in zip(values, baseline_values)
                )
    paired_deltas = {}
    for method_name in method_names:
        if method_name == "nearest_feasible_baseline":
            continue
        paired_deltas[method_name] = {}
        for metric_name, _getter in METRICS:
            values = [
                run["deltas_vs_nearest_feasible"][method_name][metric_name]
                for run in runs
            ]
            paired_deltas[method_name][metric_name] = summarize(values)
    return {
        "experiment": "repeated-seed-matched-three-way-DCE-synthetic-comparison",
        "status": "SYNTHETIC_MECHANICS_ONLY",
        "parameters": {
            "carriers": carriers,
            "seeds": list(seeds),
            "toy_quantizer_step": step,
            "candidate_radius_codes": radius_codes,
            "smoothing_alpha": smoothing_alpha,
            "distribution_weight": distribution_weight,
            "perturbation_weight": perturbation_weight,
        },
        "aggregate_by_method": aggregate,
        "strict_wins_vs_baseline": win_counts,
        "paired_deltas_vs_baseline": paired_deltas,
        "runs": runs,
        "interpretation_limit": (
            "Synthetic scalar quantizer only. Repeated seeds do not establish NF4 "
            "compatibility, model utility preservation, undetectability, cryptographic "
            "security, or checkpoint robustness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carriers", type=int, default=2000)
    parser.add_argument("--seeds", default="20261009,20261010,20261011,20261012,20261013")
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--radius-codes", type=int, default=8)
    parser.add_argument("--smoothing-alpha", type=float, default=0.5)
    parser.add_argument("--distribution-weight", type=float, default=0.5)
    parser.add_argument("--perturbation-weight", type=float, default=1.0)
    parser.add_argument("--output", default="", help="Optional JSON path; refuses to overwrite.")
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    report = run_multiseed(
        carriers=args.carriers,
        seeds=seeds,
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

#!/usr/bin/env python3
"""Repeated-seed DCE weight sweep on the synthetic scalar-quantizer benchmark.

This is a diagnostic for tuning behavior, not evidence of NF4 compatibility or
steganographic security. Every (seed, weight) run compares DCE with a matched
nearest-feasible baseline on identical synthetic inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence

from scripts.dce_candidate_benchmark import run_benchmark


METRICS = (
    "ber_after_quantization",
    "mean_squared_perturbation",
    "histogram_total_variation_distance",
    "histogram_kl_cover_to_embedded_nats",
)


def parse_csv_ints(raw: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("provide at least one unique integer")
    return values


def parse_csv_floats(raw: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values or len(values) != len(set(values)) or any(not math.isfinite(v) or v < 0 for v in values):
        raise argparse.ArgumentTypeError("provide unique finite non-negative weights")
    return values


def method_metric(report: dict, method: str, metric: str) -> float:
    if metric.startswith("histogram_"):
        return report[method]["distribution"][metric]
    return report[method][metric]


def summarize(values: Sequence[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "population_stddev": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def run_sweep(
    carriers: int = 2000,
    seeds: Iterable[int] = (20261009, 20261010, 20261011, 20261012, 20261013),
    distribution_weights: Iterable[float] = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    step: float = 0.25,
    radius_codes: int = 8,
    smoothing_alpha: float = 0.5,
    perturbation_weight: float = 1.0,
) -> dict:
    seed_list = list(seeds)
    weight_list = list(distribution_weights)
    if carriers < 1:
        raise ValueError("carriers must be >= 1")
    if not seed_list or len(seed_list) != len(set(seed_list)):
        raise ValueError("seeds must be non-empty and unique")
    if not weight_list or len(weight_list) != len(set(weight_list)):
        raise ValueError("distribution weights must be non-empty and unique")
    if any(not math.isfinite(weight) or weight < 0 for weight in weight_list):
        raise ValueError("distribution weights must be finite and non-negative")

    results = []
    for weight in weight_list:
        paired = []
        for seed in seed_list:
            report = run_benchmark(
                carriers=carriers,
                seed=seed,
                step=step,
                radius_codes=radius_codes,
                smoothing_alpha=smoothing_alpha,
                distribution_weight=weight,
                perturbation_weight=perturbation_weight,
            )
            row = {"seed": seed, "distribution_weight": weight}
            row["baseline"] = {
                metric: method_metric(report, "nearest_feasible_baseline", metric)
                for metric in METRICS
            }
            row["dce"] = {
                metric: method_metric(report, "dce", metric)
                for metric in METRICS
            }
            row["dce_minus_baseline"] = {
                metric: row["dce"][metric] - row["baseline"][metric]
                for metric in METRICS
            }
            paired.append(row)

        summary = {}
        for method in ("baseline", "dce", "dce_minus_baseline"):
            summary[method] = {
                metric: summarize([row[method][metric] for row in paired])
                for metric in METRICS
            }
        results.append({
            "distribution_weight": weight,
            "runs": paired,
            "summary": summary,
            "beats_baseline_on_all_seeds": {
                metric: all(row["dce_minus_baseline"][metric] <= 0 for row in paired)
                for metric in METRICS
            },
        })

    return {
        "experiment": "DCE repeated-seed distribution-weight sweep",
        "status": "SYNTHETIC_MECHANICS_ONLY",
        "selection_protocol": (
            "All listed weights and seeds are reported; no best weight is selected "
            "from a held-out test set. Treat this sweep as exploratory, not confirmatory."
        ),
        "parameters": {
            "carriers_per_run": carriers,
            "seeds": seed_list,
            "distribution_weights": weight_list,
            "toy_quantizer_step": step,
            "candidate_radius_codes": radius_codes,
            "smoothing_alpha": smoothing_alpha,
            "perturbation_weight": perturbation_weight,
            "runs_total": len(seed_list) * len(weight_list),
        },
        "results_by_weight": results,
        "interpretation_limit": (
            "Synthetic scalar quantizer only. Zero BER is expected by construction "
            "for feasible candidates. Results do not establish NF4 compatibility, "
            "model utility preservation, undetectability, cryptographic security, "
            "or robustness under checkpoint transformations."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carriers", type=int, default=2000)
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("20261009,20261010,20261011,20261012,20261013"))
    parser.add_argument("--distribution-weights", type=parse_csv_floats, default=parse_csv_floats("0,0.1,0.5,1,2,5,10"))
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--radius-codes", type=int, default=8)
    parser.add_argument("--smoothing-alpha", type=float, default=0.5)
    parser.add_argument("--perturbation-weight", type=float, default=1.0)
    parser.add_argument("--output", default="", help="Optional JSON report path; refuses to overwrite.")
    args = parser.parse_args()
    report = run_sweep(
        carriers=args.carriers,
        seeds=args.seeds,
        distribution_weights=args.distribution_weights,
        step=args.step,
        radius_codes=args.radius_codes,
        smoothing_alpha=args.smoothing_alpha,
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

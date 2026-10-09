#!/usr/bin/env python3
"""Sweep hard distortion budgets for batch DCE on matched synthetic seeds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.dce_three_way_comparison import run_comparison


def summarize(values: list[float]) -> dict:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "n": len(values),
        "mean": mean,
        "population_std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def run_budget_sweep(
    carriers: int = 2000,
    seeds: tuple[int, ...] = (20261009, 20261010, 20261011, 20261012, 20261013),
    budgets: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25),
) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be non-empty and unique")
    if any(not math.isfinite(b) or b < 0 for b in budgets):
        raise ValueError("budgets must be finite and >= 0")
    runs = []
    for budget in budgets:
        for seed in seeds:
            report = run_comparison(
                carriers=carriers,
                seed=seed,
                include_budgeted=True,
                distortion_budget_fraction=budget,
            )
            baseline = report["methods"]["nearest_feasible_baseline"]
            selected = report["methods"]["batch_dce_distortion_budgeted"]
            budget_detail = report["budgeted_batch_optimizer"]
            runs.append({
                "seed": seed,
                "budget_fraction": budget,
                "baseline": {
                    "mean_squared_perturbation": baseline["mean_squared_perturbation"],
                    "histogram_total_variation_distance": baseline["distribution"]["histogram_total_variation_distance"],
                    "histogram_kl_cover_to_embedded_nats": baseline["distribution"]["histogram_kl_cover_to_embedded_nats"],
                },
                "budgeted": {
                    "mean_squared_perturbation": selected["mean_squared_perturbation"],
                    "histogram_total_variation_distance": selected["distribution"]["histogram_total_variation_distance"],
                    "histogram_kl_cover_to_embedded_nats": selected["distribution"]["histogram_kl_cover_to_embedded_nats"],
                    "ber_after_quantization": selected["ber_after_quantization"],
                },
                "paired_deltas": {
                    "mean_squared_perturbation": selected["mean_squared_perturbation"] - baseline["mean_squared_perturbation"],
                    "histogram_total_variation_distance": selected["distribution"]["histogram_total_variation_distance"] - baseline["distribution"]["histogram_total_variation_distance"],
                    "histogram_kl_cover_to_embedded_nats": selected["distribution"]["histogram_kl_cover_to_embedded_nats"] - baseline["distribution"]["histogram_kl_cover_to_embedded_nats"],
                },
                "budget_check": budget_detail,
            })
    summary = {}
    for budget in budgets:
        selected_runs = [run for run in runs if run["budget_fraction"] == budget]
        summary[str(budget)] = {
            "seed_count": len(selected_runs),
            "within_budget_count": sum(run["budget_check"]["within_budget"] for run in selected_runs),
            "mean_relative_perturbation_increase": summarize([
                run["paired_deltas"]["mean_squared_perturbation"] /
                run["baseline"]["mean_squared_perturbation"]
                if run["baseline"]["mean_squared_perturbation"] else 0.0
                for run in selected_runs
            ]),
            "paired_delta_mean_squared_perturbation": summarize([
                run["paired_deltas"]["mean_squared_perturbation"] for run in selected_runs
            ]),
            "paired_delta_histogram_tv": summarize([
                run["paired_deltas"]["histogram_total_variation_distance"] for run in selected_runs
            ]),
            "paired_delta_cover_to_embedded_kl": summarize([
                run["paired_deltas"]["histogram_kl_cover_to_embedded_nats"] for run in selected_runs
            ]),
            "strict_tv_wins_vs_baseline": sum(
                run["paired_deltas"]["histogram_total_variation_distance"] < 0 for run in selected_runs
            ),
            "strict_kl_wins_vs_baseline": sum(
                run["paired_deltas"]["histogram_kl_cover_to_embedded_nats"] < 0 for run in selected_runs
            ),
        }
    return {
        "experiment": "distortion-budgeted-batch-DCE-synthetic-budget-sweep",
        "status": "SYNTHETIC_MECHANICS_ONLY",
        "parameters": {"carriers": carriers, "seeds": list(seeds), "budget_fractions": list(budgets)},
        "summary_by_budget_fraction": summary,
        "runs": runs,
        "interpretation_limit": (
            "Synthetic scalar quantizer only. Does not establish NF4 compatibility, "
            "model utility preservation, steganographic undetectability, cryptographic "
            "security, or checkpoint robustness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carriers", type=int, default=2000)
    parser.add_argument("--seeds", default="20261009,20261010,20261011,20261012,20261013")
    parser.add_argument("--budgets", default="0,0.05,0.10,0.25",
                        help="Comma-separated maximum relative perturbation increases")
    parser.add_argument("--output", default="", help="Optional JSON path; refuses to overwrite.")
    args = parser.parse_args()
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    budgets = tuple(float(x.strip()) for x in args.budgets.split(",") if x.strip())
    report = run_budget_sweep(args.carriers, seeds, budgets)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).expanduser()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\\n", encoding="utf-8")
        print(f"Report: {output.resolve()}")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

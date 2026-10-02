#!/usr/bin/env python
"""
One-command NES experiment suite.

Runs the full experiment sequence, skipping work that has already
completed, writes structured artifacts for everything, and regenerates
the final tables and summary.

Usage
-----
    # everything, every registered model
    python run_nes_experiments.py

    # one model first (recommended: validate the pipeline on 3B)
    python run_nes_experiments.py --models Qwen/Qwen2.5-3B

    # a subset of experiments
    python run_nes_experiments.py --models Qwen/Qwen2.5-3B --exp exp1 exp3

    # audit only, run nothing
    python run_nes_experiments.py --audit

    # redo work that already produced a result (archives the old one)
    python run_nes_experiments.py --models Qwen/Qwen2.5-3B --exp exp6 --force

Exit codes
----------
    0   suite completed
    1   environment or configuration error

The suite does not exit non-zero just because a gate FAILed. A failing
experiment is valid evidence, and treating it as a crash would make
"all gates pass" the only terminating condition, which invites
threshold-fiddling until everything passes.
"""

import argparse
import os
import sys
from pathlib import Path

# Resolve imports when launched from either the repo root or nes-llm/.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


DEFAULT_MODEL = "Qwen/Qwen2.5-3B"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="NES multi-model experiment suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_MODEL],
        help=(
            "Model ids to run. Defaults to a single small model so the "
            "first run validates the pipeline rather than spending hours "
            "on seven model loads."
        ),
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        default=None,
        help=(
            "Experiments to run (exp1..exp9). Default: all, in "
            "dependency order."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rerun experiments that already produced a result. Previous "
            "artifacts are archived, not overwritten in place."
        ),
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Report the manifest and matrix, then exit without running.",
    )
    parser.add_argument(
        "--train-neural-detector",
        action="store_true",
        help=(
            "Allow the Exp7 neural detector to train if no saved result "
            "exists. Off by default: it is an expensive job and must "
            "not be triggered as a side effect of a cross-model run."
        ),
    )
    parser.add_argument(
        "--download-exp9",
        action="store_true",
        help=(
            "Allow Exp9 to download GPTQ/AWQ checkpoints (multi-GB). "
            "Off by default."
        ),
    )
    parser.add_argument(
        "--recompute-exp5",
        action="store_true",
        help=(
            "Recompute Exp5's three-way PPL measurement instead of "
            "reusing the recorded result. Deliberately separate from "
            "--force: this is an hours-long MPS run, so a routine cell "
            "refresh must not trigger it."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Suite-wide random seed (default: 42).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip regenerating tables and the summary.",
    )

    return parser.parse_args(argv)


def print_environment() -> None:
    from src.experiments.environment_check import assert_ready

    info = assert_ready()

    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    print(f"  device            : {info['device']}")
    print(f"  cuda / mps        : {info['cuda_available']} / {info['mps_available']}")

    for name, version in info["software_versions"].items():
        print(f"  {name:<18}: {version}")

    for note in info["platform_notes"]:
        print(f"  note: {note}")

    print()


def print_audit() -> None:
    from src.experiments import manifest as manifest_mod
    from src.reporting.aggregate_results import aggregate

    data = aggregate()
    manifest = manifest_mod.load()

    print("=" * 70)
    print("AUDIT — CURRENT MANIFEST")
    print("=" * 70)

    coverage = manifest_mod.coverage(manifest)
    for status, count in coverage.items():
        if count:
            print(f"  {status:<18}: {count}")

    matrix = data["matrix"]
    if matrix:
        experiments = sorted(
            {
                experiment
                for row in matrix.values()
                for experiment in row
            }
        )
        print()
        print("  " + "MODEL".ljust(42) + "  " + "  ".join(experiments))
        for model_id, row in matrix.items():
            print(
                "  "
                + model_id.ljust(42)
                + "  "
                + "  ".join(
                    row.get(experiment, "-") for experiment in experiments
                )
            )

    missing = data["missing_cells"]
    print()
    print(f"  missing cells: {len(missing)}")
    for item in missing[:20]:
        print(f"    {item['model_id']:<40} {item['experiment']}")
    if len(missing) > 20:
        print(f"    ... {len(missing) - 20} more")

    print()


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        print_environment()
    except RuntimeError as exc:
        print(f"\nENVIRONMENT ERROR: {exc}", file=sys.stderr)
        return 1

    if args.audit:
        print_audit()
        return 0

    from src.experiments.runner import Runner

    try:
        runner = Runner(
            models=args.models,
            experiments=args.exp,
            force=args.force,
            seed=args.seed,
            allow_neural_training=args.train_neural_detector,
            exp9_download=args.download_exp9,
            recompute_exp5=args.recompute_exp5,
        )
    except KeyError as exc:
        print(f"\nCONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("NES EXPERIMENT SUITE")
    print("=" * 70)
    print(f"  models      : {', '.join(args.models)}")
    print(f"  experiments : {', '.join(runner.experiments)}")
    print(f"  seed        : {args.seed}")
    print(f"  force       : {args.force}")
    print()

    outcome = runner.run()

    print("\n" + "=" * 70)
    print("SUITE COMPLETE")
    print("=" * 70)
    print(f"  duration : {outcome['duration_seconds']}s")
    print(f"  executed : {len(outcome['executed'])}")
    print(f"  skipped  : {len(outcome['skipped'])} (already completed)")
    print()

    for status, count in outcome["coverage"].items():
        if count:
            print(f"  {status:<18}: {count}")

    if not args.no_report:
        from src.reporting.generate_summary import generate_all

        print()
        print("GENERATING REPORTS")
        for name, path in generate_all().items():
            print(f"  {name:<18}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
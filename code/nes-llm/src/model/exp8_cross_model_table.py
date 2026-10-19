"""
Experiment 8 — Cross-Model Ablation Table

Creates a unified results table across all tested models.
"""

import os
import json

# Enable CPU fallback for unsupported MPS operations.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


from src.model.model_loader import (
    load_model_pair,
    extract_residuals
)

from src.evaluation.nes_benchmark import NESBenchmark


# ============================================================
# MODELS
# ============================================================

MODELS = [
    ("meta-llama/Llama-3.1-8B", "llama", 32),
    ("mistralai/Mistral-7B-v0.3", "mistral", 32),
    ("google/gemma-2-9b", "gemma", 42),
    ("Qwen/Qwen2.5-7B", "qwen", 28),
    ("Qwen/Qwen2.5-3B", "qwen", 36),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", 22),
    ("microsoft/Phi-3-mini-4k-instruct", "phi3", 32),
]


# ============================================================
# RESULTS
# ============================================================

results = {}


# ============================================================
# RUN BENCHMARK FOR EACH MODEL
# Time Complexity: O(M * benchmark_cost)
# M = number of models
# ============================================================

for model_id, family, n_layers in MODELS:

    print("\n" + "=" * 70)
    print(f"Model: {model_id}")
    print(f"Family: {family}")
    print(f"Layers: {n_layers}")
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # Load NF4 + FP16 models
        # Time Complexity: O(model size)
        # ----------------------------------------------------

        nf4_model, fp16_model, _ = load_model_pair(
            model_id
        )

        # ----------------------------------------------------
        # Extract residuals
        # Time Complexity: O(L * W)
        # ----------------------------------------------------

        residuals = extract_residuals(
            nf4_model,
            fp16_model,
            family
        )

        # ----------------------------------------------------
        # Initialize benchmark
        # ----------------------------------------------------

        benchmark = NESBenchmark(
            n_layers=n_layers,
            payload_bits=50_000,
            verbose=False,
            residuals=residuals
        )

        # ----------------------------------------------------
        # Run all benchmark gates
        # Time Complexity: Depends on all benchmark tests
        # ----------------------------------------------------

        results[model_id] = benchmark.run_all()

        print(
            f"\nCompleted: {model_id}"
        )

    except Exception as e:

        print(
            f"\nFAILED: {model_id}"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        results[model_id] = {
            "status": "FAILED",
            "error": str(e)
        }


# ============================================================
# SAVE RESULTS
# Time Complexity: O(number of result entries)
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

output_file = (
    "results/cross_model_table.json"
)

with open(
    output_file,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CROSS-MODEL RESULTS")
print("=" * 70)

for model_id, result in results.items():

    print(f"\n{model_id}")
    print(result)

print(
    f"\nResults saved to: {output_file}"
)